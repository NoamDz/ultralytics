# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""
Dental-specific loss functions for YOLOv11-seg tooth detection and segmentation.

This module provides specialized losses for dental imaging:
1. Anatomical Constraint Loss - Penalizes anatomically implausible tooth detections
2. Generalized Surface Loss (GSL) - Boundary-focused loss from arXiv:2302.03868

Supports FDI tooth numbering system with 32 classes (panoramic) or 24 classes (bitewing).

References:
    Celaya et al., "A Generalized Surface Loss for Reducing the Hausdorff Distance
    in Medical Imaging Segmentation", arXiv:2302.03868, 2024.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.loss import v8SegmentationLoss


class DistanceTransform(nn.Module):
    """
    GPU-native Distance Transform Map (DTM) computation using morphological operations.

    Computes signed distance transform where:
    - Positive values = exterior (outside mask)
    - Zero = boundary
    - Negative values = interior (inside mask)

    This is an iterative approximation that converges to true Euclidean distance.
    """

    def __init__(self, max_iterations: int = 30):
        """
        Initialize DistanceTransform.

        Args:
            max_iterations (int): Maximum iterations for distance computation.
                Higher = more accurate for large objects, but slower.
        """
        super().__init__()
        self.max_iterations = max_iterations

        # 3x3 kernel for morphological operations (8-connectivity)
        kernel = torch.ones(1, 1, 3, 3)
        self.register_buffer("kernel", kernel)

    def forward(self, masks: torch.Tensor) -> torch.Tensor:
        """
        Compute signed distance transform for batch of masks.

        Args:
            masks (torch.Tensor): Binary masks, shape (N, H, W) or (N, 1, H, W).

        Returns:
            (torch.Tensor): Signed DTM, same shape as input.
                Positive outside, zero on boundary, negative inside.
        """
        # Ensure 4D: (N, 1, H, W)
        if masks.dim() == 3:
            masks = masks.unsqueeze(1)
        elif masks.dim() == 2:
            masks = masks.unsqueeze(0).unsqueeze(0)

        masks = masks.float()

        # Compute interior distance (negative inside)
        interior_dist = self._compute_interior_distance(masks)

        # Compute exterior distance (positive outside)
        exterior_dist = self._compute_exterior_distance(masks)

        # Combine: negative inside, positive outside, zero on boundary
        dtm = exterior_dist - interior_dist

        # Remove channel dimension if input was 3D
        return dtm.squeeze(1)

    def _compute_interior_distance(self, masks: torch.Tensor) -> torch.Tensor:
        """
        Compute distance from interior points to boundary (erosion-based).

        No early exit to avoid GPU synchronization. Operations on zero tensors are fast.
        """
        distance = torch.zeros_like(masks)
        current = masks.clone()

        for i in range(1, self.max_iterations + 1):
            # Erosion: min pooling (or equivalently, -max(-x))
            eroded = -F.max_pool2d(-current, 3, stride=1, padding=1)

            # Pixels that were removed in this iteration are at distance i
            removed = current - eroded
            distance = distance + removed * i

            # Update for next iteration
            current = eroded

        return distance

    def _compute_exterior_distance(self, masks: torch.Tensor) -> torch.Tensor:
        """
        Compute distance from exterior points to boundary (dilation-based).

        No early exit to avoid GPU synchronization. Operations on zero tensors are fast.
        """
        distance = torch.zeros_like(masks)
        current = masks.clone()
        exterior_mask = 1.0 - masks  # Points outside the mask

        for i in range(1, self.max_iterations + 1):
            # Dilation: max pooling
            dilated = F.max_pool2d(current, 3, stride=1, padding=1)

            # Pixels that were added in this iteration (and are in exterior)
            added = (dilated - current) * exterior_mask
            distance = distance + added * i

            # Update exterior mask (remove newly covered areas)
            exterior_mask = exterior_mask - added

            # Update for next iteration
            current = dilated

        return distance


class GeneralizedSurfaceLoss(nn.Module):
    """
    Generalized Surface Loss (GSL) for boundary-focused segmentation.

    From: "A Generalized Surface Loss for Reducing the Hausdorff Distance
    in Medical Imaging Segmentation" (arXiv:2302.03868)

    Formula (Equation 12):
        L_gsl = 1 - (Σ w_k Σ (D_i * (1 - (T_i + P_i)))²) / (Σ w_k Σ (D_i)²)

    Key properties:
    - Bounded in [0, 1] - won't dominate region-based loss
    - Only requires DTM from ground truth (not predictions)
    - Pre-computed class weights for class imbalance

    Attributes:
        dtm_transform (DistanceTransform): Module for computing DTMs.
        class_weights (torch.Tensor | None): Pre-computed class weights.
    """

    def __init__(
        self,
        max_dtm_iterations: int = 30,
        class_weights: torch.Tensor | None = None,
    ):
        """
        Initialize GeneralizedSurfaceLoss.

        Args:
            max_dtm_iterations (int): Max iterations for DTM computation.
            class_weights (torch.Tensor, optional): Pre-computed class weights.
                Shape (num_classes,). If None, uniform weights are used.
        """
        super().__init__()
        self.dtm_transform = DistanceTransform(max_iterations=max_dtm_iterations)

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(
        self,
        pred_masks: torch.Tensor,
        gt_masks: torch.Tensor,
        class_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute GSL loss for batch of mask pairs.

        Args:
            pred_masks (torch.Tensor): Predicted masks after sigmoid, shape (N, H, W).
            gt_masks (torch.Tensor): Ground truth binary masks, shape (N, H, W).
            class_indices (torch.Tensor, optional): Class index for each mask, shape (N,).
                Used for class-weighted loss. If None, uniform weights.

        Returns:
            (torch.Tensor): Scalar GSL loss in [0, 1].
        """
        if pred_masks.numel() == 0:
            return torch.tensor(0.0, device=pred_masks.device)

        # Compute DTM for ground truth masks (NOT predictions - key efficiency)
        dtm = self.dtm_transform(gt_masks)

        # Ensure same shape
        if dtm.shape != pred_masks.shape:
            dtm = dtm.view_as(pred_masks)

        # GSL formula: D * (1 - (T + P))
        # When P matches T: (1 - (T + P)) = (1 - 2T) = -1 inside, +1 outside
        # Multiplied by D: recovers |D| for perfect prediction
        term = dtm * (1.0 - (gt_masks + pred_masks))

        # Squared term (numerator component)
        term_squared = term ** 2

        # Denominator: sum of D²
        dtm_squared = dtm ** 2

        # Apply class weights if provided
        if self.class_weights is not None and class_indices is not None:
            weights = self.class_weights[class_indices]  # (N,)
            weights = weights.view(-1, 1, 1)  # (N, 1, 1) for broadcasting

            numerator = (weights * term_squared.view(term_squared.shape[0], -1).sum(dim=1, keepdim=True)).sum()
            denominator = (weights * dtm_squared.view(dtm_squared.shape[0], -1).sum(dim=1, keepdim=True)).sum()
        else:
            # Uniform weights - simple sum
            numerator = term_squared.sum()
            denominator = dtm_squared.sum()

        # GSL = 1 - numerator/denominator
        # Add epsilon to avoid division by zero
        gsl = 1.0 - numerator / (denominator + 1e-8)

        return gsl

    @staticmethod
    def compute_class_weights(class_pixel_counts: dict[int, int]) -> torch.Tensor:
        """
        Compute class weights from dataset statistics (Equation 13 from paper).

        Formula: w_k = (1 / Σ(1/N_j)) * (1/N_k)

        Args:
            class_pixel_counts (dict): Mapping of class_index -> total_pixel_count.
                Example: {0: 50000, 1: 30000, 2: 5000, ...}

        Returns:
            (torch.Tensor): Class weights, shape (num_classes,).

        Example:
            >>> counts = {0: 50000, 1: 30000, 2: 5000}  # Class 2 is rare
            >>> weights = GeneralizedSurfaceLoss.compute_class_weights(counts)
            >>> # weights[2] will be highest (rare class gets more weight)
        """
        num_classes = max(class_pixel_counts.keys()) + 1
        weights = torch.zeros(num_classes)

        # Compute sum of inverses
        inverse_sum = sum(1.0 / count for count in class_pixel_counts.values() if count > 0)

        # Compute normalized weights
        for class_idx, count in class_pixel_counts.items():
            if count > 0:
                weights[class_idx] = (1.0 / inverse_sum) * (1.0 / count)

        return weights


class AlphaScheduler:
    """
    Alpha scheduler for combining region-based and boundary-based losses.

    Combined loss: L = α * L_region + (1 - α) * L_boundary

    Alpha starts at 1.0 (pure region loss) and decays to 0.0 (pure boundary loss).
    """

    def __init__(self, total_epochs: int, schedule: str = "linear", step_size: int = 5):
        """
        Initialize AlphaScheduler.

        Args:
            total_epochs (int): Total training epochs.
            schedule (str): Schedule type - "linear", "step", or "cosine".
            step_size (int): Step size for "step" schedule.
        """
        self.total_epochs = total_epochs
        self.schedule = schedule
        self.step_size = step_size
        self.n_steps = total_epochs // step_size if step_size > 0 else total_epochs

    def get_alpha(self, epoch: int) -> float:
        """
        Get alpha value for given epoch.

        Args:
            epoch (int): Current epoch (0-indexed).

        Returns:
            (float): Alpha value in [0, 1].
        """
        t = min(epoch, self.total_epochs - 1)
        T = self.total_epochs

        if self.schedule == "linear":
            # Equation 14: α = 1 - t/T
            return 1.0 - t / T

        elif self.schedule == "step":
            # Equation 15: α = 1 - floor(t/h) / N_h
            if self.n_steps == 0:
                return 0.0
            return 1.0 - (t // self.step_size) / self.n_steps

        elif self.schedule == "cosine":
            # Equation 16: α = 0.5 * (1 + cos(πt/T))
            import math
            return 0.5 * (1.0 + math.cos(math.pi * t / T))

        else:
            return 1.0 - t / T  # Default to linear


class DentalSegmentationLoss(v8SegmentationLoss):
    """
    Optimized Dental-specific segmentation loss extending v8SegmentationLoss.

    Adds anatomical constraint loss and Generalized Surface Loss (GSL) for improved
    tooth detection and segmentation in dental imaging.

    Loss components: [box, seg, cls, dfl, anatomy, gsl]

    Key optimizations:
    - No GPU-CPU synchronization points
    - Pre-registered FDI lookup tensors
    - Vectorized batch operations
    - GSL with full iterative DTM

    Attributes:
        FDI_CLASSES (list): Full FDI tooth numbering (32 classes).
        gsl_loss (GeneralizedSurfaceLoss): Boundary-focused loss module.
        alpha_scheduler (AlphaScheduler): For loss weighting schedule.
    """

    # FDI tooth numbering system (full 32-class for panoramic)
    FDI_CLASSES = [
        11, 12, 13, 14, 15, 16, 17, 18,  # Q1 - Upper Right
        21, 22, 23, 24, 25, 26, 27, 28,  # Q2 - Upper Left
        31, 32, 33, 34, 35, 36, 37, 38,  # Q3 - Lower Left
        41, 42, 43, 44, 45, 46, 47, 48,  # Q4 - Lower Right
    ]

    UPPER_QUADRANTS = {1, 2}
    LOWER_QUADRANTS = {3, 4}

    def __init__(self, model):
        """
        Initialize DentalSegmentationLoss.

        Args:
            model: De-parallelized model with hyperparameters.
        """
        super().__init__(model)

        # Anatomical constraint parameters
        self.max_gap = 3
        self.midline_threshold = 3

        # Build FDI mappings as registered buffers
        self._build_fdi_tensors()

        # Pre-compute adjacency validity matrix
        self._build_adjacency_matrix()

        # GSL loss with full iterative DTM
        dtm_iterations = getattr(self.hyp, "dtm_iterations", 30)

        # Load class weights - supports dict, tensor, or file path
        class_weights = getattr(self.hyp, "gsl_class_weights", None)
        weights_path = getattr(self.hyp, "gsl_weights_path", None)

        if weights_path is not None:
            # Load from file
            import os
            if os.path.exists(weights_path):
                class_weights = torch.load(weights_path, weights_only=True)
                print(f"Loaded GSL class weights from: {weights_path}")
        elif isinstance(class_weights, dict):
            class_weights = GeneralizedSurfaceLoss.compute_class_weights(class_weights)

        self.gsl_loss = GeneralizedSurfaceLoss(
            max_dtm_iterations=dtm_iterations,
            class_weights=class_weights,
        )

        # Alpha scheduler (linear by default)
        total_epochs = getattr(self.hyp, "epochs", 100)
        self.alpha_scheduler = AlphaScheduler(total_epochs, schedule="linear")
        self.current_epoch = 0

    def _build_fdi_tensors(self):
        """Build FDI lookup tensors as registered buffers for GPU efficiency."""
        n = len(self.FDI_CLASSES)

        # FDI codes tensor for direct lookup
        fdi_tensor = torch.tensor(self.FDI_CLASSES, dtype=torch.long)
        self.register_buffer("fdi_codes", fdi_tensor)

        # Quadrant lookup (fdi // 10)
        quadrants = fdi_tensor // 10
        self.register_buffer("quadrant_lookup", quadrants)

        # Position lookup (fdi % 10)
        positions = fdi_tensor % 10
        self.register_buffer("position_lookup", positions)

        # Upper/Lower jaw masks
        upper_mask = (quadrants == 1) | (quadrants == 2)
        lower_mask = (quadrants == 3) | (quadrants == 4)
        self.register_buffer("upper_jaw_mask", upper_mask)
        self.register_buffer("lower_jaw_mask", lower_mask)

    def _build_adjacency_matrix(self):
        """Pre-compute adjacency validity matrix for all FDI class pairs."""
        n = len(self.FDI_CLASSES)
        adjacency = torch.zeros((n, n), dtype=torch.bool)

        for i, fdi_i in enumerate(self.FDI_CLASSES):
            q_i = fdi_i // 10
            n_i = fdi_i % 10

            for j, fdi_j in enumerate(self.FDI_CLASSES):
                if i == j:
                    continue

                q_j = fdi_j // 10
                n_j = fdi_j % 10

                # Same quadrant with small gap
                if q_i == q_j and abs(n_i - n_j) <= self.max_gap:
                    adjacency[i, j] = True

                # Cross-quadrant same jaw near midline
                elif self._same_jaw_static(q_i, q_j):
                    if n_i <= self.midline_threshold and n_j <= self.midline_threshold:
                        adjacency[i, j] = True

        self.register_buffer("adjacency_valid", adjacency)

    @staticmethod
    def _same_jaw_static(q1: int, q2: int) -> bool:
        """Check if two quadrants are in the same jaw."""
        upper = {1, 2}
        return (q1 in upper) == (q2 in upper)

    def set_epoch(self, epoch: int):
        """Update current epoch for alpha scheduling."""
        self.current_epoch = epoch

    def __call__(self, preds, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate combined loss for dental segmentation.

        Returns 6 loss components: [box, seg, cls, dfl, anatomy, gsl]
        """
        # Initialize 6-component loss vector
        loss = torch.zeros(6, device=self.device)

        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        batch_size, _, mask_h, mask_w = proto.shape

        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        from ultralytics.utils.tal import make_anchors

        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR segment dataset incorrectly formatted or not a segment dataset.\n"
                "This error can occur when incorrectly training a 'segment' model on a 'detect' dataset."
            ) from e

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Classification loss
        loss[2] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        if fg_mask.sum():
            # Bbox loss
            loss[0], loss[3] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
            )

            # Masks loss
            masks = batch["masks"].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]

            # Get alpha for current epoch (linear schedule)
            alpha = self.alpha_scheduler.get_alpha(self.current_epoch)

            # Standard segmentation loss (weighted by alpha)
            seg_loss = self.calculate_segmentation_loss(
                fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz, self.overlap
            )
            loss[1] = seg_loss

            # Anatomical constraint loss
            anatomy_weight = getattr(self.hyp, "anatomy", 0.0)
            if anatomy_weight > 0:
                loss[4] = self.compute_anatomy_loss_vectorized(
                    pred_scores, pred_bboxes * stride_tensor, target_scores, fg_mask
                )

            # GSL boundary loss (weighted by 1 - alpha)
            gsl_weight = getattr(self.hyp, "gsl", 0.0)
            if gsl_weight > 0:
                gsl_loss_val = self.compute_gsl_loss(
                    fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz,
                    target_scores=target_scores,  # Pass for class-weighted GSL
                )
                loss[5] = gsl_loss_val

        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()

        # Apply loss weights with alpha schedule for GSL
        # Paper: L = α * L_region + (1-α) * L_boundary
        # We use hyp.box as total segmentation budget, distributed by alpha
        alpha = self.alpha_scheduler.get_alpha(self.current_epoch)
        gsl_weight = getattr(self.hyp, "gsl", 0.0)

        loss[0] *= self.hyp.box
        if gsl_weight > 0:
            # When GSL is enabled, distribute box weight between seg and GSL
            # Total seg contribution = box * (alpha * seg + (1-alpha) * gsl)
            loss[1] *= self.hyp.box * alpha
            loss[5] *= self.hyp.box * (1.0 - alpha)
        else:
            # No GSL, full weight to seg loss
            loss[1] *= self.hyp.box
        loss[2] *= self.hyp.cls
        loss[3] *= self.hyp.dfl
        loss[4] *= getattr(self.hyp, "anatomy", 0.0)

        return loss * batch_size, loss.detach()

    def compute_gsl_loss(
        self,
        fg_mask: torch.Tensor,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        target_bboxes: torch.Tensor,
        batch_idx: torch.Tensor,
        proto: torch.Tensor,
        pred_masks: torch.Tensor,
        imgsz: torch.Tensor,
        target_scores: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute Generalized Surface Loss for mask quality.

        Collects all masks across batch and computes GSL in single forward pass.

        Args:
            target_scores: Optional target scores for class-weighted GSL.
        """
        all_pred_masks = []
        all_gt_masks = []
        all_class_indices = []

        for i in range(fg_mask.shape[0]):
            fg_mask_i = fg_mask[i]
            if not fg_mask_i.any():
                continue

            target_gt_idx_i = target_gt_idx[i]
            pred_masks_i = pred_masks[i]
            proto_i = proto[i]

            mask_idx = target_gt_idx_i[fg_mask_i]

            # Get ground truth masks
            if self.overlap:
                gt_mask = masks[i] == (mask_idx + 1).view(-1, 1, 1)
                gt_mask = gt_mask.float()
            else:
                gt_mask = masks[batch_idx.view(-1) == i][mask_idx]

            # Compute predicted masks
            pred_mask = torch.einsum("in,nhw->ihw", pred_masks_i[fg_mask_i], proto_i).sigmoid()

            all_pred_masks.append(pred_mask)
            all_gt_masks.append(gt_mask)

            # Get class indices for this batch item (for class weights)
            if target_scores is not None and self.gsl_loss.class_weights is not None:
                class_idx = target_scores[i, fg_mask_i].argmax(dim=-1)
                all_class_indices.append(class_idx)

        if not all_pred_masks:
            return torch.tensor(0.0, device=proto.device)

        # Concatenate all masks
        pred_masks_cat = torch.cat(all_pred_masks, dim=0)
        gt_masks_cat = torch.cat(all_gt_masks, dim=0)

        # Concatenate class indices if available
        class_indices_cat = None
        if all_class_indices:
            class_indices_cat = torch.cat(all_class_indices, dim=0)

        return self.gsl_loss(pred_masks_cat, gt_masks_cat, class_indices=class_indices_cat)

    def compute_anatomy_loss_vectorized(
        self,
        pred_scores: torch.Tensor,
        pred_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        fg_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute anatomical constraint loss - FULLY VECTORIZED.

        No .item() or .tolist() calls - everything stays on GPU.
        """
        batch_size = pred_scores.shape[0]
        total_loss = torch.tensor(0.0, device=pred_scores.device)
        valid_count = torch.tensor(0.0, device=pred_scores.device)

        for i in range(batch_size):
            fg_i = fg_mask[i]
            n_fg = fg_i.sum()

            if n_fg < 2:
                continue

            # Get predicted classes (no .tolist())
            pred_classes = target_scores[i, fg_i].argmax(dim=-1)
            bboxes = pred_bboxes[i, fg_i]

            # Compute loss components (all vectorized)
            dup_loss = self._duplicate_loss_fast(pred_classes)
            neighbor_loss = self._neighbor_loss_fast(pred_classes, bboxes)
            ordering_loss = self._ordering_loss_fast(pred_classes, bboxes)

            total_loss = total_loss + dup_loss + neighbor_loss + ordering_loss
            valid_count = valid_count + 1.0

        return total_loss / torch.clamp(valid_count, min=1.0)

    def _duplicate_loss_fast(self, pred_classes: torch.Tensor) -> torch.Tensor:
        """Penalize duplicate class predictions - fast version."""
        counts = torch.zeros(self.nc, device=pred_classes.device)
        ones = torch.ones_like(pred_classes, dtype=counts.dtype)
        counts.scatter_add_(0, pred_classes, ones)
        return F.relu(counts - 1.0).sum()

    def _neighbor_loss_fast(self, pred_classes: torch.Tensor, bboxes: torch.Tensor, k: int = 2) -> torch.Tensor:
        """Penalize invalid spatial neighbors - fast version."""
        n = pred_classes.shape[0]
        if n < 2:
            return torch.tensor(0.0, device=pred_classes.device)

        # Compute bbox centers
        centers = (bboxes[:, :2] + bboxes[:, 2:4]) / 2

        # Pairwise distances
        dists = torch.cdist(centers.unsqueeze(0), centers.unsqueeze(0)).squeeze(0)
        dists = dists + torch.eye(n, device=dists.device) * 1e6

        # Find k nearest neighbors
        k_actual = min(k, n - 1)
        _, knn_idx = dists.topk(k_actual, dim=1, largest=False)

        # Skip if class count doesn't match FDI
        if self.nc != len(self.FDI_CLASSES):
            return torch.tensor(0.0, device=pred_classes.device)

        # Vectorized validity check
        loss = torch.tensor(0.0, device=pred_classes.device)
        for j in range(k_actual):
            neighbor_idx = knn_idx[:, j]
            neighbor_classes = pred_classes[neighbor_idx]
            neighbor_dists = dists[torch.arange(n, device=dists.device), neighbor_idx]

            # Check validity using pre-registered adjacency matrix
            valid_mask = self.adjacency_valid[pred_classes, neighbor_classes]
            invalid_mask = ~valid_mask

            penalty = (invalid_mask.float() / (neighbor_dists + 1.0)).sum()
            loss = loss + penalty

        return loss

    def _ordering_loss_fast(self, pred_classes: torch.Tensor, bboxes: torch.Tensor) -> torch.Tensor:
        """Penalize implausible spatial ordering - fast version using lookup tensors."""
        n = pred_classes.shape[0]
        if n < 2 or self.nc != len(self.FDI_CLASSES):
            return torch.tensor(0.0, device=pred_classes.device)

        # Use pre-registered lookup tensors (no .tolist())
        quadrants = self.quadrant_lookup[pred_classes]
        positions = self.position_lookup[pred_classes]

        centers_x = (bboxes[:, 0] + bboxes[:, 2]) / 2

        # Upper/lower masks
        upper_mask = self.upper_jaw_mask[pred_classes]
        lower_mask = self.lower_jaw_mask[pred_classes]

        loss = torch.tensor(0.0, device=pred_classes.device)

        # Check upper row ordering
        if upper_mask.sum() >= 2:
            loss = loss + self._row_ordering_loss_fast(
                quadrants[upper_mask], positions[upper_mask], centers_x[upper_mask]
            )

        # Check lower row ordering
        if lower_mask.sum() >= 2:
            loss = loss + self._row_ordering_loss_fast(
                quadrants[lower_mask], positions[lower_mask], centers_x[lower_mask]
            )

        return loss

    def _row_ordering_loss_fast(
        self, quadrants: torch.Tensor, positions: torch.Tensor, x_coords: torch.Tensor
    ) -> torch.Tensor:
        """Compute ordering loss for teeth in a single row - fast version."""
        n = quadrants.shape[0]
        if n < 2:
            return torch.tensor(0.0, device=quadrants.device)

        # Sign: +1 for Q2/Q3 (left), -1 for Q1/Q4 (right)
        sign = torch.where((quadrants == 2) | (quadrants == 3), 1.0, -1.0)
        expected_order = sign * positions.float()

        # Soft ranking
        x_ranks = self._soft_rank_fast(x_coords)
        expected_ranks = self._soft_rank_fast(expected_order)

        return F.mse_loss(x_ranks, expected_ranks)

    @staticmethod
    def _soft_rank_fast(x: torch.Tensor) -> torch.Tensor:
        """Differentiable soft ranking - optimized version."""
        n = x.shape[0]
        diff = x.unsqueeze(1) - x.unsqueeze(0)
        ranks = torch.sigmoid(diff * 5.0).sum(dim=1)
        return ranks / n
