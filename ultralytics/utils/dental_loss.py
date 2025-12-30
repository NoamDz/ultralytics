# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""
Dental-specific loss functions for YOLOv11-seg tooth detection and segmentation.

This module provides specialized losses for dental imaging:
1. Anatomical Constraint Loss - Penalizes anatomically implausible tooth detections
2. HD95 Boundary Loss - Improves tooth contour accuracy via Hausdorff distance

Supports FDI tooth numbering system with 32 classes (panoramic) or 24 classes (bitewing).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.loss import v8SegmentationLoss
from ultralytics.utils.ops import crop_mask, xyxy2xywh


class HD95BoundaryLoss:
    """
    Efficient HD95 (95th percentile Hausdorff Distance) loss for boundary-focused segmentation.

    Computes HD95 per-instance within bounding box crops for efficiency.
    Uses differentiable soft contour extraction via gradient magnitude.

    Attributes:
        max_contour_points (int): Maximum number of contour points to sample.
        percentile (float): Percentile for Hausdorff distance (default 0.95 for HD95).
    """

    def __init__(self, max_contour_points: int = 100, percentile: float = 0.95):
        """
        Initialize HD95BoundaryLoss.

        Args:
            max_contour_points (int): Maximum contour points to sample for efficiency.
            percentile (float): Percentile for robust Hausdorff (0.95 = HD95).
        """
        self.max_contour_points = max_contour_points
        self.percentile = percentile

    def __call__(
        self,
        pred_masks: torch.Tensor,
        gt_masks: torch.Tensor,
        bboxes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute HD95 loss for a batch of mask pairs.

        Args:
            pred_masks (torch.Tensor): Predicted masks after sigmoid, shape (N, H, W).
            gt_masks (torch.Tensor): Ground truth binary masks, shape (N, H, W).
            bboxes (torch.Tensor): Bounding boxes in mask coordinates, shape (N, 4).

        Returns:
            (torch.Tensor): Scalar HD95 loss averaged over valid instances.
        """
        n = pred_masks.shape[0]
        if n == 0:
            return torch.tensor(0.0, device=pred_masks.device)

        total_loss = torch.tensor(0.0, device=pred_masks.device)
        valid_masks = 0

        for i in range(n):
            loss_i = self._single_mask_hd95(pred_masks[i], gt_masks[i], bboxes[i])
            if loss_i is not None:
                total_loss = total_loss + loss_i
                valid_masks += 1

        return total_loss / max(valid_masks, 1)

    def _single_mask_hd95(
        self,
        pred_mask: torch.Tensor,
        gt_mask: torch.Tensor,
        bbox: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """
        Compute HD95 for a single mask pair within bounding box.

        Args:
            pred_mask (torch.Tensor): Predicted mask probabilities, shape (H, W).
            gt_mask (torch.Tensor): Ground truth binary mask, shape (H, W).
            bbox (torch.Tensor): Bounding box [x1, y1, x2, y2].

        Returns:
            (torch.Tensor | None): HD95 loss or None if invalid.
        """
        # Crop to bbox region for efficiency
        x1, y1, x2, y2 = bbox.int().tolist()
        h, w = pred_mask.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        pred_crop = pred_mask[y1:y2, x1:x2]
        gt_crop = gt_mask[y1:y2, x1:x2]

        if pred_crop.numel() == 0 or gt_crop.sum() < 1:
            return None

        # Extract contour points
        pred_contour = self._extract_contour_soft(pred_crop)
        gt_contour = self._extract_contour_hard(gt_crop)

        if pred_contour is None or gt_contour is None:
            # Return high penalty for empty prediction with valid GT
            if gt_contour is not None and pred_contour is None:
                return torch.tensor(10.0, device=pred_mask.device)
            return None

        return self._compute_hd95(pred_contour, gt_contour)

    def _extract_contour_soft(self, mask: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Extract soft contour using gradient magnitude (differentiable).

        Args:
            mask (torch.Tensor): Probability mask, shape (H, W).

        Returns:
            (torch.Tensor | None): Contour points (K, 2) or None.
        """
        h, w = mask.shape
        if h < 3 or w < 3:
            return None

        # Sobel-like gradient computation
        grad_x = F.pad(mask[:, 2:] - mask[:, :-2], (1, 1, 0, 0))
        grad_y = F.pad(mask[2:, :] - mask[:-2, :], (0, 0, 1, 1))

        # Gradient magnitude as contour weight
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)

        # Threshold to focus on actual boundaries
        threshold = grad_mag.max() * 0.1
        grad_mag = grad_mag * (grad_mag > threshold).float()

        if grad_mag.sum() < 1e-6:
            return None

        # Create coordinate grid
        y_coords = torch.arange(h, device=mask.device, dtype=mask.dtype)
        x_coords = torch.arange(w, device=mask.device, dtype=mask.dtype)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")

        # Sample top-k points by gradient magnitude
        weights = grad_mag.flatten()
        k = min(self.max_contour_points, (weights > 0).sum().item())
        if k < 2:
            return None

        _, indices = weights.topk(k)

        contour_points = torch.stack([xx.flatten()[indices], yy.flatten()[indices]], dim=1)

        return contour_points

    def _extract_contour_hard(self, mask: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Extract hard contour from binary mask using morphological gradient.

        Args:
            mask (torch.Tensor): Binary mask, shape (H, W).

        Returns:
            (torch.Tensor | None): Contour points (K, 2) or None.
        """
        if mask.sum() < 1:
            return None

        # Morphological gradient: dilate - erode using max/min pooling
        mask_4d = mask.unsqueeze(0).unsqueeze(0).float()

        dilated = F.max_pool2d(mask_4d, kernel_size=3, stride=1, padding=1)
        eroded = -F.max_pool2d(-mask_4d, kernel_size=3, stride=1, padding=1)

        contour = (dilated - eroded).squeeze()

        # Get contour point coordinates
        contour_idx = torch.nonzero(contour > 0.5, as_tuple=False)

        if contour_idx.shape[0] < 2:
            return None

        # Subsample if too many points
        if contour_idx.shape[0] > self.max_contour_points:
            idx = torch.randperm(contour_idx.shape[0], device=mask.device)[: self.max_contour_points]
            contour_idx = contour_idx[idx]

        # Return (N, 2) with (x, y) coordinates
        return contour_idx[:, [1, 0]].float()

    def _compute_hd95(self, pred_points: torch.Tensor, gt_points: torch.Tensor) -> torch.Tensor:
        """
        Compute symmetric HD95 between two point sets.

        Args:
            pred_points (torch.Tensor): Predicted contour points (N, 2).
            gt_points (torch.Tensor): Ground truth contour points (M, 2).

        Returns:
            (torch.Tensor): Symmetric HD95 distance.
        """
        # Pairwise distances: (N, M)
        dists = torch.cdist(pred_points, gt_points)

        # Directed Hausdorff: pred -> gt
        min_dist_pred_to_gt = dists.min(dim=1)[0]

        # Directed Hausdorff: gt -> pred
        min_dist_gt_to_pred = dists.min(dim=0)[0]

        # Percentile computation
        hd_pred = self._percentile(min_dist_pred_to_gt, self.percentile)
        hd_gt = self._percentile(min_dist_gt_to_pred, self.percentile)

        # Symmetric HD95
        return torch.max(hd_pred, hd_gt)

    @staticmethod
    def _percentile(x: torch.Tensor, p: float) -> torch.Tensor:
        """Compute percentile of tensor values."""
        n = x.shape[0]
        k = int(n * p)
        k = min(k, n - 1)
        sorted_x, _ = x.sort()
        return sorted_x[k]


class DentalSegmentationLoss(v8SegmentationLoss):
    """
    Dental-specific segmentation loss extending v8SegmentationLoss.

    Adds anatomical constraint loss and HD95 boundary loss for improved
    tooth detection and segmentation in dental imaging.

    Loss components: [box, seg, cls, dfl, anatomy, hd95]

    Attributes:
        FDI_CLASSES (list): Full FDI tooth numbering (32 classes).
        UPPER_QUADRANTS (set): Upper jaw quadrants {1, 2}.
        LOWER_QUADRANTS (set): Lower jaw quadrants {3, 4}.
        adjacency_valid (torch.Tensor): Pre-computed validity matrix.
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
    MIDLINE_POSITIONS = {1, 2, 3}  # Incisors + canine can cross quadrants

    def __init__(self, model):
        """
        Initialize DentalSegmentationLoss.

        Args:
            model: De-parallelized model with hyperparameters.
        """
        super().__init__(model)

        # Anatomical constraint parameters
        self.max_gap = 3  # Max gap between same-quadrant neighbors
        self.midline_threshold = 3  # Tooth positions near midline

        # Build FDI class mappings
        self._build_fdi_mappings()

        # Pre-compute adjacency validity matrix
        self._build_adjacency_matrix()

        # HD95 boundary loss
        self.hd95_loss = HD95BoundaryLoss(max_contour_points=100, percentile=0.95)

    def _build_fdi_mappings(self):
        """Build mappings between FDI classes and indices."""
        self.fdi_to_idx = {fdi: idx for idx, fdi in enumerate(self.FDI_CLASSES)}
        self.idx_to_fdi = {idx: fdi for idx, fdi in enumerate(self.FDI_CLASSES)}

    def _build_adjacency_matrix(self):
        """
        Pre-compute adjacency validity matrix for all FDI class pairs.

        Valid neighbors are:
        1. Same quadrant with tooth number gap <= max_gap
        2. Cross-quadrant in same jaw with both teeth near midline
        """
        n = len(self.FDI_CLASSES)
        self.adjacency_valid = torch.zeros((n, n), dtype=torch.bool)

        for i, fdi_i in enumerate(self.FDI_CLASSES):
            q_i = fdi_i // 10
            n_i = fdi_i % 10

            for j, fdi_j in enumerate(self.FDI_CLASSES):
                if i == j:
                    continue

                q_j = fdi_j // 10
                n_j = fdi_j % 10

                # Case 1: Same quadrant - check gap
                if q_i == q_j and abs(n_i - n_j) <= self.max_gap:
                    self.adjacency_valid[i, j] = True

                # Case 2: Cross-quadrant same jaw - check midline
                elif self._same_jaw(q_i, q_j):
                    if n_i <= self.midline_threshold and n_j <= self.midline_threshold:
                        self.adjacency_valid[i, j] = True

    def _same_jaw(self, q1: int, q2: int) -> bool:
        """Check if two quadrants are in the same jaw."""
        upper1 = q1 in self.UPPER_QUADRANTS
        upper2 = q2 in self.UPPER_QUADRANTS
        return upper1 == upper2

    def __call__(self, preds, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate combined loss for dental segmentation.

        Returns 6 loss components: [box, seg, cls, dfl, anatomy, hd95]

        Args:
            preds: Model predictions (feats, pred_masks, proto).
            batch: Batch dictionary with targets.

        Returns:
            (tuple): (weighted_loss * batch_size, detached_loss)
        """
        # Initialize 6-component loss vector
        loss = torch.zeros(6, device=self.device)

        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        batch_size, _, mask_h, mask_w = proto.shape

        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
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
                "ERROR ❌ segment dataset incorrectly formatted or not a segment dataset.\n"
                "This error can occur when incorrectly training a 'segment' model on a 'detect' dataset."
            ) from e

        # Pboxes
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

            loss[1] = self.calculate_segmentation_loss(
                fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz, self.overlap
            )

            # Anatomical constraint loss
            if hasattr(self.hyp, "anatomy") and self.hyp.anatomy > 0:
                loss[4] = self.compute_anatomy_loss(
                    pred_scores, pred_bboxes * stride_tensor, target_scores, fg_mask
                )

            # HD95 boundary loss
            if hasattr(self.hyp, "hd95") and self.hyp.hd95 > 0:
                loss[5] = self.compute_hd95_loss(
                    fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz
                )

        # WARNING: prevent Multi-GPU DDP 'unused gradient' errors
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()

        # Apply loss weights
        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.box  # seg uses box weight
        loss[2] *= self.hyp.cls
        loss[3] *= self.hyp.dfl
        loss[4] *= getattr(self.hyp, "anatomy", 0.0)
        loss[5] *= getattr(self.hyp, "hd95", 0.0)

        return loss * batch_size, loss.detach()

    def compute_anatomy_loss(
        self,
        pred_scores: torch.Tensor,
        pred_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        fg_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute anatomical constraint loss on post-assignment predictions.

        Components:
        1. Duplicate penalty - same class predicted multiple times
        2. Neighbor validity - invalid spatial neighbors
        3. Ordering consistency - incorrect left-right ordering

        Args:
            pred_scores (torch.Tensor): Predicted scores (BS, N_anchors, nc).
            pred_bboxes (torch.Tensor): Predicted bboxes (BS, N_anchors, 4).
            target_scores (torch.Tensor): Assigned target scores (BS, N_anchors, nc).
            fg_mask (torch.Tensor): Foreground mask (BS, N_anchors).

        Returns:
            (torch.Tensor): Scalar anatomical constraint loss.
        """
        batch_size = pred_scores.shape[0]
        total_loss = torch.tensor(0.0, device=pred_scores.device)
        valid_batches = 0

        for i in range(batch_size):
            fg_i = fg_mask[i]
            n_fg = fg_i.sum().item()

            if n_fg < 2:
                continue

            # Get predicted classes for foreground anchors
            pred_classes = target_scores[i, fg_i].argmax(dim=-1)
            bboxes = pred_bboxes[i, fg_i]

            # Compute loss components
            dup_loss = self._duplicate_loss(pred_classes)
            neighbor_loss = self._neighbor_loss(pred_classes, bboxes)
            ordering_loss = self._ordering_loss(pred_classes, bboxes)

            total_loss = total_loss + dup_loss + neighbor_loss + ordering_loss
            valid_batches += 1

        return total_loss / max(valid_batches, 1)

    def _duplicate_loss(self, pred_classes: torch.Tensor) -> torch.Tensor:
        """
        Penalize duplicate class predictions within an image.

        Args:
            pred_classes (torch.Tensor): Predicted class indices (N,).

        Returns:
            (torch.Tensor): Duplicate penalty loss.
        """
        # Count per class using scatter
        counts = torch.zeros(self.nc, device=pred_classes.device)
        ones = torch.ones_like(pred_classes, dtype=counts.dtype)
        counts.scatter_add_(0, pred_classes, ones)

        # Penalize counts > 1
        return F.relu(counts - 1.0).sum()

    def _neighbor_loss(self, pred_classes: torch.Tensor, bboxes: torch.Tensor, k: int = 2) -> torch.Tensor:
        """
        Penalize invalid spatial neighbors.

        Args:
            pred_classes (torch.Tensor): Predicted class indices (N,).
            bboxes (torch.Tensor): Bounding boxes (N, 4).
            k (int): Number of nearest neighbors to check.

        Returns:
            (torch.Tensor): Neighbor validity loss.
        """
        n = pred_classes.shape[0]
        if n < 2:
            return torch.tensor(0.0, device=pred_classes.device)

        # Compute bbox centers
        centers = (bboxes[:, :2] + bboxes[:, 2:4]) / 2

        # Pairwise distances
        dists = torch.cdist(centers.unsqueeze(0), centers.unsqueeze(0)).squeeze(0)

        # Mask self-distances
        dists = dists + torch.eye(n, device=dists.device) * 1e6

        # Find k nearest neighbors
        k_actual = min(k, n - 1)
        _, knn_idx = dists.topk(k_actual, dim=1, largest=False)

        # Move adjacency matrix to device
        adj = self.adjacency_valid.to(pred_classes.device)

        loss = torch.tensor(0.0, device=pred_classes.device)

        for j in range(k_actual):
            neighbor_idx = knn_idx[:, j]
            neighbor_classes = pred_classes[neighbor_idx]
            neighbor_dists = dists[torch.arange(n, device=dists.device), neighbor_idx]

            # Check validity - need to map model classes to FDI indices
            # If nc matches FDI_CLASSES length, direct lookup; otherwise skip
            if self.nc == len(self.FDI_CLASSES):
                valid_mask = adj[pred_classes, neighbor_classes]
            else:
                # For non-standard class counts, skip neighbor validation
                continue

            # Penalize invalid neighbors (distance-weighted)
            invalid_mask = ~valid_mask
            penalty = (invalid_mask.float() / (neighbor_dists + 1.0)).sum()
            loss = loss + penalty

        return loss

    def _ordering_loss(self, pred_classes: torch.Tensor, bboxes: torch.Tensor) -> torch.Tensor:
        """
        Penalize implausible spatial ordering within rows.

        Args:
            pred_classes (torch.Tensor): Predicted class indices (N,).
            bboxes (torch.Tensor): Bounding boxes (N, 4).

        Returns:
            (torch.Tensor): Ordering consistency loss.
        """
        n = pred_classes.shape[0]
        if n < 2 or self.nc != len(self.FDI_CLASSES):
            return torch.tensor(0.0, device=pred_classes.device)

        # Convert class indices to FDI codes
        fdi_codes = torch.tensor([self.FDI_CLASSES[c] for c in pred_classes.tolist()], device=pred_classes.device)

        # Get quadrants and positions
        quadrants = fdi_codes // 10
        positions = fdi_codes % 10

        centers_x = (bboxes[:, 0] + bboxes[:, 2]) / 2

        # Separate by row
        upper_mask = (quadrants == 1) | (quadrants == 2)
        lower_mask = (quadrants == 3) | (quadrants == 4)

        loss = torch.tensor(0.0, device=pred_classes.device)

        # Check upper row ordering
        if upper_mask.sum() >= 2:
            loss = loss + self._row_ordering_loss(
                quadrants[upper_mask], positions[upper_mask], centers_x[upper_mask]
            )

        # Check lower row ordering
        if lower_mask.sum() >= 2:
            loss = loss + self._row_ordering_loss(
                quadrants[lower_mask], positions[lower_mask], centers_x[lower_mask]
            )

        return loss

    def _row_ordering_loss(
        self, quadrants: torch.Tensor, positions: torch.Tensor, x_coords: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute ordering loss for teeth in a single row.

        Expected ordering:
        - Q1/Q4 (right side): higher position = more towards back = higher/lower x
        - Q2/Q3 (left side): higher position = more towards back = lower/higher x

        Args:
            quadrants (torch.Tensor): Quadrant numbers (N,).
            positions (torch.Tensor): Tooth positions 1-8 (N,).
            x_coords (torch.Tensor): X coordinates of bbox centers (N,).

        Returns:
            (torch.Tensor): Ordering loss.
        """
        n = quadrants.shape[0]
        if n < 2:
            return torch.tensor(0.0, device=quadrants.device)

        # Compute expected relative x position based on anatomy
        # Q1 (upper right): teeth go from central (13) to back (18), x decreases
        # Q2 (upper left): teeth go from central (23) to back (28), x increases
        # Q3 (lower left): teeth go from central (33) to back (38), x increases
        # Q4 (lower right): teeth go from central (43) to back (48), x decreases

        # Sign: +1 for Q2/Q3 (left side), -1 for Q1/Q4 (right side)
        sign = torch.where((quadrants == 2) | (quadrants == 3), 1.0, -1.0)
        expected_order = sign * positions.float()

        # Compute soft ranks
        x_ranks = self._soft_rank(x_coords)
        expected_ranks = self._soft_rank(expected_order)

        # MSE between ranks
        return F.mse_loss(x_ranks, expected_ranks)

    @staticmethod
    def _soft_rank(x: torch.Tensor) -> torch.Tensor:
        """
        Compute differentiable soft ranking.

        Args:
            x (torch.Tensor): Values to rank (N,).

        Returns:
            (torch.Tensor): Soft ranks (N,).
        """
        n = x.shape[0]
        # Pairwise comparison with sigmoid smoothing
        diff = x.unsqueeze(1) - x.unsqueeze(0)
        ranks = torch.sigmoid(diff * 5.0).sum(dim=1)
        return ranks / n

    def compute_hd95_loss(
        self,
        fg_mask: torch.Tensor,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        target_bboxes: torch.Tensor,
        batch_idx: torch.Tensor,
        proto: torch.Tensor,
        pred_masks: torch.Tensor,
        imgsz: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute HD95 boundary loss for mask quality.

        Args:
            fg_mask (torch.Tensor): Foreground mask (BS, N_anchors).
            masks (torch.Tensor): Ground truth masks.
            target_gt_idx (torch.Tensor): GT indices for anchors.
            target_bboxes (torch.Tensor): Target bounding boxes.
            batch_idx (torch.Tensor): Batch indices.
            proto (torch.Tensor): Prototype masks (BS, 32, H, W).
            pred_masks (torch.Tensor): Predicted mask coefficients.
            imgsz (torch.Tensor): Image size (H, W).

        Returns:
            (torch.Tensor): HD95 boundary loss.
        """
        _, _, mask_h, mask_w = proto.shape
        total_loss = torch.tensor(0.0, device=proto.device)
        valid_count = 0

        # Normalize bboxes to mask coordinates
        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]
        mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)

        for i, (fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, masks_i) in enumerate(
            zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, masks)
        ):
            if not fg_mask_i.any():
                continue

            mask_idx = target_gt_idx_i[fg_mask_i]

            # Get ground truth masks
            if self.overlap:
                gt_mask = masks_i == (mask_idx + 1).view(-1, 1, 1)
                gt_mask = gt_mask.float()
            else:
                gt_mask = masks[batch_idx.view(-1) == i][mask_idx]

            # Compute predicted masks
            pred_mask = torch.einsum("in,nhw->ihw", pred_masks_i[fg_mask_i], proto_i).sigmoid()

            # Compute HD95 loss
            hd95_loss_val = self.hd95_loss(pred_mask, gt_mask, mxyxy_i[fg_mask_i])

            total_loss = total_loss + hd95_loss_val
            valid_count += 1

        return total_loss / max(valid_count, 1)
