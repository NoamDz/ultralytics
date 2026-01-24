# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""
Dental-specific loss functions for YOLOv11-seg tooth detection and segmentation.

This module provides specialized losses for dental imaging:
1. Anatomical Constraint Loss - Penalizes anatomically implausible tooth detections
2. Boundary Loss - Contour-based loss for improved boundary delineation

Supports FDI tooth numbering system with 32 classes (panoramic) or 24 classes (bitewing).

References:
    Kervadec et al., "Boundary loss for highly unbalanced segmentation",
    Medical Image Analysis, 2021. https://github.com/LIVIAETS/surface-loss
"""

from __future__ import annotations

import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils import LOGGER
from ultralytics.utils.loss import v8SegmentationLoss


def compute_signed_distance_map_batch(masks: np.ndarray) -> np.ndarray:
    """
    Compute signed distance maps for a batch of binary masks using scipy EDT.

    This is the core distance computation from the boundary loss paper.
    Uses scipy's distance_transform_edt which is highly optimized.

    Args:
        masks (np.ndarray): Binary masks, shape (N, H, W), values in {0, 1}.

    Returns:
        np.ndarray: Signed distance maps, shape (N, H, W).
            - Positive values: outside the mask (background)
            - Negative values: inside the mask (foreground)
            - Zero: on the boundary
    """
    from scipy.ndimage import distance_transform_edt

    batch_size = masks.shape[0]
    dist_maps = np.zeros_like(masks, dtype=np.float32)

    for i in range(batch_size):
        mask = masks[i]

        # Skip empty masks - return zero distance map
        if mask.sum() == 0:
            continue

        # Skip full masks - return zero distance map
        if mask.sum() == mask.size:
            continue

        # Compute distance from exterior points to boundary (positive outside)
        # EDT of inverted mask gives distance to nearest foreground pixel
        exterior_dist = distance_transform_edt(1 - mask)

        # Compute distance from interior points to boundary (negative inside)
        # EDT of mask gives distance to nearest background pixel
        interior_dist = distance_transform_edt(mask)

        # Combine: positive outside, negative inside
        # φ_G(q) = D_G(q) if q outside G, -D_G(q) if q inside G
        dist_maps[i] = exterior_dist - interior_dist

    return dist_maps


class BoundaryLoss(nn.Module):
    """
    Boundary Loss for segmentation from Kervadec et al. (2021).

    This loss operates on contours rather than regions, making it effective
    for highly unbalanced segmentation problems. It uses a signed distance
    function to weight predictions based on their distance from the boundary.

    Formula (Equation 5 from paper):
        L_B(θ) = Σ_q φ_G(q) * s_θ(q)

    Where:
        - φ_G(q) = -D_G(q) if q ∈ G (inside ground truth - negative)
        - φ_G(q) = +D_G(q) if q ∉ G (outside ground truth - positive)
        - s_θ(q) is the softmax/sigmoid probability output

    Key properties:
        - Perfect prediction minimizes loss (sums only negative values)
        - False positives add positive φ_G → increases loss
        - False negatives miss negative φ_G → increases loss
        - Pixels far from boundary have larger |φ_G| → stronger penalty

    IMPORTANT: Cannot be used alone - requires combination with regional loss.
    """

    def __init__(self):
        """Initialize BoundaryLoss."""
        super().__init__()

    def forward(
        self,
        pred_masks: torch.Tensor,
        gt_masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute boundary loss for batch of mask pairs.

        Args:
            pred_masks (torch.Tensor): Predicted masks after sigmoid, shape (N, H, W).
                Values should be in [0, 1].
            gt_masks (torch.Tensor): Ground truth binary masks, shape (N, H, W).
                Values should be in {0, 1}.

        Returns:
            torch.Tensor: Scalar boundary loss value.
        """
        if pred_masks.numel() == 0:
            return torch.tensor(0.0, device=pred_masks.device)

        # Compute signed distance maps from ground truth (on CPU using scipy)
        # This is very fast and doesn't require GPU synchronization
        gt_np = gt_masks.detach().cpu().numpy().astype(np.float32)
        dist_maps_np = compute_signed_distance_map_batch(gt_np)

        # Transfer back to GPU
        dist_maps = torch.from_numpy(dist_maps_np).to(
            device=pred_masks.device, dtype=pred_masks.dtype
        )

        # Boundary loss: L_B = mean(φ_G * pred)
        # Simple per-pixel mean as in the original paper
        # The loss can be negative (correct predictions) or positive (errors)
        # Scale by 50 to bring magnitude closer to seg_loss (~1-2)
        boundary_loss = (dist_maps * pred_masks).mean() * 50.0

        return boundary_loss


class AlphaScheduler:
    """
    Alpha scheduler for combining region-based and boundary-based losses.

    From the paper, the "rebalance" strategy is recommended:
        Combined loss: (1 - α) * L_regional + α * L_boundary

    Alpha starts small and increases over epochs, allowing regional loss
    to guide early training while boundary loss refines boundaries later.

    Strategies:
        - "rebalance": α starts at alpha_start, increases by alpha_increment each epoch
        - "linear": α = epoch / total_epochs
        - "constant": α stays fixed at alpha_start
        - "sigmoid": S-shaped curve that ramps up sharply in later epochs
    """

    def __init__(
        self,
        total_epochs: int,
        schedule: str = "rebalance",
        alpha_start: float = 0.01,
        alpha_increment: float = 0.01,
        alpha_max: float = 1.0,
        sigmoid_midpoint: float = 0.7,
        sigmoid_steepness: float = 0.1,
    ):
        """
        Initialize AlphaScheduler.

        Args:
            total_epochs (int): Total training epochs.
            schedule (str): Schedule type - "rebalance", "linear", "constant", or "sigmoid".
            alpha_start (float): Starting alpha value (for rebalance/constant).
            alpha_increment (float): Alpha increase per epoch (for rebalance).
            alpha_max (float): Maximum alpha value (cap).
            sigmoid_midpoint (float): Fraction of training where sigmoid crosses 50% (0.0-1.0).
                                      Higher values delay the ramp-up. Default 0.7 means
                                      50% weight is reached at 70% of training.
            sigmoid_steepness (float): Controls sharpness of S-curve (0.01-0.5).
                                       Lower values = sharper transition.
                                       Default 0.1 gives a smooth but decisive curve.
        """
        self.total_epochs = total_epochs
        self.schedule = schedule
        self.alpha_start = alpha_start
        self.alpha_increment = alpha_increment
        self.alpha_max = alpha_max
        self.sigmoid_midpoint = sigmoid_midpoint
        self.sigmoid_steepness = sigmoid_steepness

    def get_alpha(self, epoch: int) -> float:
        """
        Get alpha value for given epoch.

        Args:
            epoch (int): Current epoch (0-indexed).

        Returns:
            float: Alpha value in [0, alpha_max].
        """
        if self.schedule == "rebalance":
            # Paper's recommended strategy: start small, increase linearly
            alpha = self.alpha_start + epoch * self.alpha_increment
            return min(alpha, self.alpha_max)

        elif self.schedule == "linear":
            # Linear from 0 to 1
            if self.total_epochs <= 1:
                return self.alpha_max
            return min(epoch / (self.total_epochs - 1), self.alpha_max)

        elif self.schedule == "constant":
            return self.alpha_start

        elif self.schedule == "sigmoid":
            # S-shaped curve: slow start, rapid increase in later epochs, smooth finish
            # Formula: alpha_max * sigmoid((progress - midpoint) / steepness)
            # This gives:
            #   - Low values early (when progress << midpoint)
            #   - Rapid increase around midpoint
            #   - Approaches alpha_max at end
            import math

            if self.total_epochs <= 1:
                return self.alpha_max

            progress = epoch / (self.total_epochs - 1)  # 0.0 to 1.0
            x = (progress - self.sigmoid_midpoint) / self.sigmoid_steepness
            # Clamp x to avoid overflow in exp
            x = max(-20.0, min(20.0, x))
            sigmoid_value = 1.0 / (1.0 + math.exp(-x))
            return self.alpha_max * sigmoid_value

        else:
            # Default to rebalance
            alpha = self.alpha_start + epoch * self.alpha_increment
            return min(alpha, self.alpha_max)


class DentalSegmentationLoss(v8SegmentationLoss):
    """
    Dental-specific segmentation loss extending v8SegmentationLoss.

    Adds anatomical constraint loss and Boundary Loss for improved
    tooth detection and segmentation in dental imaging.

    Loss components: [box, seg, cls, dfl, anatomy, boundary]

    The boundary loss is combined with the segmentation loss using the
    "rebalance" strategy from Kervadec et al.:
        L_total = (1 - α) * L_seg + α * L_boundary

    Where α starts small (0.01) and increases by 0.01 each epoch.

    Attributes:
        FDI_CLASSES (list): Full FDI tooth numbering (32 classes).
        boundary_loss (BoundaryLoss): Contour-based loss module.
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
    ANATOMY_START_RATIO = 0.33  # Start at ~1/3 of anatomy weight
    ANATOMY_RAMP_EPOCHS = 20  # Epochs to ramp up to full weight
    ANATOMY_LATE_BOOST_START_FRAC = 0.75  # Start boosting after this fraction of epochs
    ANATOMY_LATE_BOOST_MAX = 2.2  # Max multiplier by final epoch

    # Neighbor loss schedule parameters (three-phase)
    NEIGHBOR_PHASE1_END = 40  # End of suppressed phase
    NEIGHBOR_PHASE2_END = 60  # End of ramp-up phase
    NEIGHBOR_PHASE1_MAX_MULT = 0.1  # Max multiplier at end of phase 1
    NEIGHBOR_PHASE2_MAX_MULT = 0.5  # Max multiplier at end of phase 2
    NEIGHBOR_PHASE3_MAX_MULT = 1.5  # Max multiplier at end of phase 3

    # Adaptive margin parameters
    NEIGHBOR_BASE_MARGIN = 0.3  # Base margin (30% gap requirement)
    NEIGHBOR_MAX_MARGIN = 0.5  # Max margin at final epoch (50% gap)

    def __init__(self, model):
        """
        Initialize DentalSegmentationLoss.

        Args:
            model: De-parallelized model with hyperparameters.
        """
        super().__init__(model)

        # Anatomical constraint parameters
        self.max_gap = 3  # Max position gap for same-quadrant neighbors
        self.midline_threshold = 3  # Max position for cross-quadrant same-jaw neighbors
        self.cross_jaw_gap = 3  # Max position gap for cross-jaw (vertical) neighbors

        # Missing teeth handling: distance threshold for neighbor loss
        # Neighbors beyond threshold are ignored (likely missing teeth between them)
        self.neighbor_median_multiplier = 2.5  # Global: 2.5X median of closest distances
        self.neighbor_closest_multiplier = 2.0  # Per-tooth: 2.0X closest neighbor distance

        # Duplicate loss type: "pairwise" (recommended) or "soft" (legacy)
        # - "pairwise": Penalizes pairs of detections competing for same class
        #               Cannot be "gamed" by the model, sustained gradient signal
        # - "soft": Sum-based penalty when class probability sums exceed 1.0
        #           Can be gamed by making one detection borderline
        self.duplicate_loss_type = getattr(self.hyp, "duplicate_loss_type", "pairwise")

        # Build FDI mappings
        self._build_fdi_tensors()

        # Pre-compute adjacency validity matrix
        self._build_adjacency_matrix()

        # Boundary loss (simple and efficient)
        self.boundary_loss = BoundaryLoss()

        # Alpha scheduler using paper's "rebalance" strategy
        total_epochs = getattr(self.hyp, "epochs", 100)
        self.total_epochs = total_epochs
        alpha_start = getattr(self.hyp, "boundary_alpha_start", 0.01)
        alpha_increment = getattr(self.hyp, "boundary_alpha_increment", 0.01)

        self.alpha_scheduler = AlphaScheduler(
            total_epochs=total_epochs,
            schedule="rebalance",
            alpha_start=alpha_start,
            alpha_increment=alpha_increment,
            alpha_max=0.3,
        )
        anatomy_weight = getattr(self.hyp, "anatomy", 0.0)
        # Default to "neighbor" schedule which uses three-phase multiplier for late-stage focus
        self.anatomy_schedule = getattr(self.hyp, "anatomy_schedule", "neighbor")
        self.anatomy_scheduler = None
        if anatomy_weight > 0:
            if self.anatomy_schedule == "neighbor":
                # New three-phase schedule: returns base weight, multiplier applied in loss
                # The _get_neighbor_multiplier() handles epoch-based scheduling
                self.anatomy_scheduler = AlphaScheduler(
                    total_epochs=total_epochs,
                    schedule="constant",
                    alpha_start=anatomy_weight,
                    alpha_max=anatomy_weight,
                )
            elif self.anatomy_schedule == "sigmoid":
                # Sigmoid schedule: S-shaped curve with late ramp-up
                # Reads midpoint and steepness from config
                sigmoid_midpoint = getattr(self.hyp, "anatomy_sigmoid_midpoint", 0.7)
                sigmoid_steepness = getattr(self.hyp, "anatomy_sigmoid_steepness", 0.1)
                self.anatomy_scheduler = AlphaScheduler(
                    total_epochs=total_epochs,
                    schedule="sigmoid",
                    alpha_max=anatomy_weight,
                    sigmoid_midpoint=sigmoid_midpoint,
                    sigmoid_steepness=sigmoid_steepness,
                )
            elif self.anatomy_schedule == "linear":
                # Linear schedule: 0 to anatomy_weight over all epochs
                self.anatomy_scheduler = AlphaScheduler(
                    total_epochs=total_epochs,
                    schedule="linear",
                    alpha_max=anatomy_weight,
                )
            elif self.anatomy_schedule == "constant":
                # Constant: use anatomy_weight throughout
                self.anatomy_scheduler = AlphaScheduler(
                    total_epochs=total_epochs,
                    schedule="constant",
                    alpha_start=anatomy_weight,
                    alpha_max=anatomy_weight,
                )
            else:
                # Default: rebalance (legacy behavior)
                start = anatomy_weight * self.ANATOMY_START_RATIO
                ramp_epochs = min(self.ANATOMY_RAMP_EPOCHS, total_epochs) if total_epochs else self.ANATOMY_RAMP_EPOCHS
                if ramp_epochs <= 0:
                    increment = 0.0
                else:
                    increment = (anatomy_weight - start) / ramp_epochs
                self.anatomy_scheduler = AlphaScheduler(
                    total_epochs=total_epochs,
                    schedule="rebalance",
                    alpha_start=start,
                    alpha_increment=increment,
                    alpha_max=anatomy_weight,
                )
        self.current_epoch = 0
        timing_env = os.getenv("ULTRA_DENTAL_LOSS_TIMING", "")
        self._timing_enabled = timing_env not in ("", "0", "false", "False")
        if self._timing_enabled:
            self._timing_every = int(os.getenv("ULTRA_DENTAL_LOSS_TIMING_EVERY", "50"))
            sync_env = os.getenv("ULTRA_DENTAL_LOSS_TIMING_SYNC", "")
            self._timing_sync = sync_env not in ("", "0", "false", "False")
            self._timing = {"boundary": 0.0, "anatomy": 0.0, "batches": 0}

        # Neighbor loss logging (enabled via environment variable)
        neighbor_log_env = os.getenv("ULTRA_NEIGHBOR_LOSS_LOG", "")
        self._neighbor_log_enabled = neighbor_log_env not in ("", "0", "false", "False")
        self._neighbor_log = {"batches": 0, "raw_loss_sum": 0.0, "count": 0}
        if self._neighbor_log_enabled:
            self._neighbor_log_every = int(os.getenv("ULTRA_NEIGHBOR_LOSS_LOG_EVERY", "100"))

    def _build_fdi_tensors(self):
        """Build FDI lookup tensors for GPU efficiency."""
        # FDI codes tensor for direct lookup
        fdi_tensor = torch.tensor(self.FDI_CLASSES, dtype=torch.long, device=self.device)
        self.fdi_codes = fdi_tensor

        # Quadrant lookup (fdi // 10)
        quadrants = fdi_tensor // 10
        self.quadrant_lookup = quadrants

        # Position lookup (fdi % 10)
        positions = fdi_tensor % 10
        self.position_lookup = positions

        # Upper/Lower jaw masks
        upper_mask = (quadrants == 1) | (quadrants == 2)
        lower_mask = (quadrants == 3) | (quadrants == 4)
        self.upper_jaw_mask = upper_mask
        self.lower_jaw_mask = lower_mask

    def _build_adjacency_matrix(self):
        """
        Pre-compute adjacency validity matrix for all FDI class pairs.

        Three types of valid neighbors:
        1. Same quadrant: teeth within max_gap positions (e.g., 14-17)
        2. Cross-quadrant same jaw: midline teeth across Q1-Q2 or Q3-Q4 (e.g., 11-21)
        3. Cross-jaw same side: vertical neighbors from opposite jaws (e.g., 18-48, 17-48)
        """
        n = len(self.FDI_CLASSES)
        adjacency = torch.zeros((n, n), dtype=torch.bool, device=self.device)

        for i, fdi_i in enumerate(self.FDI_CLASSES):
            q_i = fdi_i // 10
            n_i = fdi_i % 10

            for j, fdi_j in enumerate(self.FDI_CLASSES):
                if i == j:
                    continue

                q_j = fdi_j // 10
                n_j = fdi_j % 10

                # Rule 1: Same quadrant with small gap
                # Example: 14 and 16 (same Q1, gap=2)
                if q_i == q_j and abs(n_i - n_j) <= self.max_gap:
                    adjacency[i, j] = True

                # Rule 2: Cross-quadrant same jaw near midline
                # Example: 11 and 21 (Q1-Q2 both upper, positions 1-1)
                elif self._same_jaw_static(q_i, q_j):
                    if n_i <= self.midline_threshold and n_j <= self.midline_threshold:
                        adjacency[i, j] = True

                # Rule 3: Cross-jaw same side (vertical neighbors)
                # Example: 18 and 48 (Q1-Q4 right side, same position 8)
                # Example: 17 and 48 (Q1-Q4 right side, positions 7-8, gap=1)
                elif self._same_side_static(q_i, q_j):
                    if abs(n_i - n_j) <= self.cross_jaw_gap:
                        adjacency[i, j] = True

        self.adjacency_valid = adjacency
        # Precompute float invalid adjacency matrix for soft neighbor loss (efficiency)
        # This avoids bool->float conversion on every forward pass
        self.invalid_adjacency = (~adjacency).float()

    @staticmethod
    def _same_jaw_static(q1: int, q2: int) -> bool:
        """Check if two quadrants are in the same jaw."""
        upper = {1, 2}
        return (q1 in upper) == (q2 in upper)

    @staticmethod
    def _same_side_static(q1: int, q2: int) -> bool:
        """
        Check if two quadrants are on the same side of the mouth (vertical alignment).

        This identifies teeth that can be cross-jaw neighbors:
        - Q1 (upper right) aligns with Q4 (lower right)
        - Q2 (upper left) aligns with Q3 (lower left)

        Returns:
            True if quadrants are on the same side but different jaws.
        """
        right_side = {1, 4}  # Upper right (Q1) and Lower right (Q4)
        left_side = {2, 3}  # Upper left (Q2) and Lower left (Q3)
        # Must be different jaws (one upper, one lower) AND same side
        same_jaw = (q1 in {1, 2}) == (q2 in {1, 2})
        same_side = (q1 in right_side and q2 in right_side) or (q1 in left_side and q2 in left_side)
        return (not same_jaw) and same_side

    def set_epoch(self, epoch: int):
        """Update current epoch for alpha scheduling."""
        self.current_epoch = epoch

    def _get_anatomy_weight(self) -> float:
        """Return scheduled anatomy weight for the current epoch."""
        anatomy_weight = getattr(self.hyp, "anatomy", 0.0)
        if self.anatomy_scheduler is not None:
            anatomy_weight = self.anatomy_scheduler.get_alpha(self.current_epoch)

        # Apply late boost only for legacy schedules (not sigmoid or neighbor)
        # Sigmoid already incorporates late ramp-up in its S-curve
        # Neighbor schedule uses _get_neighbor_multiplier() for late-stage focus
        if self.anatomy_schedule not in ("sigmoid", "neighbor"):
            if anatomy_weight > 0 and self.total_epochs:
                start_epoch = int(self.total_epochs * self.ANATOMY_LATE_BOOST_START_FRAC)
                if self.current_epoch >= start_epoch:
                    last_epoch = max(self.total_epochs - 1, start_epoch)
                    denom = max(1, last_epoch - start_epoch)
                    progress = min(1.0, (self.current_epoch - start_epoch) / denom)
                    anatomy_weight *= 1.0 + progress * (self.ANATOMY_LATE_BOOST_MAX - 1.0)

        return float(anatomy_weight)

    def _get_neighbor_multiplier(self) -> float:
        """
        Get weight multiplier for neighbor loss based on three-phase schedule.

        Phase 1 (Epochs 0-40):   SUPPRESSED  - multiplier = 0.0 → 0.1
        Phase 2 (Epochs 40-60):  RAMP-UP     - multiplier = 0.1 → 0.5
        Phase 3 (Epochs 60-100): FULL FORCE  - multiplier = 0.5 → 1.5 (exponential)

        Returns:
            float: Weight multiplier to apply to base anatomy weight.
        """
        epoch = self.current_epoch
        total = self.total_epochs if self.total_epochs else 100

        phase1_end = self.NEIGHBOR_PHASE1_END
        phase2_end = self.NEIGHBOR_PHASE2_END
        phase1_max = self.NEIGHBOR_PHASE1_MAX_MULT
        phase2_max = self.NEIGHBOR_PHASE2_MAX_MULT
        phase3_max = self.NEIGHBOR_PHASE3_MAX_MULT

        if epoch < phase1_end:
            # Phase 1: Linear from 0 to phase1_max
            progress = epoch / phase1_end
            multiplier = phase1_max * progress
        elif epoch < phase2_end:
            # Phase 2: Linear from phase1_max to phase2_max
            progress = (epoch - phase1_end) / (phase2_end - phase1_end)
            multiplier = phase1_max + (phase2_max - phase1_max) * progress
        else:
            # Phase 3: Exponential from phase2_max to phase3_max
            remaining = total - phase2_end
            if remaining <= 0:
                multiplier = phase3_max
            else:
                progress = (epoch - phase2_end) / remaining
                # Exponential curve: smoother ramp to peak
                exp_factor = (math.exp(2 * progress) - 1) / (math.exp(2) - 1)
                multiplier = phase2_max + (phase3_max - phase2_max) * exp_factor

        return float(multiplier)

    def _get_adaptive_margin(self) -> float:
        """
        Get adaptive margin for neighbor loss based on training progress.

        Margin increases linearly from base_margin to max_margin over training.
        This demands larger class separation as predictions sharpen.

        Returns:
            float: Margin value for softplus penalty.
        """
        epoch = self.current_epoch
        total = self.total_epochs if self.total_epochs else 100

        progress = epoch / total if total > 0 else 0.0
        margin = self.NEIGHBOR_BASE_MARGIN + (self.NEIGHBOR_MAX_MARGIN - self.NEIGHBOR_BASE_MARGIN) * progress

        return float(margin)

    def __call__(self, preds, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate combined loss for dental segmentation.

        Returns 6 loss components: [box, seg, cls, dfl, anatomy, boundary]
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
        anatomy_weight = self._get_anatomy_weight()

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

            # Standard segmentation loss
            boundary_weight = getattr(self.hyp, "boundary", 0.0)
            rep_positions = None
            rep_gt_indices = None
            if boundary_weight > 0:
                rep_positions = []
                rep_gt_indices = []
                for i in range(batch_size):
                    fg_mask_i = fg_mask[i]
                    if not fg_mask_i.any():
                        rep_positions.append(torch.empty(0, dtype=torch.long, device=fg_mask.device))
                        rep_gt_indices.append(torch.empty(0, dtype=torch.long, device=fg_mask.device))
                        continue

                    fg_idx = fg_mask_i.nonzero(as_tuple=False).squeeze(1)
                    gt_idx = target_gt_idx[i, fg_idx]
                    unique_gt, inv = gt_idx.unique(return_inverse=True)

                    if unique_gt.numel() == 0:
                        rep_positions.append(torch.empty(0, dtype=torch.long, device=fg_mask.device))
                        rep_gt_indices.append(torch.empty(0, dtype=torch.long, device=fg_mask.device))
                        continue

                    anchor_scores = target_scores[i, fg_idx].max(dim=1).values
                    rep_pos = torch.empty_like(unique_gt)
                    for j in range(unique_gt.numel()):
                        mask = inv == j
                        candidates = torch.arange(inv.numel(), device=fg_mask.device)[mask]
                        scores = anchor_scores[mask]
                        rep_pos[j] = candidates[scores.argmax()]

                    rep_positions.append(rep_pos)
                    rep_gt_indices.append(unique_gt)

            if rep_positions is not None:
                seg_loss, rep_pred_masks = self.calculate_segmentation_loss(
                    fg_mask,
                    masks,
                    target_gt_idx,
                    target_bboxes,
                    batch_idx,
                    proto,
                    pred_masks,
                    imgsz,
                    self.overlap,
                    rep_positions=rep_positions,
                )
            else:
                seg_loss = self.calculate_segmentation_loss(
                    fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz, self.overlap
                )
            loss[1] = seg_loss

            # Anatomical constraint loss
            if anatomy_weight > 0:
                if self._timing_enabled:
                    if self._timing_sync and pred_scores.is_cuda:
                        torch.cuda.synchronize()
                    t_start = time.perf_counter()
                loss[4] = self.compute_anatomy_loss_vectorized(
                    pred_scores, pred_bboxes * stride_tensor, target_scores, target_gt_idx, fg_mask
                )
                if self._timing_enabled:
                    if self._timing_sync and pred_scores.is_cuda:
                        torch.cuda.synchronize()
                    self._timing["anatomy"] += time.perf_counter() - t_start

            # Boundary loss
            if boundary_weight > 0:
                if self._timing_enabled:
                    if self._timing_sync and pred_scores.is_cuda:
                        torch.cuda.synchronize()
                    t_start = time.perf_counter()
                boundary_loss_val = self.compute_boundary_loss(
                    fg_mask,
                    masks,
                    target_gt_idx,
                    target_bboxes,
                    batch_idx,
                    proto,
                    pred_masks,
                    imgsz,
                    target_scores,
                    rep_pred_masks=rep_pred_masks,
                    rep_gt_indices=rep_gt_indices,
                )
                loss[5] = boundary_loss_val
                if self._timing_enabled:
                    if self._timing_sync and pred_scores.is_cuda:
                        torch.cuda.synchronize()
                    self._timing["boundary"] += time.perf_counter() - t_start

        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()

        # Apply loss weights
        # For boundary loss, use rebalance strategy: (1-α)*L_seg + α*L_boundary
        boundary_weight = getattr(self.hyp, "boundary", 0.0)

        loss[0] *= self.hyp.box  # Box loss

        if boundary_weight > 0:
            # Get alpha for current epoch
            alpha = self.alpha_scheduler.get_alpha(self.current_epoch)

            # Rebalance: total seg contribution = box * [(1-α)*seg + α*boundary]
            # This keeps total segmentation contribution constant across epochs
            loss[1] *= self.hyp.box * (1.0 - alpha)  # Seg loss weighted by (1-α)
            loss[5] *= self.hyp.box * alpha  # Boundary loss weighted by α
        else:
            # No boundary loss, full weight to seg
            loss[1] *= self.hyp.box

        loss[2] *= self.hyp.cls  # Classification loss
        loss[3] *= self.hyp.dfl  # DFL loss
        loss[4] *= anatomy_weight  # Anatomy loss

        if self._timing_enabled:
            self._timing["batches"] += 1
            if self._timing["batches"] % self._timing_every == 0:
                LOGGER.info(
                    f"Dental loss timing (last {self._timing_every} batches): "
                    f"boundary={self._timing['boundary']:.3f}s, "
                    f"anatomy={self._timing['anatomy']:.3f}s, "
                    f"sync={self._timing_sync}"
                )
                self._timing["boundary"] = 0.0
                self._timing["anatomy"] = 0.0

        return loss * batch_size, loss.detach()

    def compute_boundary_loss(
        self,
        fg_mask: torch.Tensor,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        target_bboxes: torch.Tensor,
        batch_idx: torch.Tensor,
        proto: torch.Tensor,
        pred_masks: torch.Tensor,
        imgsz: torch.Tensor,
        target_scores: torch.Tensor,
        rep_pred_masks: list[torch.Tensor] | None = None,
        rep_gt_indices: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Compute boundary loss for mask quality.

        Collects masks across batch and computes boundary loss in a single forward pass.
        Uses a single representative anchor per ground-truth instance to reduce EDT cost.

        Args:
            fg_mask: Foreground mask for each batch item.
            masks: Ground truth masks.
            target_gt_idx: Target ground truth indices.
            target_bboxes: Target bounding boxes.
            batch_idx: Batch indices.
            proto: Prototype masks.
            pred_masks: Predicted mask coefficients.
            imgsz: Image size.
            target_scores: Target scores per anchor for selection.
            rep_pred_masks: Optional per-image predicted mask logits for representative anchors.
            rep_gt_indices: Optional per-image GT indices for representative anchors.

        Returns:
            Scalar boundary loss value.
        """
        all_pred_masks = []
        all_gt_masks = []

        use_cached = rep_pred_masks is not None and rep_gt_indices is not None

        for i in range(fg_mask.shape[0]):
            if use_cached:
                pred_mask_logits = rep_pred_masks[i]
                gt_idx = rep_gt_indices[i]
                if pred_mask_logits.numel() == 0:
                    continue

                if self.overlap:
                    gt_mask = masks[i] == (gt_idx + 1).view(-1, 1, 1)
                    gt_mask = gt_mask.float()
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][gt_idx]

                all_pred_masks.append(pred_mask_logits.sigmoid())
                all_gt_masks.append(gt_mask)
            else:
                fg_mask_i = fg_mask[i]
                if not fg_mask_i.any():
                    continue

                target_gt_idx_i = target_gt_idx[i]
                pred_masks_i = pred_masks[i]
                proto_i = proto[i]

                fg_idx = fg_mask_i.nonzero(as_tuple=False).squeeze(1)
                gt_idx = target_gt_idx_i[fg_idx]

                unique_gt, inv = gt_idx.unique(return_inverse=True)
                if unique_gt.numel() == 0:
                    continue

                # Select one representative anchor per GT to avoid per-anchor EDT cost.
                anchor_scores = target_scores[i, fg_idx].max(dim=1).values
                rep_anchor = torch.empty_like(unique_gt)
                for j in range(unique_gt.numel()):
                    mask = inv == j
                    candidates = fg_idx[mask]
                    scores = anchor_scores[mask]
                    rep_anchor[j] = candidates[scores.argmax()]

                if self.overlap:
                    gt_mask = masks[i] == (unique_gt + 1).view(-1, 1, 1)
                    gt_mask = gt_mask.float()
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][unique_gt]

                # Compute predicted masks (apply sigmoid for probability).
                pred_mask = torch.einsum("in,nhw->ihw", pred_masks_i[rep_anchor], proto_i).sigmoid()

                all_pred_masks.append(pred_mask)
                all_gt_masks.append(gt_mask)

        if not all_pred_masks:
            return torch.tensor(0.0, device=proto.device)

        # Concatenate all masks
        pred_masks_cat = torch.cat(all_pred_masks, dim=0)
        gt_masks_cat = torch.cat(all_gt_masks, dim=0)

        return self.boundary_loss(pred_masks_cat, gt_masks_cat)

    def compute_anatomy_loss_vectorized(
        self,
        pred_scores: torch.Tensor,
        pred_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_gt_idx: torch.Tensor,
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
            if not fg_i.any():
                continue

            fg_idx = fg_i.nonzero(as_tuple=False).squeeze(1)
            gt_idx = target_gt_idx[i, fg_idx]

            unique_gt, inv = gt_idx.unique(return_inverse=True)
            if unique_gt.numel() < 2:
                continue

            # Select one representative anchor per GT (highest scoring)
            anchor_scores = target_scores[i, fg_idx].max(dim=1).values
            rep_anchor = torch.empty_like(unique_gt)
            for j in range(unique_gt.numel()):
                mask = inv == j
                candidates = fg_idx[mask]
                scores = anchor_scores[mask]
                rep_anchor[j] = candidates[scores.argmax()]

            # Get raw scores and bboxes for representative anchors
            pred_scores_subset = pred_scores[i, rep_anchor]  # (n_teeth, num_classes)
            bboxes = pred_bboxes[i, rep_anchor]

            # Compute loss components
            # Duplicate loss (temporarily disabled for neighbor loss experiments)
            # TODO: Re-enable after neighbor loss experiments
            # if self.duplicate_loss_type == "pairwise":
            #     dup_loss = self._duplicate_loss_pairwise(pred_scores_subset)
            # else:
            #     dup_loss = self._duplicate_loss_soft(pred_scores_subset)
            dup_loss = torch.tensor(0.0, device=pred_scores_subset.device)

            # Margin-based neighbor loss with softplus (new implementation)
            # Returns raw loss; weight multiplier applied below
            neighbor_loss = self._neighbor_loss_margin(pred_scores_subset, bboxes)

            # Legacy soft neighbor loss (disabled - kept for reference)
            # neighbor_loss = self._neighbor_loss_soft_legacy(pred_scores_subset, bboxes)

            # Ordering loss (currently disabled - uses hard class assignments)
            ordering_loss = torch.tensor(0.0, device=pred_scores_subset.device)

            # Apply three-phase weight multiplier to neighbor loss
            # This implements late-stage focus: suppressed early, full force late
            neighbor_multiplier = self._get_neighbor_multiplier()

            # Log raw loss before multiplier (for monitoring)
            if self._neighbor_log_enabled:
                self._neighbor_log["raw_loss_sum"] += neighbor_loss.detach().item()
                self._neighbor_log["count"] += 1

            neighbor_loss = neighbor_loss * neighbor_multiplier

            # Note: sqrt normalization removed for margin loss since _neighbor_loss_margin
            # already returns mean() over all pairs (proper normalization).
            # Legacy losses (dup_loss, ordering_loss) would need sqrt normalization if re-enabled.
            total_loss = total_loss + (dup_loss + neighbor_loss + ordering_loss)
            valid_count = valid_count + 1.0

        # Periodic logging of neighbor loss statistics
        if self._neighbor_log_enabled:
            self._neighbor_log["batches"] += 1
            if self._neighbor_log["batches"] % self._neighbor_log_every == 0:
                avg_raw = self._neighbor_log["raw_loss_sum"] / max(1, self._neighbor_log["count"])
                margin = self._get_adaptive_margin()
                multiplier = self._get_neighbor_multiplier()
                LOGGER.info(
                    f"Neighbor loss [epoch {self.current_epoch}]: "
                    f"raw={avg_raw:.4f}, margin={margin:.3f}, mult={multiplier:.3f}, "
                    f"effective={avg_raw * multiplier:.4f}"
                )
                self._neighbor_log["raw_loss_sum"] = 0.0
                self._neighbor_log["count"] = 0

        return total_loss / torch.clamp(valid_count, min=1.0)

    def _duplicate_loss_fast(self, pred_classes: torch.Tensor) -> torch.Tensor:
        """Penalize duplicate class predictions - fast version (hard counting)."""
        counts = torch.zeros(self.nc, device=pred_classes.device)
        ones = torch.ones_like(pred_classes, dtype=counts.dtype)
        counts.scatter_add_(0, pred_classes, ones)
        return F.relu(counts - 1.0).sum()

    def _duplicate_loss_soft(self, pred_scores_subset: torch.Tensor) -> torch.Tensor:
        """
        Penalize duplicate class predictions using soft probabilistic counting.

        Unlike _duplicate_loss_fast which uses hard argmax (non-differentiable),
        this method uses softmax probabilities allowing gradients to flow back
        to the classification head.

        Args:
            pred_scores_subset: Raw logits for representative anchors, shape (n_teeth, num_classes).

        Returns:
            Differentiable duplicate loss scalar.

        Example:
            If two detections each assign 0.8 probability to tooth 12:
            class_prob_sums[tooth_12] = 1.6 → penalty = ReLU(1.6 - 1.0) = 0.6
        """
        # Convert logits to probabilities
        pred_probs = pred_scores_subset.softmax(dim=-1)  # (n_teeth, num_classes)

        # Sum probabilities for each class across all detections
        class_prob_sums = pred_probs.sum(dim=0)  # (num_classes,)

        # Penalize when sum > 1.0 (indicates duplicate predictions)
        soft_dup_loss = F.relu(class_prob_sums - 1.0).sum()

        return soft_dup_loss

    def _duplicate_loss_pairwise(self, pred_scores_subset: torch.Tensor) -> torch.Tensor:
        """
        Pairwise contrastive duplicate loss.

        Penalizes when two detections both have high probability for the same class.
        The product prob[i,c] * prob[j,c] is high only when BOTH detections want
        class c, directly targeting the duplicate problem.

        Unlike _duplicate_loss_soft which can be "gamed" by the model (making one
        detection confident and another borderline), this approach requires the
        model to actually predict different classes to reduce the loss.

        Mathematical interpretation:
            L = Σ_c Σ_{i≠j} prob[i,c] * prob[j,c]
              = Σ_c (Σ_i prob[i,c])² - Σ_c Σ_i prob[i,c]²

        This penalizes concentrated probability mass on any single class across
        multiple detections.

        Args:
            pred_scores_subset: Raw logits for representative anchors, shape (n_teeth, num_classes).

        Returns:
            Differentiable duplicate loss scalar.

        Example:
            Detection A: [0.90, 0.05, 0.05] for class 0 (confident)
            Detection B: [0.55, 0.40, 0.05] for class 0 (borderline, but argmax=0)

            Pairwise penalty for class 0 = 0.90 × 0.55 = 0.495
            This is HIGH even though B is borderline.

            To reduce this, B must shift probability away from class 0,
            which will change its argmax prediction - eliminating the duplicate.
        """
        pred_probs = pred_scores_subset.softmax(dim=-1)  # (n, C)
        n = pred_probs.shape[0]

        if n < 2:
            return torch.tensor(0.0, device=pred_probs.device)

        # Compute pairwise products for all classes at once
        # pair_products[i,j,c] = prob[i,c] * prob[j,c]
        # High value means both detection i and j want class c
        pair_products = pred_probs.unsqueeze(1) * pred_probs.unsqueeze(0)  # (n, n, C)

        # Create mask to exclude diagonal (self-pairs add bias, not competition)
        mask = 1.0 - torch.eye(n, device=pred_probs.device)  # (n, n)
        mask = mask.unsqueeze(-1)  # (n, n, 1) for broadcasting

        # Sum all pairwise competitions, normalized by number of detections
        # Dividing by n (not n*(n-1)) keeps loss magnitude comparable to soft loss
        loss = (pair_products * mask).sum() / n

        return loss

    def _neighbor_loss_fast(self, pred_classes: torch.Tensor, bboxes: torch.Tensor, k: int = 2) -> torch.Tensor:
        """Penalize invalid spatial neighbors - fast version.

        Includes distance threshold to handle missing teeth: neighbors beyond
        a dynamic threshold are ignored (likely missing teeth between them).
        """
        n = pred_classes.shape[0]
        if n < 2:
            return torch.tensor(0.0, device=pred_classes.device)

        # Compute bbox centers
        centers = (bboxes[:, :2] + bboxes[:, 2:4]) / 2

        # Pairwise distances (torch.cdist does not support fp16 on CUDA)
        centers_f = centers.float()
        dists = torch.cdist(centers_f.unsqueeze(0), centers_f.unsqueeze(0)).squeeze(0)
        dists = dists + torch.eye(n, device=dists.device) * 1e6

        # Find k nearest neighbors
        k_actual = min(k, n - 1)
        _, knn_idx = dists.topk(k_actual, dim=1, largest=False)

        # Skip if class count doesn't match FDI
        if self.nc != len(self.FDI_CLASSES):
            return torch.tensor(0.0, device=pred_classes.device)

        # Compute distance thresholds for missing teeth handling
        # Closest distance for each tooth
        closest_dists = dists.min(dim=1).values  # Shape: [n]

        # Global threshold: multiplier × median of closest distances
        median_closest = closest_dists.median()
        global_threshold = self.neighbor_median_multiplier * median_closest

        # Per-tooth threshold: multiplier × each tooth's closest distance
        per_tooth_threshold = self.neighbor_closest_multiplier * closest_dists

        # Combined threshold: minimum of global and per-tooth
        threshold = torch.minimum(global_threshold.expand(n), per_tooth_threshold)

        # Vectorized validity check
        loss = torch.tensor(0.0, device=pred_classes.device)
        for j in range(k_actual):
            neighbor_idx = knn_idx[:, j]
            neighbor_classes = pred_classes[neighbor_idx]
            neighbor_dists = dists[torch.arange(n, device=dists.device), neighbor_idx]

            # Check validity using pre-registered adjacency matrix
            valid_mask = self.adjacency_valid[pred_classes, neighbor_classes]

            # Only penalize invalid neighbors within threshold (skip far neighbors - missing teeth)
            within_threshold = neighbor_dists <= threshold
            invalid_mask = ~valid_mask & within_threshold

            penalty = (invalid_mask.float() / (neighbor_dists + 1.0)).sum()
            loss = loss + penalty

        return loss

    def _neighbor_loss_margin(self, pred_scores: torch.Tensor, bboxes: torch.Tensor, k: int = 2) -> torch.Tensor:
        """
        Margin-based neighbor loss using softplus penalty.

        For each detection, computes the gap between probability on valid classes
        vs. invalid classes (given the neighbor's predicted class). Uses softplus
        to create a smooth, differentiable penalty that never fully vanishes.

        Formula:
            gap = max(prob_valid) - max(prob_invalid)
            penalty = softplus(margin - gap) = log(1 + exp(margin - gap))

        The penalty is high when:
        - Detection predicts an invalid neighbor class (gap < 0)
        - Detection is correct but not confident enough (0 < gap < margin)

        The penalty is low (but never zero) when:
        - Detection confidently predicts a valid class (gap > margin)

        Args:
            pred_scores: Raw logits for detections, shape (n_teeth, num_classes).
            bboxes: Bounding boxes in xyxy format, shape (n_teeth, 4).
            k: Number of nearest neighbors to check (default 2).

        Returns:
            Differentiable neighbor loss scalar (unweighted - weight applied externally).
        """
        n = pred_scores.shape[0]
        dtype = pred_scores.dtype
        device = pred_scores.device

        if n < 2:
            return torch.tensor(0.0, device=device, dtype=dtype)

        # Skip if class count doesn't match FDI
        if self.nc != len(self.FDI_CLASSES):
            return torch.tensor(0.0, device=device, dtype=dtype)

        # Get adaptive margin for current epoch
        margin = self._get_adaptive_margin()

        # Convert logits to probabilities
        probs = pred_scores.softmax(dim=-1)  # (n, C)

        # Compute bbox centers
        centers = (bboxes[:, :2] + bboxes[:, 2:4]) / 2

        # Pairwise distances (torch.cdist does not support fp16 on CUDA)
        centers_f = centers.float()
        dists = torch.cdist(centers_f.unsqueeze(0), centers_f.unsqueeze(0)).squeeze(0)
        dists = dists + torch.eye(n, device=device) * 1e6  # Exclude self

        # Find k nearest neighbors
        k_actual = min(k, n - 1)
        _, knn_idx = dists.topk(k_actual, dim=1, largest=False)

        # Compute distance thresholds for missing teeth handling
        closest_dists = dists.min(dim=1).values
        median_closest = closest_dists.median()
        global_threshold = self.neighbor_median_multiplier * median_closest
        per_tooth_threshold = self.neighbor_closest_multiplier * closest_dists
        threshold = torch.minimum(global_threshold.expand(n), per_tooth_threshold)

        # Get adjacency matrices (cast for mixed precision)
        invalid_adjacency = self.invalid_adjacency.to(dtype=dtype)  # (C, C)
        valid_adjacency = 1.0 - invalid_adjacency  # (C, C)

        # Pre-compute values used in loop (efficiency optimization)
        arange_n = torch.arange(n, device=device)
        likely_class_self = probs.argmax(dim=1)  # (n,) - same for all neighbors

        # Pre-allocate penalty tensor
        all_penalties = torch.zeros(k_actual, n, device=device, dtype=dtype)

        for j in range(k_actual):
            neighbor_idx = knn_idx[:, j]  # (n,) indices of j-th nearest neighbor
            neighbor_probs = probs[neighbor_idx]  # (n, C)
            neighbor_dists = dists[arange_n, neighbor_idx]  # (n,)

            # Get most likely class for each neighbor (conditioning)
            likely_class_neighbor = neighbor_probs.argmax(dim=1)  # (n,)

            # For each detection i, get invalid/valid masks based on neighbor's class
            invalid_mask = invalid_adjacency[likely_class_neighbor]  # (n, C)
            valid_mask = valid_adjacency[likely_class_neighbor]  # (n, C)

            # Compute max probability on valid and invalid classes
            # probs * mask gives 0 where mask is 0, so max of zeros = 0 (correct behavior)
            max_valid_prob = (probs * valid_mask).max(dim=1).values  # (n,)
            max_invalid_prob = (probs * invalid_mask).max(dim=1).values  # (n,)

            # Compute gap: positive when valid class dominates
            gap = max_valid_prob - max_invalid_prob  # (n,)

            # Softplus penalty: log(1 + exp(margin - gap))
            penalty = F.softplus(margin - gap)  # (n,)

            # Apply distance threshold (ignore far neighbors - likely missing teeth)
            within_threshold = (neighbor_dists <= threshold).to(dtype=dtype)

            # Symmetric: compute penalty for neighbor conditioned on this detection
            invalid_mask_neighbor = invalid_adjacency[likely_class_self]  # (n, C)
            valid_mask_neighbor = valid_adjacency[likely_class_self]  # (n, C)

            max_valid_prob_neighbor = (neighbor_probs * valid_mask_neighbor).max(dim=1).values
            max_invalid_prob_neighbor = (neighbor_probs * invalid_mask_neighbor).max(dim=1).values

            gap_neighbor = max_valid_prob_neighbor - max_invalid_prob_neighbor
            penalty_neighbor = F.softplus(margin - gap_neighbor)

            # Average symmetric penalties, apply threshold mask
            all_penalties[j] = 0.5 * (penalty + penalty_neighbor) * within_threshold

        # Average over all neighbor pairs
        loss = all_penalties.mean()

        return loss

    def _neighbor_loss_soft_legacy(self, pred_scores: torch.Tensor, bboxes: torch.Tensor, k: int = 2) -> torch.Tensor:
        """
        LEGACY: Soft differentiable neighbor loss using joint class probabilities.

        NOTE: This method is disabled but kept for reference. It suffers from
        vanishing gradients as predictions sharpen (joint probability → 0).
        Use _neighbor_loss_margin instead.

        Unlike _neighbor_loss_fast which uses hard argmax (non-differentiable),
        this method computes the expected probability that two spatially adjacent
        detections predict an invalid neighbor pair, allowing gradients to flow
        back to the classification head.

        Formula:
            For detection i and its k nearest spatial neighbors j:
            invalid_prob[i,j] = prob_i^T @ invalid_adjacency @ prob_j
            penalty[i,j] = invalid_prob[i,j] / (distance[i,j] + 1.0)

        Args:
            pred_scores: Raw logits for detections, shape (n_teeth, num_classes).
            bboxes: Bounding boxes in xyxy format, shape (n_teeth, 4).
            k: Number of nearest neighbors to check (default 2).

        Returns:
            Differentiable neighbor loss scalar.
        """
        n = pred_scores.shape[0]
        if n < 2:
            return torch.tensor(0.0, device=pred_scores.device, dtype=pred_scores.dtype)

        # Skip if class count doesn't match FDI
        if self.nc != len(self.FDI_CLASSES):
            return torch.tensor(0.0, device=pred_scores.device, dtype=pred_scores.dtype)

        # Convert logits to probabilities
        probs = pred_scores.softmax(dim=-1)  # (n, C)

        # Compute bbox centers
        centers = (bboxes[:, :2] + bboxes[:, 2:4]) / 2

        # Pairwise distances
        centers_f = centers.float()
        dists = torch.cdist(centers_f.unsqueeze(0), centers_f.unsqueeze(0)).squeeze(0)
        dists = dists + torch.eye(n, device=dists.device) * 1e6  # Exclude self

        # Find k nearest neighbors
        k_actual = min(k, n - 1)
        _, knn_idx = dists.topk(k_actual, dim=1, largest=False)

        # Compute distance thresholds for missing teeth handling
        closest_dists = dists.min(dim=1).values
        median_closest = closest_dists.median()
        global_threshold = self.neighbor_median_multiplier * median_closest
        per_tooth_threshold = self.neighbor_closest_multiplier * closest_dists
        threshold = torch.minimum(global_threshold.expand(n), per_tooth_threshold)

        # Use precomputed invalid adjacency matrix (efficiency optimization)
        # Cast to same dtype as probs for mixed precision training compatibility
        invalid_adjacency = self.invalid_adjacency.to(dtype=probs.dtype)  # (C, C)

        # Compute soft neighbor loss
        loss = torch.tensor(0.0, device=pred_scores.device, dtype=probs.dtype)

        for j in range(k_actual):
            neighbor_idx = knn_idx[:, j]  # (n,) indices of j-th nearest neighbor
            neighbor_probs = probs[neighbor_idx]  # (n, C)
            neighbor_dists = dists[torch.arange(n, device=dists.device), neighbor_idx]  # (n,)

            # Compute expected invalidity for each pair
            # invalid_prob[i] = probs[i] @ invalid_adjacency @ neighbor_probs[i]
            # Vectorized: probs @ invalid_adjacency -> (n, C), then element-wise with neighbor_probs
            probs_times_invalid = probs @ invalid_adjacency  # (n, C)
            invalid_prob = (probs_times_invalid * neighbor_probs).sum(dim=1)  # (n,)

            # Apply hard distance threshold (missing teeth handling)
            within_threshold = (neighbor_dists <= threshold).to(dtype=probs.dtype)

            # Distance-weighted penalty with threshold mask
            penalty = (invalid_prob * within_threshold / (neighbor_dists + 1.0)).sum()
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
