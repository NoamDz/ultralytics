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
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from ultralytics.utils import LOGGER
from ultralytics.utils.loss import v8DetectionLoss, v8SegmentationLoss


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


class _MarginCRFFunction(torch.autograd.Function):
    """Custom autograd for margin CRF loss with explicit forward-backward marginals."""

    @staticmethod
    def forward(ctx, emissions, gt_pos_sorted, marginals, raw_loss_val, margin):
        """
        Args:
            emissions: (N, C) requires_grad — the tensor we need gradient for.
            gt_pos_sorted: (N,) long — sorted GT spatial positions.
            marginals: (N, C) detached — precomputed from forward-backward.
            raw_loss_val: scalar — (log_Z - gt_score) / N, detached.
            margin: float.
        """
        loss_val = max(0.0, raw_loss_val.item() - margin)
        loss = torch.tensor(loss_val, device=emissions.device, dtype=emissions.dtype)

        ctx.save_for_backward(marginals, gt_pos_sorted)
        ctx.active = loss_val > 0
        ctx.N = emissions.shape[0]
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        if not ctx.active:
            return (torch.zeros_like(ctx.saved_tensors[0]),) + (None,) * 4

        marginals, gt_pos_sorted = ctx.saved_tensors
        N = ctx.N

        # Analytical CRF gradient: (marginals - GT_indicator) / N
        grad = marginals / N
        arange = torch.arange(N, device=grad.device)
        grad[arange, gt_pos_sorted.long()] -= 1.0 / N

        return (grad_output * grad,) + (None,) * 4


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

    # Neighbor loss schedule parameters (constant + late boost)
    # No suppression: starts at 1.0, ramp 70-80 to 5.0, max from 80 onwards
    NEIGHBOR_BOOST_START = 70  # Epoch to start boosting
    NEIGHBOR_BOOST_END = 80  # Epoch to reach maximum (most effectiveness from here)
    NEIGHBOR_MAX_MULT = 5.0  # Maximum multiplier (reached at BOOST_END, held until end)

    # CRF warmup schedule: let model learn basics first, then introduce CRF
    # Epochs 0 to CRF_WARMUP_START: multiplier = 0 (no CRF)
    # Epochs CRF_WARMUP_START to CRF_WARMUP_END: linear ramp 0 → 1.0
    # Epochs CRF_WARMUP_END to end: multiplier = 1.0 (full force)
    CRF_WARMUP_START = 30  # Start introducing CRF after model has learned basics
    CRF_WARMUP_END = 60  # Full CRF force from this epoch onwards

    # Adaptive margin parameters
    NEIGHBOR_BASE_MARGIN = 0.3  # Base margin (30% gap requirement)
    NEIGHBOR_MAX_MARGIN = 0.5  # Max margin at final epoch (50% gap)

    # Ordinal classification loss parameters
    ORDINAL_SAME_QUAD_BASE = 0  # Base distance for same quadrant (position diff added)
    ORDINAL_ADJ_QUAD_BASE = 4   # Base distance for adjacent quadrants (midline/vertical)
    ORDINAL_OPP_QUAD_BASE = 8   # Base distance for opposite quadrants (Q1-Q3, Q2-Q4)

    # Distance Regularization (DR) Loss parameters (Chung et al., 2021)
    # Empirically determined from validation dataset analysis
    DR_UPPER_MIDLINE_GAP = 1.18  # Upper midline is WIDER than baseline (92.80/78.95)
    DR_LOWER_MIDLINE_GAP = 0.63  # Lower midline is NARROWER than baseline (49.54/78.95)
    DR_MIN_TEETH_PER_SEGMENT = 4  # Minimum teeth for Laplacian (need 3 distances → 1 value)

    def __init__(self, model):
        """
        Initialize DentalSegmentationLoss.

        Args:
            model: De-parallelized model with hyperparameters.
        """
        super().__init__(model)
        self._init_dental_components(model)

    def _init_dental_components(self, model):
        """Initialize dental-specific components (shared by segmentation and detection variants)."""
        # Anatomical constraint parameters
        self.max_gap = 3  # Max position gap for same-quadrant neighbors
        self.midline_threshold = 3  # Max position for cross-quadrant same-jaw neighbors
        self.cross_jaw_gap = 3  # Max position gap for cross-jaw (vertical) neighbors

        # Missing teeth handling: distance threshold for neighbor loss
        # Neighbors beyond threshold are ignored (likely missing teeth between them)
        self.neighbor_median_multiplier = 2.5  # Global: 2.5X median of closest distances
        self.neighbor_closest_multiplier = 2.0  # Per-tooth: 2.0X closest neighbor distance

        # Duplicate loss type options:
        # - "soft": Sum-based penalty when class probability sums exceed 1.0
        #           Can be gamed by making one detection borderline
        # - "pairwise": Penalizes pairs of detections competing for same class
        #               Cannot be "gamed" by the model, sustained gradient signal
        # - "hungarian": DETR-style bipartite matching for optimal 1:1 assignment (DEFAULT)
        #                Explicit unique targets, strongest gradient signal
        #                Reference: Carion et al., ECCV 2020
        # - "hungarian_violation_gated": OPH cost term only on anchors currently
        #                                involved in ordering/duplicate violations.
        #                                Order term magnitude is scaled to be a
        #                                light fraction of classification cost.
        # - "hungarian_order_margin": Hungarian CE + violation-gated ordered-competitor
        #                             margin (HOCM). Preserves Hungarian strengths while
        #                             adding targeted ordering pressure only on unordered
        #                             assignments.
        self.duplicate_loss_type = getattr(self.hyp, "duplicate_loss_type", "hungarian")

        # Anatomy loss type options:
        # - "components": Use separate duplicate loss + neighbor loss (DEFAULT, current behavior)
        # - "ordered": Use ordered assignment loss only (replaces both duplicate and neighbor)
        #              Orders anchors by GT class FDI position, finds optimal monotonic assignment
        #              Combines uniqueness (like Hungarian) with ordering constraint
        self.anatomy_loss_type = getattr(self.hyp, "anatomy_loss_type", "components")

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

        # GT-conditioned neighbor loss (aligns training with evaluation metric)
        # Default: False - GT-conditioned was found to hurt performance
        # Uses soft neighbor loss (original implementation) instead
        self.neighbor_gt_conditioned = getattr(self.hyp, "neighbor_gt_conditioned", False)

        # Ordinal classification loss settings
        # Default: 0.3 - adds 30% extra penalty per unit of class distance
        # This penalizes distant class errors more than nearby errors
        self.ordinal_alpha = getattr(self.hyp, "ordinal_alpha", 0.3)
        if self.ordinal_alpha > 0:
            self._build_distance_matrix()

        # Distance Regularization (DR) Loss (Chung et al., 2021)
        # Enforces smooth inter-tooth spacing via Laplacian regularization
        # Set dr > 0 in config to enable (e.g., dr: 0.1)
        self.dr_weight = getattr(self.hyp, "dr", 0.0)
        self.dr_enabled = self.dr_weight > 0

        # Always build spatial position lookup (needed by CRF loss and optionally DR loss)
        self._build_dr_linear_position_lookup()

        # CRF temperature: prevents emission scores from degenerating when BCE
        # drives logits to large magnitudes. Without this, CRF loss vanishes by epoch 30-40.
        # T=5 gives "moderate signal" at typical late-training logit scales (±10-12).
        # Lower T → weaker signal (T=4 is marginal). Higher T → stronger but noisier.
        self.crf_temperature = getattr(self.hyp, "crf_temperature", 5.0)
        self.crf_margin = getattr(self.hyp, "crf_margin", 0.05)

        # Order-Penalized Hungarian (OPH) lambda parameter
        # Controls ordering bias strength in Hungarian cost matrix.
        # With -prob cost: lambda=0.03 gives ~3% ordering bias at trained logit scales.
        # With -log(prob) cost: lambda=0.03 gives calibrated bias relative to NLL scale.
        # Set oph_lambda=0 to disable (pure Hungarian).
        self.oph_lambda = getattr(self.hyp, "oph_lambda", 0.0)
        self.oph_start_epoch = getattr(self.hyp, "oph_start_epoch", 0)
        self.oph_ramp_epochs = getattr(self.hyp, "oph_ramp_epochs", 0)
        self.oph_use_log_cost = getattr(self.hyp, "oph_use_log_cost", False)

        # Violation-gated OPH settings (duplicate_loss_type="hungarian_violation_gated")
        # Penalty is:
        #   cost_ord = vgoph_lambda * mean(cost_cls_row) * normalized_distance
        # so order signal is capped to a small fraction (vgoph_lambda) of cls cost scale.
        self.vgoph_lambda = getattr(self.hyp, "vgoph_lambda", 0.08)
        self.vgoph_use_log_cost = getattr(self.hyp, "vgoph_use_log_cost", True)
        self.vgoph_include_duplicate_gate = getattr(self.hyp, "vgoph_include_duplicate_gate", True)

        # Hungarian Ordered-Competitor Margin (HOCM) settings.
        # Active only when duplicate_loss_type == "hungarian_order_margin".
        self.hocm_beta = getattr(self.hyp, "hocm_beta", 0.5)
        self.hocm_margin = getattr(self.hyp, "hocm_margin", 0.05)
        self.hocm_use_oph_base = getattr(self.hyp, "hocm_use_oph_base", False)
        self.hocm_oph_lambda = getattr(self.hyp, "hocm_oph_lambda", 0.0)
        self.hocm_use_log_cost = getattr(self.hyp, "hocm_use_log_cost", True)
        self.hocm_severity_power = getattr(self.hyp, "hocm_severity_power", 0.0)

        # Neighbor loss weight multiplier (0 to disable, 1.0 = default)
        self.neighbor_weight = getattr(self.hyp, "neighbor_weight", 1.0)

        # Violation ordering loss: penalizes consecutive anchor pairs whose predicted
        # soft spatial positions violate GT ordering. Separate from Hungarian (additive),
        # gated to activate after epoch K with linear ramp.
        self.violation_weight = getattr(self.hyp, "violation_weight", 0.05)
        self.violation_start_epoch = getattr(self.hyp, "violation_start_epoch", 20)
        self.violation_ramp_epochs = getattr(self.hyp, "violation_ramp_epochs", 10)
        self.violation_margin = getattr(self.hyp, "violation_margin", 0.5)
        self.violation_tau = getattr(self.hyp, "violation_tau", 3.0)

        # CRF spatial loss constants (precomputed once)
        self._build_crf_constants()

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

    def _build_distance_matrix(self):
        """
        Precompute distance matrix between all FDI class pairs for ordinal loss.

        Distance reflects anatomical proximity:
        - Same quadrant: |pos_a - pos_b|  (0-7)
        - Same jaw or same side: ORDINAL_ADJ_QUAD_BASE + |pos_a - pos_b|  (4-11)
        - Opposite corner: ORDINAL_OPP_QUAD_BASE + |pos_a - pos_b|  (8-15)

        This is used by ordinal classification loss to penalize distant errors more.
        """
        n = len(self.FDI_CLASSES)
        distance = torch.zeros((n, n), dtype=torch.float32, device=self.device)

        for i, fdi_i in enumerate(self.FDI_CLASSES):
            quad_i = fdi_i // 10
            pos_i = fdi_i % 10

            for j, fdi_j in enumerate(self.FDI_CLASSES):
                if i == j:
                    distance[i, j] = 0
                    continue

                quad_j = fdi_j // 10
                pos_j = fdi_j % 10

                pos_diff = abs(pos_i - pos_j)

                if quad_i == quad_j:
                    # Same quadrant: just position difference
                    distance[i, j] = self.ORDINAL_SAME_QUAD_BASE + pos_diff
                else:
                    # Different quadrant - check relationship
                    same_jaw = self._same_jaw_static(quad_i, quad_j)
                    same_side = self._same_side_static(quad_i, quad_j)

                    if same_jaw or same_side:
                        # Adjacent quadrants (midline or vertical)
                        distance[i, j] = self.ORDINAL_ADJ_QUAD_BASE + pos_diff
                    else:
                        # Opposite corners (Q1-Q3 or Q2-Q4)
                        distance[i, j] = self.ORDINAL_OPP_QUAD_BASE + pos_diff

        self.class_distance_matrix = distance

    def _build_dr_linear_position_lookup(self):
        """
        Precompute mapping from class index (0-31) to linear position (0-15).

        Linear position represents left-to-right order in panoramic X-ray,
        separately for upper (0-15) and lower (0-15) rows.

        Layout:
            Upper: 18(0) 17(1) ... 11(7) | 21(8) 22(9) ... 28(15)
            Lower: 48(0) 47(1) ... 41(7) | 31(8) 32(9) ... 38(15)
        """
        linear_positions = torch.zeros(32, dtype=torch.float32, device=self.device)

        for idx, fdi in enumerate(self.FDI_CLASSES):
            quadrant = fdi // 10
            position = fdi % 10

            if quadrant == 1:  # Upper right: 18→0, 17→1, ..., 11→7
                linear_positions[idx] = 8 - position
            elif quadrant == 2:  # Upper left: 21→8, 22→9, ..., 28→15
                linear_positions[idx] = 7 + position
            elif quadrant == 4:  # Lower right: 48→0, 47→1, ..., 41→7
                linear_positions[idx] = 8 - position
            elif quadrant == 3:  # Lower left: 31→8, 32→9, ..., 38→15
                linear_positions[idx] = 7 + position

        self.dr_linear_position_lookup = linear_positions

    def _build_crf_constants(self):
        """
        Precompute constants for CRF spatial loss (whole-mouth, C=32).

        Builds:
        - crf_full_position_lookup: class index (0-31) -> full-mouth spatial position (0-31)
          Upper jaw positions 0-15 (L-to-R in panoramic), lower jaw positions 16-31 (L-to-R).
        - spatial_to_class_full: inverse mapping, full-mouth position -> class index
        - crf_transition_full: (32, 32) matrix encoding strict monotonicity constraint

        Per-jaw spatial ordering is preserved because positions 0-15 (upper) are all
        strictly less than positions 16-31 (lower), so any monotonically increasing
        path through {0,...,31} restricts to monotonically increasing subsequences
        within each jaw.
        """
        C_full = 32  # full-mouth spatial positions

        # Full-mouth position lookup: class -> full-mouth spatial position (0-31)
        # Upper jaw (classes 0-15): keep per-jaw positions 0-15
        # Lower jaw (classes 16-31): offset per-jaw positions by 16 -> 16-31
        self.crf_full_position_lookup = self.dr_linear_position_lookup.clone()
        self.crf_full_position_lookup[16:] += 16

        # Inverse mapping: full-mouth spatial position -> class index
        spatial_to_class_full = torch.zeros(C_full, dtype=torch.long, device=self.device)
        for cls_idx in range(32):
            sp = int(self.crf_full_position_lookup[cls_idx].item())
            spatial_to_class_full[sp] = cls_idx
        self.spatial_to_class_full = spatial_to_class_full

        # Transition matrix (32x32): T[k, j] = 0 if k < j (valid), -1e9 if k >= j (invalid)
        # Encodes strict monotonicity: next position must be greater than current
        # Uses -1e9 instead of -inf to avoid NaN gradients in logsumexp backward pass
        T = torch.full((C_full, C_full), -1e9, device=self.device)
        mask = torch.triu(torch.ones(C_full, C_full, device=self.device), diagonal=1).bool()
        T[mask] = 0.0
        self.crf_transition = T

    def _dr_compute_effective_gaps(
        self,
        sorted_pos: torch.Tensor,
        is_upper_jaw: bool,
    ) -> torch.Tensor:
        """
        Compute effective gaps between consecutive positions for DR loss.

        Handles midline crossing with jaw-specific factors determined empirically:
        - Upper midline (11↔21): 1.18x baseline (wider gap)
        - Lower midline (31↔41): 0.63x baseline (narrower/crowded)

        Args:
            sorted_pos: (m,) linear positions sorted left-to-right
            is_upper_jaw: True for upper jaw, False for lower

        Returns:
            (m-1,) tensor of effective gaps between consecutive teeth
        """
        m = sorted_pos.shape[0]
        device = sorted_pos.device
        dtype = sorted_pos.dtype

        gaps = torch.zeros(m - 1, device=device, dtype=dtype)
        midline_gap = self.DR_UPPER_MIDLINE_GAP if is_upper_jaw else self.DR_LOWER_MIDLINE_GAP

        for i in range(m - 1):
            pos_a = sorted_pos[i].item()
            pos_b = sorted_pos[i + 1].item()

            min_pos, max_pos = min(pos_a, pos_b), max(pos_a, pos_b)

            # Check if crosses midline (position 7 to 8)
            if min_pos <= 7 < max_pos:
                # Crosses midline: split into components
                left_part = 7 - min_pos  # slots from min_pos to position 7
                right_part = max_pos - 8  # slots from position 8 to max_pos
                gaps[i] = left_part + midline_gap + right_part
            else:
                # Same side of midline: simple difference
                gaps[i] = max_pos - min_pos

        return gaps

    def _dr_row_loss(
        self,
        centers: torch.Tensor,
        linear_pos: torch.Tensor,
        is_upper_jaw: bool,
    ) -> torch.Tensor:
        """
        Compute Distance Regularization loss for a single row (upper or lower).

        Implements Laplacian regularization on gap-normalized inter-tooth distances.
        The Laplacian (second derivative) measures "bumpiness" in the distance sequence
        and should be near zero for evenly spaced teeth.

        Formula: L_dr = mean(∇²d)² where ∇²d[i] = d[i+1] - 2·d[i] + d[i-1]

        Args:
            centers: (m, 2) bbox centers for teeth in this row
            linear_pos: (m,) linear positions (0-15)
            is_upper_jaw: True for upper jaw, False for lower

        Returns:
            Scalar DR loss for this row.
        """
        m = centers.shape[0]
        device = centers.device
        dtype = centers.dtype

        # Need at least 4 teeth to compute Laplacian (3 distances → 1 Laplacian value)
        if m < self.DR_MIN_TEETH_PER_SEGMENT:
            return torch.tensor(0.0, device=device, dtype=dtype)

        # Sort detections by linear position (left to right in image)
        sort_idx = linear_pos.argsort()
        sorted_centers = centers[sort_idx]
        sorted_pos = linear_pos[sort_idx]

        # Compute Euclidean distances between consecutive detections
        diffs = sorted_centers[1:] - sorted_centers[:-1]  # (m-1, 2)
        distances = torch.norm(diffs, dim=1)  # (m-1,)

        # Compute effective gaps with midline handling
        effective_gaps = self._dr_compute_effective_gaps(sorted_pos, is_upper_jaw)

        # Normalize distances by effective gap to handle missing teeth
        # This makes d(11,13)/2 comparable to d(13,14)/1
        normalized_distances = distances / effective_gaps.clamp(min=0.5)

        # Compute Laplacian (second derivative of distance sequence)
        # L[i] = d[i+1] - 2*d[i] + d[i-1]
        # Measures "bumpiness" - should be near zero for evenly spaced teeth
        laplacian = (
            normalized_distances[2:]
            - 2.0 * normalized_distances[1:-1]
            + normalized_distances[:-2]
        )

        if laplacian.numel() == 0:
            return torch.tensor(0.0, device=device, dtype=dtype)

        # L2 regularization of Laplacian (squared mean)
        dr_loss = (laplacian**2).mean()

        return dr_loss

    def _distance_regularization_loss(
        self,
        pred_bboxes: torch.Tensor,
        gt_classes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Distance Regularization (DR) loss adapted for YOLO.

        Enforces smooth spacing between detected teeth by penalizing
        irregular inter-tooth distances via Laplacian regularization.

        Based on: Chung et al., "Individual Tooth Detection and Identification
        from Dental Panoramic X-Ray Images via Point-wise Localization and
        Distance Regularization", Artificial Intelligence in Medicine, 2021.

        Adaptations for YOLO:
        1. Uses GT class assignment to order detections by FDI position
        2. Normalizes distances by effective gap to handle missing teeth
        3. Processes upper and lower jaws separately
        4. Uses empirically-determined midline factors (upper: 1.18, lower: 0.63)

        Args:
            pred_bboxes: (n_teeth, 4) predicted bboxes (x1, y1, x2, y2)
            gt_classes: (n_teeth,) GT class indices (0-31)

        Returns:
            Scalar DR loss value.
        """
        n = pred_bboxes.shape[0]
        device = pred_bboxes.device
        dtype = pred_bboxes.dtype

        # Need minimum teeth for meaningful DR loss
        if n < self.DR_MIN_TEETH_PER_SEGMENT:
            return torch.tensor(0.0, device=device, dtype=dtype)

        # Compute bbox centers
        centers = (pred_bboxes[:, :2] + pred_bboxes[:, 2:4]) / 2

        # Get linear positions from precomputed lookup
        gt_classes_long = gt_classes.long()
        linear_pos = self.dr_linear_position_lookup[gt_classes_long]

        # Separate upper and lower jaws using precomputed masks
        upper_mask = self.upper_jaw_mask[gt_classes_long]
        lower_mask = self.lower_jaw_mask[gt_classes_long]

        loss = torch.tensor(0.0, device=device, dtype=dtype)
        count = 0

        # Process upper jaw
        n_upper = upper_mask.sum()
        if n_upper >= self.DR_MIN_TEETH_PER_SEGMENT:
            upper_loss = self._dr_row_loss(
                centers[upper_mask],
                linear_pos[upper_mask],
                is_upper_jaw=True,
            )
            loss = loss + upper_loss
            count += 1

        # Process lower jaw
        n_lower = lower_mask.sum()
        if n_lower >= self.DR_MIN_TEETH_PER_SEGMENT:
            lower_loss = self._dr_row_loss(
                centers[lower_mask],
                linear_pos[lower_mask],
                is_upper_jaw=False,
            )
            loss = loss + lower_loss
            count += 1

        # Average over jaws (if both have enough teeth)
        if count > 0:
            loss = loss / count

        return loss

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
        Get weight multiplier for neighbor loss based on constant + late boost schedule.

        No suppression - full weight from epoch 1, steep ramp 70-80, max from 80+.

        Epochs 0 to BOOST_START (70):   multiplier = 1.0 (constant, full weight)
        Epochs BOOST_START to BOOST_END (70-80): multiplier = 1.0 → MAX_MULT (steep ramp)
        Epochs BOOST_END to end (80-100): multiplier = MAX_MULT (maximum effectiveness)

        Returns:
            float: Weight multiplier to apply to base anatomy weight.
        """
        epoch = self.current_epoch

        boost_start = self.NEIGHBOR_BOOST_START
        boost_end = self.NEIGHBOR_BOOST_END
        max_mult = self.NEIGHBOR_MAX_MULT

        if epoch < boost_start:
            # Constant phase: full weight from start
            multiplier = 1.0
        elif epoch < boost_end:
            # Ramp phase: steep linear increase from 1.0 to max_mult
            ramp_duration = boost_end - boost_start
            if ramp_duration <= 0:
                multiplier = max_mult
            else:
                progress = (epoch - boost_start) / ramp_duration
                multiplier = 1.0 + (max_mult - 1.0) * progress
        else:
            # Maximum phase: hold at max_mult for most effectiveness
            multiplier = max_mult

        return float(multiplier)

    def _get_crf_multiplier(self) -> float:
        """
        Get warmup multiplier for CRF spatial loss.

        Implements a "learn first, enforce later" schedule:
        - Epochs 0 to CRF_WARMUP_START: 0.0 (let model learn basics)
        - CRF_WARMUP_START to CRF_WARMUP_END: linear ramp 0.0 → 1.0
        - CRF_WARMUP_END to end: 1.0 (full CRF force)

        Returns:
            float: Multiplier in [0.0, 1.0].
        """
        epoch = self.current_epoch
        warmup_start = self.CRF_WARMUP_START
        warmup_end = self.CRF_WARMUP_END

        if epoch < warmup_start:
            return 0.0
        elif epoch < warmup_end:
            ramp_duration = warmup_end - warmup_start
            if ramp_duration <= 0:
                return 1.0
            progress = (epoch - warmup_start) / ramp_duration
            return float(progress)
        else:
            return 1.0

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

        # Classification loss (standard or ordinal-weighted)
        if self.ordinal_alpha > 0:
            # Ordinal-weighted: penalizes distant class errors more
            loss[2] = self._ordinal_classification_loss(pred_scores, target_scores, fg_mask)
        else:
            # Standard BCE
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

            # Extract GT classes for representative anchors (for GT-conditioned loss)
            # target_scores has soft labels where argmax = GT class
            gt_classes_subset = target_scores[i, rep_anchor].argmax(dim=-1)  # (n_teeth,)

            # Compute loss based on anatomy_loss_type
            if self.anatomy_loss_type == "ordered":
                # Pairwise ordering loss: penalizes P(violation) between adjacent anchors
                anatomy_loss_value = self._pairwise_ordering_loss(
                    pred_scores_subset, gt_classes_subset
                )
                total_loss = total_loss + anatomy_loss_value
            elif self.anatomy_loss_type == "dp_ordered":
                # Pure DP ordered assignment: monotonic argmax via dynamic programming + CE
                anatomy_loss_value = self._ordered_assignment_loss(
                    pred_scores_subset, gt_classes_subset
                )
                total_loss = total_loss + anatomy_loss_value

            # Distance Regularization (DR) Loss - independent, can be combined with any anatomy loss
            # Enforces smooth inter-tooth spacing via Laplacian regularization
            if self.dr_enabled:
                dr_loss = self._distance_regularization_loss(bboxes, gt_classes_subset)
                total_loss = total_loss + self.dr_weight * dr_loss

            if self.anatomy_loss_type not in ("ordered", "dp_ordered"):
                # Components mode (default): separate duplicate + neighbor losses
                # Duplicate loss: enforces unique class predictions
                if self.duplicate_loss_type == "hungarian":
                    # DETR-style bipartite matching (Carion et al., ECCV 2020)
                    # Pass gt_classes for OPH ordering bias (uses oph_lambda > 0)
                    dup_loss = self._duplicate_loss_hungarian(pred_scores_subset, gt_classes_subset)
                elif self.duplicate_loss_type == "hungarian_violation_gated":
                    # Apply OPH ordering penalty only to anchors participating
                    # in current ordering/duplicate violations.
                    dup_loss = self._duplicate_loss_hungarian_violation_gated(pred_scores_subset, gt_classes_subset)
                elif self.duplicate_loss_type == "hungarian_order_margin":
                    # HOCM: keep Hungarian CE, add margin only when assignment is unordered
                    dup_loss = self._duplicate_loss_hungarian_order_margin(pred_scores_subset, gt_classes_subset)
                elif self.duplicate_loss_type == "pairwise":
                    # Pairwise contrastive loss
                    dup_loss = self._duplicate_loss_pairwise(pred_scores_subset)
                elif self.duplicate_loss_type == "soft":
                    # Sum-based soft loss
                    dup_loss = self._duplicate_loss_soft(pred_scores_subset)
                else:
                    # Disabled (for ablation studies)
                    dup_loss = torch.tensor(0.0, device=pred_scores_subset.device)

                # Neighbor loss: choose between GT-conditioned and soft (original)
                if self.neighbor_weight > 0:
                    if self.neighbor_gt_conditioned:
                        # GT-conditioned: uses GT neighbor class (found to hurt performance)
                        neighbor_loss = self._neighbor_loss_gt_conditioned(
                            pred_scores_subset, bboxes, gt_classes_subset
                        )
                    else:
                        # Soft neighbor loss (original implementation)
                        # Uses joint probability of invalid neighbor pairs
                        neighbor_loss = self._neighbor_loss_soft_legacy(pred_scores_subset, bboxes)

                    # Apply three-phase weight multiplier to neighbor loss
                    neighbor_multiplier = self._get_neighbor_multiplier()

                    # Log raw loss before multiplier (for monitoring)
                    if self._neighbor_log_enabled:
                        self._neighbor_log["raw_loss_sum"] += neighbor_loss.detach().item()
                        self._neighbor_log["count"] += 1

                    neighbor_loss = neighbor_loss * neighbor_multiplier * self.neighbor_weight
                else:
                    neighbor_loss = torch.tensor(0.0, device=pred_scores_subset.device)

                # Violation ordering loss: penalizes soft spatial position inversions
                # Gated to activate after violation_start_epoch with linear ramp
                viol_w = self._get_violation_weight()
                if viol_w > 0:
                    ordering_loss = viol_w * self._violation_ordering_loss(
                        pred_scores_subset, gt_classes_subset
                    )
                else:
                    ordering_loss = torch.tensor(0.0, device=pred_scores_subset.device)

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

    def _hungarian_assign_classes(
        self,
        pred_scores_subset: torch.Tensor,
        gt_classes: torch.Tensor = None,
        oph_lambda: Optional[float] = None,
        use_log_prob_cost: bool = False,
        violation_gate: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Hungarian assignment classes for each detection.

        Args:
            pred_scores_subset: Raw logits, shape (n_teeth, num_classes).
            gt_classes: Optional GT classes for order-biased costs.
            oph_lambda: Optional override for ordering-bias lambda.
            use_log_prob_cost: If True, uses -log(prob) cost. Else uses -prob.
            violation_gate: Optional (n_teeth,) binary/float mask. If provided,
                            ordering penalty is applied only where gate>0.

        Returns:
            assigned_classes: (n_teeth,) class index assigned to each detection.
            probs: Softmax probabilities (n_teeth, num_classes).
        """
        n_teeth = pred_scores_subset.shape[0]
        n_classes = self.nc
        device = pred_scores_subset.device

        probs = pred_scores_subset.softmax(dim=-1)  # (n_teeth, n_classes)

        if use_log_prob_cost:
            cost_cls = -torch.log(probs.detach().clamp_min(1e-8)).cpu().numpy()
        else:
            cost_cls = -probs.detach().cpu().numpy()
        cost_cls = np.nan_to_num(cost_cls, nan=50.0, posinf=50.0, neginf=0.0)

        lambda_eff = self.oph_lambda if oph_lambda is None else oph_lambda
        if lambda_eff > 0 and gt_classes is not None:
            spatial_lookup = self.crf_full_position_lookup.to(device=device)
            gt_spatial = spatial_lookup[gt_classes.long()].detach().cpu().numpy()  # (n_teeth,)
            class_spatial = spatial_lookup.detach().cpu().numpy()  # (n_classes,)

            # cost_ord[i,j] = normalized spatial distance between class_j and anchor_i GT position
            cost_ord = np.abs(class_spatial[np.newaxis, :] - gt_spatial[:, np.newaxis]) / 32.0
            if violation_gate is not None:
                # Keep order term lightweight and scale-aware:
                # penalty <= lambda_eff * mean(classification cost row)
                # when normalized distance <= 1.
                gate_np = violation_gate.detach().float().clamp(0, 1).cpu().numpy().reshape(-1, 1)
                row_scale = cost_cls.mean(axis=1, keepdims=True).clip(min=1e-6)
                cost_ord = float(lambda_eff) * row_scale * cost_ord * gate_np
                cost_ord = np.nan_to_num(cost_ord, nan=0.0, posinf=0.0, neginf=0.0)
                cost_cls = cost_cls + cost_ord
            else:
                cost_cls = cost_cls + float(lambda_eff) * cost_ord
            cost_cls = np.nan_to_num(cost_cls, nan=50.0, posinf=50.0, neginf=0.0)

        # Hungarian expects square matrix
        if n_teeth < n_classes:
            padding = np.zeros((n_classes - n_teeth, n_classes))
            cost_matrix = np.vstack([cost_cls, padding])
        elif n_teeth > n_classes:
            padding = np.zeros((n_teeth, n_teeth - n_classes))
            cost_matrix = np.hstack([cost_cls, padding])
        else:
            cost_matrix = cost_cls
        cost_matrix = np.nan_to_num(cost_matrix, nan=50.0, posinf=50.0, neginf=0.0)

        try:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
        except ValueError:
            fallback = probs.argmax(dim=-1)
            return fallback, probs
        valid_mask = (row_ind < n_teeth) & (col_ind < n_classes)

        assigned_classes = torch.full((n_teeth,), -1, device=device, dtype=torch.long)
        if valid_mask.any():
            det_indices = torch.tensor(row_ind[valid_mask], device=device, dtype=torch.long)
            class_indices = torch.tensor(col_ind[valid_mask], device=device, dtype=torch.long)
            assigned_classes[det_indices] = class_indices

        # Safety fallback for unmatched detections (rare in this setting)
        if (assigned_classes < 0).any():
            fallback = probs.argmax(dim=-1)
            assigned_classes = torch.where(assigned_classes >= 0, assigned_classes, fallback)

        return assigned_classes, probs

    def _duplicate_loss_hungarian(
        self,
        pred_scores_subset: torch.Tensor,
        gt_classes: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Order-Penalized Hungarian (OPH) duplicate loss.

        Uses the Hungarian algorithm to find optimal 1:1 assignment of detections
        to classes, with an optional spatial ordering bias in the cost matrix.

        When oph_lambda > 0 and gt_classes is provided, the cost matrix becomes:
            C[i,j] = -prob[i,j] + oph_lambda * |spatial_pos(j) - gt_spatial(i)| / 32

        This makes spatially implausible assignments more expensive, biasing the
        matching toward order-consistent assignments when classification confidence
        is low, while respecting confident predictions that may violate ordering.

        Inspired by DETR (Carion et al., ECCV 2020) for bipartite matching and
        Scott & Nowak (IEEE TIP 2006) for order-preserving assignment.

        Args:
            pred_scores_subset: Raw logits for representative anchors,
                                shape (n_teeth, num_classes).
            gt_classes: GT class indices for each anchor, shape (n_teeth,).
                        Required when oph_lambda > 0 for ordering bias.

        Returns:
            Differentiable duplicate loss scalar.
        """
        n_teeth = pred_scores_subset.shape[0]
        dtype = pred_scores_subset.dtype

        # Edge case: no teeth or single tooth
        if n_teeth < 2:
            return torch.tensor(0.0, device=pred_scores_subset.device, dtype=dtype)

        assigned_classes, probs = self._hungarian_assign_classes(
            pred_scores_subset,
            gt_classes=gt_classes,
            oph_lambda=self._get_oph_lambda(),
            use_log_prob_cost=self.oph_use_log_cost,
        )
        det_indices = torch.arange(n_teeth, device=pred_scores_subset.device)
        assigned_probs = probs[det_indices, assigned_classes]
        loss = -torch.log(assigned_probs + 1e-8).mean()

        return loss

    def _vgoph_violation_gate(self, pred_scores_subset: torch.Tensor, gt_classes: torch.Tensor) -> torch.Tensor:
        """
        Build per-anchor binary gate for violation-gated OPH.

        Gate=1 for anchors participating in:
        1) Adjacent ordering inversions in GT-sorted anchor order.
        2) Duplicate predicted classes (optional).
        """
        n = pred_scores_subset.shape[0]
        device = pred_scores_subset.device
        if n < 2:
            return torch.zeros(n, device=device, dtype=torch.float32)

        spatial_lookup = self.crf_full_position_lookup.to(device)
        pred_classes = pred_scores_subset.detach().argmax(dim=-1)

        # Adjacent inversion detection in GT-sorted anchor order
        gt_spatial = spatial_lookup[gt_classes.long()]
        sort_idx = gt_spatial.argsort()
        pred_spatial_sorted = spatial_lookup[pred_classes.long()][sort_idx]
        pair_viol = pred_spatial_sorted[:-1] >= pred_spatial_sorted[1:]

        gate_sorted = torch.zeros(n, device=device, dtype=torch.bool)
        if pair_viol.any():
            v_idx = pair_viol.nonzero(as_tuple=False).squeeze(1)
            gate_sorted[v_idx] = True
            gate_sorted[v_idx + 1] = True

        gate = torch.zeros(n, device=device, dtype=torch.bool)
        gate[sort_idx] = gate_sorted

        if self.vgoph_include_duplicate_gate:
            uniq, counts = pred_classes.unique(return_counts=True)
            dup_classes = uniq[counts > 1]
            if dup_classes.numel() > 0:
                dup_mask = torch.isin(pred_classes, dup_classes)
                gate = gate | dup_mask

        return gate.float()

    def _duplicate_loss_hungarian_violation_gated(
        self,
        pred_scores_subset: torch.Tensor,
        gt_classes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Violation-gated OPH duplicate loss.

        Uses Hungarian CE as base objective. Adds OPH cost bias only for anchors
        currently involved in ordering/duplicate violations. Order term is scaled
        to remain a weak signal relative to classification cost.
        """
        n_teeth = pred_scores_subset.shape[0]
        dtype = pred_scores_subset.dtype
        device = pred_scores_subset.device

        if n_teeth < 2:
            return torch.tensor(0.0, device=device, dtype=dtype)

        gate = self._vgoph_violation_gate(pred_scores_subset, gt_classes)

        assigned_classes, probs = self._hungarian_assign_classes(
            pred_scores_subset,
            gt_classes=gt_classes,
            oph_lambda=self.vgoph_lambda,
            use_log_prob_cost=self.vgoph_use_log_cost,
            violation_gate=gate,
        )

        det_indices = torch.arange(n_teeth, device=device)
        assigned_probs = probs[det_indices, assigned_classes]
        loss = -torch.log(assigned_probs + 1e-8).mean()

        return loss

    def _ordered_dp_assignment_classes(self, pred_scores_subset: torch.Tensor, gt_classes: torch.Tensor) -> torch.Tensor:
        """
        Compute best ordered class assignment via DP and return assigned class indices.

        Args:
            pred_scores_subset: Raw logits, shape (n_teeth, num_classes).
            gt_classes: GT class indices, shape (n_teeth,).

        Returns:
            (n_teeth,) class assignment enforcing strict monotonic order in spatial space.
        """
        n = pred_scores_subset.shape[0]
        device = pred_scores_subset.device
        C = 32

        if n < 2:
            return pred_scores_subset.argmax(dim=-1)
        if n > C:
            return pred_scores_subset.argmax(dim=-1)

        spatial_lookup = self.crf_full_position_lookup.to(device)
        s2c = self.spatial_to_class_full.to(device)

        gt_spatial = spatial_lookup[gt_classes.long()]
        sort_idx = gt_spatial.argsort()
        ordered_preds = pred_scores_subset[sort_idx]

        # Use log-probability emissions so path scores are comparable across anchors.
        emissions = F.log_softmax(ordered_preds.float(), dim=-1)[:, s2c]

        NEG_INF = -1e9
        V = torch.full((n, C), NEG_INF, device=device, dtype=torch.float32)
        backptr = torch.zeros((n, C), device=device, dtype=torch.long)

        max_first = C - n
        V[0, : max_first + 1] = emissions[0, : max_first + 1]

        for i in range(1, n):
            remaining = n - 1 - i
            max_pos_i = C - 1 - remaining

            prev_row = V[i - 1]
            running_max = NEG_INF
            running_idx = 0

            for p in range(C):
                cummax_val_p = running_max
                cummax_idx_p = running_idx

                if prev_row[p].item() > running_max:
                    running_max = prev_row[p].item()
                    running_idx = p

                if 1 <= p <= max_pos_i:
                    V[i, p] = cummax_val_p + emissions[i, p]
                    backptr[i, p] = cummax_idx_p

        min_final = n - 1
        valid_final = V[n - 1, min_final:]
        if valid_final.numel() > 0 and valid_final.max() > NEG_INF:
            best_final = min_final + valid_final.argmax()
        else:
            best_final = V[n - 1].argmax()

        assignment_pos = torch.zeros(n, device=device, dtype=torch.long)
        assignment_pos[n - 1] = best_final

        current = best_final.item()
        for i in range(n - 2, -1, -1):
            prev = backptr[i + 1, current].item()
            assignment_pos[i] = prev
            current = prev

        ordered_classes = s2c[assignment_pos]
        inverse_sort = sort_idx.argsort()
        return ordered_classes[inverse_sort]

    def _ordering_violation_ratio(self, assigned_classes: torch.Tensor, gt_classes: torch.Tensor) -> torch.Tensor:
        """Return fraction of adjacent-order violations in GT-sorted anchor order."""
        n = assigned_classes.shape[0]
        device = assigned_classes.device
        if n < 2:
            return torch.tensor(0.0, device=device, dtype=torch.float32)

        spatial_lookup = self.crf_full_position_lookup.to(device)
        gt_spatial = spatial_lookup[gt_classes.long()]
        sort_idx = gt_spatial.argsort()

        assigned_spatial = spatial_lookup[assigned_classes.long()][sort_idx]
        violations = (assigned_spatial[:-1] >= assigned_spatial[1:]).float()
        return violations.mean()

    def _duplicate_loss_hungarian_order_margin(
        self,
        pred_scores_subset: torch.Tensor,
        gt_classes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Hungarian Ordered-Competitor Margin (HOCM) loss.

        Keeps Hungarian CE as the base objective and adds ordering pressure only when
        Hungarian's assignment is unordered. The margin compares the score of the
        Hungarian assignment to the score of the best ordered DP assignment.

        Args:
            pred_scores_subset: Raw logits for representative anchors, shape (n_teeth, num_classes).
            gt_classes: GT class indices for each anchor, shape (n_teeth,).

        Returns:
            Scalar duplicate loss with optional ordered-competitor margin.
        """
        n = pred_scores_subset.shape[0]
        device = pred_scores_subset.device
        dtype = pred_scores_subset.dtype

        if n < 2:
            return torch.tensor(0.0, device=device, dtype=dtype)

        oph_lambda = self.hocm_oph_lambda if self.hocm_use_oph_base else 0.0
        assigned_h, probs = self._hungarian_assign_classes(
            pred_scores_subset,
            gt_classes=gt_classes if oph_lambda > 0 else None,
            oph_lambda=oph_lambda,
            use_log_prob_cost=self.hocm_use_log_cost,
        )

        idx = torch.arange(n, device=device)
        base_loss = -torch.log(probs[idx, assigned_h] + 1e-8).mean()

        if self.hocm_beta <= 0:
            return base_loss

        violation_ratio = self._ordering_violation_ratio(assigned_h, gt_classes)
        if violation_ratio.item() <= 0:
            return base_loss

        assigned_o = self._ordered_dp_assignment_classes(pred_scores_subset.detach(), gt_classes)

        # Compare mean log-probability per anchor for scale compatibility with CE mean.
        log_probs = F.log_softmax(pred_scores_subset.float(), dim=-1)
        score_h = log_probs[idx, assigned_h].mean()
        score_o = log_probs[idx, assigned_o].mean()
        margin_loss = F.relu(self.hocm_margin + score_h - score_o)

        if self.hocm_severity_power > 0:
            margin_loss = margin_loss * violation_ratio.pow(self.hocm_severity_power)

        return base_loss + self.hocm_beta * margin_loss.to(dtype)

    def _violation_ordering_loss(
        self,
        pred_scores_subset: torch.Tensor,
        gt_classes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Violation ordering loss: penalizes consecutive anchors whose predicted
        soft spatial positions violate GT spatial ordering.

        Computes a differentiable "expected spatial position" per anchor via
        softmax-weighted sum over spatial positions, then applies a hinge loss
        on consecutive pairs sorted by GT spatial position.

        Args:
            pred_scores_subset: Raw logits, shape (n_teeth, num_classes).
            gt_classes: GT class indices per anchor, shape (n_teeth,).

        Returns:
            Scalar violation loss (0 if all pairs are correctly ordered).
        """
        n_teeth = pred_scores_subset.shape[0]
        device = pred_scores_subset.device
        dtype = pred_scores_subset.dtype

        if n_teeth < 2:
            return torch.tensor(0.0, device=device, dtype=dtype)

        # Spatial position lookup: class index -> full-mouth position (0-31)
        spatial_lookup = self.crf_full_position_lookup.to(device)  # (32,)

        # Step 1: Sort anchors by GT spatial position
        gt_spatial = spatial_lookup[gt_classes.long()]  # (n_teeth,)
        sort_idx = gt_spatial.argsort()  # ascending spatial order

        # Step 2: Compute soft spatial position for each anchor
        # soft_pos(i) = sum_c softmax(logits_i / tau)[c] * spatial_position[c]
        probs = torch.softmax(pred_scores_subset.float() / self.violation_tau, dim=-1)
        soft_pos = (probs * spatial_lookup.unsqueeze(0)).sum(dim=-1)  # (n_teeth,)

        # Step 3: Get soft positions in GT-sorted order
        sorted_soft_pos = soft_pos[sort_idx]  # (n_teeth,)

        # Step 4: Hinge loss on consecutive pairs
        # For each pair (i, i+1): loss = max(0, soft_pos[i] - soft_pos[i+1] + margin)
        # Violation = predicted position of earlier tooth >= predicted position of later tooth
        diffs = sorted_soft_pos[:-1] - sorted_soft_pos[1:]  # (n_teeth-1,)
        violations = torch.relu(diffs + self.violation_margin)  # (n_teeth-1,)

        return violations.mean()

    def _get_violation_weight(self) -> float:
        """Get current violation loss weight based on epoch gating schedule."""
        if self.violation_weight <= 0:
            return 0.0
        epoch = self.current_epoch
        start = self.violation_start_epoch
        ramp = self.violation_ramp_epochs
        if epoch < start:
            return 0.0
        if ramp <= 0 or epoch >= start + ramp:
            return self.violation_weight
        # Linear ramp from 0 to violation_weight over ramp epochs
        return self.violation_weight * (epoch - start) / ramp

    def _get_oph_lambda(self) -> float:
        """Get current OPH lambda based on epoch gating schedule.

        Returns 0.0 before oph_start_epoch, linearly ramps over oph_ramp_epochs,
        then returns full oph_lambda. When oph_start_epoch=0 and oph_ramp_epochs=0,
        returns oph_lambda from the start (backwards compatible).
        """
        if self.oph_lambda <= 0:
            return 0.0
        epoch = self.current_epoch
        start = self.oph_start_epoch
        ramp = self.oph_ramp_epochs
        if epoch < start:
            return 0.0
        if ramp <= 0 or epoch >= start + ramp:
            return self.oph_lambda
        # Linear ramp from 0 to oph_lambda over ramp epochs
        return self.oph_lambda * (epoch - start) / ramp

    def _ordered_assignment_loss(
        self,
        pred_scores_subset: torch.Tensor,
        gt_classes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Ordered assignment loss using DP + cross-entropy.

        Finds the optimal monotonic class assignment via dynamic programming in
        spatial-position space (0-31 whole mouth), then computes cross-entropy
        loss between predictions and the DP assignment for all anchors.

        Args:
            pred_scores_subset: Raw logits for representative anchors,
                                shape (n_teeth, num_classes).
            gt_classes: GT class indices for each anchor, shape (n_teeth,).

        Returns:
            Differentiable ordered assignment loss scalar.
        """
        n = pred_scores_subset.shape[0]
        device = pred_scores_subset.device
        dtype = pred_scores_subset.dtype
        C = 32  # spatial positions (whole mouth)

        if n < 2:
            return torch.tensor(0.0, device=device, dtype=dtype)
        if n > C:
            return torch.tensor(0.0, device=device, dtype=dtype)

        # --- Step 1: Sort anchors by GT spatial position (not class index!) ---
        spatial_lookup = self.crf_full_position_lookup.to(device)
        s2c = self.spatial_to_class_full.to(device)

        gt_spatial = spatial_lookup[gt_classes.long()]  # (n,) GT spatial positions
        sort_idx = gt_spatial.argsort()
        ordered_preds = pred_scores_subset[sort_idx]  # (n, 32) logits in spatial order

        # --- Step 2: DP in spatial-position space ---
        emissions = ordered_preds[:, s2c].float()  # (n, C) float32 for DP stability

        NEG_INF = -1e9
        V = torch.full((n, C), NEG_INF, device=device, dtype=torch.float32)
        backptr = torch.zeros((n, C), device=device, dtype=torch.long)

        max_first = C - n
        V[0, : max_first + 1] = emissions[0, : max_first + 1]

        for i in range(1, n):
            remaining = n - 1 - i
            max_pos_i = C - 1 - remaining

            prev_row = V[i - 1]
            running_max = NEG_INF
            running_idx = 0

            for p in range(C):
                cummax_val_p = running_max
                cummax_idx_p = running_idx

                if prev_row[p].item() > running_max:
                    running_max = prev_row[p].item()
                    running_idx = p

                if 1 <= p <= max_pos_i:
                    V[i, p] = cummax_val_p + emissions[i, p]
                    backptr[i, p] = cummax_idx_p

        # --- Step 3: Backtrack to find optimal assignment ---
        min_final = n - 1
        valid_final = V[n - 1, min_final:]

        if valid_final.numel() > 0 and valid_final.max() > NEG_INF:
            best_final = min_final + valid_final.argmax()
        else:
            best_final = V[n - 1].argmax()

        assignment_pos = torch.zeros(n, device=device, dtype=torch.long)
        assignment_pos[n - 1] = best_final

        current = best_final.item()
        for i in range(n - 2, -1, -1):
            prev = backptr[i + 1, current].item()
            assignment_pos[i] = prev
            current = prev

        # Convert spatial positions back to class indices
        dp_classes_sorted = s2c[assignment_pos]

        # Reorder back to original anchor order
        inverse_sort = sort_idx.argsort()
        dp_classes = dp_classes_sorted[inverse_sort]

        # --- Step 4: Cross-entropy loss against DP assignment ---
        probs = pred_scores_subset.softmax(dim=-1)
        assigned_probs = probs[torch.arange(n, device=device), dp_classes]
        loss = -torch.log(assigned_probs + 1e-8).mean()

        return loss

    def _pairwise_ordering_loss(
        self,
        pred_scores_subset: torch.Tensor,
        gt_classes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pairwise ordering loss based on violation probability between adjacent anchors.

        For each pair of spatially adjacent anchors (sorted by GT spatial position),
        computes the probability that the left anchor predicts a class at a spatial
        position >= the right anchor's predicted class — i.e., an ordering violation.

        P(violation) = Σ_{p} left(p) · CDF_right(p)

        where left(p) is the left anchor's probability at spatial position p, and
        CDF_right(p) = Σ_{q<=p} right(q) is the right anchor's cumulative probability
        up to position p.

        Key properties:
        - No GT class target: GT is only used for sorting anchors spatially
        - No exponential scaling: each pair evaluated independently
        - Naturally reaches ~0 when predictions are correct and confident
        - Fully differentiable via softmax → cumsum → products

        Args:
            pred_scores_subset: Raw logits for representative anchors,
                                shape (n_teeth, num_classes).
            gt_classes: GT class indices for each anchor, shape (n_teeth,).

        Returns:
            Differentiable pairwise ordering loss scalar.
        """
        n = pred_scores_subset.shape[0]
        device = pred_scores_subset.device
        dtype = pred_scores_subset.dtype

        if n < 2:
            return torch.tensor(0.0, device=device, dtype=dtype)

        # Sort anchors by GT spatial position (not class index — fixes Q1/Q4 reversal)
        spatial_lookup = self.crf_full_position_lookup.to(device)
        s2c = self.spatial_to_class_full.to(device)

        gt_spatial = spatial_lookup[gt_classes.long()]  # (n,) GT spatial positions
        sort_idx = gt_spatial.argsort()

        # Softmax probabilities remapped to spatial position order
        probs = pred_scores_subset[sort_idx].float().softmax(dim=-1)  # (n, 32)
        spatial_probs = probs[:, s2c]  # (n, 32) index = spatial position

        # Adjacent pairs in spatial order
        left = spatial_probs[:-1]   # (n-1, 32)
        right = spatial_probs[1:]   # (n-1, 32)

        # For each pair: P(violation) = Σ_p left(p) · CDF_right(p)
        # CDF_right(p) = Σ_{q=0}^{p} right(q) = probability right anchor is at position ≤ p
        # If left is at position p and right is at position ≤ p, that's a violation
        cum_right = right.cumsum(dim=-1)  # (n-1, 32)
        pair_violation = (left * cum_right).sum(dim=-1)  # (n-1,)

        return pair_violation.mean()

    def _crf_spatial_loss(
        self,
        pred_scores_subset: torch.Tensor,
        gt_classes_subset: torch.Tensor,
    ) -> torch.Tensor:
        """
        Margin CRF spatial loss with forward-backward (whole-mouth, C=32).

        Computes L = max(0, (-score_GT + log_Z) / N - margin), where Z is the partition
        function over ALL valid monotonically increasing spatial orderings across all 32
        tooth positions. Uses explicit forward-backward for numerically stable marginals.

        Args:
            pred_scores_subset: Raw logits for representative anchors, shape (n_teeth, 32).
            gt_classes_subset: GT class indices for each anchor, shape (n_teeth,).

        Returns:
            Margin CRF spatial loss scalar.
        """
        n_teeth = pred_scores_subset.shape[0]
        if n_teeth < 2:
            return torch.tensor(0.0, device=pred_scores_subset.device, dtype=pred_scores_subset.dtype)
        return self._margin_crf_loss(pred_scores_subset, gt_classes_subset)

    def _crf_forward_dp(
        self,
        logits: torch.Tensor,
        gt_classes: torch.Tensor,
        spatial_to_class: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward DP for CRF loss (whole-mouth, C=32).

        Args:
            logits: Raw logits, shape (N, 32).
            gt_classes: GT class indices, shape (N,).
            spatial_to_class: Mapping from spatial position to class index, shape (C,).

        Returns:
            CRF loss scalar (normalized by N).
        """
        device = logits.device
        # Force float32: CRF uses -1e9 as "practical -inf" which overflows fp16
        # (fp16 max ≈ 65504), becoming true -inf. In logsumexp backward,
        # exp(-inf - (-inf)) = exp(NaN) = NaN, which makes GradScaler skip
        # ALL optimizer steps (not just anatomy), causing total training failure.
        logits = logits.float()
        dtype = logits.dtype  # always float32
        C = spatial_to_class.shape[0]  # 32 for whole-mouth
        N = logits.shape[0]

        # Sort detections by GT full-mouth spatial position (0-31)
        # Upper jaw: positions 0-15 (L-to-R), Lower jaw: positions 16-31 (L-to-R)
        pos_lookup = self.crf_full_position_lookup.to(device)
        gt_spatial = pos_lookup[gt_classes].long()  # (N,)
        sort_idx = gt_spatial.argsort()
        sorted_logits = logits[sort_idx]  # (N, 32)
        gt_sp_sorted = gt_spatial[sort_idx]  # (N,) strictly increasing

        # Remap logits to spatial position space with temperature-scaled log_softmax.
        # Raw logits cause CRF loss to vanish as BCE drives logits to large magnitudes
        # (e.g., +/-10-15), making the partition function degenerate to a single path.
        # log_softmax normalizes per-position and temperature controls effective scale.
        S = F.log_softmax(sorted_logits / self.crf_temperature, dim=-1)[:, spatial_to_class]  # (N, C)

        # GT path score: sum of scores at GT spatial positions
        gt_score = S[torch.arange(N, device=device), gt_sp_sorted].sum()

        # Edge case: N >= C (all 32 teeth present) means only one valid path.
        # Extremely rare (requires all wisdom teeth). Return 0 to avoid degenerate DP.
        if N >= C:
            return torch.tensor(0.0, device=device, dtype=dtype)

        # Forward DP with LogSumExp to compute log Z (partition function)
        # Use large negative finite value instead of -inf to avoid NaN gradients
        # (logsumexp on all-inf inputs produces NaN gradients; 0 * NaN = NaN in torch.where)
        NEG_INF = torch.tensor(-1e9, device=device, dtype=dtype)

        T = self.crf_transition.to(device=device, dtype=dtype)  # (C, C), precomputed

        # Base case: detection 0 can be at positions 0..C-N
        max_start = C - N
        alpha = NEG_INF.expand(C).clone()
        alpha[: max_start + 1] = S[0, : max_start + 1]

        # Forward pass
        for i in range(1, N):
            # alpha_expanded[k, j] = alpha[k] + T[k, j]
            # T[k,j] = 0 if k < j, -inf if k >= j
            # So logsumexp over dim=0 gives logsumexp(alpha[k] for k < j) for each j
            alpha_expanded = alpha.unsqueeze(1) + T  # (C, C)
            alpha_new = torch.logsumexp(alpha_expanded, dim=0) + S[i]  # (C,)

            # Feasibility: detection i needs at least i positions before it
            # and N-1-i positions after it
            min_pos = i
            max_pos = C - N + i
            feasible = torch.zeros(C, dtype=torch.bool, device=device)
            feasible[min_pos : max_pos + 1] = True
            alpha = torch.where(feasible, alpha_new, NEG_INF)

        # Partition function: sum over all valid final positions
        log_Z = torch.logsumexp(alpha, dim=0)

        # CRF loss: negative GT score + log partition function
        loss = (-gt_score + log_Z) / N

        return loss

    def _crf_backward(self, S, N, C, T, NEG_INF):
        """
        Backward algorithm for CRF.

        Computes beta[i, p] = logsumexp over q > p of (S[i+1, q] + beta[i+1, q]).

        Args:
            S: Emissions (N, C) in spatial position space, float32.
            N: Number of anchors.
            C: Number of spatial positions (32).
            T: Transition matrix (C, C), T[p,q] = 0 if p < q, -1e9 otherwise.
            NEG_INF: Sentinel value (-1e9).

        Returns:
            beta: (N, C) backward log-probabilities.
        """
        device, dtype = S.device, S.dtype
        beta = torch.full((N, C), NEG_INF.item(), device=device, dtype=dtype)

        # Base case: last anchor, feasible positions N-1 to C-1
        min_last = N - 1
        max_last = C - 1
        beta[N - 1, min_last : max_last + 1] = 0.0

        # Backward pass
        for i in range(N - 2, -1, -1):
            # combine[q] = S[i+1, q] + beta[i+1, q]
            combine = S[i + 1] + beta[i + 1]  # (C,)

            # beta[i, p] = logsumexp_q (combine[q] + T[p, q])
            # T[p, q] = 0 if p < q (valid: next position q > current p)
            # So this computes logsumexp_{q > p} combine[q]
            beta_expanded = combine.unsqueeze(0) + T  # (C, C): [p, q]
            beta_new = torch.logsumexp(beta_expanded, dim=1)  # (C,)

            # Explicit feasibility masking
            min_pos = i
            max_pos = C - N + i
            feasible = torch.zeros(C, dtype=torch.bool, device=device)
            feasible[min_pos : max_pos + 1] = True
            beta[i] = torch.where(feasible, beta_new, NEG_INF)

        return beta

    def _margin_crf_loss(self, pred_scores_subset, gt_classes_subset):
        """
        Margin CRF loss with explicit forward-backward.

        Computes L = max(0, (log_Z - score_GT) / N - margin) with analytical gradients
        from forward-backward marginals instead of autograd through the forward chain.

        Args:
            pred_scores_subset: Raw logits (n_teeth, 32).
            gt_classes_subset: GT class indices (n_teeth,).

        Returns:
            Margin CRF loss scalar.
        """
        device = pred_scores_subset.device
        logits = pred_scores_subset.float()
        C = 32
        N = logits.shape[0]

        # Sort by GT spatial position
        pos_lookup = self.crf_full_position_lookup.to(device)
        spatial_to_class = self.spatial_to_class_full.to(device)
        gt_spatial = pos_lookup[gt_classes_subset].long()
        sort_idx = gt_spatial.argsort()
        sorted_logits = logits[sort_idx]
        gt_sp_sorted = gt_spatial[sort_idx]

        # Temperature-scaled log-softmax emissions
        S = F.log_softmax(sorted_logits / self.crf_temperature, dim=-1)[:, spatial_to_class]

        # Edge case: N >= C (all 32 teeth present, only 1 valid path)
        if N >= C:
            return torch.tensor(0.0, device=device, dtype=logits.dtype)

        NEG_INF = torch.tensor(-1e9, device=device, dtype=logits.dtype)
        T = self.crf_transition.to(device=device, dtype=logits.dtype)

        # === Forward algorithm (compute alpha table and log_Z) ===
        alpha_table = torch.full((N, C), NEG_INF.item(), device=device, dtype=logits.dtype)
        alpha_table[0, : C - N + 1] = S[0, : C - N + 1]

        for i in range(1, N):
            alpha_expanded = alpha_table[i - 1].unsqueeze(1) + T  # (C, C)
            alpha_new = torch.logsumexp(alpha_expanded, dim=0) + S[i]  # (C,)
            min_pos, max_pos = i, C - N + i
            feasible = torch.zeros(C, dtype=torch.bool, device=device)
            feasible[min_pos : max_pos + 1] = True
            alpha_table[i] = torch.where(feasible, alpha_new, NEG_INF)

        log_Z = torch.logsumexp(alpha_table[N - 1], dim=0)

        # === Backward algorithm (compute beta) ===
        beta_table = self._crf_backward(S, N, C, T, NEG_INF)

        # === Marginals ===
        log_marginals = alpha_table + beta_table - log_Z  # (N, C)
        marginals = log_marginals.exp().detach()

        # === GT score ===
        gt_score = S[torch.arange(N, device=device), gt_sp_sorted].sum()
        raw_loss = (-gt_score + log_Z) / N

        # === Margin CRF loss with custom gradient ===
        loss = _MarginCRFFunction.apply(S, gt_sp_sorted, marginals, raw_loss.detach(), self.crf_margin)
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

    def _neighbor_loss_gt_conditioned(
        self,
        pred_scores: torch.Tensor,
        bboxes: torch.Tensor,
        gt_classes: torch.Tensor,
        k: int = 2,
    ) -> torch.Tensor:
        """
        GT-conditioned neighbor loss using softplus penalty.

        KEY DIFFERENCE from _neighbor_loss_margin:
        - Uses GT class of neighbor instead of predicted class
        - This aligns training with the evaluation metric (which uses GT neighbors)

        For each detection A with spatial neighbor B:
        - Check if pred_A is valid neighbor of gt_B (NOT pred_B)
        - Penalize using softplus if the gap is below margin

        This catches cases where both predictions are wrong but "consistent" -
        the pred-conditioned loss misses these because it only checks pred vs pred.

        Args:
            pred_scores: Raw logits for detections, shape (n_teeth, num_classes).
            bboxes: Bounding boxes in xyxy format, shape (n_teeth, 4).
            gt_classes: GT class indices (0-31), shape (n_teeth,).
            k: Number of nearest neighbors to check (default 2).

        Returns:
            Differentiable neighbor loss scalar.
        """
        n = pred_scores.shape[0]
        dtype = pred_scores.dtype
        device = pred_scores.device

        if n < 2:
            return torch.tensor(0.0, device=device, dtype=dtype)

        if self.nc != len(self.FDI_CLASSES):
            return torch.tensor(0.0, device=device, dtype=dtype)

        margin = self._get_adaptive_margin()
        probs = pred_scores.softmax(dim=-1)  # (n, C)

        # Compute bbox centers and distances
        centers = (bboxes[:, :2] + bboxes[:, 2:4]) / 2
        centers_f = centers.float()
        dists = torch.cdist(centers_f.unsqueeze(0), centers_f.unsqueeze(0)).squeeze(0)
        dists = dists + torch.eye(n, device=device) * 1e6

        k_actual = min(k, n - 1)
        _, knn_idx = dists.topk(k_actual, dim=1, largest=False)

        # Distance thresholds for missing teeth handling
        closest_dists = dists.min(dim=1).values
        median_closest = closest_dists.median()
        global_threshold = self.neighbor_median_multiplier * median_closest
        per_tooth_threshold = self.neighbor_closest_multiplier * closest_dists
        threshold = torch.minimum(global_threshold.expand(n), per_tooth_threshold)

        # Adjacency matrices
        invalid_adjacency = self.invalid_adjacency.to(dtype=dtype)
        valid_adjacency = 1.0 - invalid_adjacency

        arange_n = torch.arange(n, device=device)
        all_penalties = torch.zeros(k_actual, n, device=device, dtype=dtype)

        for j in range(k_actual):
            neighbor_idx = knn_idx[:, j]
            neighbor_probs = probs[neighbor_idx]
            neighbor_dists = dists[arange_n, neighbor_idx]

            # KEY CHANGE: Use GT class of neighbor instead of predicted class
            gt_class_neighbor = gt_classes[neighbor_idx]  # (n,)

            # Get valid/invalid masks based on GT neighbor class
            invalid_mask = invalid_adjacency[gt_class_neighbor]  # (n, C)
            valid_mask = valid_adjacency[gt_class_neighbor]      # (n, C)

            # Compute max probability on valid vs invalid classes
            max_valid_prob = (probs * valid_mask).max(dim=1).values
            max_invalid_prob = (probs * invalid_mask).max(dim=1).values

            gap = max_valid_prob - max_invalid_prob
            penalty = F.softplus(margin - gap)

            within_threshold = (neighbor_dists <= threshold).to(dtype=dtype)

            # Symmetric: check neighbor's prediction against this detection's GT
            gt_class_self = gt_classes  # (n,)
            invalid_mask_neighbor = invalid_adjacency[gt_class_self]
            valid_mask_neighbor = valid_adjacency[gt_class_self]

            max_valid_neighbor = (neighbor_probs * valid_mask_neighbor).max(dim=1).values
            max_invalid_neighbor = (neighbor_probs * invalid_mask_neighbor).max(dim=1).values

            gap_neighbor = max_valid_neighbor - max_invalid_neighbor
            penalty_neighbor = F.softplus(margin - gap_neighbor)

            all_penalties[j] = 0.5 * (penalty + penalty_neighbor) * within_threshold

        return all_penalties.mean()

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

    def _ordinal_classification_loss(
        self,
        pred_scores: torch.Tensor,
        target_scores: torch.Tensor,
        fg_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Ordinal-weighted classification loss.

        Penalizes predictions based on distance from GT class:
        - Nearby wrong class (off-by-1): baseline penalty × (1 + alpha)
        - Distant wrong class (different quadrant): baseline penalty × (1 + alpha * distance)

        This makes the model prefer "close" mistakes over "far" mistakes,
        which should help reduce violations caused by cross-quadrant confusions.

        The weight formula is: weight[k] = 1 + alpha * distance(k, gt_class)
        where distance is computed by _build_distance_matrix().

        Args:
            pred_scores: Raw classification logits, shape (batch, num_anchors, num_classes).
            target_scores: Soft target labels from assigner, shape (batch, num_anchors, num_classes).
            fg_mask: Foreground mask, shape (batch, num_anchors).

        Returns:
            Weighted classification loss scalar.
        """
        batch_size = pred_scores.shape[0]
        dtype = pred_scores.dtype
        device = pred_scores.device

        # Standard BCE (unreduced)
        bce_raw = F.binary_cross_entropy_with_logits(
            pred_scores,
            target_scores.to(dtype),
            reduction='none'
        )  # (batch, num_anchors, num_classes)

        # Initialize weights to 1.0 (baseline - same as standard BCE)
        weights = torch.ones_like(bce_raw)

        # Pre-convert distance matrix to match weights dtype (not pred_scores dtype!)
        # In AMP, bce_raw may be Half even if pred_scores is Float32 due to autocast
        # We must match weights.dtype for the assignment to work
        distance_matrix = self.class_distance_matrix.to(dtype=weights.dtype)  # (32, 32)

        # Only apply ordinal weighting to foreground anchors
        for i in range(batch_size):
            fg_i = fg_mask[i]
            if not fg_i.any():
                continue

            fg_idx = fg_i.nonzero(as_tuple=False).squeeze(-1)
            if fg_idx.dim() == 0:
                fg_idx = fg_idx.unsqueeze(0)

            # Get GT class for each foreground anchor (argmax of soft labels)
            gt_classes = target_scores[i, fg_idx].argmax(dim=-1)  # (num_fg,)

            # Get distances from each GT class to all classes (already correct dtype)
            distances = distance_matrix[gt_classes]  # (num_fg, num_classes)

            # Compute weights: 1 + alpha * distance
            # Result stays in correct dtype since distances is already converted
            anchor_weights = 1.0 + self.ordinal_alpha * distances  # (num_fg, num_classes)

            # Assign weights to foreground positions
            weights[i, fg_idx] = anchor_weights

        # Apply weights and reduce
        weighted_bce = bce_raw * weights

        # Normalize by sum of target scores (same as standard YOLO cls loss)
        target_scores_sum = max(target_scores.sum(), 1)
        loss = weighted_bce.sum() / target_scores_sum

        return loss


class DentalDetectionLoss(DentalSegmentationLoss):
    """Detection-only variant of DentalSegmentationLoss.

    Inherits all dental methods (CRF, anatomy, neighbor, etc.) from DentalSegmentationLoss
    but uses v8DetectionLoss as the base (no segmentation head required).

    Returns 4 loss components: [box, cls, dfl, anatomy]
    """

    def __init__(self, model):
        """Initialize DentalDetectionLoss with detection base + dental components."""
        # Skip DentalSegmentationLoss/v8SegmentationLoss __init__, go straight to detection
        v8DetectionLoss.__init__(self, model)
        self._init_dental_components(model)

    def __call__(self, preds, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate detection loss + anatomy. Returns [box, cls, dfl, anatomy]."""
        loss = torch.zeros(4, device=self.device)
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        from ultralytics.utils.tal import make_anchors

        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        # Capture target_gt_idx (v8DetectionLoss discards it)
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

        # Classification loss (standard or ordinal-weighted)
        if self.ordinal_alpha > 0:
            loss[1] = self._ordinal_classification_loss(pred_scores, target_scores, fg_mask)
        else:
            loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        # Bbox loss
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
            )

            # Anatomy loss
            if anatomy_weight > 0:
                if self._timing_enabled:
                    if self._timing_sync and pred_scores.is_cuda:
                        torch.cuda.synchronize()
                    t_start = time.perf_counter()
                loss[3] = self.compute_anatomy_loss_vectorized(
                    pred_scores, pred_bboxes * stride_tensor, target_scores, target_gt_idx, fg_mask
                )
                if self._timing_enabled:
                    if self._timing_sync and pred_scores.is_cuda:
                        torch.cuda.synchronize()
                    self._timing["anatomy"] += time.perf_counter() - t_start

        # Apply loss weights
        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain
        loss[3] *= anatomy_weight  # anatomy gain

        if self._timing_enabled:
            self._timing["batches"] += 1
            if self._timing["batches"] % self._timing_every == 0:
                LOGGER.info(
                    f"Dental loss timing (last {self._timing_every} batches): "
                    f"anatomy={self._timing['anatomy']:.3f}s, "
                    f"sync={self._timing_sync}"
                )
                self._timing["anatomy"] = 0.0

        return loss * batch_size, loss.detach()  # loss(box, cls, dfl, anatomy)
