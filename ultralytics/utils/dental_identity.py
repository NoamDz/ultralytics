"""Identity-aware attribution loss (column-uniqueness) for dental FDI numbering.

Design spec: Teeth_Segmentation/docs/superpowers/specs/2026-07-05-odin-identity-aware-loss-design.md

Honest framing: this is a geometry-gated, present-class-restricted, confidence-proportional
hard-negative reweighting of BCE. For each GT tooth j (a "column"), geometry (box IoU)
names the rightful owner anchor; every OTHER representative anchor currently claiming
tooth j's FDI number is suppressed on that one class channel only.

Probability convention: per-class sigmoid p = sigmoid(logit), matching the base YOLO BCE
cls loss (NOT softmax — soft-dup uses softmax; the two terms are intentionally independent).

Gradient (spec 3.4, autograd-verified):  dL_neg/dz[i, m_j] = (1/Z) * (1 - A[i,j]) * g[j] * p_i(m_j)
- confidence-seeking: scales with the impostor's p (opposite of soft-dup's vanishing gradient)
- bounded by (1/Z)(1-A)g <= 1; reaches ONLY class logits (S, A, g all detached)
- owner-protected: (1 - A) ~ 0 for the geometric owner
- gated: ambiguous columns (near-tied IoU) abstain

Stability: -log(1 - sigmoid(z)) is computed as softplus(z) (and -log(sigmoid(z)) as
softplus(-z)); the naive form NaNs on saturated logits — the exact population targeted.
"""

import torch
import torch.nn.functional as F

__all__ = ["pairwise_iou_xyxy", "identity_loss"]


def pairwise_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Pairwise IoU between two xyxy box sets. Returns (len(boxes1), len(boxes2))."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area1[:, None] + area2[None, :] - inter + eps)


def identity_loss(
    pred_logits: torch.Tensor,
    pred_boxes: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_classes: torch.Tensor,
    *,
    mode: str = "column",
    tau: float = 0.1,
    iou_floor: float = 0.5,
    pos_weight: float = 0.0,
    p2_sharpen: bool = False,
):
    """Identity-aware attribution loss on representative anchors.

    Args:
        pred_logits: (n, nc) raw class logits, one representative anchor per GT tooth.
        pred_boxes: (n, 4) xyxy predicted boxes, PIXEL scale (pred_bboxes * stride).
        gt_boxes: (n, 4) xyxy GT boxes, PIXEL scale, same order (anchor i represents GT i).
        gt_classes: (n,) long, FDI class index per GT.
        mode: "column" (IoU ownership + ambiguity gate) | "column_nogeo" (TAL-diagonal
            ownership, no gate — the no-geometry control arm) | "local" (per-anchor CE
            toward its best-IoU GT class — plumbing scaffold).
        tau: ownership softmax temperature (column mode).
        iou_floor: minimum best-IoU for a column (or anchor, in local mode) to participate.
        pos_weight: weight of the optional positive owner-CE stabilizer (0 = off).
        p2_sharpen: multiply the anti-intruder term by detached p (focal-style sharpening).

    Returns:
        (loss, diag): scalar loss tensor and a diagnostics dict
        {"tal_iou_disagree", "gate_off_frac", "n_teeth"} ({} when n < 2).
    """
    n = pred_logits.shape[0]
    zero = pred_logits.sum() * 0.0  # keeps graph/device/dtype
    if n < 2:
        return zero, {}

    with torch.no_grad():
        S = pairwise_iou_xyxy(pred_boxes, gt_boxes)  # (n_pred, n_gt); S[i, j] = IoU(pred_i, gt_j)
        # Diagnostic: how often the best-IoU owner disagrees with the TAL owner (index i == j).
        disagree = (S.argmax(dim=0) != torch.arange(n, device=S.device)).float().mean().item()

    if mode == "local":
        with torch.no_grad():
            s_max, j_star = S.max(dim=1)  # best-overlap GT per anchor
            g_i = (s_max >= iou_floor).to(pred_logits.dtype)
            z_i = g_i.sum().clamp(min=1.0)
        target_cls = gt_classes[j_star]  # (n,)
        z_star = pred_logits.gather(1, target_cls.unsqueeze(1)).squeeze(1)
        loss = (g_i * F.softplus(-z_star)).sum() / z_i  # -log p toward the overlapped tooth's class
        diag = {"tal_iou_disagree": disagree, "gate_off_frac": float((g_i <= 0).float().mean()), "n_teeth": n}
        return loss, diag

    with torch.no_grad():
        if mode == "column_nogeo":
            # No-geometry control: ownership = TAL assignment (anchor i owns GT i), no gate.
            A = torch.eye(n, device=S.device, dtype=pred_logits.dtype)
            g = torch.ones(n, device=S.device, dtype=pred_logits.dtype)
        elif mode == "column":
            A = torch.softmax(S / tau, dim=0)  # each GT column distributes 1.0 of ownership
            top2 = S.topk(2, dim=0).values  # (2, n); n >= 2 guaranteed above
            g = (top2[0] >= iou_floor).to(pred_logits.dtype) * (top2[0] - top2[1])  # abstention gate
        else:
            raise ValueError(f"unknown identity_loss mode: {mode!r}")
        z_norm = g.sum().clamp(min=1.0)  # Z in the spec
        gate_off = float((g <= 0).float().mean())

    z_at_m = pred_logits[:, gt_classes]  # (n_pred, n_gt); z_at_m[i, j] = logit of anchor i for class m_j
    neg = F.softplus(z_at_m)  # stable -log(1 - sigmoid(z))
    if p2_sharpen:
        neg = neg * torch.sigmoid(z_at_m).detach()
    loss = ((1.0 - A) * neg * g.unsqueeze(0)).sum() / z_norm

    if pos_weight > 0:
        pos = F.softplus(-z_at_m)  # stable -log(sigmoid(z))
        loss = loss + pos_weight * ((A * pos * g.unsqueeze(0)).sum() / z_norm)

    diag = {"tal_iou_disagree": disagree, "gate_off_frac": gate_off, "n_teeth": n}
    return loss, diag
