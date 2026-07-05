"""Tests for the identity-aware attribution loss (dental_identity.py) and its config keys."""

import sys

import pytest
import torch

WORKTREE = "E:/ultralytics-identity"


def test_imports_worktree():
    """Guard: make sure we test the worktree copy, not an installed ultralytics."""
    import ultralytics

    assert "ultralytics-identity" in ultralytics.__file__.replace("\\", "/"), ultralytics.__file__


def test_cfg_accepts_identity_keys():
    from ultralytics.cfg import get_cfg

    cfg = get_cfg(
        overrides=dict(
            ordinal_alpha=0.6,
            identity_loss_type="column",
            identity_weight=0.5,
            identity_tau=0.1,
            identity_iou_floor=0.5,
            identity_pos_weight=0.0,
            identity_warmup_iters=500,
            identity_p2_sharpen=False,
        )
    )
    assert cfg.identity_loss_type == "column"
    assert cfg.identity_weight == 0.5
    assert cfg.ordinal_alpha == 0.6


def test_cfg_identity_default_off_is_string():
    """YAML `off` parses as bool False — the default must be the *string* 'off' (quoted in yaml)."""
    from ultralytics.cfg import get_cfg

    cfg = get_cfg()
    assert cfg.identity_loss_type == "off", (
        f"identity_loss_type default is {cfg.identity_loss_type!r} — "
        "did default.yaml write bare `off` (YAML bool) instead of quoted \"off\"?"
    )
    assert cfg.identity_weight == 0.0


# ---------------------------------------------------------------------------
# dental_identity.py — pure-function math tests
# ---------------------------------------------------------------------------


def _rand_case(n=5, nc=32, seed=42, dtype=torch.float64):
    """Random but reproducible case: n teeth, boxes roughly on a row, distinct classes."""
    g = torch.Generator().manual_seed(seed)
    x0 = torch.arange(n, dtype=dtype) * 60.0
    gt_boxes = torch.stack([x0, torch.zeros(n, dtype=dtype), x0 + 50.0, torch.full((n,), 80.0, dtype=dtype)], dim=1)
    jitter = torch.rand((n, 4), generator=g, dtype=dtype) * 8.0 - 4.0
    pred_boxes = gt_boxes + jitter
    logits = torch.randn((n, nc), generator=g, dtype=dtype) * 2.0
    classes = torch.arange(n, dtype=torch.long)  # distinct FDI indices 0..n-1
    return logits, pred_boxes, gt_boxes, classes


def _closed_form_grad(z, A, g, m, Z):
    """Spec §3.4: dL/dz[i, m_j] = (1/Z) * (1 - A[i,j]) * g[j] * sigmoid(z[i, m_j]), summed over j."""
    grad = torch.zeros_like(z)
    p = torch.sigmoid(z)
    n = z.shape[0]
    for j in range(n):
        for i in range(n):
            grad[i, m[j]] += (1.0 - A[i, j]) * g[j] * p[i, m[j]] / Z
    return grad


def _recompute_Ag(pred_boxes, gt_boxes, tau, iou_floor):
    from ultralytics.utils.dental_identity import pairwise_iou_xyxy

    S = pairwise_iou_xyxy(pred_boxes, gt_boxes)
    A = torch.softmax(S / tau, dim=0)
    top2 = S.topk(2, dim=0).values
    g = (top2[0] >= iou_floor).to(S.dtype) * (top2[0] - top2[1])
    Z = g.sum().clamp(min=1.0)
    return A, g, Z


def test_pairwise_iou_basic():
    from ultralytics.utils.dental_identity import pairwise_iou_xyxy

    a = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    b = torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0], [0.0, 0.0, 5.0, 10.0]])
    iou = pairwise_iou_xyxy(a, b)
    assert iou.shape == (1, 3)
    assert torch.allclose(iou[0, 0], torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(iou[0, 1], torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(iou[0, 2], torch.tensor(0.5), atol=1e-6)


def test_pairwise_iou_fp16_large_boxes_no_overflow():
    """fp16 boxes >256px overflow area to inf without the float cast -> IoU degrades to 0.
    With the cast, IoU is finite and correct (identical boxes -> 1.0)."""
    from ultralytics.utils.dental_identity import pairwise_iou_xyxy

    a = torch.tensor([[0.0, 0.0, 800.0, 800.0]], dtype=torch.float16)
    b = torch.tensor([[0.0, 0.0, 800.0, 800.0]], dtype=torch.float16)
    iou = pairwise_iou_xyxy(a, b)
    assert torch.isfinite(iou).all(), iou
    assert torch.allclose(iou[0, 0].float(), torch.tensor(1.0), atol=1e-3), iou


def test_gradient_matches_closed_form():
    from ultralytics.utils.dental_identity import identity_loss

    logits, pred_boxes, gt_boxes, classes = _rand_case()
    logits.requires_grad_(True)
    loss, _ = identity_loss(logits, pred_boxes, gt_boxes, classes, mode="column", tau=0.1, iou_floor=0.1)
    loss.backward()
    A, g, Z = _recompute_Ag(pred_boxes, gt_boxes, tau=0.1, iou_floor=0.1)
    expected = _closed_form_grad(logits.detach(), A, g, classes, Z)
    assert torch.allclose(logits.grad, expected, atol=1e-6), (logits.grad - expected).abs().max()


def test_gradient_matches_closed_form_duplicate_classes():
    """Two GT teeth sharing an FDI index: gradients on the shared logit column must SUM over j."""
    from ultralytics.utils.dental_identity import identity_loss

    logits, pred_boxes, gt_boxes, classes = _rand_case()
    classes = classes.clone()
    classes[1] = classes[0]  # duplicate class
    logits.requires_grad_(True)
    loss, _ = identity_loss(logits, pred_boxes, gt_boxes, classes, mode="column", tau=0.1, iou_floor=0.1)
    loss.backward()
    A, g, Z = _recompute_Ag(pred_boxes, gt_boxes, tau=0.1, iou_floor=0.1)
    expected = _closed_form_grad(logits.detach(), A, g, classes, Z)
    assert torch.allclose(logits.grad, expected, atol=1e-6)


def test_saturation_no_nan():
    """Confident impostor (z=50): loss finite, gradient == closed form (== bounded), no NaN."""
    from ultralytics.utils.dental_identity import identity_loss

    logits, pred_boxes, gt_boxes, classes = _rand_case()
    logits = logits.detach().clone()
    logits[2, int(classes[3])] = 50.0  # anchor 2 confidently claims tooth 3's number
    logits.requires_grad_(True)
    loss, _ = identity_loss(logits, pred_boxes, gt_boxes, classes, mode="column", tau=0.1, iou_floor=0.1)
    assert torch.isfinite(loss), loss
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    A, g, Z = _recompute_Ag(pred_boxes, gt_boxes, tau=0.1, iou_floor=0.1)
    expected = _closed_form_grad(logits.detach(), A, g, classes, Z)
    assert torch.allclose(logits.grad, expected, atol=1e-6)


def test_owner_protected():
    """Perfect boxes -> ownership ~one-hot -> the owner's gradient on its own class is ~0."""
    from ultralytics.utils.dental_identity import identity_loss

    logits, _, gt_boxes, classes = _rand_case()
    logits = logits.detach().clone()
    logits += 1.0  # make every anchor claim things a bit
    logits.requires_grad_(True)
    loss, _ = identity_loss(logits, gt_boxes.clone(), gt_boxes, classes, mode="column", tau=0.1, iou_floor=0.1)
    loss.backward()
    n = gt_boxes.shape[0]
    owner_grads = torch.stack([logits.grad[i, classes[i]] for i in range(n)])
    intruder_grads = torch.stack(
        [logits.grad[i, classes[j]] for i in range(n) for j in range(n) if j != i]
    )
    assert owner_grads.abs().max() < 1e-3 * intruder_grads.abs().max().clamp(min=1e-9)


def test_ambiguity_gate_zeros_tied_columns():
    """Two identical GT boxes (tied top-2 IoU) -> margin ~0 -> those columns contribute ~0."""
    from ultralytics.utils.dental_identity import identity_loss

    logits, _, gt_boxes, classes = _rand_case(n=4)
    gt_boxes = gt_boxes.clone()
    gt_boxes[1] = gt_boxes[0]  # tooth 1 sits exactly on tooth 0
    pred_boxes = gt_boxes.clone()
    logits = logits.detach().clone().requires_grad_(True)
    loss, diag = identity_loss(logits, pred_boxes, gt_boxes, classes, mode="column", tau=0.1, iou_floor=0.1)
    loss.backward()
    # columns 0 and 1 are ambiguous (two preds at IoU 1.0 each) -> gated off ->
    # no gradient into classes[0]/classes[1] from any anchor
    assert logits.grad[:, classes[0]].abs().max() < 1e-9
    assert logits.grad[:, classes[1]].abs().max() < 1e-9
    assert diag["gate_off_frac"] >= 0.5


def test_no_gradient_into_boxes():
    from ultralytics.utils.dental_identity import identity_loss

    logits, pred_boxes, gt_boxes, classes = _rand_case()
    pred_boxes = pred_boxes.detach().clone().requires_grad_(True)
    logits = logits.detach().clone().requires_grad_(True)
    loss, _ = identity_loss(logits, pred_boxes, gt_boxes, classes, mode="column", tau=0.1, iou_floor=0.1)
    loss.backward()
    assert pred_boxes.grad is None or pred_boxes.grad.abs().max() == 0.0


def test_nogeo_mode_matches_diagonal_closed_form():
    """column_nogeo: A = I (TAL ownership), g = 1, Z = n."""
    from ultralytics.utils.dental_identity import identity_loss

    logits, pred_boxes, gt_boxes, classes = _rand_case()
    logits.requires_grad_(True)
    loss, _ = identity_loss(logits, pred_boxes, gt_boxes, classes, mode="column_nogeo")
    loss.backward()
    n = logits.shape[0]
    A = torch.eye(n, dtype=logits.dtype)
    g = torch.ones(n, dtype=logits.dtype)
    expected = _closed_form_grad(logits.detach(), A, g, classes, torch.tensor(float(n)))
    assert torch.allclose(logits.grad, expected, atol=1e-6)


def test_local_mode_tiny_case():
    """local: -(1/Z) sum_i g_i log p_i(m_{j*(i)}) with j* = best-IoU GT per anchor."""
    from ultralytics.utils.dental_identity import identity_loss

    logits, pred_boxes, gt_boxes, classes = _rand_case(n=3)
    logits.requires_grad_(True)
    loss, _ = identity_loss(logits, pred_boxes, gt_boxes, classes, mode="local", iou_floor=0.1)
    # jitter is small -> j*(i) = i and every anchor passes the floor
    z_star = torch.stack([logits[i, classes[i]] for i in range(3)])
    expected = torch.nn.functional.softplus(-z_star).sum() / 3.0
    assert torch.allclose(loss, expected, atol=1e-9)


def test_single_tooth_returns_zero():
    from ultralytics.utils.dental_identity import identity_loss

    logits = torch.randn(1, 32, requires_grad=True)
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    loss, diag = identity_loss(logits, boxes, boxes, torch.tensor([5]), mode="column")
    assert float(loss.detach()) == 0.0
    assert diag == {}


def test_diag_disagreement_detects_swap():
    """Anchor boxes swapped across two GT teeth -> best-IoU owner != TAL owner for both columns."""
    from ultralytics.utils.dental_identity import identity_loss

    logits, _, gt_boxes, classes = _rand_case(n=2)
    pred_boxes = gt_boxes.flip(0).clone()  # anchor 0 sits on tooth 1 and vice versa
    loss, diag = identity_loss(logits.detach(), pred_boxes, gt_boxes, classes, mode="column")
    assert diag["tal_iou_disagree"] == 1.0
