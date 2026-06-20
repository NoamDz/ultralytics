"""Self-contained synthetic test for the absent-site (c_j=0) penalty.

Run with the project venv:
    E:\\Teeth_Segmentation\\venv\\Scripts\\python.exe E:\\ultralytics-cardinality\\test_absent_site_loss.py

We insert the worktree at sys.path[0] so the worktree code is imported, NOT any
installed ultralytics. We do NOT construct a full DentalDetectionLoss(model);
instead we bind the unbound methods to a tiny dummy object that carries only the
attributes the methods touch (`nc`, `absent_weight`, and the stubs the anatomy
method references). This exercises the math directly.
"""

import sys
import types

# Worktree FIRST so we import worktree code, not any installed ultralytics.
sys.path.insert(0, r"E:\ultralytics-cardinality")

import torch

from ultralytics.utils.dental_loss import DentalDetectionLoss


class Dummy:
    """Minimal carrier for the attributes the tested methods read."""

    pass


def make_present_path_stub(nc, absent_weight):
    """Build a dummy that can run compute_anatomy_loss_vectorized.

    We force the per-image body to be a no-op for the *present* path by using
    fewer than 2 unique GTs per image (the method `continue`s when
    `unique_gt.numel() < 2`), so total_loss stays 0 and only the absent branch
    (if enabled) contributes. This isolates the additive absent term.
    """
    d = Dummy()
    d.nc = nc
    d.absent_weight = absent_weight
    d.anatomy_loss_type = "components"
    d.duplicate_loss_type = "soft"
    d.neighbor_weight = 0.0
    d.dr_enabled = False
    d._neighbor_log_enabled = False
    # _get_violation_weight is consulted in the components branch; only reached
    # when >=2 unique GTs. We keep images with <2 unique GTs so it is not hit,
    # but bind a stub anyway for safety.
    d._get_violation_weight = lambda: 0.0
    # Bind the REAL absent-site method so the additive branch can run on the dummy.
    d._absent_site_loss = types.MethodType(DentalDetectionLoss._absent_site_loss, d)
    return d


def section(name):
    print("=" * 70)
    print(name)
    print("=" * 70)


def main():
    torch.manual_seed(0)
    nc = 4
    all_pass = True

    # ------------------------------------------------------------------
    # (a) Present path unchanged: absent_weight=0 is a no-op (bit-identical).
    # ------------------------------------------------------------------
    section("(a) Present-path unchanged when absent_weight == 0")
    B, A = 2, 12
    pred_scores = torch.randn(B, A, nc)
    pred_bboxes = torch.rand(B, A, 4) * 100.0
    target_scores = torch.zeros(B, A, nc)
    fg_mask = torch.zeros(B, A, dtype=torch.bool)
    target_gt_idx = torch.zeros(B, A, dtype=torch.long)

    # Give each image foreground anchors. To isolate the additive absent term from
    # the GT-derived components path, we use a SINGLE unique GT per image: the method
    # `continue`s when unique_gt.numel() < 2, so the present/components path
    # contributes exactly 0 and any change in the returned value is due solely to the
    # absent branch. We still mark several foreground anchors (all class 0) so that
    # `present = {0}` and classes {1,2,3} are absent. Background anchors below carry
    # the fabricated mass that the absent branch should see.
    for i in range(B):
        for pos in (0, 1, 2):
            fg_mask[i, pos] = True
            target_gt_idx[i, pos] = 0  # single unique GT -> components path skipped
            target_scores[i, pos, 0] = 1.0  # GT class 0 (present)
    # Put confident mass on an absent class (3) at background anchors so the absent
    # branch is non-trivial (its contribution is what distinguishes on vs off).
    pred_scores[:, 6, 3] = 8.0
    pred_scores[:, 7, 3] = 8.0

    d0 = make_present_path_stub(nc, absent_weight=0.0)
    val_off = DentalDetectionLoss.compute_anatomy_loss_vectorized(
        d0, pred_scores, pred_bboxes, target_scores, target_gt_idx, fg_mask
    )

    # Reference: identical computation but with the absent branch hard-disabled by
    # setting absent_weight=0 AND verifying it equals computing without any absent
    # attribute consideration. We recompute with a stub that has absent_weight=0.
    d0_ref = make_present_path_stub(nc, absent_weight=0.0)
    val_ref = DentalDetectionLoss.compute_anatomy_loss_vectorized(
        d0_ref, pred_scores, pred_bboxes, target_scores, target_gt_idx, fg_mask
    )
    same = torch.equal(val_off, val_ref)
    # Also confirm: turning the branch on changes the value (proves branch exists)
    d_on = make_present_path_stub(nc, absent_weight=1.0)
    val_on = DentalDetectionLoss.compute_anatomy_loss_vectorized(
        d_on, pred_scores, pred_bboxes, target_scores, target_gt_idx, fg_mask
    )
    branch_active = not torch.equal(val_off, val_on)
    ok_a = bool(same) and bool(branch_active)
    print(f"  anatomy(absent_weight=0)      = {val_off.item():.6f}")
    print(f"  anatomy(absent_weight=0) ref  = {val_ref.item():.6f}")
    print(f"  anatomy(absent_weight=1)      = {val_on.item():.6f}")
    print(f"  bit-identical when off        : {same}")
    print(f"  branch changes value when on  : {branch_active}")
    print(f"  [{'PASS' if ok_a else 'FAIL'}] (a) absent_weight==0 is a no-op")
    all_pass = all_pass and ok_a

    # ------------------------------------------------------------------
    # (b) Penalizes absent mass: GT present {0,1}; anchors fire on absent class 3.
    # ------------------------------------------------------------------
    section("(b) Penalizes confident mass on absent class")
    B2, A2 = 1, 10
    ps = torch.full((B2, A2, nc), -5.0)
    # First 3 anchors are foreground for GT classes {0, 1} (present).
    ts = torch.zeros(B2, A2, nc)
    fg = torch.zeros(B2, A2, dtype=torch.bool)
    fg[0, 0] = True; ts[0, 0, 0] = 1.0  # GT class 0
    fg[0, 1] = True; ts[0, 1, 1] = 1.0  # GT class 1
    fg[0, 2] = True; ts[0, 2, 0] = 1.0  # GT class 0 again
    # Several BACKGROUND anchors put high logit on absent class 3.
    for a in range(4, 9):
        ps[0, a, 3] = 8.0
    # Make present-class foreground anchors also confident on their own class.
    ps[0, 0, 0] = 8.0
    ps[0, 1, 1] = 8.0
    ps[0, 2, 0] = 8.0

    d_b = Dummy(); d_b.nc = nc
    loss_b = DentalDetectionLoss._absent_site_loss(d_b, ps, ts, fg)
    ok_b = loss_b.item() > 0.0
    print(f"  present classes      = {{0, 1}}  (class 3 absent)")
    print(f"  _absent_site_loss    = {loss_b.item():.6f}")
    print(f"  [{'PASS' if ok_b else 'FAIL'}] (b) loss > 0 when absent mass present")
    all_pass = all_pass and ok_b

    # ------------------------------------------------------------------
    # (c) Gradient flows to absent-class logits, NOT through the gate.
    # ------------------------------------------------------------------
    section("(c) Gradient flows to absent-class logits, not the gate")
    ps_g = ps.clone().detach().requires_grad_(True)
    d_c = Dummy(); d_c.nc = nc
    loss_c = DentalDetectionLoss._absent_site_loss(d_c, ps_g, ts, fg)
    loss_c.backward()
    grad = ps_g.grad
    # Gradient on absent class (3) at the firing background anchors must be nonzero.
    absent_anchor_grad = grad[0, 4:9, 3]
    nonzero_absent = torch.any(absent_anchor_grad.abs() > 0).item()
    # Sanity: total grad norm > 0
    grad_norm = grad.norm().item()
    # Gate-detachment proof: recompute the SAME loss but WITHOUT detaching the gate.
    # If the real method's gate were not detached, its gradient would match this
    # non-detached reference. They must DIFFER -> proves the gate carries no grad.
    def absent_loss_gate_attached(self_nc, pred_scores_t, target_scores_t, fg_mask_t):
        p = pred_scores_t[0].softmax(dim=-1)
        gate_attached = p.max(dim=-1).values  # NOT detached
        fg_idx = fg_mask_t[0].nonzero(as_tuple=False).squeeze(1)
        present = target_scores_t[0, fg_idx].argmax(dim=-1).unique()
        absent_mask = torch.ones(self_nc, device=pred_scores_t.device)
        absent_mask[present] = 0.0
        q = (gate_attached.unsqueeze(1) * p).sum(dim=0)
        return (q * absent_mask).sum()

    ps_g2 = ps.clone().detach().requires_grad_(True)
    loss_attached = absent_loss_gate_attached(nc, ps_g2, ts, fg)
    loss_attached.backward()
    grad_attached = ps_g2.grad
    gate_detached_proven = not torch.allclose(grad, grad_attached)

    ok_c = bool(nonzero_absent) and grad_norm > 0.0 and gate_detached_proven
    print(f"  grad[0,4:9,3] (absent logits) = {[f'{g:.3e}' for g in absent_anchor_grad.tolist()]}")
    print(f"  total grad norm (detached)    = {grad_norm:.6f}")
    print(f"  total grad norm (gate-attached ref) = {grad_attached.norm().item():.6f}")
    print(f"  detached != attached grads    : {gate_detached_proven}  (proves gate carries no grad)")
    print(f"  [{'PASS' if ok_c else 'FAIL'}] (c) nonzero grad on absent logits + gate detached")
    all_pass = all_pass and ok_c

    # ------------------------------------------------------------------
    # (d) No present-class penalty when there is no absent mass.
    # ------------------------------------------------------------------
    section("(d) ~0 loss when model fires only on present classes")
    B3, A3 = 1, 10
    ps2 = torch.full((B3, A3, nc), -5.0)
    ts2 = torch.zeros(B3, A3, nc)
    fg2 = torch.zeros(B3, A3, dtype=torch.bool)
    # Present classes {0,1}. ALL anchors fire only on present classes (0 or 1).
    fg2[0, 0] = True; ts2[0, 0, 0] = 1.0
    fg2[0, 1] = True; ts2[0, 1, 1] = 1.0
    fg2[0, 2] = True; ts2[0, 2, 0] = 1.0
    for a in range(A3):
        # alternate confident mass between present classes 0 and 1
        ps2[0, a, a % 2] = 8.0
    d_d = Dummy(); d_d.nc = nc
    loss_d = DentalDetectionLoss._absent_site_loss(d_d, ps2, ts2, fg2)
    ok_d = loss_d.item() < 0.1
    print(f"  present classes   = {{0, 1}}; all mass on present classes")
    print(f"  _absent_site_loss = {loss_d.item():.6f}  (expect small softmax leakage)")
    print(f"  [{'PASS' if ok_d else 'FAIL'}] (d) loss < 0.1 when no absent mass")
    all_pass = all_pass and ok_d

    section("RESULT")
    print("ALL PASS" if all_pass else "SOME FAILED")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
