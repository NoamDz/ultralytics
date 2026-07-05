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
