"""Issue #81: boundary-regression head and CDF segment loss for the sequence TCN (training options, off by default)."""
# ruff: noqa: E501
from __future__ import annotations

import importlib.util
import sys

import numpy as np
import pytest

from mtah2027.config import REPO_ROOT

torch = pytest.importorskip("torch")
sys.path.insert(0, str(REPO_ROOT / "scripts"))
_spec = importlib.util.spec_from_file_location("train_sequence_81", REPO_ROOT / "scripts" / "train_sequence.py")
ts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ts)


def _y(rows):
    return torch.tensor(rows, dtype=torch.int64)


def test_boundary_target_peaks_at_direct_changes_only():
    y = _y([[0] * 20 + [1] * 20 + [2] * 10 + [0] * 10 + [3] * 5])
    tg = ts.boundary_target(y, sigma=2.0)
    assert tg.shape == y.shape
    assert tg[0, 20].item() == pytest.approx(1.0)                 # H -> A at frame 20
    assert tg[0, 21].item() == pytest.approx(np.exp(-0.5 / 4.0), abs=1e-6) and tg[0, 19].item() == pytest.approx(tg[0, 21].item())
    assert tg[0, 40].item() == 0.0 and tg[0, 50].item() == 0.0    # A -> D and D -> H are not direct changes
    assert tg[0, 0].item() < 1e-3 and tg.min().item() >= 0.0 and tg.max().item() <= 1.0


def test_boundary_target_takes_the_max_of_close_events_and_ignores_invalid():
    y = _y([[0, 0, 1, 1, 0, 0, 3, 3, 1, 1]])
    tg = ts.boundary_target(y, sigma=1.0)
    assert tg[0, 2].item() == 1.0 and tg[0, 4].item() == 1.0 and tg[0, 3].item() == pytest.approx(np.exp(-0.5))
    assert tg[0, 8].item() < 1.0 and tg[0, 6].item() == pytest.approx(0.0, abs=0.2)   # 0 -> 3 -> 1: no direct change


def test_boundary_bce_is_lower_for_the_right_logits():
    y = _y([[0] * 30 + [1] * 30])
    tg = ts.boundary_target(y, sigma=3.0)
    valid = y < 3
    good = torch.where(tg > 0.5, 6.0, -6.0)
    bad = -good
    assert ts.boundary_bce(good, tg, valid).item() < ts.boundary_bce(bad, tg, valid).item()
    assert ts.boundary_bce(good, tg, torch.zeros_like(valid)).item() == 0.0


def _logits_from(labels, conf=8.0):
    oh = torch.nn.functional.one_hot(labels.clamp(max=2), 3).float()
    return (oh * conf).float()


def test_segment_loss_is_zero_when_perfect_and_grows_with_boundary_shift():
    y = _y([[0] * 100 + [1] * 100 + [0] * 56])
    valid = y < 3
    perfect = ts.segment_cdf_loss(_logits_from(y, 30.0), y, valid, win=64, stride=32).item()
    small = _y([[0] * 103 + [1] * 97 + [0] * 56])
    big = _y([[0] * 125 + [1] * 75 + [0] * 56])
    l_small = ts.segment_cdf_loss(_logits_from(small, 30.0), y, valid, win=64, stride=32).item()
    l_big = ts.segment_cdf_loss(_logits_from(big, 30.0), y, valid, win=64, stride=32).item()
    assert perfect < 1e-4 < l_small < l_big


def test_segment_loss_penalises_a_merged_possession_more_than_framewise_noise():
    y = _y([[0] * 64 + [1] * 64 + [0] * 64 + [1] * 64])
    valid = y < 3
    merged = _y([[0] * 192 + [1] * 64])                 # a whole A possession missed (one error segment)
    noise = y.clone()
    noise[0, ::16] = 1 - noise[0, ::16]                 # 16 isolated wrong frames
    lm = ts.segment_cdf_loss(_logits_from(merged, 20.0), y, valid, win=128, stride=64).item()
    ln = ts.segment_cdf_loss(_logits_from(noise, 20.0), y, valid, win=128, stride=64).item()
    assert lm > 3 * ln


def test_segment_loss_ignores_invalid_frames_and_has_gradient():
    y = _y([[0] * 60 + [3] * 68])
    lg = torch.zeros(1, 128, 3, requires_grad=True)
    loss = ts.segment_cdf_loss(lg, y, y < 3, win=64, stride=32)
    loss.backward()
    assert torch.isfinite(loss) and lg.grad is not None and lg.grad[0, 100:].abs().sum().item() == 0.0
    short = ts.segment_cdf_loss(torch.zeros(1, 10, 3), _y([[0] * 10]), torch.ones(1, 10, dtype=torch.bool))
    assert torch.isfinite(short)


def test_boundary_head_is_training_only_and_default_network_unchanged():
    torch.manual_seed(0)
    base = ts.TCN(6, 8, dilations=(1, 2))
    torch.manual_seed(0)
    withh = ts.TCN(6, 8, dilations=(1, 2), boundary_head=True)
    assert not hasattr(base, "hbd") and hasattr(withh, "hbd")
    for (k, a), (k2, b) in zip(base.state_dict().items(), withh.state_dict().items()):
        assert k == k2 and torch.equal(a, b)                       # same init: head created last
    sm = withh.export(np.zeros(6, np.float32), np.ones(6, np.float32), [f"f{i}" for i in range(6)], (1, 2))
    assert not any("hbd" in k or k.startswith("w_b") for k in sm.w)
    x = torch.randn(2, 40, 6)
    h = withh.trunk(x)
    assert withh.boundary_logits(h).shape == (2, 40)
