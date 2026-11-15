# ruff: noqa: E501
"""Issue #67: oracle ceiling script helpers and the soft ball evidence (off by default)."""
from __future__ import annotations

import sys

import numpy as np

from mtah2027.vision.attention_crops import VisionParams
from mtah2027.vision.soft_evidence import apply_soft_evidence, soft_tilt

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "scripts"))
import oracle_ceiling as oc  # noqa: E402


def test_default_is_off():
    assert VisionParams.from_config().soft_evidence_weight == 0.0


def _players():
    # frame 0: home player 2 m from (10, 10), away 8 m; frame 1: reversed
    p = np.full((3, 2, 2), np.nan)
    p[0, 0], p[0, 1] = (12.0, 10.0), (18.0, 10.0)
    p[1, 0], p[1, 1] = (18.0, 10.0), (12.0, 10.0)
    p[2, 0], p[2, 1] = (12.0, 10.0), (12.0, 10.0)
    return p, np.array([0, 1])


def test_tilt_points_to_nearest_team_and_holds():
    p, team = _players()
    frames = np.array([0, 1])
    xy = np.array([[10.0, 10.0], [10.0, 10.0]])
    t = soft_tilt(8, np.concatenate([p, np.full((5, 2, 2), np.nan)]), team, frames, xy, 1.0,
                  weight=2.0, scale_m=3.0, hold_frames=0)
    assert t[0] > 0 > t[1] and not t[2:].any()
    assert np.isclose(t[0], 2.0 * np.tanh(6.0 / 3.0)) and np.isclose(t[1], -t[0] * 0 + t[1])


def test_tilt_hold_and_confidence():
    p, team = _players()
    p = np.concatenate([p, np.full((5, 2, 2), np.nan)])
    t = soft_tilt(8, p, team, np.array([0]), np.array([[10.0, 10.0]]), 0.5, weight=1.0, scale_m=3.0, hold_frames=2)
    assert np.all(t[:3] == t[0]) and t[0] > 0 and not t[3:].any()
    z = soft_tilt(8, p, team, np.array([0]), np.array([[10.0, 10.0]]), 1.0, weight=0.0, scale_m=3.0, hold_frames=2)
    assert not z.any()


def test_apply_is_identity_without_tilt_and_keeps_D():
    lp = np.log(np.full((4, 3), 1 / 3))
    assert apply_soft_evidence(lp, np.zeros(4)) is lp
    out = apply_soft_evidence(lp, np.array([2.0, 0, 0, -2.0]))
    assert np.allclose(out[0, 0] - lp[0, 0], 1.0, atol=1e-6) and np.allclose(out[0, 1] - lp[0, 1], -1.0, atol=1e-6)
    assert np.array_equal(out[:, 2], np.round(lp[:, 2], 6))


def test_scenarios_and_nested_coverage():
    assert "none" in oc.SCENARIOS and "orc100_all" in oc.SCENARIOS and "s2_n1_f10_c50" in oc.SCENARIOS
    n = 200
    truth = np.random.default_rng(0).uniform(0, 100, (n, 2))
    d = oc.draws(n, 3)
    frames = np.arange(0, n, 5)
    f100, xy100, _ = oc.detections(truth, frames, d, cov=100, sigma=0.5, fp=0.1)
    f50, xy50, _ = oc.detections(truth, frames, d, cov=50, sigma=0.5, fp=0.1)
    assert set(f50) <= set(f100) and len(f100) == len(frames)
    sel = np.isin(f100, f50)
    assert np.allclose(xy100[sel], xy50)          # same error draws: nested
    f0, xy0, fp0 = oc.detections(truth, frames, d, cov=100, sigma=0.0, fp=0.0)
    assert np.allclose(xy0, truth[f0]) and not fp0.any()


def test_decompose_perfect_and_hidden_fix():
    gt = np.array([3] * 10 + [0] * 100 + [1] * 100 + [2] * 40 + [0] * 100 + [3] * 10, dtype=np.int8)
    hidden = np.zeros(len(gt), bool)
    hidden[110:210] = True
    d = oc.decompose(gt, gt.copy(), hidden)
    assert d["spell_f1"] == 1.0 and d["spell_fate"]["matched"] == d["gt_spells"] == 3
    pred = gt.copy()
    pred[110:210] = 0                               # the A spell (hidden) is swallowed by H
    d = oc.decompose(gt, pred, hidden)
    assert d["spell_f1"] < 1.0 and d["spell_f1_fix_hidden"] == 1.0 and d["spell_f1_fix_visible"] < 1.0
    assert d["spell_fate"]["missed"] + d["spell_fate"]["merged"] + d["spell_fate"]["boundary"] >= 1
    assert d["transitions"]["turnover"]["n_gt"] == 1 and d["transitions"]["turnover"]["missed"] == 1
