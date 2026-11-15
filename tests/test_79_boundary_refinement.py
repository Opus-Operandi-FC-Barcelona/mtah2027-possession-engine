"""Issue #79: boundary refinement with the turnover head (scoring itself needs external data)."""
# ruff: noqa: E501
from __future__ import annotations

import sys

import numpy as np
import yaml

from mtah2027.config import REPO_ROOT, route_config
from mtah2027.postprocessing.boundary_refine import BoundaryRefineParams, refine_boundaries

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import boundary_refine_eval as be  # noqa: E402

H, A, D = 0, 1, 2
P = BoundaryRefineParams(enabled=True, peak_threshold=0.7, window_s=1.0, margin=1.0, min_piece_s=0.5)


def _lp(labels, strength=3.0):
    """Emission log-probs favouring ``labels`` by ``strength``."""
    lp = np.full((len(labels), 3), -strength)
    lp[np.arange(len(labels)), labels] = 0.0
    return lp


def _head(n, peaks, kind=1):
    tp = np.zeros((n, 3))
    tp[:, 0] = 1.0
    for t in peaks:
        tp[t, 0], tp[t, kind] = 0.0, 0.95
    return tp


def test_grid_is_the_preregistered_12_cells():
    assert len(be.CELLS) == 12 and len(set(be.CELLS)) == 12
    assert {g["boundary_refine.peak_threshold"] for g in be.GRID.values()} == {0.5, 0.7, 0.9}
    assert {g["boundary_refine.window_s"] for g in be.GRID.values()} == {1.0, 2.0}
    assert {g["boundary_refine.margin"] for g in be.GRID.values()} == {0.5, 1.0}


def test_off_by_default_and_read_by_the_learned_route():
    raw = yaml.safe_load((REPO_ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    assert not raw["boundary_refine"]["enabled"] and "boundary_refine" not in raw["route_overrides"]["specialists"]
    assert not BoundaryRefineParams.from_config().enabled
    for name, g in be.GRID.items():
        cfg = route_config("specialists", be.variant_config(raw, name))
        p = BoundaryRefineParams.from_config(cfg)
        assert p.enabled and p.peak_threshold == g["boundary_refine.peak_threshold"], name
        assert p.window_s == g["boundary_refine.window_s"] and p.margin == g["boundary_refine.margin"], name


def test_split_at_a_peak_with_evidence_for_the_other_team():
    n = 200
    codes = np.full(n, H)
    truth = np.where(np.arange(n) < 100, H, A)
    out, st = refine_boundaries(codes, play_blocks=[(0, n - 1)], turnover_prob=_head(n, [100]), log_probs=_lp(truth),
                                params=P)
    assert (out[:100] == H).all() and (out[100:] == A).all()
    assert st["splits"] == 1 and st["frames_relabelled"] == 100
    assert (codes == H).all()                      # the input is not modified


def test_no_split_without_evidence_or_below_threshold():
    n = 200
    codes = np.full(n, H)
    out, st = refine_boundaries(codes, play_blocks=[(0, n - 1)], turnover_prob=_head(n, [100]),
                                log_probs=_lp(np.full(n, H)), params=P)
    assert (out == codes).all() and st["candidates"] == 1 and st["splits"] == 0
    truth = np.where(np.arange(n) < 100, H, A)
    out, _ = refine_boundaries(codes, play_blocks=[(0, n - 1)], turnover_prob=_head(n, [100]) * 0.5,
                               log_probs=_lp(truth), params=P)
    assert (out == codes).all()


def test_evidence_flip_back_ends_the_relabelled_piece():
    n = 300
    codes = np.full(n, A)
    truth = np.where((np.arange(n) >= 100) & (np.arange(n) < 180), H, A)
    out, st = refine_boundaries(codes, play_blocks=[(0, n - 1)], turnover_prob=_head(n, [100], kind=2),
                                log_probs=_lp(truth), params=P)
    assert (out[100:165] == H).all() and (out[:100] == A).all() and (out[185:] == A).all()   # the look-ahead window flips back a little early
    assert st["flip_back"] == 1


def test_never_touches_dead_other_possessions_or_tiny_pieces():
    n = 200
    codes = np.concatenate([np.full(90, H), np.full(10, D), np.full(100, A)])
    truth = np.where(np.arange(n) < 100, H, A)
    out, _ = refine_boundaries(codes, play_blocks=[(0, n - 1)], turnover_prob=_head(n, [95, 150], kind=1),
                               log_probs=_lp(truth), params=P)
    assert (out == codes).all()                    # peak in D, and an H->A head inside an A possession (wrong direction)
    codes = np.full(n, H)
    out, _ = refine_boundaries(codes, play_blocks=[(0, n - 1)], turnover_prob=_head(n, [3]), log_probs=_lp(np.full(n, A)),
                               params=P)
    assert (out == codes).all()                    # left piece shorter than min_piece_s


def test_missing_head_or_disabled_is_identity_and_selection_adoption():
    codes = np.full(50, H)
    assert (refine_boundaries(codes, play_blocks=[(0, 49)], turnover_prob=None, log_probs=_lp(codes), params=P)[0] == codes).all()
    assert (refine_boundaries(codes, play_blocks=[(0, 49)], turnover_prob=_head(50, [25]), log_probs=_lp(codes))[0] == codes).all()
    table = {("b33", 0): {"m": {n: {"spell_f1": 0.7} for n in be.CELLS}}}
    table[("b33", 0)]["m"]["T0.9_W2_M1.0"]["spell_f1"] = 0.8
    assert be.select_cell(table)[0] == "T0.9_W2_M1.0"
    assert be.adoption({"1": 0.012, "2": 0.011}, {"1": 0.0, "2": 0.0}, 0.0)["adopted"]
    assert not be.adoption({"1": 0.012, "2": 0.011}, {"1": -0.001, "2": 0.0}, 0.0)["adopted"]
    assert not be.adoption({"1": 0.012, "2": 0.011}, {"1": 0.0, "2": 0.0}, -0.0001)["adopted"]
