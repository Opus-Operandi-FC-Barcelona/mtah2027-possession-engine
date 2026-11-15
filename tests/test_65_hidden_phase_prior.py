# ruff: noqa: E501
"""Issue #65: phase prior on the H-vs-A log-odds of hidden-ball frames only (``w_phase_hidden``)."""
from __future__ import annotations

import numpy as np

from mtah2027.models.heuristic_emissions import (
    EmissionParams,
    auto_phase_hidden_weight,
    hidden_run_mask,
)
from mtah2027.models.specialist_fusion import apply_priors


def _setup(n=60):
    rng = np.random.default_rng(0)
    prob = rng.dirichlet([2, 2, 1], size=n)
    phase = np.where(np.arange(n) % 2 == 0, 1.0, -0.5)
    vis = np.arange(n) < 20
    return prob, phase, vis


def test_off_is_identical():
    prob, phase, vis = _setup()
    ep = EmissionParams()
    assert ep.w_phase_hidden == 0.0 and ep.phase_hidden_min_s == 0.0 and not ep.auto_calibrate_phase_hidden
    a, za = apply_priors(prob, phase=phase, owner=None, status_dead=None, ep=ep, visible=vis)
    b, zb = apply_priors(prob, phase=phase, owner=None, status_dead=None, ep=ep, visible=None)
    assert np.array_equal(a, b) and np.array_equal(za, zb)


def test_on_keeps_p_dead_and_only_hidden_frames_move():
    prob, phase, vis = _setup()
    off, zoff = apply_priors(prob, phase=phase, owner=None, status_dead=None, ep=EmissionParams(), visible=vis)
    on, zon = apply_priors(prob, phase=phase, owner=None, status_dead=None,
                           ep=EmissionParams(w_phase_hidden=1.5), visible=vis)
    assert np.array_equal(on[:, 2], off[:, 2])                      # p_dead exactly unchanged
    assert np.array_equal(on[vis], off[vis]) and np.array_equal(zon[vis], zoff[vis])
    assert np.allclose(zon[~vis] - zoff[~vis], 1.5 * phase[~vis])
    assert np.allclose(on.sum(axis=1), 1.0)


def test_min_hidden_run_selects_long_runs():
    vis = np.ones(100, bool)
    vis[10:15] = False          # 5 frames
    vis[40:80] = False          # 40 frames
    m = hidden_run_mask(vis, 25)
    assert not m[10:15].any() and m[40:80].all() and m.sum() == 40
    assert np.array_equal(hidden_run_mask(vis, 0), ~vis)
    prob, phase, _ = _setup(100)
    off, _ = apply_priors(prob, phase=phase, owner=None, status_dead=None, ep=EmissionParams(), visible=vis)
    on, _ = apply_priors(prob, phase=phase, owner=None, status_dead=None,
                         ep=EmissionParams(w_phase_hidden=2.0, phase_hidden_min_s=1.0), visible=vis, fps=25.0)
    changed = ~np.isclose(on[:, 0], off[:, 0])
    assert changed[40:80].any() and not changed[10:15].any() and not changed[:10].any()


def test_calibration_deterministic_bounded_and_label_free_guard():
    n = 3000
    rng = np.random.default_rng(1)
    lab = rng.integers(0, 2, n)
    phase = np.where(lab == 0, 1.0, -1.0)
    flip = rng.random(n) < 0.1
    phase = np.where(flip, -phase, phase)           # 90 % agreement
    vis = np.ones(n, bool)
    ep = EmissionParams()
    w1, i1 = auto_phase_hidden_weight(phase, lab, vis, params=ep)
    w2, _ = auto_phase_hidden_weight(phase, lab, vis, params=ep)
    assert w1 == w2 and 0.0 < w1 <= ep.phase_hidden_cap
    assert abs(w1 - (np.log(0.9 / 0.1) - ep.w_phase)) < 0.2 and i1["n_usable"] == n
    w_rand, _ = auto_phase_hidden_weight(-phase, lab, vis, params=ep)    # anti-correlated -> 0
    assert w_rand == 0.0
    w_few, _ = auto_phase_hidden_weight(phase[:50], lab[:50], vis[:50], params=ep)
    assert w_few == 0.0
    w_perfect, _ = auto_phase_hidden_weight(np.where(lab == 0, 1.0, -1.0), lab, vis, params=ep)
    assert w_perfect == ep.phase_hidden_cap
