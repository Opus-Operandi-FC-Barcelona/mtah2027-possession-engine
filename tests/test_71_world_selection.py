"""Issue #71: world selection helpers on synthetic data."""
from __future__ import annotations

import numpy as np

from mtah2027.validation import world_selection as ws


def _synthetic_tracks(n=600, speed=3.0, seed=0, hidden=(100, 400)):
    rng = np.random.default_rng(seed)
    base = rng.uniform([5, 5], [100, 63], size=(22, 2))
    steps = rng.normal(0, speed / 25.0, size=(n, 22, 2)).cumsum(axis=0)
    players = (base[None] + steps).astype(np.float32)
    team = np.array([0] * 11 + [1] * 11, np.int8)
    ball = np.tile(np.array([50.0, 34.0], np.float32), (n, 1))
    ball[hidden[0]:hidden[1]] = np.nan
    return players, team, ball


def test_gap_structure_basic():
    vis = np.ones(10, bool)
    vis[3:7] = False
    glen, since, until = ws.gap_structure(vis, 1.0)
    assert glen.tolist() == [0, 0, 0, 4, 4, 4, 4, 0, 0, 0]
    assert since[3:7].tolist() == [1, 2, 3, 4] and until[3:7].tolist() == [4, 3, 2, 1]


def test_features_shape_ball_free_and_deterministic():
    players, team, ball = _synthetic_tracks()
    x, idx = ws.hidden_ball_features(players, team, ball, 25.0, graded=np.ones(len(ball), bool), stride=5)
    assert x.shape == (len(idx), len(ws.FEATURE_NAMES)) and len(idx) == 60
    assert idx.min() >= 100 and idx.max() < 400
    assert np.isfinite(x[:, ws.FEATURE_NAMES.index("n_tracked")]).all()
    assert (x[:, ws.FEATURE_NAMES.index("n_tracked_h")] == 11).all()
    ball2 = ball.copy()
    ball2[:100] += 7.0               # only the visibility of the ball matters, never its coordinates
    x2, _ = ws.hidden_ball_features(players, team, ball2, 25.0, graded=np.ones(len(ball), bool), stride=5)
    assert np.array_equal(np.nan_to_num(x), np.nan_to_num(x2))


def test_features_separate_slow_from_fast_worlds():
    slow = ws.hidden_ball_features(*_synthetic_tracks(speed=0.2), 25.0, stride=5)[0]
    fast = ws.hidden_ball_features(*_synthetic_tracks(speed=4.0), 25.0, stride=5)[0]
    j = ws.FEATURE_NAMES.index("speed_med_1s")
    assert np.nanmedian(slow[:, j]) < np.nanmedian(fast[:, j])


def test_auc_known_values():
    y = np.array([0, 0, 1, 1])
    assert ws.auc_score(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert ws.auc_score(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert ws.auc_score(y, np.array([0.5, 0.5, 0.5, 0.5])) == 0.5


def test_cross_fit_detects_shift_and_chance():
    rng = np.random.default_rng(1)
    n = 1500
    groups = np.repeat(np.arange(30), n // 15)[: 2 * n // 2 * 1][: n]
    groups = np.concatenate([groups, groups + 100])
    a = rng.normal(0, 1, (n, 3))
    b_same = rng.normal(0, 1, (n, 3))
    b_shift = rng.normal(0, 1, (n, 3)) + np.array([2.0, 0, 0])
    y = np.concatenate([np.ones(n), np.zeros(n)])
    p_same, _ = ws.cross_fit_probabilities(np.concatenate([a, b_same]), y, groups, n_folds=3, max_iter=40)
    p_shift, imp = ws.cross_fit_probabilities(np.concatenate([a, b_shift]), y, groups, n_folds=3, max_iter=40)
    assert abs(ws.auc_score(y, p_same) - 0.5) < 0.06
    assert ws.auc_score(y, p_shift) > 0.85
    assert int(np.argmax(imp)) == 0


def test_density_ratio_and_mixture_recovers_weights():
    rng = np.random.default_rng(2)
    # world densities N(0,1) and N(4,1); real = 0.7 N(0,1) + 0.3 N(4,1)
    z = np.where(rng.random(20000) < 0.7, rng.normal(0, 1, 20000), rng.normal(4, 1, 20000))
    pdf = lambda x, m: np.exp(-0.5 * (x - m) ** 2) / np.sqrt(2 * np.pi)          # noqa: E731
    p_real = 0.7 * pdf(z, 0) + 0.3 * pdf(z, 4)
    ratios = np.stack([p_real / pdf(z, 0), p_real / pdf(z, 4)], axis=1)
    pi = ws.fit_mixture(ratios)
    assert abs(pi[0] - 0.7) < 0.03 and abs(pi.sum() - 1.0) < 1e-9
    r = ws.density_ratio(np.array([0.5, 0.9, 0.0]))
    assert np.isclose(r[0], 1.0) and r[1] > r[0] and np.isfinite(r[2])


def test_weights_ess_and_weighted_metric():
    assert np.isclose(ws.effective_sample_size(np.ones(10)), 10.0)
    assert ws.effective_sample_size(np.array([1.0, 0, 0, 0])) == 1.0
    mw = ws.match_weights(np.array([1.0, 1.0, 3.0, 3.0]), np.array(["a", "a", "b", "b"]))
    assert np.isclose(mw["a"], 0.25) and np.isclose(mw["b"], 0.75)
    scores = {"w1": {"a": 0.5, "b": 0.7}, "w2": {"c": 0.4}}
    m = ws.weighted_metric(scores, {"w1": 0.5, "w2": 0.5}, {"w1": {"a": 0.25, "b": 0.75}, "w2": {"c": 1.0}})
    assert np.isclose(m, 0.5 * (0.25 * 0.5 + 0.75 * 0.7) + 0.5 * 0.4)
