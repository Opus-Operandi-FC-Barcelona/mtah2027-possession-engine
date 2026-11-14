"""Issue #48: hidden-ball <-> dead-ball affinity of the degrader and its label-free estimator (synthetic)."""
# ruff: noqa: E501
from __future__ import annotations

import dataclasses

import numpy as np

from mtah2027.models.external_degrader import DegraderParams, degrade_match
from mtah2027.validation import gap_affinity as ga

FPS = 25.0


def _match(seed=0, n=60000):
    """Alternating alive / dead stretches of 20-60 s; players run when alive and stand still when dead."""
    rng = np.random.default_rng(seed)
    gt = np.zeros(n, dtype=np.int8)
    t, state = 0, 0
    while t < n:
        dur = int(rng.uniform(20, 60) * FPS)
        gt[t:t + dur] = 2 if state else int(rng.integers(0, 2))
        t, state = t + dur, 1 - state
    walk = np.cumsum(rng.normal(0, 0.15, (n, 22, 2)), axis=0) * (gt != 2)[:, None, None]
    base = rng.uniform([5, 5], [100, 63], (22, 2))
    players = (base[None] + np.where((gt != 2)[:, None, None], walk, 0)).astype(np.float32)
    ball = players[:, 0] + 1.0
    return players, np.repeat([0, 1], 11).astype(np.int8), ball.astype(np.float32), gt


def _params(**kw):
    return dataclasses.replace(DegraderParams(), gap_median_s=10.0, guard_s=4.0, **kw)


def test_default_affinity_is_blind_and_unchanged():
    pl, tm, ball, gt = _match()
    a = degrade_match(pl, tm, ball, gt, seed=3, params=_params(), artefacts=False)
    b = degrade_match(pl, tm, ball, gt, seed=3, params=_params(dead_affinity=1.0), artefacts=False)
    assert np.array_equal(a.hidden, b.hidden)
    assert 0.5 < a.report.realised_odds_ratio < 2.0


def test_affinity_couples_hidden_ball_with_dead_ball_and_keeps_missing_share():
    pl, tm, ball, gt = _match()
    blind = degrade_match(pl, tm, ball, gt, seed=3, params=_params(), artefacts=False).report
    coupled = degrade_match(pl, tm, ball, gt, seed=3, params=_params(dead_affinity=30.0), artefacts=False).report
    assert coupled.realised_odds_ratio > 4 * blind.realised_odds_ratio
    assert coupled.hidden_dead_share > coupled.visible_dead_share + 0.2
    assert abs(coupled.hidden_graded_share - blind.hidden_graded_share) < 0.04


def test_affinity_is_deterministic_and_validated():
    pl, tm, ball, gt = _match(n=20000)
    a = degrade_match(pl, tm, ball, gt, seed=1, params=_params(dead_affinity=8.0)).hidden
    b = degrade_match(pl, tm, ball, gt, seed=1, params=_params(dead_affinity=8.0)).hidden
    assert np.array_equal(a, b)
    try:
        DegraderParams(dead_affinity=0.0)
    except ValueError:
        return
    raise AssertionError("dead_affinity <= 0 must be rejected")


def test_em_prior_recovers_a_shifted_prior():
    rng = np.random.default_rng(0)
    train_prior, target_prior = 0.4, 0.8
    y = rng.random(40000) < target_prior
    x = np.where(y, rng.normal(1.0, 1.0, y.size), rng.normal(-1.0, 1.0, y.size))
    like1, like0 = np.exp(-(x - 1) ** 2 / 2), np.exp(-(x + 1) ** 2 / 2)
    post = train_prior * like1 / (train_prior * like1 + (1 - train_prior) * like0)
    assert abs(ga.em_prior(post, train_prior) - target_prior) < 0.03
    assert np.isnan(ga.em_prior(np.zeros(0), 0.4))


def test_estimator_tracks_the_planted_affinity_on_a_synthetic_match():
    pl, tm, ball, gt = _match(seed=1)
    pl2, _, ball2, gt2 = _match(seed=2)
    g, g2 = gt != 3, gt2 != 3
    clf = ga.fit_dead_classifier(ga.player_features(pl, FPS)[g], (gt == 2)[g])
    est = {}
    for aff in (1.0, 40.0):
        dm = degrade_match(pl2, tm, ball2, gt2, seed=5, params=_params(dead_affinity=aff), artefacts=False)
        inter, vis = ga.gap_masks(dm.ball_xy, g2, FPS)
        e = ga.estimate(clf, ga.player_features(dm.players_xy, FPS), inter, vis)
        t = ga.true_shares(gt2, inter, vis)
        assert abs(e["d_share_hidden"] - t["d_share_hidden"]) < 0.12
        est[aff] = e["odds_ratio"]
    assert est[40.0] > 3 * est[1.0]
    assert set(e["strata"]) == {"hidden", "visible"}


def test_gap_masks_and_boundary_features():
    n = 600
    ball = np.tile(np.array([[50.0, 0.5]]), (n, 1))
    ball[100:200] = np.nan
    ball[300:310] = np.nan                       # shorter than 1 s: ignored
    graded = np.ones(n, bool)
    inter, vis = ga.gap_masks(ball, graded, FPS)
    assert inter[100:200].all() and not inter[300:310].any() and vis.sum() == n - 110
    pl = np.full((n, 3, 2), np.nan, np.float32)
    pl[:, 0] = [52.0, 5.0]
    b = ga.boundary_features(ball, pl, graded, FPS)
    assert b["last"]["n"] == 1 and b["first"]["n"] == 1
    assert b["last"]["near_line_1p5m"] == 1.0 and b["last"]["slow_ball"] == 1.0


def test_hidden_row_weights_can_differ_from_visible_row_weights():
    from mtah2027.models.specialist_data import MatchRows, TrainingSet

    def part(src, n):
        x = np.zeros((n, 2), np.float32)
        y = np.zeros(n, np.int64)
        return MatchRows(match=f"{src}_1", seed=0, swap=False, names=["a", "b"], x_vis=x, y_vis=y, x_hid=x,
                         y_hid=y, source=src)

    ts = TrainingSet(names=["a", "b"], parts=[part("idsse", 3), part("pff", 2)], weights={"pff": 0.5},
                     weights_b={"pff": 0.15})
    _, _, _, w_vis = ts.stack_weighted("vis")
    _, _, _, w_hid = ts.stack_weighted("hid")
    assert list(w_vis) == [1, 1, 1, 0.5, 0.5] and list(w_hid) == [1, 1, 1, 0.15, 0.15]
    same = ts.select(["idsse", "pff"], {"pff": 0.5})
    assert list(same.stack_weighted("hid")[3]) == [1, 1, 1, 0.5, 0.5]
    assert same.weights_b == {}


def test_adopted_training_configuration_and_model_variants():
    """The adopted default trains on the calibrated degrader with the estimated affinity (Issue #48)."""
    import json

    from mtah2027.config import REPO_ROOT, load_section

    cfg = load_section("specialist_data")
    assert cfg["degrader_section"] == "external_degrader" and float(cfg["dead_affinity"]) == 33.0
    assert float(load_section("external_degrader")["dead_affinity"]) == 1.0      # evaluation world stays blind
    source = (REPO_ROOT / "scripts" / "generate_submission.py").read_text(encoding="utf-8")
    assert '"gap"' in source
    default = REPO_ROOT / "models" / "specialists.json"
    gap = REPO_ROOT / "models" / "specialists_gap.json"
    if default.is_file() and gap.is_file():
        assert default.read_bytes() == gap.read_bytes()
        assert json.loads(default.read_text(encoding="utf-8"))["training"]["source_weights"]["pff"] == 0.5
