# ruff: noqa: E501
"""Issue #49: restart-team rules, wider restart-team window, t* offset prior and Long Ball flags."""
from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np

from mtah2027.config import ConfigError, params_from_mapping
from mtah2027.decoding.transition_matrix import TransitionParams, build_log_transitions
from mtah2027.features import restart_rules as rr
from mtah2027.features import setpiece_anchor as sa
from mtah2027.io import smart_tagging_loader as stl
from mtah2027.io.smart_tagging_sim import NOISE_LEVELS, simulate_smart_tagging
from mtah2027.models.heuristic_emissions import EmissionParams, compute_emissions
from mtah2027.models.specialist_fusion import apply_priors

FPS = 25.0
TEAM = np.array([0] * 11 + [1] * 11)
H, A, D = 0, 1, 2


def _anchor(typ, team, t_star, tag_start=None, **kw):
    a = tag_start if tag_start is not None else t_star - 200
    return sa.SetPieceAnchor(type=typ, team=team, t_star=t_star, confidence=1.0, method="ball", tag_start=a,
                             tag_end=a + 875, window=(a, a + 875), dead_onset=-1, onset_method="none", **kw)


def _result(anchors, n=20000):
    return sa.AnchorResult(anchors=anchors, restart_mask=np.zeros(n, bool), forced_dead_intervals=[],
                           forced_dead_mask=np.zeros(n, bool))


def _tracking(n, controllers):
    """Ball at (50, 34); a controller of the given team 1 m away from ``t`` on; the others far away."""
    players = np.zeros((n, 22, 2), dtype=np.float32)
    players[:, :11, 0] = 10.0
    players[:, 11:, 0] = 95.0
    players[:, :, 1] = np.linspace(5, 60, 22)[None, :]
    ball = np.tile(np.array([50.0, 34.0], dtype=np.float32), (n, 1))
    for t, team in controllers:
        slot = 0 if team == H else 11
        players[t:t + 40, slot] = (49.0, 34.0)
    return players, ball


# ---------------------------------------------------------------------------------------- config
def test_yaml_sections_default_off_and_reject_typos():
    p = rr.RestartRuleParams.from_config()
    assert p.enabled is False and p.calibrate_weight is False and p.auto_long_ball_weight is False
    assert sa.AnchorParams.from_config().offset_prior is False
    tp = TransitionParams.from_config()
    assert tp.restart_team_before_s == 0.0 and tp.restart_team_after_s == 0.0
    assert EmissionParams.from_config().w_long_ball == 0.0
    try:
        params_from_mapping(rr.RestartRuleParams, {"enable": True})
    except ConfigError:
        pass
    else:
        raise AssertionError("typo accepted")


# ---------------------------------------------------------------------------------------- rules
def test_tracking_team_first_clear_controller():
    players, ball = _tracking(2000, [(1000, A)])
    p = rr.RestartRuleParams()
    assert rr.tracking_team(ball, players, TEAM, 1000, fps=FPS, params=p) == A
    assert rr.tracking_team(ball, players, TEAM, 500, fps=FPS, params=p) == -1      # nobody near the ball
    players[1000:1040, 1] = (49.5, 34.0)                                             # a contested ball
    assert rr.tracking_team(ball, players, TEAM, 1000, fps=FPS, params=p) == -1


def test_kickoff_after_goal_is_the_team_that_conceded():
    players, ball = _tracking(8000, [])
    anchors = _result([_anchor("kick off (after goal)", -1, 3000, tag_start=2800)])
    goals = [stl.Goal(team=0, start_frame=2000, end_frame=2100, half=1)]       # H scored -> A concedes and kicks off
    out, rep = rr.apply_team_rules(anchors, goals, ball, players, TEAM, fps=FPS, params=rr.RestartRuleParams())
    assert out.anchors[0].team == A and rep["rule_goal"] == 1 and rep["filled"] == 1
    far = [stl.Goal(team=0, start_frame=100, end_frame=150, half=1)]           # beyond goal_link_s
    late = _result([_anchor("kick off (after goal)", -1, 6200, tag_start=6000)])
    out, rep = rr.apply_team_rules(late, far, ball, players, TEAM, fps=FPS, params=rr.RestartRuleParams())
    assert out.anchors[0].team == -1 and rep["rule_goal"] == 0


def test_second_half_kickoff_opposite_of_first_and_dissent_lowers_confidence():
    players, ball = _tracking(40000, [(1000, H), (20000, A)])

    def ko(team, t):
        return _anchor("kick off (start)", team, t, tag_start=t - 90)

    p = rr.RestartRuleParams()
    out, _ = rr.apply_team_rules(_result([ko(H, 1000), ko(A, 20000)], 40000), [], ball, players, TEAM, fps=FPS, params=p)
    assert [a.team for a in out.anchors] == [H, A] and all(a.team_conf == 1.0 for a in out.anchors)
    # the half-2 tag repeats the half-1 team: rule + tracking out-vote it
    out, rep = rr.apply_team_rules(_result([ko(H, 1000), ko(H, 20000)], 40000), [], ball, players, TEAM, fps=FPS, params=p)
    assert [a.team for a in out.anchors] == [H, A] and out.anchors[1].team_source == "vote" and rep["changed"] == 1
    assert out.anchors[1].team_conf == p.unreliable_conf
    # a half-2 tag of N/A is filled from the rule
    out, rep = rr.apply_team_rules(_result([ko(H, 1000), ko(-1, 20000)], 40000), [], ball, players, TEAM, fps=FPS, params=p)
    assert out.anchors[1].team == A and rep["filled"] == 1
    off = dataclasses.replace(p, second_half_opposite=False, tracking_vote=False)
    out, _ = rr.apply_team_rules(_result([ko(H, 1000), ko(H, 20000)], 40000), [], ball, players, TEAM, fps=FPS, params=off)
    assert [a.team for a in out.anchors] == [H, H]


def test_tracking_disagreement_keeps_the_tag_with_lower_confidence():
    players, ball = _tracking(5000, [(1000, A)])
    out, rep = rr.apply_team_rules(_result([_anchor("corner kick", H, 1000)]), [], ball, players, TEAM,
                                   fps=FPS, params=rr.RestartRuleParams())
    a = out.anchors[0]
    assert a.team == H and a.team_conf == 0.5 and rep["unreliable"] == 1 and rep["tracking_agree"] == 0


def test_calibrated_weight_from_confident_states_only():
    anchors = _result([_anchor("corner kick", H, 100 * k) for k in range(2, 10)], 3000)
    lab = np.full(3000, -1, dtype=np.int8)
    for k in range(2, 10):
        lab[100 * k + 10:100 * k + 40] = H if k < 8 else A      # 6 of 8 agree
    w, n = rr.calibrate_team_weight(anchors, lab, fps=FPS, params=rr.RestartRuleParams())
    assert n == 8 and abs(w - (2 * (7 / 10) - 1)) < 1e-9
    assert rr.calibrate_team_weight(anchors, np.full(3000, -1, np.int8), fps=FPS)[0] == 1.0


# ---------------------------------------------------------------------------------------- transitions
def test_wide_window_penalises_other_team_and_weights_scale_it():
    n = 3000
    kw = dict(restart_frames=np.array([1000]), restart_teams=np.array([H]), fps=FPS)
    tp = TransitionParams(restart_team_before_s=4.0, restart_team_after_s=2.0)
    base = build_log_transitions(n, params=TransitionParams(), **kw)
    wide = build_log_transitions(n, params=tp, **kw)
    f = 1000 - 60
    assert wide[f, D, A] < base[f, D, A] and wide[f, D, H] == base[f, D, H]
    assert np.array_equal(wide[1000], base[1000])                          # core window untouched
    assert np.array_equal(wide[:800], base[:800]) and np.array_equal(wide[1100:], base[1100:])
    half = build_log_transitions(n, params=tp, restart_weights=np.array([0.5]), **kw)
    zero = build_log_transitions(n, params=tp, restart_weights=np.array([0.0]), **kw)
    assert wide[f, D, A] < half[f, D, A] < zero[f, D, A] == base[f, D, A]


def test_default_params_reproduce_the_previous_matrix_with_unit_weights():
    n = 2000
    kw = dict(restart_frames=np.array([500, 1200]), restart_teams=np.array([H, A]), fps=FPS)
    a = build_log_transitions(n, params=TransitionParams(), **kw)
    b = build_log_transitions(n, params=TransitionParams(), restart_weights=np.ones(2), **kw)
    assert np.array_equal(a, b)
    assert a[500, D, H] == TransitionParams().restart_open and a[500, D, A] < a[500, D, H]


# ---------------------------------------------------------------------------------------- offset prior
def test_offset_bands_learned_from_ball_anchors():
    sps = [stl.SetPiece(type="corner kick", team=0, start_frame=a, end_frame=a + 875, half=1) for a in (0, 1500, 3000)]
    found = [(300, 1.0, 1.0), (1800, 1.0, 1.0), (3340, 1.0, 1.0)]
    p = sa.AnchorParams(offset_prior=True, offset_prior_min_n=3, offset_prior_margin_s=1.0)
    bands = sa.learn_offset_bands(sps, found, p, FPS)
    lo, hi = bands["corner kick"]
    assert lo <= 300 - 25 and hi >= 330 + 25 and bands["*"] == bands["corner kick"]
    assert sa.learn_offset_bands(sps, found[:2], p, FPS) == {}                # too few anchors -> no band


def _collective_scene(n=9000):
    """No ball at all; the players start to move at frame 2500 (a jump of the collective speed)."""
    xy = np.zeros((n, 22, 2), dtype=np.float32)
    xy[:, :11, 0], xy[:, 11:, 0] = 30.0, 70.0
    speed = np.zeros(n)
    speed[2500:] = 3.0
    xy[:, :, 1] = np.linspace(2, 40, 22)[None, :] + (np.cumsum(speed) / FPS)[:, None]
    return xy, np.full((n, 2), np.nan, dtype=np.float32)


def test_offset_prior_restricts_the_collective_search_to_the_learned_band():
    n = 9000
    xy, ball = _collective_scene(n)
    # three ball-resolved set pieces teach the band (rest -> launch at offset ~300 of the window)
    for a in (4000, 5200, 6400):
        ball[a + 280:a + 300, 0] = 20.0
        ball[a + 280:a + 300, 1] = 30.0
        ball[a + 300:a + 330, 0] = 20.0 + np.arange(30) * 0.5
        ball[a + 300:a + 330, 1] = 30.0
    sps = [stl.SetPiece(type="throw-in", team=-1, start_frame=a, end_frame=a + 875, half=1) for a in (4000, 5200, 6400)]
    sps.append(stl.SetPiece(type="throw-in", team=-1, start_frame=1800, end_frame=2675, half=1))
    ds = np.ones(n)
    p = sa.AnchorParams(collective_dead_lag_s=0.0)
    off = sa.anchor_set_pieces(ball, xy, TEAM, sps, [], params=p, dead_score=ds)
    on = sa.anchor_set_pieces(ball, xy, TEAM, sps, [], params=dataclasses.replace(
        p, offset_prior=True, offset_prior_min_n=3, offset_prior_margin_s=0.4), dead_score=ds)
    assert [a.method for a in on.anchors[:3]] == [a.method for a in off.anchors[:3]]    # ball anchors untouched
    if off.anchors[3].resolved:     # the collective t* at ~2500 lies at offset 700, outside a band around ~300
        assert off.anchors[3].method == "collective"
        assert not on.anchors[3].resolved                # nothing is invented when the band excludes it


def test_offset_prior_off_is_identical_without_ball_anchors():
    xy, ball = _collective_scene()
    sps = [stl.SetPiece(type="throw-in", team=-1, start_frame=1800, end_frame=2675, half=1)]
    p = sa.AnchorParams(collective_dead_lag_s=0.0)
    off = sa.anchor_set_pieces(ball, xy, TEAM, sps, [], params=p, dead_score=np.ones(9000))
    on = sa.anchor_set_pieces(ball, xy, TEAM, sps, [], params=dataclasses.replace(p, offset_prior=True),
                              dead_score=np.ones(9000))
    assert off.anchors[0].t_star == on.anchors[0].t_star       # no ball anchors -> no band -> unchanged


# ---------------------------------------------------------------------------------------- Long Ball
def test_long_ball_mask_and_weight_only_on_hidden_frames():
    ev = [stl.TagEvent("long ball", 100, 300, 4.0, 12.0, 0, 1, "", {}),
          stl.TagEvent("progression", 400, 500, 16.0, 20.0, 0, 1, "", {}),
          stl.TagEvent("ball in the box", 450, 600, 18.0, 24.0, 1, 1, "", {})]
    m = rr.long_ball_mask(SimpleNamespace(events=ev), 1000)
    assert m[100] and m[299] and not m[300] and not m[400] and m[500] and m[599] and not m[600]
    n = 40
    ds, e0, v0 = np.full(n, 0.6), np.zeros(n), np.zeros(n, bool)
    vis = np.arange(n) < 20
    lb = np.ones(n, bool)
    plain = compute_emissions(ds, e0, v0, vis, params=EmissionParams())
    off = compute_emissions(ds, e0, v0, vis, long_ball=lb, params=EmissionParams())
    assert np.array_equal(plain.log_probs, off.log_probs)
    on = compute_emissions(ds, e0, v0, vis, long_ball=lb, params=EmissionParams(w_long_ball=1.5))
    assert np.array_equal(on.p_dead[:20], plain.p_dead[:20]) and (on.p_dead[20:] < plain.p_dead[20:] - 0.05).all()
    prob = np.tile([0.2, 0.2, 0.6], (n, 1))
    out, _ = apply_priors(prob, phase=None, owner=None, status_dead=None, ep=EmissionParams(), long_ball=lb,
                          visible=vis, w_long_ball=1.0)
    assert np.array_equal(out[:20], prob[:20]) and (out[20:, 2] < 0.6).all()


def test_auto_long_ball_weight_from_confident_visible_frames():
    n = 4000
    mask = np.zeros(n, bool)
    mask[:2000] = True
    lab = np.full(n, -1, dtype=np.int8)
    rng = np.random.default_rng(0)
    lab[:2000] = np.where(rng.random(2000) < 0.8, H, D)           # alive 80 % inside
    lab[2000:] = np.where(rng.random(2000) < 0.6, A, D)           # alive 60 % outside
    vis = np.ones(n, bool)
    p = rr.RestartRuleParams()
    w, info = rr.auto_long_ball_weight(mask, lab, vis, params=p)
    assert 0.8 < w < 1.5 and info["alive_inside"] > info["alive_outside"]
    assert rr.auto_long_ball_weight(mask, np.where(np.arange(n) < 100, lab, -1), vis, params=p)[0] == 0.0
    inv = np.where(lab == D, H, np.where(lab == H, D, lab))        # less alive inside -> never negative
    assert rr.auto_long_ball_weight(mask, inv, vis, params=p)[0] == 0.0


# ---------------------------------------------------------------------------------------- simulator
def _sim_labels():
    codes = np.full(30000, 3, dtype=np.int8)
    for k, (a, b) in enumerate([(100, 3000), (3000, 3300), (3300, 6000), (6000, 6400), (6400, 9000), (9000, 9300),
                                (9300, 12000)]):
        codes[a:b] = [H, D, A, D, H, D, A][k]
    codes[16000:20000] = A
    return codes, [(100, 11999), (16000, 19999)]


def test_simulator_restart_options_are_off_by_default_and_deterministic():
    codes, periods = _sim_labels()
    base = simulate_smart_tagging(codes, periods, FPS, noise="low", seed=1, n_frames=30000)
    again = simulate_smart_tagging(codes, periods, FPS, noise="low", seed=1, n_frames=30000)
    key = lambda s: [(x.type, x.team, x.start_frame) for x in s.set_pieces]          # noqa: E731
    assert key(base) == key(again)
    assert all(s.team == -1 for s in base.set_pieces if "kick off" in s.type)
    named = simulate_smart_tagging(codes, periods, FPS, seed=1, n_frames=30000,
                                   noise=dataclasses.replace(NOISE_LEVELS["low"], kickoff_teams=True))
    assert [(s.type, s.start_frame) for s in named.set_pieces] == [(s.type, s.start_frame) for s in base.set_pieces]
    teams = [s.team for s in named.set_pieces if "kick off" in s.type]
    assert teams and set(teams) <= {H, A}
    quiet = dataclasses.replace(NOISE_LEVELS["low"], fn_rate=0.0, fp_rate=0.0)
    w = simulate_smart_tagging(codes, periods, FPS, noise=dataclasses.replace(quiet, wrong_team_rate=1.0), seed=2, n_frames=30000)
    r = simulate_smart_tagging(codes, periods, FPS, noise=quiet, seed=2, n_frames=30000)
    assert len(w.set_pieces) == len(r.set_pieces) > 0
    for sw, sr in zip(w.set_pieces, r.set_pieces):
        assert (sw.team == -1 and sr.team == -1) or sw.team == 1 - sr.team
