"""Issue #18: velocity phase coherence, line advance and ball memory."""
from __future__ import annotations

import time

import numpy as np
import pytest

from mtah2027.config import ConfigError, params_from_mapping
from mtah2027.features import ball_memory as bm

FPS = 25.0
PARAMS = bm.BallMemoryParams()
TEAM = np.array([0, 0, 0, 0, 1, 1, 1, 1])


def _vel(n, home, away):
    """Constant velocities: ``home`` / ``away`` are [4, 2] arrays."""
    v = np.concatenate([np.asarray(home, float), np.asarray(away, float)])
    return np.broadcast_to(v, (n, 8, 2)).copy()


def _static(n, home_xy, away_xy):
    xy = np.concatenate([np.asarray(home_xy, float), np.asarray(away_xy, float)])
    return np.broadcast_to(xy, (n, 8, 2)).astype(np.float32).copy()


# ---------------------------------------------------------------------------------- config
def test_yaml_section_matches_defaults_and_rejects_typos():
    assert bm.BallMemoryParams.from_config() == PARAMS
    with pytest.raises(ConfigError):
        params_from_mapping(bm.BallMemoryParams, {"cap": 3}, "ball_memory")


# ---------------------------------------------------------------------------------- coherence
def test_phase_coherence_known_cases():
    aligned = [[5, 0]] * 4
    opposite = [[5, 0], [-5, 0], [0, 5], [0, -5]]
    n = 3
    vel = _vel(n, aligned, opposite)
    r_home = bm.phase_coherence(vel, TEAM, 0)
    r_away = bm.phase_coherence(vel, TEAM, 1)
    np.testing.assert_allclose(r_home, 5.0 / 5.001, atol=1e-6)
    np.testing.assert_allclose(r_away, 0.0, atol=1e-6)
    # still players contribute almost nothing: 2 aligned + 2 still -> R = 0.5 * 5/5.001
    mixed = _vel(n, [[5, 0], [5, 0], [0, 0], [0, 0]], opposite)
    np.testing.assert_allclose(bm.phase_coherence(mixed, TEAM, 0), 0.5 * 5.0 / 5.001, atol=1e-6)


def test_phase_coherence_is_immune_to_the_heading_discontinuity():
    eps = 0.01
    headings = [np.pi - eps, -np.pi + eps, np.pi - eps / 2, -np.pi + eps / 2]
    v = [[6 * np.cos(h), 6 * np.sin(h)] for h in headings]
    r = bm.phase_coherence(_vel(2, v, v), TEAM, 0)
    assert (r > 0.999).all()                                    # all point towards -x despite the +-pi wrap


def test_coherence_nan_with_too_few_players_and_line_advance_signs():
    n = 120
    vel = _vel(n, [[2, 1], [2, -1], [4, 0], [3, 0]], [[-1, 0], [-1, 1], [-3, 0], [-1, -1]])
    vel[:, 3] = np.nan                                           # one home player lost: 3 < min_players
    xy, ball = _static(n, [[10, 10]] * 4, [[60, 10]] * 4), np.full((n, 2), 30.0, np.float32)
    r = bm.compute_ball_memory(xy, TEAM, ball, players_vel=vel, params=PARAMS)
    assert np.isnan(r.phase_R_home).all() and np.isnan(r.line_vx_home).all()      # 3 valid < min_players
    assert np.isfinite(r.phase_R_away).all()
    assert r.line_vx_away[60] == pytest.approx(-1.5)                              # mean(-1,-1,-3,-1)
    p3 = bm.BallMemoryParams(min_players=3)
    r3 = bm.compute_ball_memory(xy, TEAM, ball, players_vel=vel, params=p3)
    assert r3.line_vx_home[60] == pytest.approx((2 + 2 + 4) / 3)
    assert r3.line_vx_diff[60] == pytest.approx((2 + 2 + 4) / 3 - 1.5)     # home gains on the away block


# ---------------------------------------------------------------------------------- visibility memory
def _gap_case(n=60, hidden=(10, 30)):
    ball = np.stack([np.linspace(10, 40, n), np.full(n, 30.0)], axis=1).astype(np.float32)
    ball[hidden[0]:hidden[1]] = np.nan
    players = _static(n, [[20, 20], [20, 40], [90, 5], [90, 60]], [[22, 30], [50, 50], [40, 30], [95, 34]])
    return players, ball


def test_since_until_gap_fraction_and_seen_positions():
    players, ball = _gap_case()
    r = bm.compute_ball_memory(players, TEAM, ball, params=PARAMS)
    assert r.since_seen_s[9] == 0 and r.until_seen_s[30] == 0 and r.ball_visible[9] == 1
    assert r.since_seen_s[10] == pytest.approx(1 / FPS) and r.since_seen_s[29] == pytest.approx(20 / FPS)
    assert r.until_seen_s[10] == pytest.approx(20 / FPS) and r.until_seen_s[29] == pytest.approx(1 / FPS)
    assert r.gap_frac[20] == pytest.approx(11 / 21) and np.isnan(r.gap_frac[5])
    np.testing.assert_allclose(r.last_seen_xy[20], ball[9])
    np.testing.assert_allclose(r.next_seen_xy[20], ball[30])
    assert r.gap_dist_m[20] == pytest.approx(np.hypot(*(ball[30] - ball[9])), rel=1e-5)
    vx = r.last_seen_vel[20, 0]
    assert vx == pytest.approx(30.0 / 59 * FPS, rel=1e-3)                  # speed of the linear ball path


def test_open_ended_gaps_and_cap():
    n = 400
    ball = np.full((n, 2), np.nan, np.float32)
    ball[100:120] = [50.0, 30.0]
    r = bm.compute_ball_memory(_static(n, [[1, 1]] * 4, [[2, 2]] * 4), TEAM, ball, params=PARAMS)
    assert np.isnan(r.since_seen_s[50]) and r.until_seen_s[50] == pytest.approx(50 / FPS)
    assert np.isnan(r.until_seen_s[300]) and np.isnan(r.gap_frac[300]) and np.isnan(r.last_seen_vel[50]).all()
    assert r.since_seen_s[399] == pytest.approx(280 / FPS)                  # 11.2 s, below the 120 s cap
    capped = bm.compute_ball_memory(_static(n, [[1, 1]] * 4, [[2, 2]] * 4), TEAM, ball,
                                    params=bm.BallMemoryParams(cap_s=5.0))
    assert capped.since_seen_s[399] == pytest.approx(5.0)


# ---------------------------------------------------------------------------------- touches / clear possessor
def test_last_next_touch_and_clear_possessor_teams():
    players, ball = _gap_case(n=80, hidden=(15, 60))
    n = 80
    touch = np.zeros(n, bool)
    touch[[5, 12, 65]] = True
    delta = np.zeros(n)
    delta[5], delta[12], delta[65] = 2.0, 0.2, -1.0     # home touch, ambiguous touch (ignored), away touch
    evidence = np.zeros(n, np.float32)
    evidence[8], evidence[70] = -0.8, 0.5               # clear away at 8, clear home at 70 (weak ignored)
    evidence[10] = 0.2
    r = bm.compute_ball_memory(players, TEAM, ball, touch_mask=touch, delta_d=delta,
                               possession_evidence=evidence, params=PARAMS)
    f = 30
    assert r.last_touch_team[f] == 0 and r.next_touch_team[f] == 1
    assert r.since_last_touch_s[f] == pytest.approx((f - 5) / FPS)
    assert r.until_next_touch_s[f] == pytest.approx((65 - f) / FPS)
    assert r.last_clear_team[f] == 1 and r.next_clear_team[f] == 0
    assert r.since_last_clear_s[f] == pytest.approx((f - 8) / FPS)
    assert r.until_next_clear_s[f] == pytest.approx((70 - f) / FPS)
    assert r.last_touch_team[2] == -1 and np.isnan(r.since_last_touch_s[2])        # nothing before the first
    assert r.next_touch_team[70] == -1 and np.isnan(r.until_next_touch_s[70])      # nothing after the last
    assert r.last_touch_team.dtype == np.int8
    # without the optional inputs the touch / clear features are empty
    plain = bm.compute_ball_memory(players, TEAM, ball, params=PARAMS)
    assert (plain.last_touch_team == -1).all() and np.isnan(plain.until_next_clear_s).all()


# ---------------------------------------------------------------------------------- pressure
def test_pressure_around_last_seen_ball_and_advanced_players():
    n = 40
    ball = np.full((n, 2), np.nan, np.float32)
    ball[:5] = [50.0, 30.0]
    home = [[5, 34], [30, 30], [60, 20], [52, 28]]      # slot 0 goalkeeper, slot 2 the most advanced
    away = [[100, 34], [70, 30], [40, 30], [49, 33]]    # slot 4 goalkeeper, slot 6 the most advanced
    players = _static(n, home, away)
    r = bm.compute_ball_memory(players, TEAM, ball, params=PARAMS)
    f = 20
    # around (50, 30) within 5 m: home (52, 28) -> 2.83 m; away (49, 33) -> 3.16 m, (40,30) is 10 m away
    assert r.press_last_home[f] == 1 and r.press_last_away[f] == 1
    assert r.dist_last_home[f] == pytest.approx(np.hypot(2, 2), rel=1e-5)
    assert r.dist_last_away[f] == pytest.approx(np.hypot(1, 3), rel=1e-5)
    assert np.isnan(r.press_next_home[f]) and np.isnan(r.dist_next_away[f])        # never seen again
    assert r.adv_x_home[f] == 60.0 and r.adv_x_away[f] == 40.0
    # opponents within 5 m of home's (60, 20): none; of away's (40, 30): none (52,28 is 12 m away)
    assert r.adv_press_home[f] == 0 and r.adv_press_away[f] == 0
    close = _static(n, [[5, 34], [30, 30], [60, 20], [52, 28]], [[100, 34], [62, 22], [40, 30], [49, 33]])
    r2 = bm.compute_ball_memory(close, TEAM, ball, params=PARAMS)
    assert r2.adv_press_home[f] == 1                              # away (62, 22) is 2.83 m from (60, 20)


def test_matrix_export_and_input_validation():
    players, ball = _gap_case()
    r = bm.compute_ball_memory(players, TEAM, ball, params=PARAMS)
    m, names = r.matrix()
    assert m.shape == (len(ball), len(names)) and m.dtype == np.float32
    assert "last_seen_xy_x" in names and "press_next_away" in names and "phase_R_home" in names
    with pytest.raises(ValueError):
        bm.compute_ball_memory(players, TEAM, ball[:-1], params=PARAMS)


# ---------------------------------------------------------------------------------- synthetic match
def test_synthetic_match_memory_agrees_with_the_true_team(synthetic_kin):
    from mtah2027.features.ball_kinematics import compute_ball_kinematics
    kin, td, spec = synthetic_kin["kin"], synthetic_kin["td"], synthetic_kin["spec"]
    team = td.team_of_slot
    bk = compute_ball_kinematics(kin.players_xy, team, kin.ball_xy, players_vel=kin.players_vel,
                                 ball_vel=kin.ball_vel, ball_acc=kin.ball_acc, fps=FPS)
    ball = kin.ball_xy.copy()
    a, b = spec.half1
    ball[a + 1000:a + 1500] = np.nan                                  # artificial 20 s occlusion
    r = bm.compute_ball_memory(kin.players_xy, team, ball, players_vel=kin.players_vel, ball_vel=kin.ball_vel,
                               touch_mask=bk.touch_mask, delta_d=bk.delta_d,
                               possession_evidence=bk.possession_evidence, params=PARAMS)
    mid = a + 1250
    assert r.since_seen_s[mid] == pytest.approx(251 / FPS) and r.until_seen_s[mid] == pytest.approx(250 / FPS)
    assert np.isfinite(r.phase_R_home[a:b + 1]).mean() > 0.9
    assert (r.phase_R_home[np.isfinite(r.phase_R_home)] <= 1.0 + 1e-6).all()
    # the last clear possessor before a visible frame agrees with the true labels more often than chance
    labels = np.asarray(synthetic_kin["labels"])[a:b + 1]
    code = np.where(labels == "H", 0, np.where(labels == "A", 1, -1))
    known = (code >= 0) & (r.last_clear_team[a:b + 1] >= 0)
    assert known.sum() > 500
    assert (code[known] == r.last_clear_team[a:b + 1][known]).mean() > 0.6


# ---------------------------------------------------------------------------------- performance
@pytest.mark.slow
def test_full_size_computation_is_fast():
    n, s = 199319, 46
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 100, (n, s, 2)).astype(np.float32)
    ball = rng.uniform(0, 100, (n, 2)).astype(np.float32)
    ball[rng.random(n) < 0.4] = np.nan
    team = np.array([0] * 23 + [1] * 23)
    touch = rng.random(n) < 0.01
    from mtah2027.features._utils import velocity_from_xy
    pvel, bvel = velocity_from_xy(xy, FPS, 1), velocity_from_xy(ball, FPS, 2)       # the pipeline supplies them
    t0 = time.perf_counter()
    r = bm.compute_ball_memory(xy, team, ball, players_vel=pvel, ball_vel=bvel, touch_mask=touch, delta_d=rng.normal(0, 3, n),
                               possession_evidence=rng.uniform(-1, 1, n), params=PARAMS)
    assert time.perf_counter() - t0 < 5.0
    assert r.phase_R_home.shape == (n,)
