"""Issue #16: ball-to-player distances, control weights, touches, possession evidence."""
from __future__ import annotations

import time

import numpy as np
import pytest

from mtah2027.config import ConfigError, params_from_mapping
from mtah2027.features import ball_kinematics as bk

FPS = 25.0
TEAM6 = np.array([0, 0, 0, 1, 1, 1])


def _static(ball, home, away):
    """Frames with fixed players: ``home``/``away`` are [3, 2] arrays; ``ball`` is [N, 2]."""
    ball = np.asarray(ball, dtype=np.float32)
    n = len(ball)
    players = np.concatenate([np.asarray(home), np.asarray(away)])[None].repeat(n, axis=0).astype(np.float32)
    return players, ball


def _comp(players, ball, team=TEAM6, **kw):
    return bk.compute_ball_kinematics(players, team, ball, params=bk.BallKinParams(), **kw)


# ---------------------------------------------------------------------------------- config
def test_yaml_section_matches_defaults_and_rejects_typos():
    assert bk.BallKinParams.from_config() == bk.BallKinParams()
    with pytest.raises(ConfigError):
        params_from_mapping(bk.BallKinParams, {"touch_accel": 12}, "ball_kinematics")


# ---------------------------------------------------------------------------------- distances
def test_knn_distances_sorted_per_team_and_delta_d():
    home = [[10, 10], [20, 10], [30, 10]]
    away = [[50, 10], [41, 10], [60, 10]]
    players, ball = _static([[22.0, 10.0], [36.0, 10.0]], home, away)
    r = _comp(players, ball)
    np.testing.assert_allclose(r.d_home[0], [2.0, 8.0, 12.0], atol=1e-5)
    np.testing.assert_allclose(r.d_away[0], [19.0, 28.0, 38.0], atol=1e-5)
    np.testing.assert_allclose(r.d_home[1], [6.0, 16.0, 26.0], atol=1e-5)
    np.testing.assert_allclose(r.d_away[1], [5.0, 14.0, 24.0], atol=1e-5)
    np.testing.assert_allclose(r.delta_d, [17.0, -1.0], atol=1e-5)
    np.testing.assert_array_equal(r.nearest_team, [0, 1])
    np.testing.assert_array_equal(r.nearest_slot_home, [1, 2])
    np.testing.assert_array_equal(r.nearest_slot_away, [4, 4])
    assert r.d_home.shape == (2, 3) and r.nearest_team.dtype == np.int8


def test_tie_missing_ball_missing_team_and_short_teams():
    home = [[10, 10], [20, 10], [30, 10]]
    away = [[50, 10], [41, 10], [60, 10]]
    players, ball = _static([[25.0, 10.0], [np.nan, np.nan], [22.0, 10.0]], home, away)
    ball[0] = [25.0, 10.0]
    # tie in frame 0: home slot 1 at 5, away slot 4 at 16 -> make a real tie
    players[0, 4] = [30.0, 10.0]                               # away at 5 m, home (20) at 5 m
    players[2, 3:] = np.nan                                    # away team has no valid player
    players[2, 1] = np.nan                                     # and home only two players left
    r = _comp(players, ball)
    assert r.nearest_team[0] == -1 and r.delta_d[0] == pytest.approx(0.0, abs=1e-5)
    assert r.nearest_team[1] == -1 and np.isnan(r.delta_d[1]) and np.isnan(r.d_home[1]).all()
    assert r.nearest_slot_home[1] == -1 and not r.ball_visible[1]
    assert r.nearest_team[2] == -1 and np.isnan(r.d_away[2]).all()
    assert np.isnan(r.d_home[2, 2]) and np.isfinite(r.d_home[2, :2]).all()      # 2 players left: 3rd is NaN
    assert r.possession_evidence[1] == 0.0 and not r.evidence_valid[1] and not r.evidence_valid[2]


def test_relative_speed_and_closing_speed():
    n = 40
    t = np.arange(n) / FPS
    ball = np.stack([np.full(n, 50.0), np.full(n, 30.0)], axis=1).astype(np.float32)
    players = np.zeros((n, 6, 2), dtype=np.float32)
    players[:, :, 1] = 10.0
    players[:, 0, 0] = 40.0 + 4.0 * t                         # home 0 runs towards the ball at 4 m/s
    players[:, 0, 1] = 30.0
    players[:, 1:3] = [[0.0, 0.0], [5.0, 0.0]]
    players[:, 3:] = [[90.0, 5.0], [91.0, 5.0], [92.0, 5.0]]
    vel = np.zeros_like(players)
    vel[:, 0, 0] = 4.0
    r = _comp(players, ball, players_vel=vel, ball_vel=np.zeros((n, 2), np.float32),
              ball_acc=np.zeros((n, 2), np.float32))
    np.testing.assert_allclose(r.rel_speed_home, 4.0, atol=1e-5)
    np.testing.assert_allclose(r.closing_speed_home[5:-5], 4.0, atol=1e-3)
    np.testing.assert_allclose(r.closing_speed_away[5:-5], 0.0, atol=1e-3)
    assert r.rel_speed_away[0] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------------- control weight
def test_control_weight_matches_the_documented_formula_and_is_monotone():
    p = bk.BallKinParams()
    home = [[10, 10], [60, 60], [70, 60]]
    away = [[80, 10], [85, 60], [90, 60]]
    players, ball = _static([[12.0, 10.0]], home, away)
    zero2 = np.zeros((1, 2), np.float32)
    r = _comp(players, ball, players_vel=np.zeros_like(players), ball_vel=zero2, ball_acc=zero2)
    w = np.exp(-4.0 / (2 * p.control_sigma_m ** 2))                  # d = 2, v_rel = 0
    expected = 1.0 - (1 - w) * (1 - 0.0) * (1 - 0.0)
    assert r.control_home[0] == pytest.approx(expected, rel=1e-4)
    assert r.control_away[0] < 1e-6 and r.control_diff[0] > 0.99
    # decreasing with distance and with relative speed
    dists = np.array([[11.0, 10.0], [13.0, 10.0], [16.0, 10.0]], dtype=np.float32)
    players3, ball3 = _static(dists, home, away)
    rr = _comp(players3, ball3, players_vel=np.zeros_like(players3), ball_vel=np.zeros((3, 2), np.float32),
               ball_acc=np.zeros((3, 2), np.float32))
    assert rr.control_home[0] > rr.control_home[1] > rr.control_home[2]
    fast = np.zeros_like(players3)
    fast[:, 0, 0] = 8.0
    rf = _comp(players3, ball3, players_vel=fast, ball_vel=np.zeros((3, 2), np.float32),
               ball_acc=np.zeros((3, 2), np.float32))
    assert (rf.control_home < rr.control_home).all()
    assert ((rf.control_home >= 0) & (rf.control_home <= 1)).all()


def test_possession_evidence_sign_range_and_smoothing():
    n = 80
    home = [[30, 30], [10, 60], [10, 5]]
    away = [[70, 30], [90, 60], [90, 5]]
    ball = np.zeros((n, 2), dtype=np.float32)
    ball[:40] = [32.0, 30.0]            # home player on the ball
    ball[40:] = [68.0, 30.0]            # away player on the ball
    players, ball = _static(ball, home, away)
    r = _comp(players, ball)
    ev = r.possession_evidence
    assert (ev[:30] > 0.8).all() and (ev[50:] < -0.8).all()
    assert ((ev >= -1) & (ev <= 1)).all()
    assert abs(ev[39]) < 0.9 and abs(ev[40]) < 0.9            # the 15-frame window blends the boundary
    assert r.evidence_valid.all()
    assert (r.nearest_team[:40] == 0).all() and (r.nearest_team[40:] == 1).all()


def test_nan_ball_gives_no_evidence_and_no_touch():
    n = 30
    players, ball = _static(np.full((n, 2), np.nan), [[10, 10], [20, 20], [30, 30]], [[50, 10], [60, 20],
                                                                                     [70, 30]])
    r = _comp(players, ball)
    assert (r.nearest_team == -1).all() and (r.possession_evidence == 0).all()
    assert not r.touch_mask.any() and (r.touch_score == 0).all() and not r.evidence_valid.any()
    assert np.isnan(r.d_home).all() and (r.control_diff == 0).all()


# ---------------------------------------------------------------------------------- touches
def _turn_scene(n=100, turn=50, player_near=True, angle_deg=90.0, speed=10.0):
    """Ball rolling at ``speed`` m/s along +x, changing direction by ``angle_deg`` at frame ``turn``."""
    t = np.arange(n)
    d = speed / FPS
    ang = np.radians(angle_deg)
    x = np.where(t <= turn, 20 + d * t, 20 + d * turn + d * np.cos(ang) * (t - turn))
    y = np.where(t <= turn, 30.0, 30.0 + d * np.sin(ang) * (t - turn))
    ball = np.stack([x, y], axis=1).astype(np.float32)
    px, py = 20 + d * turn, 30.0
    players = np.zeros((n, 6, 2), dtype=np.float32)
    players[:] = [[5, 5], [10, 60], [15, 5], [95, 5], [90, 60], [85, 40]]
    if player_near:
        players[:, 0] = [px, py + 0.8]
    return players, ball, turn


def test_touch_on_direction_change_with_player_close():
    players, ball, turn = _turn_scene()
    r = _comp(players, ball)
    idx = np.flatnonzero(r.touch_mask)
    assert len(idx) == 1 and abs(idx[0] - turn) <= 2
    assert 0.5 <= r.touch_score[idx[0]] <= 1.0 and r.touch_score.sum() == r.touch_score[idx[0]]
    assert r.turn_angle_deg[turn] > 60


def test_no_touch_without_player_close_or_when_aerial():
    players, ball, turn = _turn_scene(player_near=False)
    assert not _comp(players, ball).touch_mask.any()
    players, ball, turn = _turn_scene()
    aerial = np.zeros(len(ball), dtype=bool)
    aerial[turn - 3:turn + 4] = True
    assert not _comp(players, ball, is_aerial=aerial).touch_mask.any()
    assert _comp(players, ball, is_aerial=None).touch_mask.any()


def test_acceleration_peak_kick_from_rest_and_straight_roll():
    n = 100
    ball = np.full((n, 2), 40.0, dtype=np.float32)
    ball[:, 1] = 30.0
    ball[60:, 0] = 40.0 + 15.0 * (np.arange(60, n) - 60) / FPS     # kicked at 15 m/s at frame 60
    players = np.zeros((n, 6, 2), dtype=np.float32)
    players[:] = [[39.0, 30.0], [10, 60], [15, 5], [95, 5], [90, 60], [85, 40]]
    r = _comp(players, ball)
    idx = np.flatnonzero(r.touch_mask)
    assert len(idx) >= 1 and abs(idx[0] - 60) <= 4
    # a ball rolling straight at constant speed produces no touch even with a player alongside
    roll = np.stack([20.0 + 6.0 * np.arange(n) / FPS, np.full(n, 30.0)], axis=1).astype(np.float32)
    players2 = players.copy()
    players2[:, 0] = roll + [0.0, 1.0]
    assert not _comp(players2, roll).touch_mask.any()


def test_thresholds_are_taken_from_params():
    players, ball, _ = _turn_scene(angle_deg=30.0)
    strict = bk.BallKinParams(touch_accel_mps2=1e6, touch_angle_deg=45.0)
    assert not bk.compute_ball_kinematics(players, TEAM6, ball, params=strict).touch_mask.any()
    loose = bk.BallKinParams(touch_accel_mps2=1e6, touch_angle_deg=20.0)
    assert bk.compute_ball_kinematics(players, TEAM6, ball, params=loose).touch_mask.any()


def test_input_validation_and_matrix():
    players, ball, _ = _turn_scene()
    with pytest.raises(ValueError):
        _comp(players, ball[:-1])
    with pytest.raises(ValueError):
        _comp(players, ball, is_aerial=np.zeros(3, dtype=bool))
    with pytest.raises(ValueError):
        _comp(players, ball, team=np.array([0, 1]))
    mat, names = _comp(players, ball).matrix(["d_home", "delta_d", "touch_score"])
    assert mat.shape == (len(ball), 5) and names[:3] == ["d_home_1", "d_home_2", "d_home_3"]


# ---------------------------------------------------------------------------------- synthetic match
def test_synthetic_nearest_team_and_evidence_agree_with_truth(synthetic_kin):
    k, labels, td = synthetic_kin["kin"], synthetic_kin["labels"], synthetic_kin["td"]
    r = bk.compute_ball_kinematics(k.players_xy, td.team_of_slot, k.ball_xy, players_vel=k.players_vel,
                                   ball_vel=k.ball_vel, ball_acc=k.ball_acc)
    vis = r.ball_visible > 0
    h, a = (labels == "H") & vis, (labels == "A") & vis
    assert h.sum() > 500 and a.sum() > 500
    assert (r.nearest_team[h] == 0).mean() > 0.75 and (r.nearest_team[a] == 1).mean() > 0.75
    alive = h | a
    truth = np.where(labels == "H", 0, 1)
    assert (r.nearest_team[alive] == truth[alive]).mean() > 0.8
    sign_ok = np.where(labels == "H", r.possession_evidence > 0, r.possession_evidence < 0)
    assert sign_ok[alive].mean() > 0.8
    assert r.possession_evidence[h].mean() > 0.3 > -0.3 > r.possession_evidence[a].mean()
    assert ((r.possession_evidence >= -1) & (r.possession_evidence <= 1)).all()
    # touches happen while the ball is in play, and plenty of them
    assert r.touch_mask[alive].sum() > 40 and r.touch_mask[labels == "D"].sum() < 0.1 * r.touch_mask.sum()
    # no ball -> no evidence
    assert (r.possession_evidence[~vis] == 0).all() and (r.nearest_team[~vis] == -1).all()
    # delta_d consistent with the nearest team
    both = np.isfinite(r.delta_d)
    assert ((r.delta_d[both] > 0) == (r.nearest_team[both] == 0)).all()


def test_full_size_timing():
    n = 199319
    rng = np.random.default_rng(0)
    players = rng.uniform(0, 100, (n, 46, 2)).astype(np.float32)
    players[:, 40:] = np.nan
    team = np.r_[np.zeros(23), np.ones(23)].astype(int)
    ball = rng.uniform(0, 100, (n, 2)).astype(np.float32)
    ball[rng.random(n) < 0.4] = np.nan
    pv = rng.normal(0, 3, (n, 46, 2)).astype(np.float32)
    bv = rng.normal(0, 3, (n, 2)).astype(np.float32)
    ba = rng.normal(0, 8, (n, 2)).astype(np.float32)
    t0 = time.perf_counter()
    r = bk.compute_ball_kinematics(players, team, ball, players_vel=pv, ball_vel=bv, ball_acc=ba,
                                   params=bk.BallKinParams())
    elapsed = time.perf_counter() - t0
    assert r.nearest_team.shape == (n,) and r.d_home.shape == (n, 3)
    assert elapsed < 20.0, elapsed
    print(f"ball_kinematics full-size: {elapsed:.2f}s")
