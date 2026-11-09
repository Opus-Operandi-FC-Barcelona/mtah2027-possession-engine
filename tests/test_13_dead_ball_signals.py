"""Issue #13: continuous dead-ball signals."""
from __future__ import annotations

import time

import numpy as np
import pytest

from mtah2027.config import ConfigError, load_config, load_section, params_from_mapping
from mtah2027.features import _utils as U
from mtah2027.features import dead_ball_signals as dbs

FPS = 25.0
L, W = 105.3, 68.0
TEAM = np.array([0] * 11 + [1] * 11)


def _players(n, speed=0.0):
    """22 players on a line, optionally all moving at ``speed`` m/s along +y (so they never leave)."""
    xy = np.zeros((n, 22, 2), dtype=np.float32)
    xy[:, :11, 0] = 30.0
    xy[:, 11:, 0] = 70.0
    xy[:, :, 1] = np.linspace(5, 60, 22)[None, :]
    if speed:
        xy[:, :, 1] += (np.arange(n) / FPS * speed)[:, None]
    return xy


def _signals(players, ball, **kw):
    return dbs.compute_dead_ball_signals(players, TEAM, np.asarray(ball, dtype=np.float32),
                                         params=dbs.DeadBallParams(), **kw)


# ---------------------------------------------------------------------------------- utilities
def test_utils_row_percentile_matches_numpy():
    rng = np.random.default_rng(1)
    a = rng.uniform(0, 8, (200, 22))
    a[rng.random(a.shape) < 0.3] = np.nan
    a[5] = np.nan
    ref = np.array([np.nanpercentile(r, 80) if np.isfinite(r).any() else np.nan for r in a])
    np.testing.assert_allclose(U.row_percentile(a, 0.8), ref, rtol=1e-5, equal_nan=True)
    assert np.isnan(U.row_percentile(a, 0.8, min_count=30)).all()


def test_utils_runs_and_moving_mean():
    m = np.array([0, 1, 1, 1, 0, 1, 1], dtype=bool)
    np.testing.assert_array_equal(U.run_length_before(m), [0, 1, 2, 3, 0, 1, 2])
    np.testing.assert_array_equal(U.frames_since(m), [-1, 0, 0, 0, 1, 0, 0])
    np.testing.assert_array_equal(U.frames_until(m), [1, 0, 0, 0, 1, 0, 0])
    x = np.array([1.0, np.nan, 3.0, 5.0, np.nan])
    out = U.moving_mean(x, 3)
    np.testing.assert_allclose(out, [1.0, 2.0, 4.0, 4.0, 5.0])
    assert np.isnan(U.moving_mean(np.full(4, np.nan), 3)).all()


def test_speed_from_xy_straight_line_and_gaps():
    t = np.arange(40) / FPS
    xy = np.stack([3.0 * t, np.zeros(40)], axis=1)
    xy[10:12] = np.nan
    sp = U.speed_from_xy(xy, FPS, half_span=2)
    np.testing.assert_allclose(sp[np.isfinite(sp)], 3.0, atol=1e-4)
    assert np.isnan(sp[10]) and np.isnan(sp[11])
    assert np.isfinite(sp[9]) and np.isfinite(sp[12])        # one-sided differences next to a gap


# ---------------------------------------------------------------------------------- config
def test_yaml_section_matches_dataclass_defaults():
    from_yaml = dbs.DeadBallParams.from_config()
    assert from_yaml == dbs.DeadBallParams()
    assert isinstance(from_yaml.status_channels, tuple)


def test_unknown_key_in_section_is_rejected():
    with pytest.raises(ConfigError):
        params_from_mapping(dbs.DeadBallParams, {"still_speed": 1.0}, "dead_ball_signals")
    assert "bias" in load_section("dead_ball_signals")
    assert load_section("does_not_exist") == {}
    assert load_config().seed == 0       # the extra section does not break the typed loader


# ---------------------------------------------------------------------------------- hand-built
def test_out_of_bounds_indicator_with_margin():
    n = 6
    ball = np.array([[50, 30], [-0.2, 30], [-0.5, 30], [50, 68.4], [110.0, 72.0], [np.nan, np.nan]])
    s = _signals(_players(n), ball, ball_speed=np.zeros(n))
    np.testing.assert_array_equal(s.ball_out, [0, 0, 1, 1, 1, 0])
    assert s.ball_out_dist[0] == 0
    assert s.ball_out_dist[2] == pytest.approx(0.5, abs=1e-5)
    assert s.ball_out_dist[4] == pytest.approx(np.hypot(110.0 - L, 72.0 - W), abs=1e-4)
    assert np.isnan(s.ball_out_dist[5])
    np.testing.assert_array_equal(s.ball_visible, [1, 1, 1, 1, 1, 0])


def test_still_time_accumulates_resets_and_is_nan_aware():
    n = 40
    bs = np.zeros(n, dtype=np.float32)
    bs[10:12] = 2.0                  # moving: reset
    bs[25] = np.nan                  # unknown speed
    ball = np.tile([50.0, 30.0], (n, 1))
    ball[25] = np.nan
    s = _signals(_players(n), ball, ball_speed=bs)
    assert s.ball_still_s[0] == pytest.approx(1 / FPS)
    assert s.ball_still_s[9] == pytest.approx(10 / FPS)
    assert s.ball_still_s[10] == 0 and s.ball_still_s[11] == 0
    assert s.ball_still_s[12] == pytest.approx(1 / FPS)
    assert np.isnan(s.ball_still_s[25])
    assert s.ball_still_s[26] == pytest.approx(1 / FPS)       # a gap restarts the count
    assert s.ball_still_s[39] == pytest.approx(14 / FPS)


def test_ball_speed_derived_from_positions_when_not_given():
    n = 60
    t = np.arange(n) / FPS
    ball = np.stack([20.0 + 5.0 * t, np.full(n, 30.0)], axis=1)
    ball[30:] = ball[30]                                         # stops at frame 30
    s = _signals(_players(n), ball)
    assert (s.ball_still_s[:25] == 0).all()
    assert s.ball_still_s[-1] > 1.0


def test_player_speed_mean_p80_and_walking_fraction():
    n = 8
    speeds = np.zeros((n, 22), dtype=np.float32)
    speeds[:] = np.arange(22, dtype=np.float32)[None, :] * 0.5      # 0 .. 10.5
    speeds[3, :] = np.nan                                            # no data
    speeds[4, 5:] = np.nan                                           # only 5 active: below the minimum
    s = _signals(_players(n), np.tile([50.0, 30.0], (n, 1)), players_speed=speeds, ball_speed=np.zeros(n))
    assert s.speed_mean[0] == pytest.approx(np.mean(speeds[0]), rel=1e-5)
    assert s.speed_p80[0] == pytest.approx(np.percentile(speeds[0], 80), rel=1e-5)
    assert s.frac_walking[0] == pytest.approx(np.mean(speeds[0] < 1.7), rel=1e-5)   # 4 of 22
    assert np.isnan(s.speed_mean[3]) and np.isnan(s.speed_p80[3]) and np.isnan(s.frac_walking[3])
    assert np.isnan(s.speed_mean[4]) and s.n_active[4] == 5
    assert s.n_active[0] == 22


def test_visibility_timing():
    ball = np.full((12, 2), np.nan)
    ball[3:5] = [50, 30]
    ball[9] = [50, 30]
    s = _signals(_players(12), ball, ball_speed=np.zeros(12))
    np.testing.assert_allclose(s.t_since_ball[3:8] * FPS, [0, 0, 1, 2, 3], atol=1e-5)
    assert s.t_since_ball[0] == s.t_to_ball[-1] == 60.0             # none yet / none left: capped
    np.testing.assert_allclose(s.t_to_ball[:4] * FPS, [3, 2, 1, 0], atol=1e-5)
    assert s.t_to_ball[5] * FPS == pytest.approx(4)


def test_set_piece_geometry_indicators():
    ball = np.array([[0.3, 0.4],            # corner
                     [104.8, 67.5],         # corner (other)
                     [3.0, 34.0],           # goal area
                     [L - 5.0, 40.0],       # goal area, other side
                     [L / 2 + 0.5, W / 2],  # centre spot
                     [40.0, 0.5],           # touchline
                     [40.0, 67.0],          # touchline
                     [40.0, 30.0],          # nothing
                     [3.0, 20.0]],          # in the box depth but outside its width
                    dtype=np.float32)
    n = len(ball)
    s = _signals(_players(n), ball, ball_speed=np.zeros(n))
    np.testing.assert_array_equal(s.geom_corner, [1, 1, 0, 0, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(s.geom_goal_area, [0, 0, 1, 1, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(s.geom_kickoff, [0, 0, 0, 0, 1, 0, 0, 0, 0])
    np.testing.assert_array_equal(s.geom_touchline, [0, 0, 0, 0, 0, 1, 1, 0, 0])
    np.testing.assert_array_equal(s.geom_any, [1, 1, 1, 1, 1, 1, 1, 0, 0])


def test_kickoff_geometry_needs_teams_in_own_halves():
    n = 4
    ball = np.tile([L / 2, W / 2], (n, 1))
    pl = _players(n)
    pl[2, :6, 0] = 80.0                    # six home players in the away half
    pl[3, 11:, :] = np.nan                 # away team missing: cannot confirm
    s = _signals(pl, ball, ball_speed=np.zeros(n))
    np.testing.assert_array_equal(s.geom_kickoff, [1, 1, 0, 0])


def test_dead_score_ordering_on_hand_built_scenes():
    n = 100
    play = _signals(_players(n, speed=4.0), np.stack([np.linspace(20, 80, n), np.full(n, 30.0)], 1))
    dead = _signals(_players(n, speed=0.8), np.tile([-1.5, 20.0], (n, 1)), ball_speed=np.zeros(n))
    assert play.dead_score[50] < 0.15
    assert dead.dead_score[50] > 0.9
    assert ((play.dead_score >= 0) & (play.dead_score <= 1)).all()
    # without a ball only the collective speed speaks (weaker, but ordered)
    nb_dead = _signals(_players(n, speed=0.8), np.full((n, 2), np.nan))
    nb_play = _signals(_players(n, speed=4.0), np.full((n, 2), np.nan))
    assert nb_dead.dead_score[50] > nb_play.dead_score[50] + 0.2
    assert np.isfinite(nb_dead.dead_score).all()


def test_status_channel_numeric_and_coded():
    n = 10
    dead_flag = np.array([0, 0, 0, 1, 1, 1, np.nan, 0, 0, 0], dtype=np.float32)
    ball = np.tile([50.0, 30.0], (n, 1))
    s = _signals(_players(n, speed=3.0), ball, ball_speed=np.full(n, 8.0), ball_channels={"dead": dead_flag})
    assert s.has_status and s.status_channel == "dead"
    np.testing.assert_array_equal(np.nan_to_num(s.status_dead, nan=-1), [0, 0, 0, 1, 1, 1, -1, 0, 0, 0])
    assert s.dead_score[4] > 0.9 > 0.2 > s.dead_score[1]
    alive = np.array([1, 1, 0, 0, 1, 1, 1, 1, 1, 1], dtype=np.float32)
    s2 = _signals(_players(n), ball, ball_speed=np.full(n, 8.0), ball_channels={"Alive": alive})
    np.testing.assert_array_equal(s2.status_dead[:4], [0, 0, 1, 1])
    codes = np.array([0, 0, 1, 1, 2, 2, 0, 0, 0, 0], dtype=np.float32)
    s3 = _signals(_players(n), ball, ball_speed=np.full(n, 8.0), ball_channels={"status": codes},
                  ball_channel_codes={"status": {"Alive": 0, "Dead": 1, "Unknown": 2}})
    np.testing.assert_array_equal(np.nan_to_num(s3.status_dead, nan=-1), [0, 0, 1, 1, -1, -1, 0, 0, 0, 0])
    # a numeric channel with an ambiguous name is not interpreted, speed/x/y are never status
    s4 = _signals(_players(n), ball, ball_speed=np.full(n, 8.0),
                  ball_channels={"speed": codes, "state": codes})
    assert not s4.has_status and np.isnan(s4.status_dead).all()


def test_matrix_and_shapes():
    n = 20
    s = _signals(_players(n), np.tile([50.0, 30.0], (n, 1)))
    mat, names = s.matrix()
    assert mat.shape == (n, len(names)) and "dead_score" in names and mat.dtype == np.float32
    sub, sel = s.matrix(["dead_score", "ball_out"])
    assert sub.shape == (n, 2) and sel == ["dead_score", "ball_out"]
    with pytest.raises(ValueError):
        dbs.compute_dead_ball_signals(_players(n), TEAM, np.zeros((n + 1, 2), dtype=np.float32))


# ---------------------------------------------------------------------------------- synthetic match
def test_synthetic_dead_score_separates_dead_from_alive(synthetic_kin):
    k, labels = synthetic_kin["kin"], synthetic_kin["labels"]
    td = synthetic_kin["td"]
    s = dbs.compute_dead_ball_signals(k.players_xy, td.team_of_slot, k.ball_xy, players_speed=k.players_speed,
                                      ball_speed=k.ball_speed)
    graded = (labels == "H") | (labels == "A") | (labels == "D")
    d = labels == "D"
    alive = (labels == "H") | (labels == "A")
    assert d.sum() > 500 and alive.sum() > 500
    margin = float(np.nanmean(s.dead_score[d]) - np.nanmean(s.dead_score[alive]))
    assert margin > 0.3, margin
    # rank separation (AUC) with a clear margin, with and without a visible ball
    vis = s.ball_visible > 0

    def auc(mask_pos, mask_neg):
        pos, neg = s.dead_score[mask_pos], s.dead_score[mask_neg]
        ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
        return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

    assert auc(d & vis & graded, alive & vis) > 0.85
    assert auc(d & ~vis & graded, alive & ~vis) > 0.55 or (d & ~vis).sum() < 50
    # features behave: dead frames have the ball stopped and many players walking
    assert np.nanmean(s.ball_still_s[d & vis]) > 5 * np.nanmean(s.ball_still_s[alive & vis])
    assert np.nanmean(s.frac_walking[d]) > np.nanmean(s.frac_walking[alive])
    assert np.nanmean(s.ball_out[d & vis]) > 0.1 > np.nanmean(s.ball_out[alive & vis])
    assert np.isfinite(s.dead_score).all()


def test_full_size_timing():
    n = 199319
    rng = np.random.default_rng(0)
    players = rng.uniform(0, 100, (n, 46, 2)).astype(np.float32)
    players[:, 40:] = np.nan
    players[::7, :10] = np.nan
    team = np.r_[np.zeros(23), np.ones(23)].astype(int)
    ball = rng.uniform(0, 100, (n, 2)).astype(np.float32)
    ball[rng.random(n) < 0.4] = np.nan
    t0 = time.perf_counter()
    s = dbs.compute_dead_ball_signals(players, team, ball, params=dbs.DeadBallParams())
    elapsed = time.perf_counter() - t0
    assert s.dead_score.shape == (n,) and np.isfinite(s.dead_score).all()
    assert elapsed < 20.0, elapsed
    print(f"dead_ball_signals full-size: {elapsed:.2f}s")
