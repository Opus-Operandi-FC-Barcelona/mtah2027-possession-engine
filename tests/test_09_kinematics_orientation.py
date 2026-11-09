"""Issue #9: Savitzky-Golay kinematics and attack orientation."""
from __future__ import annotations

import numpy as np
import pytest

from mtah2027.io import atd_loader as al
from mtah2027.preprocessing import kinematics_and_orientation as ko
from mtah2027.preprocessing.clean_tracks import clean_tracks

FPS = 25.0
PITCH = (105.3, 68.0)


def test_interpolate_gaps_limits():
    a = np.full((30, 2), np.nan, dtype=np.float32)
    a[:, 0] = np.arange(30, dtype=np.float32)
    a[:, 1] = 2.0 * np.arange(30)
    a[0:2] = np.nan                 # leading margin: never filled
    a[5:10, :] = np.nan             # gap of 5: filled
    a[15:21, :] = np.nan            # gap of 6: not filled
    a[28:] = np.nan                 # trailing margin
    out, filled = ko.interpolate_gaps(a, 5)
    np.testing.assert_allclose(out[5:10, 0], np.arange(5, 10))
    np.testing.assert_allclose(out[5:10, 1], 2.0 * np.arange(5, 10))
    assert filled[5:10].all() and filled.sum() == 10
    assert np.isnan(out[15:21]).all() and np.isnan(out[:2]).all() and np.isnan(out[28:]).all()
    assert np.isnan(a[5, 0])        # input untouched
    out0, filled0 = ko.interpolate_gaps(a, 0)
    assert np.isnan(out0[5:10]).all() and not filled0.any()


def test_savgol_exact_on_quadratic_including_run_edges():
    t = np.arange(60) / FPS
    x, y = 3.0 + 2.0 * t + 0.5 * 4.0 * t ** 2, 10.0 - 1.5 * t
    xy = np.stack([x, y], axis=1)[:, None, :].astype(np.float32)
    vel, acc = ko.savgol_kinematics(xy, FPS)
    np.testing.assert_allclose(vel[:, 0, 0], 2.0 + 4.0 * t, atol=2e-3)
    np.testing.assert_allclose(vel[:, 0, 1], -1.5, atol=2e-3)
    np.testing.assert_allclose(acc[:, 0, 0], 4.0, atol=0.05)
    np.testing.assert_allclose(acc[:, 0, 1], 0.0, atol=0.05)


def test_runs_are_independent_and_short_runs_use_finite_differences():
    t = np.arange(60) / FPS
    xy = np.full((60, 1, 2), np.nan, dtype=np.float32)
    xy[0:20, 0, 0] = 1.0 * t[0:20]                                 # vx = 1
    xy[0:20, 0, 1] = 0.0
    xy[25:45, 0, 0] = 100.0 + 5.0 * t[25:45]                       # vx = 5, after a 5-frame NaN gap
    xy[25:45, 0, 1] = 0.0
    xy[50:54, 0, 0] = 3.0 * t[50:54]                               # 4-sample run (< window): vx = 3
    xy[50:54, 0, 1] = 0.0
    xy[56:57, 0, :] = 1.0                                          # single sample
    xy[58:60, 0, 0] = [0.0, 0.4]                                   # two samples: v = 0.4 * 25 = 10
    xy[58:60, 0, 1] = 0.0
    vel, acc = ko.savgol_kinematics(xy, FPS)
    np.testing.assert_allclose(vel[0:20, 0, 0], 1.0, atol=1e-3)
    np.testing.assert_allclose(vel[25:45, 0, 0], 5.0, atol=1e-3)         # no contamination by the 1.0 run
    np.testing.assert_allclose(vel[50:54, 0, 0], 3.0, atol=1e-3)
    np.testing.assert_allclose(acc[50:54, 0, 0], 0.0, atol=1e-3)
    assert np.isnan(vel[56]).all() and np.isnan(acc[56]).all()
    np.testing.assert_allclose(vel[58:60, 0, 0], 10.0, atol=1e-4)
    assert np.isnan(acc[58:60]).all()
    assert np.isnan(vel[20:25]).all() and np.isnan(vel[45:50]).all()


def test_savgol_parameter_validation():
    with pytest.raises(ValueError):
        ko.savgol_kinematics(np.zeros((10, 1, 2), dtype=np.float32), FPS, window=6)


def _oriented(td, spec, cleaned=True):
    xy = td.players_xy
    if cleaned:
        xy = clean_tracks(xy, td.team_of_slot).players_xy
    halves = (spec.half1, spec.half2_play)
    return ko.compute_kinematics_and_orientation(xy, td.ball_xy, td.team_of_slot, halves=halves), halves


def test_synthetic_directions_and_home_attacks_plus_x(synthetic_dir, synthetic_info):
    spec = synthetic_info["spec"]
    td = al.load_atd(synthetic_dir, cache_dir=None)
    out, halves = _oriented(td, spec)
    # the generator: home attacks +x in the 1st half and -x in the 2nd
    assert out.direction.home_dir == (1, -1)
    assert out.flip_halves == (False, True)
    assert out.direction.method == ("kickoff-window", "kickoff-window")
    mid = (spec.half1[1] + spec.half2_play[0]) // 2
    assert not out.flipped[:mid + 1].any() and out.flipped[mid + 1:].all()
    # after normalisation the home team sits left of the away team at the start of both halves
    team = td.team_of_slot
    for first, _ in halves:
        sl = slice(first, first + 750)
        home = np.nanmean(out.players_xy[sl][:, team == 0, 0])
        away = np.nanmean(out.players_xy[sl][:, team == 1, 0])
        assert home < away - 5.0
    # shapes / dtypes
    n, s = td.n_frames, td.n_slots
    assert out.players_xy.shape == (n, s, 2) and out.players_vel.shape == (n, s, 2)
    assert out.ball_xy.shape == (n, 2) and out.ball_speed.shape == (n,)
    assert out.players_xy.dtype == np.float32 and out.players_speed.dtype == np.float32


def test_mirror_invariance_and_velocity_sign(synthetic_dir, synthetic_info):
    spec = synthetic_info["spec"]
    td = al.load_atd(synthetic_dir, cache_dir=None)
    halves = (spec.half1, spec.half2_play)
    cleaned = clean_tracks(td.players_xy, td.team_of_slot).players_xy
    base = ko.compute_kinematics_and_orientation(cleaned, td.ball_xy, td.team_of_slot, halves=halves)
    # the same match recorded with both teams attacking the other way
    flipped_in = cleaned.copy()
    flipped_in[..., 0] = PITCH[0] - flipped_in[..., 0]
    flipped_in[..., 1] = PITCH[1] - flipped_in[..., 1]
    ball_in = td.ball_xy.copy()
    ball_in[:, 0] = PITCH[0] - ball_in[:, 0]
    ball_in[:, 1] = PITCH[1] - ball_in[:, 1]
    other = ko.compute_kinematics_and_orientation(flipped_in, ball_in, td.team_of_slot, halves=halves)
    assert other.direction.home_dir == (-1, 1) and other.flip_halves == (True, False)
    np.testing.assert_allclose(other.players_xy, base.players_xy, atol=2e-4, equal_nan=True)
    np.testing.assert_allclose(other.players_vel, base.players_vel, atol=2e-3, equal_nan=True)
    np.testing.assert_allclose(other.players_acc, base.players_acc, atol=5e-2, equal_nan=True)
    np.testing.assert_allclose(other.ball_xy, base.ball_xy, atol=2e-4, equal_nan=True)


def test_orientation_on_a_hand_made_pitch():
    n = 40
    s = 4
    xy = np.full((n, s, 2), np.nan, dtype=np.float32)
    # half 1 = frames 0..9: home (slots 0,1) on the left; half 2 = frames 30..39: home on the right
    sides = {**{i: (20.0, 80.0) for i in range(0, 10)}, **{i: (85.0, 25.0) for i in range(30, 40)}}
    for f, (hx, ax) in sides.items():
        xy[f, 0], xy[f, 1] = [hx, 30.0], [hx + 2, 40.0]
        xy[f, 2], xy[f, 3] = [ax, 30.0], [ax - 2, 40.0]
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    ball[30:40] = [85.0, 34.0]
    team = np.array([0, 0, 1, 1])
    out = ko.compute_kinematics_and_orientation(xy, ball, team, halves=((0, 9), (30, 39)), fps=25.0,
                                                min_samples=10)
    assert out.direction.home_dir == (1, -1) and out.flip_halves == (False, True)
    np.testing.assert_allclose(out.players_xy[2, 0], [20.0, 30.0])
    np.testing.assert_allclose(out.players_xy[32, 0], [PITCH[0] - 85.0, PITCH[1] - 30.0], atol=1e-4)
    np.testing.assert_allclose(out.ball_xy[32], [PITCH[0] - 85.0, PITCH[1] - 34.0], atol=1e-4)
    assert out.flipped[:20].sum() == 0 and out.flipped[20:].all()


def test_unknown_direction_leaves_data_untouched():
    xy = np.full((50, 2, 2), np.nan, dtype=np.float32)
    out = ko.compute_kinematics_and_orientation(xy, np.full((50, 2), np.nan, dtype=np.float32),
                                                np.array([0, 1]), halves=((0, 20), (30, 49)))
    assert out.direction.home_dir == (0, 0) and out.direction.method == ("unknown", "unknown")
    assert out.flip_halves == (False, False) and not out.flipped.any()
