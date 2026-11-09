"""Issue #8: bounds filter and duplicate-track merge."""
from __future__ import annotations

import numpy as np

from mtah2027.io import atd_loader as al
from mtah2027.preprocessing.clean_tracks import clean_tracks, run_lengths


def _tracks(n, s=4):
    xy = np.full((n, s, 2), np.nan, dtype=np.float32)
    return xy


def test_run_lengths():
    v = np.array([[1, 0], [1, 1], [0, 1], [1, 1], [1, 1], [1, 0]], dtype=bool)
    out = run_lengths(v)
    assert out[:, 0].tolist() == [2, 2, 0, 3, 3, 3]
    assert out[:, 1].tolist() == [0, 4, 4, 4, 4, 0]


def test_out_of_bounds_masked_in_metres():
    xy = _tracks(3, 2)
    xy[:] = [[10.0, 10.0], [50.0, 30.0]]
    xy[0, 0] = [-2.0, 0.0]            # exactly on the margin: kept
    xy[1, 0] = [-2.01, 0.0]           # outside
    xy[2, 0] = [10.0, 70.01]          # outside
    xy[2, 1] = [107.3, 70.0]          # on the corner of the margin: kept
    speed = np.ones((3, 2), dtype=np.float32)
    out = clean_tracks(xy, np.array([0, 1]), speed)
    assert out.active.tolist() == [[True, True], [False, True], [False, True]]
    assert np.isnan(out.players_xy[1, 0]).all() and np.isnan(out.players_speed[1, 0])
    assert out.removed_out_of_bounds.sum() == 2 and not out.removed_duplicate.any()
    assert out.n_active.tolist() == [2, 1, 1] and out.n_active_by_team.tolist() == [[1, 1], [0, 1], [0, 1]]
    assert xy[1, 0, 0] == np.float32(-2.01)          # input not modified


def test_duplicate_keeps_longer_continuous_track():
    n = 40
    xy = _tracks(n, 3)
    xy[:, 0] = [20.0, 30.0]                    # long track
    xy[10:30, 1] = [20.3, 30.2]                # 0.36 m away, shorter track
    xy[:, 2] = [60.0, 10.0]                    # unrelated
    out = clean_tracks(xy, np.array([0, 0, 0]))
    assert out.removed_duplicate[10:30, 1].all() and out.removed_duplicate.sum() == 20
    assert out.active[:, 0].all() and out.active[:, 2].all() and not out.active[:, 1].any()
    assert np.isnan(out.players_xy[10:30, 1]).all()


def test_duplicate_longer_slot_can_be_the_higher_index_and_ties_keep_low_index():
    n = 20
    xy = _tracks(n, 2)
    xy[:5, 0] = [20.0, 30.0]
    xy[:, 1] = [20.1, 30.0]
    out = clean_tracks(xy, np.array([0, 0]))
    assert out.removed_duplicate[:5, 0].all() and out.active[:, 1].all()    # slot 1 is longer
    xy[:] = np.nan
    xy[:, 0] = [20.0, 30.0]
    xy[:, 1] = [20.1, 30.0]
    out = clean_tracks(xy, np.array([0, 0]))
    assert out.active[:, 0].all() and out.removed_duplicate[:, 1].all()     # tie -> lower index


def test_other_team_and_threshold_not_merged():
    xy = _tracks(5, 4)
    xy[:, 0] = [30.0, 30.0]
    xy[:, 1] = [30.2, 30.0]                    # other team: kept
    xy[:, 2] = [50.0, 30.0]
    xy[:, 3] = [50.7, 30.0]                    # same team but 0.7 m apart: kept
    out = clean_tracks(xy, np.array([0, 1, 0, 0]))
    assert out.active.all() and not out.removed_duplicate.any()


def test_chain_keeps_independent_player():
    n = 10
    xy = _tracks(n, 3)
    xy[:, 0] = [30.0, 30.0]
    xy[:, 1] = [30.5, 30.0]
    xy[:, 2] = [31.0, 30.0]       # 1.0 m from slot 0: near slot 1 only
    xy[5:, 2] = np.nan            # make slot 2 shorter than slot 1
    xy[:3, 1] = np.nan            # slot 1: 7 frames; slot 0: 10 frames; slot 2: 5 frames
    out = clean_tracks(xy, np.array([0, 0, 0]))
    # slot 0 (best) removes slot 1 where it is near; slot 2 is >0.6 m from slot 0 and survives
    assert out.active[:5, 2].all()
    assert out.removed_duplicate[3:, 1].all()


def test_synthetic_match_is_cleaned(synthetic_dir, synthetic_info):
    td = al.load_atd(synthetic_dir, cache_dir=None)
    before = np.stack([(~np.isnan(td.players_xy[:, td.team_of_slot == t, 0])).sum(axis=1) for t in (0, 1)], 1)
    out = clean_tracks(td.players_xy, td.team_of_slot, td.players_speed)
    assert out.players_xy.shape == td.players_xy.shape and out.players_xy.dtype == np.float32
    assert before.max() > 11                                  # staff + duplicates present in the raw data
    assert out.n_active_by_team.max() <= 11
    assert out.removed_out_of_bounds.any() and out.removed_duplicate.any()
    tracked = out.n_active > 0
    assert out.n_active[tracked].max() <= 22 and np.median(out.n_active[tracked]) == 22
    # every remaining position is on (or within the margin of) the pitch
    x, y = out.players_xy[..., 0], out.players_xy[..., 1]
    assert np.nanmin(x) >= -2.0 and np.nanmax(x) <= 107.3 and np.nanmin(y) >= -2.0 and np.nanmax(y) <= 70.0
    # speed is masked exactly where the position is
    removed = out.removed_out_of_bounds | out.removed_duplicate
    assert np.isnan(out.players_speed[removed]).all()
    # unaffected frames are bit-identical
    keep = ~removed
    np.testing.assert_array_equal(out.players_xy[keep], td.players_xy[keep])
