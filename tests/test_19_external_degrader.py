"""Issue #19: realistic degradation of clean external matches."""
from __future__ import annotations

import time

import numpy as np
import pytest

from mtah2027.config import ConfigError, params_from_mapping
from mtah2027.models import external_degrader as ed
from mtah2027.models.external_degrader import DegraderParams, degrade_match

FPS = 25.0
# Issue #19 behaviour (classes of 5-20 / 20-60 / 60-120 s, no guard); the default is now the Issue #38 one
LEGACY = DegraderParams(hide_frac=0.41, gap_model="classes", guard_s=0.0, drop_rate_per_min=0.5,
                        drop_duration_s=(2.0, 20.0), group_drop_rate_per_min=0.0)


def _match(n=60000, s=12, crowd_first_half=False, seed=0):
    """Clean synthetic match: slots on a grid wandering slowly, the ball moving among them.

    gt: X for the first 1000 and the 2000 frames around the middle (break), H/A blocks and a few D."""
    rng = np.random.default_rng(seed)
    team = np.array([0] * (s // 2) + [1] * (s // 2))
    base = np.stack([rng.uniform(10, 95, s), rng.uniform(5, 63, s)], axis=1)
    walk = np.cumsum(rng.normal(0, 0.02, (n, s, 2)), axis=0)
    players = (base[None] + walk).astype(np.float32)
    ball = (np.array([52.0, 34.0]) + np.cumsum(rng.normal(0, 0.03, (n, 2)), axis=0)).astype(np.float32)
    if crowd_first_half:                       # first half: every player sits within 3 m of the ball
        players[: n // 2] = ball[: n // 2, None, :] + rng.uniform(-2, 2, (n // 2, s, 2))
    gt = np.where((np.arange(n) // 700) % 2 == 0, "H", "A").astype("<U1")
    gt[(np.arange(n) // 2300) % 7 == 3] = "D"
    gt[:1000] = "X"
    gt[n // 2 - 1000:n // 2 + 1000] = "X"
    return players, team, ball, gt


@pytest.fixture(scope="module")
def match():
    return _match()


# ---------------------------------------------------------------------------------- config
def test_yaml_section_matches_defaults_and_rejects_typos():
    assert DegraderParams.from_config() == DegraderParams()
    with pytest.raises(ConfigError):
        params_from_mapping(DegraderParams, {"hide_fraction": 0.4}, "external_degrader")
    with pytest.raises(ValueError):
        DegraderParams(hide_frac=1.2)
    with pytest.raises(ValueError):
        DegraderParams(dur_edges_s=(5.0, 20.0), dur_probs=(0.5, 0.5))


# ---------------------------------------------------------------------------------- hidden ball
def test_hidden_share_of_alive_frames_is_met_exactly(match):
    players, team, ball, gt = match
    out = degrade_match(players, team, ball, gt, fps=FPS, seed=3, params=LEGACY)
    alive = np.isin(gt, ["H", "A"])
    assert out.report.hidden_alive_share == pytest.approx(0.41, abs=2e-3)
    assert out.hidden[alive].mean() == pytest.approx(0.41, abs=2e-3)
    assert not out.hidden[gt == "X"].any()                       # nothing hidden outside the graded frames
    assert np.isnan(out.ball_xy[out.hidden]).all()
    assert np.isfinite(out.ball_xy[~out.hidden]).all()           # visible ball keeps (jittered) positions


def test_blocks_are_contiguous_inside_one_period_and_follow_duration_classes():
    players, team, ball, gt = _match(n=400000, s=8, seed=1)
    out = degrade_match(players, team, ball, gt, fps=FPS, seed=0, hide_frac=0.3, artefacts=False,
                        params=LEGACY)
    edges = np.diff(np.concatenate([[0], out.hidden.astype(np.int8), [0]]))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    graded = gt != "X"
    for a, b in zip(starts, ends):                                # never spans the graded/ungraded border
        assert graded[a:b].all()
    secs = np.array(out.report.block_seconds)
    assert out.report.n_blocks >= 10
    assert secs.min() >= 1.0 and secs.max() <= 120.5
    share = [(secs < 20).mean(), ((secs >= 20) & (secs < 60)).mean(), (secs >= 60).mean()]
    # classes of the spec (33 / 48 / 19 %), loose on ~20 blocks; the last block is trimmed
    assert share[1] >= 0.2 and share[0] <= 0.6 and share[2] <= 0.45


def test_duration_sampler_follows_the_configured_mix():
    rng = np.random.default_rng(0)
    p = LEGACY
    d = np.array([ed.sample_block_frames(rng, p, FPS) for _ in range(6000)]) / FPS
    assert (d >= 5.0).all() and (d <= 120.0 + 1e-6).all()
    assert (d < 20).mean() == pytest.approx(0.33, abs=0.03)
    assert ((d >= 20) & (d < 60)).mean() == pytest.approx(0.48, abs=0.03)
    assert (d >= 60).mean() == pytest.approx(0.19, abs=0.03)


def test_placement_prefers_crowded_ball_areas():
    players, team, ball, gt = _match(crowd_first_half=True)
    out = degrade_match(players, team, ball, gt, fps=FPS, seed=5, hide_frac=0.15, artefacts=False)
    n = len(gt)
    first = out.hidden[: n // 2].sum()
    second = out.hidden[n // 2:].sum()
    assert first > 2 * second                                     # crowded half is hidden far more often
    assert out.report.mean_crowding_hidden > out.report.mean_crowding_all


def test_zero_hide_fraction_hides_nothing(match):
    players, team, ball, gt = match
    out = degrade_match(players, team, ball, gt, fps=FPS, hide_frac=0.0, artefacts=False)
    assert not out.hidden.any() and out.report.n_blocks == 0
    np.testing.assert_array_equal(out.ball_xy, ball)
    np.testing.assert_array_equal(out.players_xy, players)


# ---------------------------------------------------------------------------------- determinism
def test_deterministic_by_seed_and_inputs_untouched(match):
    players, team, ball, gt = match
    p0, b0 = players.copy(), ball.copy()
    a = degrade_match(players, team, ball, gt, fps=FPS, seed=7)
    b = degrade_match(players, team, ball, gt, fps=FPS, seed=7)
    c = degrade_match(players, team, ball, gt, fps=FPS, seed=8)
    np.testing.assert_array_equal(a.players_xy, b.players_xy)
    np.testing.assert_array_equal(a.ball_xy, b.ball_xy)
    np.testing.assert_array_equal(a.hidden, b.hidden)
    assert not np.array_equal(a.hidden, c.hidden)
    np.testing.assert_array_equal(players, p0)
    np.testing.assert_array_equal(ball, b0)


# ---------------------------------------------------------------------------------- artefacts
def test_artefacts_only_when_requested(match):
    players, team, ball, gt = match
    clean = degrade_match(players, team, ball, gt, fps=FPS, seed=1, artefacts=False)
    np.testing.assert_array_equal(clean.players_xy, players)
    dirty = degrade_match(players, team, ball, gt, fps=FPS, seed=1)
    vis = ~dirty.hidden
    err = (dirty.ball_xy[vis] - ball[vis])
    assert err.std() == pytest.approx(0.08, rel=0.1)
    assert dirty.report.n_cuts > 0 and dirty.report.n_drops > 0


def test_noise_is_stationary_and_temporally_correlated():
    rng = np.random.default_rng(0)
    x = ed.correlated_noise((200000, 2), 0.1, 0.5, FPS, rng)
    assert x.std() == pytest.approx(0.1, rel=0.05)
    lag1 = np.corrcoef(x[:-1, 0], x[1:, 0])[0, 1]
    assert lag1 == pytest.approx(np.exp(-1.0 / 12.5), abs=0.02)
    white = ed.correlated_noise((100000,), 0.1, 0.0, FPS, rng)
    assert abs(np.corrcoef(white[:-1], white[1:])[0, 1]) < 0.02


def test_identity_cuts_permute_within_a_team_and_leave_gaps():
    n, s = 20000, 8
    team = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    players = np.zeros((n, s, 2), dtype=np.float32)
    players[:, :, 0] = np.arange(s)[None] * 10.0                 # slot i always at x = 10 i (distinct)
    ball = np.full((n, 2), 50.0, dtype=np.float32)
    gt = np.full(n, "H", dtype="<U1")
    p = DegraderParams(hide_frac=0.0, noise_player_m=0.0, noise_ball_m=0.0, cut_rate_per_min=2.0,
                       dup_rate_per_min=0.0, drop_rate_per_min=0.0)
    out = degrade_match(players, team, ball, gt, fps=FPS, seed=2, params=p)
    assert out.report.n_cuts > 3
    x = out.players_xy[..., 0]
    for side in (0, 1):
        cols = np.flatnonzero(team == side)
        want = np.sort(players[0, cols, 0])
        for t in np.flatnonzero(np.isfinite(x[:, cols]).all(axis=1))[::997]:
            np.testing.assert_allclose(np.sort(x[t, cols]), want)   # still the same set of tracks per team
    assert np.isnan(x).any()                                      # fragmentation gaps
    assert (x[-1] != players[-1, :, 0]).any() or np.isnan(x[-1]).any()   # identities have been switched


def test_duplicates_and_drops_use_spare_slots_and_blank_tracks():
    n, s = 30000, 8
    team = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    players = np.zeros((n, s, 2), dtype=np.float32)
    players[:, :, 0] = 10.0 + np.arange(s)[None] * 10.0
    players[:, [3, 7]] = np.nan                                    # one spare slot per team
    ball = np.full((n, 2), 50.0, dtype=np.float32)
    gt = np.full(n, "A", dtype="<U1")
    p = DegraderParams(hide_frac=0.0, noise_player_m=0.0, noise_ball_m=0.0, cut_rate_per_min=0.0,
                       dup_rate_per_min=1.0, drop_rate_per_min=0.0, group_drop_rate_per_min=0.0,
                       dup_offset_m=0.4)
    out = degrade_match(players, team, ball, gt, fps=FPS, seed=4, params=p)
    assert out.report.n_duplicates > 3 and out.report.n_drops == 0
    for spare in (3, 7):
        dup = np.isfinite(out.players_xy[:, spare, 0])
        assert dup.any()
        src = out.players_xy[dup, spare, 0][:, None] - out.players_xy[dup][:, :s, 0]
        assert (np.nanmin(np.abs(src[:, [i for i in range(s) if i != spare]]), axis=1) <= 0.41).all()
    p = DegraderParams(hide_frac=0.0, noise_player_m=0.0, noise_ball_m=0.0, cut_rate_per_min=0.0,
                       dup_rate_per_min=0.0, drop_rate_per_min=1.0, group_drop_rate_per_min=0.0)
    out = degrade_match(players, team, ball, gt, fps=FPS, seed=4, params=p)
    assert out.report.n_drops > 3
    assert np.isnan(out.players_xy[:, [0, 1, 2, 4, 5, 6], 0]).any()     # dropped tracks


def test_shape_validation(match):
    players, team, ball, gt = match
    with pytest.raises(ValueError):
        degrade_match(players, team, ball[:-1], gt)


# ---------------------------------------------------------------------------------- performance
@pytest.mark.slow
def test_full_size_degradation_is_fast():
    n, s = 199319, 46
    rng = np.random.default_rng(0)
    players = rng.uniform(0, 100, (n, s, 2)).astype(np.float32)
    ball = rng.uniform(0, 100, (n, 2)).astype(np.float32)
    gt = np.full(n, "H", dtype="<U1")
    gt[:11185] = "X"
    gt[90652:110808] = "X"
    gt[190198:] = "X"
    team = np.array([0] * 23 + [1] * 23)
    t0 = time.perf_counter()
    out = degrade_match(players, team, ball, gt, fps=FPS, seed=0, params=LEGACY)
    assert time.perf_counter() - t0 < 5.0
    assert out.report.hidden_alive_share == pytest.approx(0.41, abs=2e-3)
