"""Issue #14: exact restart anchoring inside Smart Tagging windows."""
from __future__ import annotations

import time

import numpy as np
import pytest

from mtah2027.config import ConfigError, params_from_mapping
from mtah2027.features import setpiece_anchor as sa
from mtah2027.io.smart_tagging_loader import Goal, SetPiece

FPS = 25.0
TEAM = np.array([0] * 11 + [1] * 11)


def _players(n, speed_fn=None):
    """22 players on two lines; ``speed_fn(t)`` gives their common speed in m/s along +y."""
    xy = np.zeros((n, 22, 2), dtype=np.float32)
    xy[:, :11, 0] = 30.0
    xy[:, 11:, 0] = 70.0
    base = np.linspace(2, 40, 22)[None, :]
    if speed_fn is None:
        xy[:, :, 1] = base
    else:
        dist = np.cumsum(speed_fn(np.arange(n))) / FPS
        xy[:, :, 1] = base + dist[:, None] * 0.5
    return xy


def _ball_path(n, launch, rest_from=0, speed=10.0, hidden=None):
    x = np.full(n, 50.0)
    x[launch + 1:] = 50.0 + speed * (np.arange(launch + 1, n) - launch) / FPS
    ball = np.stack([x, np.full(n, 30.0)], axis=1).astype(np.float32)
    if rest_from > 0:
        ball[:rest_from, 0] = np.linspace(40.0, 50.0, rest_from)       # carried slowly before the rest
    if hidden is not None:
        ball[hidden[0]:hidden[1]] = np.nan
    return ball


def _run(ball, players, sps, goals=(), **kw):
    return sa.anchor_set_pieces(ball, players, TEAM, sps, goals, params=sa.AnchorParams(), **kw)


def _sp(kind, a, b, team=0):
    return SetPiece(type=kind, team=team, start_frame=a, end_frame=b, half=1)


# ---------------------------------------------------------------------------------- config
def test_yaml_section_matches_defaults_and_rejects_typos():
    assert sa.AnchorParams.from_config() == sa.AnchorParams()
    with pytest.raises(ConfigError):
        params_from_mapping(sa.AnchorParams, {"pre_margin": 1.0}, "setpiece_anchor")


# ---------------------------------------------------------------------------------- ball method
def test_ball_rest_to_launch_anchors_within_two_frames():
    n = 400
    r = _run(_ball_path(n, 200), _players(n), [_sp("corner kick", 150, 230)])
    (a,) = r.anchors
    assert a.method == "ball" and a.resolved and abs(a.t_star - 201) <= 3
    assert a.confidence > 0.9 and a.rest_s > 3.0
    assert r.restart_mask.sum() == 1 and r.restart_mask[a.t_star]
    np.testing.assert_array_equal(r.t_star, [a.t_star])
    assert a.window == (125, 230 - 1 + 50)


def test_start_frame_is_never_used_as_restart():
    """A Kick Off window opens early: t* must come from the tracking, not from start_frame."""
    n = 500
    r = _run(_ball_path(n, 300), _players(n), [_sp("kick off (start)", 215, 330)])
    assert abs(r.anchors[0].t_star - 301) <= 3 and r.anchors[0].tag_start == 215


def test_longest_rest_wins_over_short_pauses():
    n = 600
    ball = _ball_path(n, 300)
    ball[:260, 0] = 50.0                      # long rest...
    ball[240:260, 0] += np.linspace(0, 12, 20)   # ...but a fast touch at 240..260 (speed ~15 m/s)
    ball[260:301, 0] = ball[259, 0]           # short rest (41 frames) then a launch at 300
    ball[301:, 0] = ball[259, 0] + 10.0 * (np.arange(301, n) - 300) / FPS
    r = _run(ball, _players(n), [_sp("free kick", 200, 320)])
    a = r.anchors[0]
    assert a.method == "ball" and abs(a.t_star - 240) <= 3       # the long (dead-ball) rest wins


def test_launch_without_rest_is_not_a_restart():
    n = 400
    t = np.arange(n)
    ball = np.stack([40.0 + 2.5 * t / FPS, np.full(n, 30.0)], axis=1)       # 2.5 m/s throughout
    ball[200:, 0] = ball[200, 0] + 12.0 * (t[200:] - 200) / FPS             # sudden 12 m/s
    r = _run(ball.astype(np.float32), _players(n), [_sp("free kick", 150, 230)])
    assert r.anchors[0].method == "unresolved" and r.anchors[0].t_star == -1
    assert not r.restart_mask.any() and not r.forced_dead_mask.any()
    assert r.n_resolved == 0 and len(r.t_star) == 0


def test_supplied_ball_speed_is_used():
    n = 300
    bs = np.zeros(n, dtype=np.float32)
    bs[150:] = 9.0
    ball = np.tile([50.0, 30.0], (n, 1)).astype(np.float32)
    r = _run(ball, _players(n), [_sp("goal kick", 100, 180)], ball_speed=bs)
    assert r.anchors[0].t_star == 150          # onset = first frame after the last rest frame


# ---------------------------------------------------------------------------------- collective method
def test_collective_speed_jump_when_ball_is_missing():
    n = 400
    players = _players(n, lambda t: np.where(t < 200, 0.8, 3.5))
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    r = _run(ball, players, [_sp("throw-in", 150, 230, team=-1)])
    a = r.anchors[0]
    assert a.method == "collective" and abs(a.t_star - 200) <= 3
    assert 0 < a.confidence <= sa.AnchorParams().collective_conf_max
    assert a.team == -1


def test_no_jump_and_no_ball_is_unresolved_not_invented():
    n = 400
    r = _run(np.full((n, 2), np.nan, dtype=np.float32), _players(n, lambda t: np.full(len(t), 1.0)),
             [_sp("throw-in", 150, 230)])
    assert r.anchors[0].method == "unresolved"


def test_ball_beats_collective_when_both_available():
    n = 400
    players = _players(n, lambda t: np.where(t < 180, 0.8, 3.5))
    r = _run(_ball_path(n, 220), players, [_sp("corner kick", 150, 260)])
    assert r.anchors[0].method == "ball" and abs(r.anchors[0].t_star - 221) <= 3


# ---------------------------------------------------------------------------------- no tags
def test_without_tags_nothing_is_invented():
    n = 300
    r = _run(_ball_path(n, 150), _players(n), [])
    assert r.anchors == [] and not r.restart_mask.any() and not r.forced_dead_mask.any()
    assert r.forced_dead_intervals == [] and r.restart_mask.shape == (n,)


# ---------------------------------------------------------------------------------- forced dead
def test_forced_dead_interval_runs_from_dead_evidence_onset_to_restart():
    n = 500
    ds = np.zeros(n, dtype=np.float32)
    ds[100:300] = 0.9
    ds[180:190] = 0.1                       # a short gap is bridged
    r = _run(_ball_path(n, 300), _players(n), [_sp("corner kick", 250, 320)], dead_score=ds)
    a = r.anchors[0]
    assert a.onset_method == "dead-score" and a.dead_onset == 100
    (iv,) = r.forced_dead_intervals
    assert (iv.start, iv.end, iv.kind) == (100, a.t_star - 1, "pre-restart")
    assert r.forced_dead_mask[100] and r.forced_dead_mask[a.t_star - 1] and not r.forced_dead_mask[a.t_star]
    assert not r.forced_dead_mask[99]


def test_forced_dead_respects_max_back_and_falls_back_to_tag_window():
    n = 3000
    ds = np.full(n, 0.9, dtype=np.float32)            # dead "forever"
    r = _run(_ball_path(n, 2000), _players(n), [_sp("goal kick", 1900, 2050)], dead_score=ds)
    a = r.anchors[0]
    assert a.t_star - a.dead_onset == int(sa.AnchorParams().max_dead_back_s * FPS)
    r2 = _run(_ball_path(n, 2000), _players(n), [_sp("goal kick", 1900, 2050)],
              dead_score=np.zeros(n, dtype=np.float32))
    assert r2.anchors[0].onset_method == "fallback" and r2.anchors[0].t_star - r2.anchors[0].dead_onset == 12
    legacy = sa.anchor_set_pieces(
        _ball_path(n, 2000), _players(n), TEAM, [_sp("goal kick", 1900, 2050)], (), dead_score=np.zeros(n),
        params=sa.AnchorParams(fallback_tag_window=True))          # Issue #50: legacy behaviour by config
    assert legacy.anchors[0].onset_method == "tag-window" and legacy.anchors[0].dead_onset == 1900


def test_goal_dead_until_next_kickoff_after_goal():
    n = 800
    ds = np.zeros(n, dtype=np.float32)
    ds[205:500] = 0.95
    ball = _ball_path(n, 500)
    sps = [_sp("throw-in", 50, 90, team=-1), _sp("kick off (after goal)", 440, 520, team=1)]
    r = _run(ball, _players(n), sps, goals=[Goal(team=0, start_frame=180, end_frame=260, half=1)],
             dead_score=ds)
    kick = r.anchors[1]
    assert kick.method == "ball" and abs(kick.t_star - 501) <= 3
    goal_ivs = [d for d in r.forced_dead_intervals if d.kind == "goal"]
    assert len(goal_ivs) == 1
    assert (goal_ivs[0].start, goal_ivs[0].end) == (205, kick.t_star - 1)
    assert goal_ivs[0].anchor_index == 1
    assert r.forced_dead_mask[205:kick.t_star].all() and not r.forced_dead_mask[kick.t_star]
    assert not r.forced_dead_mask[:200].any()


def test_goal_without_kickoff_tag_ends_with_dead_evidence_and_far_kickoff_is_not_linked():
    n = 800
    ds = np.zeros(n, dtype=np.float32)
    ds[205:400] = 0.95
    r = _run(_ball_path(n, 700), _players(n), [], goals=[Goal(0, 180, 260, 1)], dead_score=ds)
    (iv,) = r.forced_dead_intervals
    assert (iv.start, iv.end, iv.anchor_index) == (205, 399, -1)
    short = sa.AnchorParams(max_goal_to_kickoff_s=10.0)
    kick = [_sp("kick off (after goal)", 650, 720)]
    far = sa.anchor_set_pieces(_ball_path(n, 700), _players(n), TEAM, kick,
                               [Goal(0, 180, 260, 1)], dead_score=ds, params=short)
    goal_far = [d for d in far.forced_dead_intervals if d.kind == "goal"]
    assert [(d.start, d.end, d.anchor_index) for d in goal_far] == [(205, 399, -1)]


def test_restart_mask_halfwidth_and_edges():
    n = 300
    params = sa.AnchorParams(restart_mask_halfwidth=2)
    r = sa.anchor_set_pieces(_ball_path(n, 150), _players(n), TEAM, [_sp("corner kick", 100, 170),
                                                                      _sp("throw-in", 0, 5),
                                                                      _sp("throw-in", 290, 400)],
                             params=params)
    a = r.anchors[0]
    assert r.restart_mask[a.t_star - 2:a.t_star + 3].all() and r.restart_mask.sum() == 5
    assert r.anchors[1].window[0] == 0 and r.anchors[2].window[1] == n - 1       # clipped, no crash


def test_input_validation():
    with pytest.raises(ValueError):
        sa.anchor_set_pieces(np.zeros((10, 2), np.float32), _players(11), TEAM, [], params=sa.AnchorParams())


# ---------------------------------------------------------------------------------- synthetic match
def test_synthetic_anchors_match_true_restarts(synthetic_kin):
    k, labels, st = synthetic_kin["kin"], synthetic_kin["labels"], synthetic_kin["st"]
    td = synthetic_kin["td"]
    r = sa.anchor_set_pieces(k.ball_xy, k.players_xy, td.team_of_slot, st.set_pieces, st.goals,
                             players_speed=k.players_speed, ball_speed=k.ball_speed)
    alive = (labels == "H") | (labels == "A")
    true = np.flatnonzero(alive[1:] & ~alive[:-1]) + 1
    assert len(st.set_pieces) >= 8 and len(r.anchors) == len(st.set_pieces)
    resolved = [a for a in r.anchors if a.resolved]
    assert len(resolved) >= 0.7 * len(r.anchors)
    err = np.array([np.min(np.abs(true - a.t_star)) for a in resolved])
    assert (err <= 10).mean() >= 0.8, err
    ball_err = np.array([np.min(np.abs(true - a.t_star)) for a in resolved if a.method == "ball"])
    assert len(ball_err) >= 5 and (ball_err <= 6).all(), ball_err
    assert all(a.confidence > 0.5 for a in resolved if a.method == "ball")
    assert all(a.confidence <= 0.5 for a in resolved if a.method == "collective")
    assert r.restart_mask.sum() == len(resolved)
    # forced dead frames are mostly truly dead and cover much of the dead time
    d = labels == "D"
    m = r.forced_dead_mask
    assert d[m].mean() > 0.6 and m[d].mean() > 0.6
    # goal: D from the goal to the kick-off after it
    goal_ivs = [iv for iv in r.forced_dead_intervals if iv.kind == "goal"]
    assert len(goal_ivs) == len(st.goals) >= 1
    assert all(d[iv.start:iv.end + 1].mean() > 0.9 for iv in goal_ivs)


def test_full_size_timing():
    n = 199319
    rng = np.random.default_rng(0)
    players = rng.uniform(0, 100, (n, 46, 2)).astype(np.float32)
    players[:, 40:] = np.nan
    team = np.r_[np.zeros(23), np.ones(23)].astype(int)
    ball = rng.uniform(0, 100, (n, 2)).astype(np.float32)
    ball[rng.random(n) < 0.4] = np.nan
    starts = np.sort(rng.choice(np.arange(500, n - 500), 59, replace=False))
    sps = [SetPiece("throw-in", -1, int(s), int(s) + 100, 1) for s in starts]
    goals = [Goal(0, int(s) + 10, int(s) + 60, 1) for s in starts[:3]]
    t0 = time.perf_counter()
    r = sa.anchor_set_pieces(ball, players, team, sps, goals)
    elapsed = time.perf_counter() - t0
    assert len(r.anchors) == 59 and r.restart_mask.shape == (n,)
    assert elapsed < 20.0, elapsed
    print(f"setpiece_anchor full-size (59 set pieces): {elapsed:.2f}s")
