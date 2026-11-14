"""Issue #50: bounded tag-window fallback and period-start clamp of the Kick Off (Start) anchors."""
from __future__ import annotations

import dataclasses

import numpy as np

from mtah2027.config import load_config
from mtah2027.features import setpiece_anchor as sa
from mtah2027.io.smart_tagging_loader import SetPiece

FPS = 25.0
TEAM = np.array([0] * 11 + [1] * 11)
N = 800


def _players_moving():
    """Players keep moving at ~3 m/s all the time (live play, no dead evidence anywhere)."""
    xy = np.zeros((N, 22, 2), dtype=np.float32)
    t = np.arange(N)[:, None] / FPS
    xy[:, :, 0] = 20.0 + np.linspace(0, 60, 22)[None, :] + 3.0 * t
    xy[:, :, 1] = 10.0 + np.linspace(0, 40, 22)[None, :]
    return xy


def _ball(launch):
    """Slow carried ball, a 1.2 s rest just before ``launch`` and then a fast pass."""
    x = np.linspace(20.0, 24.0, N)
    x[launch - 30:launch + 1] = x[launch - 30]
    x[launch + 1:] = x[launch] + 10.0 * (np.arange(launch + 1, N) - launch) / FPS
    return np.stack([x, np.full(N, 30.0)], axis=1).astype(np.float32)


def _sp(kind, a, b, team=0):
    return SetPiece(type=kind, team=team, start_frame=a, end_frame=b, half=1)


def _run(sps, **kw):
    params = kw.pop("params", sa.AnchorParams())
    return sa.anchor_set_pieces(_ball(400), _players_moving(), TEAM, sps, (), params=params,
                                dead_score=np.zeros(N), **kw)


def test_wide_window_with_live_play_is_not_forced_to_d():
    r = _run([_sp("free kick", 20, 20 + 875 // 5 * 3)])          # window opens ~15 s before t*
    (a,) = r.anchors
    assert a.resolved and a.onset_method == "fallback"
    forced = r.forced_dead_mask
    assert forced.sum() <= int(round(sa.AnchorParams().fallback_dead_s * FPS))
    assert not forced[:a.t_star - 20].any()                       # live play before t* is untouched
    assert forced[a.t_star - 1] and not forced[a.t_star]


def test_legacy_tag_window_fallback_reachable_by_config():
    p = dataclasses.replace(sa.AnchorParams(), fallback_tag_window=True)
    r = _run([_sp("free kick", 20, 500)], params=p)
    (a,) = r.anchors
    assert a.onset_method == "tag-window" and a.dead_onset == 20
    assert r.forced_dead_mask[20:a.t_star].all()


def test_zero_fallback_leaves_decision_to_viterbi():
    p = dataclasses.replace(sa.AnchorParams(), fallback_dead_s=0.0)
    r = _run([_sp("free kick", 20, 500)], params=p)
    assert r.anchors[0].onset_method == "none" and not r.forced_dead_mask.any()


def test_fallback_never_precedes_tag_start():
    r = _run([_sp("free kick", 395, 500)])
    (a,) = r.anchors
    assert a.dead_onset >= 395


def test_dead_score_evidence_still_wins():
    ds = np.zeros(N)
    ds[200:395] = 1.0
    r = sa.anchor_set_pieces(_ball(400), _players_moving(), TEAM, [_sp("free kick", 20, 500)], (),
                             params=sa.AnchorParams(), dead_score=ds)
    a = r.anchors[0]
    assert a.onset_method == "dead-score" and a.dead_onset == 200


def test_kickoff_start_anchor_clamped_to_block_start():
    # the ball launches at 400 but the play block starts at 410: the anchor moves to the block start
    r = _run([_sp("kick off (start)", 330, 420)], period_starts=[410, 700])
    (a,) = r.anchors
    assert a.t_star == 410 and a.resolved
    assert r.restart_mask[410] and r.restart_mask.sum() == 1


def test_kickoff_clamp_ignores_far_period_start_and_other_types():
    r = _run([_sp("kick off (start)", 330, 420), _sp("kick off (after goal)", 330, 420, team=1),
              _sp("free kick", 330, 420)], period_starts=[600])
    assert all(a.t_star != 600 for a in r.anchors)
    r = _run([_sp("kick off (after goal)", 330, 420, team=1), _sp("free kick", 330, 420)],
             period_starts=[410])
    assert all(a.t_star != 410 for a in r.anchors)


def test_clamp_can_be_disabled_and_unresolved_kickoff_gets_block_start():
    p = dataclasses.replace(sa.AnchorParams(), clamp_period_starts=False)
    r = _run([_sp("kick off (start)", 330, 420)], params=p, period_starts=[410])
    assert r.anchors[0].t_star != 410
    still = np.full((N, 2), 50.0, dtype=np.float32)
    r = sa.anchor_set_pieces(still, np.full((N, 22, 2), 30.0, dtype=np.float32), TEAM,
                             [_sp("kick off (start)", 330, 420)], (), params=sa.AnchorParams(),
                             dead_score=np.zeros(N), period_starts=[410])
    a = r.anchors[0]
    assert a.method == "period-start" and a.t_star == 410


def test_config_period_starts_are_the_official_ones():
    m = load_config().match
    assert (m.half1[0], m.half2_play[0]) == (11185, 113742)
    assert sa.AnchorParams.from_config() == sa.AnchorParams()
