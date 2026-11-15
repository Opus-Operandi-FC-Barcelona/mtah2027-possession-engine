# ruff: noqa: E501
"""Issue #61: post-Viterbi boundary snap to the touch / restart frame."""
import numpy as np

from mtah2027.postprocessing.boundary_snap import BoundarySnapParams, snap_boundaries

H, A, D, X = 0, 1, 2, 3
P = BoundarySnapParams(enabled=True, window_frames=12)


def _setup(n=200):
    return dict(touch_score=np.zeros(n, np.float32), nearest_team=np.full(n, -1), ball_visible=np.ones(n, bool))


def _codes(*runs, n=200):
    c = np.full(n, X, np.int8)
    pos = 0
    for lab, ln in runs:
        c[pos:pos + ln] = lab
        pos += ln
    return c


def test_disabled_is_identity():
    c = _codes((H, 100), (A, 100))
    kw = _setup()
    kw["touch_score"][95] = 1.0
    kw["nearest_team"][95] = A
    out, _ = snap_boundaries(c, play_blocks=[(0, 199)], params=BoundarySnapParams(), **kw)
    assert np.array_equal(out, c)


def test_ha_moves_to_gaining_team_touch_both_directions():
    kw = _setup()
    kw["touch_score"][95] = 1.0
    kw["nearest_team"][95] = A
    out, st = snap_boundaries(_codes((H, 100), (A, 100)), play_blocks=[(0, 199)], params=P, **kw)
    assert (out[:95] == H).all() and (out[95:] == A).all() and st["ha_moved"] == 1
    kw["touch_score"][95] = 0.0
    kw["touch_score"][108] = 1.0
    kw["nearest_team"][108] = A
    out, _ = snap_boundaries(_codes((H, 100), (A, 100)), play_blocks=[(0, 199)], params=P, **kw)
    assert (out[:108] == H).all() and (out[108:] == A).all()


def test_ha_ignores_wrong_team_aerial_hidden_and_far():
    c = _codes((H, 100), (A, 100))
    for mut in ("team", "aerial", "hidden", "far"):
        kw = _setup()
        t = 120 if mut == "far" else 95
        kw["touch_score"][t] = 1.0
        kw["nearest_team"][t] = H if mut == "team" else A
        aer = np.zeros(200, bool)
        if mut == "aerial":
            aer[t] = True
        if mut == "hidden":
            kw["ball_visible"][t] = False
        out, st = snap_boundaries(c, play_blocks=[(0, 199)], is_aerial=aer, params=P, **kw)
        assert np.array_equal(out, c) and st["ha_moved"] == 0


def test_nearest_peak_wins_and_never_crosses_neighbours():
    kw = _setup()
    for t in (90, 97):
        kw["touch_score"][t] = 1.0
        kw["nearest_team"][t] = A
    out, _ = snap_boundaries(_codes((H, 100), (A, 100)), play_blocks=[(0, 199)], params=P, **kw)
    assert out[97] == A and out[96] == H
    # short run H(3) between A runs: the A boundary may not cross the previous boundary
    c = _codes((A, 50), (H, 3), (A, 147))
    kw = _setup()
    kw["touch_score"][45] = 1.0
    kw["nearest_team"][45] = H
    kw["touch_score"][49] = 1.0
    kw["nearest_team"][49] = A
    out, _ = snap_boundaries(c, play_blocks=[(0, 199)], params=P, **kw)
    runs = np.flatnonzero(np.diff(out) != 0)
    assert len(runs) == 2 and (np.diff(np.concatenate([[0], runs + 1, [200]])) >= 1).all()


def test_restart_snap_and_anchor_used_once_and_forced_respected():
    c = _codes((H, 50), (D, 40), (A, 110))
    kw = _setup()
    out, st = snap_boundaries(c, play_blocks=[(0, 199)], restart_frames=np.array([95]), params=P, **kw)
    assert (out[50:95] == D).all() and (out[95:] == A).all() and st["restart_moved"] == 1
    forced = np.zeros(200, bool)
    forced[50:90] = True
    out, st = snap_boundaries(c, play_blocks=[(0, 199)], restart_frames=np.array([85]),
                              forced_dead_mask=forced, params=P, **kw)
    assert np.array_equal(out, c)       # would relabel forced dead frames
    out, _ = snap_boundaries(c, play_blocks=[(0, 199)], restart_frames=np.array([300]), params=P, **kw)
    assert np.array_equal(out, c)


def test_alive_to_dead_block_edges_and_x_unchanged_and_deterministic():
    c = _codes((H, 50), (D, 50), (A, 50), n=200)   # last 50 frames X
    kw = _setup()
    kw["touch_score"][48] = 1.0
    kw["nearest_team"][48] = D
    out1, _ = snap_boundaries(c, play_blocks=[(0, 149)], restart_frames=np.array([30]), params=P, **kw)
    out2, _ = snap_boundaries(c, play_blocks=[(0, 149)], restart_frames=np.array([30]), params=P, **kw)
    assert np.array_equal(out1, c) and np.array_equal(out1, out2) and (out1[150:] == X).all()


def test_default_config_is_off():
    assert BoundarySnapParams.from_config().enabled is False
