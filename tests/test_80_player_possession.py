"""Issue #80: per-player possession zones (synthetic arrays)."""
from __future__ import annotations

import time

import numpy as np
import pytest

from mtah2027.config import load_section
from mtah2027.features.player_possession import (
    PlayerPossessionParams,
    apply_emission_tilt,
    compute_player_possession,
    emission_tilt,
    opening_masks,
)

FPS = 25.0


def _params(**kw):
    return PlayerPossessionParams(enabled=True, **kw)


def _dist(n, home=(), away=()):
    """d_home / d_away [n, 3]: 0.5 m inside the given (start, end) intervals, 20 m elsewhere."""
    dh, da = np.full(n, 20.0), np.full(n, 20.0)
    for a, b in home:
        dh[a:b] = 0.5
    for a, b in away:
        da[a:b] = 0.5
    return np.stack([dh] * 3, 1), np.stack([da] * 3, 1)


def _run(n, home=(), away=(), touch=None, vis=None, aerial=None, **kw):
    dh, da = _dist(n, home, away)
    ts = np.zeros(n) if touch is None else touch
    vis = np.ones(n, bool) if vis is None else vis
    return compute_player_possession(dh, da, ts, vis, aerial, fps=FPS, params=_params(**kw))


def test_default_is_off_and_section_valid():
    assert PlayerPossessionParams().enabled is False
    assert load_section("player_possession")["enabled"] is False
    assert PlayerPossessionParams.from_config().enabled is False
    with pytest.raises(ValueError):
        PlayerPossessionParams.from_config({"player_possession": {"usage": "nope"}})


def test_zone_radius():
    dh, da = _dist(100, home=[(0, 100)])
    dh[:, 0] = 1.2
    ts, vis = np.zeros(100), np.ones(100, bool)
    near = compute_player_possession(dh, da, ts, vis, None, params=_params(radius_m=1.5, confirm_s=0.4))
    far = compute_player_possession(dh, da, ts, vis, None, params=_params(radius_m=1.0, confirm_s=0.4))
    assert (near.control == 0).all() and (far.control == -1).all()


def test_control_confirmed_after_c_seconds():
    c = _run(200, home=[(10, 19)], confirm_s=0.4)        # 9 frames < 10 needed
    assert (c.control == -1).all()
    c = _run(200, home=[(10, 20)], confirm_s=0.4)        # exactly 10 frames
    assert (c.control[10:20] == 0).all() and (c.control[:10] == -1).all() and (c.control[20:] == -1).all()
    c = _run(200, away=[(10, 20)], confirm_s=0.8)        # 0.8 s = 20 frames needed
    assert (c.control == -1).all()
    c = _run(200, away=[(10, 40)], confirm_s=0.8)
    assert (c.control[10:40] == 1).all()


def test_duel_gives_no_control():
    c = _run(200, home=[(10, 60)], away=[(10, 60)], confirm_s=0.4)
    assert (c.control == -1).all() and c.n_changes == 0


def test_bridge_small_gap_but_not_over_the_other_team():
    c = _run(200, home=[(10, 25), (27, 45)], confirm_s=0.8)       # 2-frame gap bridged -> 35 frames
    assert (c.control[10:45] == 0).all()
    c = _run(200, home=[(10, 25), (26, 40)], away=[(25, 26)], confirm_s=0.8)
    assert c.control[25] != 0


def test_two_consecutive_same_team_touches_confirm_control():
    n = 300
    ts = np.zeros(n)
    ts[50] = ts[80] = 0.9
    home = [(48, 52), (78, 82)]                                    # zone runs too short for the C rule
    c = _run(n, home=home, touch=ts, confirm_s=0.8)
    assert (c.control[50:81] == 0).all() and (c.control[:50] == -1).all()
    ts2 = ts.copy()
    ts2[80] = 0.0
    assert (_run(n, home=home, touch=ts2, confirm_s=0.8).control == -1).all()
    # opposite-team touches do not pair
    c = _run(n, home=[(48, 52)], away=[(78, 82)], touch=ts, confirm_s=0.8)
    assert (c.control == -1).all()
    # touches too far apart do not pair
    ts3 = np.zeros(n)
    ts3[10] = ts3[200] = 0.9
    assert (_run(n, home=[(8, 12), (198, 202)], touch=ts3, confirm_s=0.8).control == -1).all()


def test_only_visible_and_not_aerial_frames():
    vis = np.ones(200, bool)
    vis[20:30] = False
    c = _run(200, home=[(10, 60)], vis=vis, confirm_s=0.4)
    assert (c.control[20:30] == -1).all() and (c.control[30:60] == 0).all()
    aer = np.zeros(200, bool)
    aer[10:60] = True
    assert (_run(200, home=[(10, 60)], aerial=aer, confirm_s=0.4).control == -1).all()


def test_change_detection_openings_and_tilt():
    c = _run(300, home=[(10, 60)], away=[(100, 150)], confirm_s=0.4)
    assert c.n_changes == 1 and c.change_to[100] == 1 and (c.change_to >= 0).sum() == 1
    p = _params(confirm_s=0.4, opening_halfwidth=7)
    to_h, to_a = opening_masks(c, p)
    assert to_a[93:108].all() and not to_a[:93].any() and not to_a[108:].any() and not to_h.any()
    tilt = emission_tilt(c, p)
    assert tilt[20] > 0 > tilt[120] and tilt[80] == 0
    assert tilt[20] == pytest.approx(0.5 * np.log(4.0))
    lp = np.log(np.full((300, 3), 1 / 3))
    out = apply_emission_tilt(lp, tilt)
    assert out[20, 0] - out[20, 1] == pytest.approx(tilt[20], abs=1e-5)
    assert np.allclose(out[:, 2], lp[:, 2]) and apply_emission_tilt(lp, np.zeros(300)) is lp


def test_vectorised_and_fast_on_full_length():
    n = 199319
    rng = np.random.default_rng(0)
    dh = np.abs(rng.normal(3, 2, (n, 3)))
    da = np.abs(rng.normal(3, 2, (n, 3)))
    ts = (rng.random(n) > 0.99) * 0.9
    vis = rng.random(n) > 0.3
    t0 = time.perf_counter()
    c = compute_player_possession(dh, da, ts, vis, None, fps=FPS, params=_params())
    assert time.perf_counter() - t0 < 5.0
    assert c.control.shape == (n,) and set(np.unique(c.control)) <= {-1, 0, 1}
