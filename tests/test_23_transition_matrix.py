"""Issue #23: dynamic log transition matrix."""
from __future__ import annotations

import numpy as np
import pytest

from mtah2027.config import ConfigError, params_from_mapping
from mtah2027.decoding import transition_matrix as tm

P = tm.TransitionParams()
N = 400
H, A, D = tm.H, tm.A, tm.D


def _build(**kw):
    return tm.build_log_transitions(N, params=P, **kw)


def test_yaml_section_matches_defaults_and_rejects_typos():
    assert tm.TransitionParams.from_config() == P
    with pytest.raises(ConfigError):
        params_from_mapping(tm.TransitionParams, {"ha_close": -3}, "transition_matrix")


def test_shape_dtype_rounding_and_stay():
    lt = _build()
    assert lt.shape == (N, 3, 3) and lt.dtype == np.float64
    assert np.all(lt[:, [0, 1, 2], [0, 1, 2]] == P.stay)
    finite = lt[1:][np.isfinite(lt[1:])]
    assert np.array_equal(finite, np.round(finite, 6))
    assert np.isfinite(lt[1:]).all()


def test_ha_closed_at_rest_and_no_inputs():
    lt = _build()
    assert np.all(lt[1:, H, A] == P.ha_closed) and np.all(lt[1:, A, H] == P.ha_closed)


def test_ha_opens_only_around_contested_touch():
    touch = np.zeros(N)
    touch[200] = 0.9
    near = np.full(N, 10.0)
    near[198:203] = 1.2
    lt = _build(touch_score=touch, d_home=near, d_away=near)
    assert lt[200, H, A] == P.ha_open and lt[195, A, H] == P.ha_open and lt[205, H, A] == P.ha_open
    assert lt[194, H, A] == P.ha_closed and lt[206, H, A] == P.ha_closed
    # only the home team is at the ball: it can win it (A -> H opens) but the away team cannot (H -> A closed)
    far = np.full(N, 10.0)
    lt2 = _build(touch_score=touch, d_home=near, d_away=far)
    assert np.all(lt2[1:, H, A] == P.ha_closed) and lt2[200, A, H] == P.ha_open
    # strict reading: only a contested touch (both teams close) opens anything
    strict = tm.TransitionParams(require_both_teams=True)
    lt_s = tm.build_log_transitions(N, touch_score=touch, d_home=near, d_away=far, params=strict)
    assert np.all(lt_s[1:, H, A] == P.ha_closed) and np.all(lt_s[1:, A, H] == P.ha_closed)
    lt_b = tm.build_log_transitions(N, touch_score=touch, d_home=near, d_away=near, params=strict)
    assert lt_b[200, H, A] == P.ha_open and lt_b[200, A, H] == P.ha_open
    # an aerial ball never opens it
    aer = np.zeros(N, bool)
    aer[200] = True
    lt3 = _build(touch_score=touch, d_home=near, d_away=near, is_aerial=aer)
    assert np.all(lt3[1:, H, A] == P.ha_closed)


def test_long_occlusion_moderately_open_short_one_not():
    vis = np.ones(N, bool)
    vis[100:160] = False          # 60 frames = 2.4 s -> occlusion
    vis[300:320] = False          # 0.8 s -> not
    lt = _build(ball_visible=vis)
    assert np.all(lt[100:160, H, A] == P.ha_occluded)
    assert lt[96, H, A] == P.ha_occluded and lt[164, H, A] == P.ha_occluded
    assert lt[90, H, A] == P.ha_closed and lt[170, H, A] == P.ha_closed
    assert np.all(lt[300:320, H, A] == P.ha_closed)


def test_dead_entry_modulated_by_dead_score_and_ball_out():
    ds = np.zeros(N)
    ds[100:200] = 1.0
    out = np.zeros(N)
    out[300:310] = 1.0
    lt = _build(dead_score=ds, ball_out=out)
    assert lt[50, H, D] == P.dead_closed
    assert lt[150, H, D] == P.dead_open and lt[150, A, D] == P.dead_open
    assert P.dead_closed < lt[101, H, D] < P.dead_open or lt[101, H, D] == P.dead_open
    assert lt[305, H, D] == P.dead_open
    assert np.all(np.diff(lt[95:106, H, D]) >= 0)


def test_dead_exit_closed_until_evidence_drops_and_restart_opens():
    ds = np.full(N, 1.0)
    ds[300:] = 0.0
    lt = _build(dead_score=ds, restart_frames=np.array([150]), restart_teams=np.array([-1]))
    assert lt[100, D, H] == P.d_exit_closed
    assert lt[150, D, H] == P.restart_open and lt[150, D, A] == P.restart_open
    assert lt[147, D, H] == P.restart_open and lt[153, D, A] == P.restart_open
    assert lt[146, D, H] == P.d_exit_closed and lt[154, D, H] == P.d_exit_closed
    assert lt[350, D, H] == P.d_exit_open


def test_known_restart_team_biases_the_other_column():
    ds = np.full(N, 1.0)
    lt = _build(dead_score=ds, restart_frames=np.array([150, 250]), restart_teams=np.array([1, 0]))
    assert lt[150, D, A] == P.restart_open
    assert lt[150, D, H] == pytest.approx(P.restart_open + P.restart_other_team)
    assert lt[250, D, H] == P.restart_open
    assert lt[250, D, A] == pytest.approx(P.restart_open + P.restart_other_team)


def test_input_validation_and_determinism():
    with pytest.raises(ValueError):
        _build(touch_score=np.zeros(N + 1))
    with pytest.raises(ValueError):
        _build(restart_frames=np.array([1, 2]), restart_teams=np.array([0]))
    ds = np.random.default_rng(0).random(N)
    assert np.array_equal(_build(dead_score=ds), _build(dead_score=ds))


def test_on_synthetic_match_runs_fast(synthetic_kin):
    from mtah2027.features.ball_kinematics import compute_ball_kinematics
    from mtah2027.features.dead_ball_signals import compute_dead_ball_signals

    kin, td = synthetic_kin["kin"], synthetic_kin["td"]
    bk = compute_ball_kinematics(kin.players_xy, td.team_of_slot, kin.ball_xy, players_vel=kin.players_vel,
                                 ball_vel=kin.ball_vel, ball_acc=kin.ball_acc)
    sig = compute_dead_ball_signals(kin.players_xy, td.team_of_slot, kin.ball_xy,
                                    players_speed=kin.players_speed, ball_speed=kin.ball_speed)
    lt = tm.build_log_transitions(len(kin.ball_xy), touch_score=bk.touch_score, d_home=bk.d_home[:, 0],
                                  d_away=bk.d_away[:, 0], ball_visible=bk.ball_visible > 0,
                                  dead_score=sig.dead_score, ball_out=sig.ball_out, params=P)
    assert lt.shape == (len(kin.ball_xy), 3, 3)
    assert (lt[1:, H, A] > P.ha_closed).any()       # some turnover windows opened
    assert (lt[1:, H, D] > P.dead_closed).any()
