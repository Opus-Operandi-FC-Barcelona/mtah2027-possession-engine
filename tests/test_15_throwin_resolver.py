"""Issue #15: restarting team of Team = N/A throw-ins from the tracking."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from mtah2027.config import ConfigError, MatchConfig, params_from_mapping
from mtah2027.features import throwin_resolver as tr
from mtah2027.features.setpiece_anchor import AnchorParams, AnchorResult, SetPieceAnchor, anchor_set_pieces
from mtah2027.pipeline import run_pipeline

FPS = 25.0
W = 68.0
TEAM = np.array([0] * 11 + [1] * 11)
T_STAR = 100
N = 200


def _scene(taker_team=0, side="bottom", taker_isolation=6.0, ball_after="taker_team", ball_visible=True,
           taker_on_line=True):
    """22 players in midfield; the taker stands on a touchline with the ball, rivals ``taker_isolation`` away.

    After ``t*`` the ball flies to a player of ``ball_after`` ('taker_team' / 'other_team' / 'none').
    """
    players = np.zeros((N, 22, 2), dtype=np.float32)
    players[:, :11, 0] = np.linspace(20, 60, 11)[None, :]
    players[:, 11:, 0] = np.linspace(22, 62, 11)[None, :]
    players[:, :11, 1] = 30.0 + np.arange(11)[None, :] % 3 * 3.0
    players[:, 11:, 1] = 34.0 + np.arange(11)[None, :] % 3 * 3.0
    y_line = 0.5 if side == "bottom" else W - 0.5
    inward = 1.0 if side == "bottom" else -1.0
    taker = 0 if taker_team == 0 else 11
    opp = slice(11, 22) if taker_team == 0 else slice(0, 11)
    if taker_on_line:
        players[:, taker] = (50.0, y_line)
        # every opponent keeps ``taker_isolation`` m or more away from the taker
        players[:, opp, 0] = 50.0 + taker_isolation + np.arange(11)[None, :] * 3.0
        players[:, opp, 1] = y_line + inward * (taker_isolation + 4.0)
    ball = np.tile(np.array([50.0, y_line], dtype=np.float32), (N, 1))
    if ball_after != "none":
        recv = taker + 1 if ball_after == "taker_team" else (11 if taker_team == 0 else 0)
        recv_team_slots = slice(0, 11) if (recv < 11) else slice(11, 22)
        players[T_STAR:, recv] = (50.0, y_line + inward * 5.0)
        # other players of the receiver's team stay far from the ball
        far = [s for s in range(22) if s != recv and TEAM[s] == TEAM[recv]]
        players[T_STAR:, far, 1] = 20.0 if side == "bottom" else 48.0
        players[T_STAR:, far, 0] = np.linspace(10.0, 90.0, len(far))[None, :]
        if ball_after == "other_team":
            near_old = [s for s in range(22) if TEAM[s] != TEAM[recv]]
            players[T_STAR:, near_old, 0] = np.linspace(5.0, 95.0, len(near_old))[None, :]
            players[T_STAR:, near_old, 1] = 25.0 if side == "bottom" else 43.0
        ramp = np.clip((np.arange(N) - T_STAR) / 10.0, 0.0, 1.0)[:, None]
        ball = (ball * (1 - ramp) + (players[:, recv] + np.array([0.3, 0.0])) * ramp).astype(np.float32)
        del recv_team_slots
    if not ball_visible:
        ball[:] = np.nan
    return players, ball


def _resolve(players, ball, **kw):
    params = dataclasses.replace(tr.ThrowInParams(), **kw)
    return tr.resolve_throw_in(players, TEAM, ball, tag_start=T_STAR - 80, t_star=T_STAR, params=params)


# ---------------------------------------------------------------------------------- config
def test_yaml_section_matches_defaults_and_rejects_typos():
    assert tr.ThrowInParams.from_config() == tr.ThrowInParams()
    with pytest.raises(ConfigError):
        params_from_mapping(tr.ThrowInParams, {"band_m": 1.5}, "throwin_resolver")


# ---------------------------------------------------------------------------------- hand-built scenes
@pytest.mark.parametrize("team", [0, 1])
@pytest.mark.parametrize("side", ["bottom", "top"])
def test_taker_and_controller_agree(team, side):
    res = _resolve(*_scene(taker_team=team, side=side))
    assert res.team == team and res.resolved
    assert res.method == "taker+controller"
    assert res.taker_team == team and res.controller_team == team
    assert res.confidence > 0.5


def test_taker_alone_decides_when_ball_is_hidden():
    res = _resolve(*_scene(taker_team=1, ball_visible=False))
    assert (res.team, res.method) == (1, "taker")
    assert res.controller_team == -1


def test_controller_alone_decides_without_a_player_on_the_touchline():
    players, ball = _scene(taker_team=0, taker_on_line=False, ball_after="taker_team")
    # the ball is thrown to a player of team 0 in the field of play, nobody stands on the line
    players[:, 0] = (50.0, 20.0)
    res = _resolve(players, ball)
    assert res.taker_team == -1
    assert res.team in (0, -1)
    if res.resolved:
        assert res.method == "controller"


def test_disagreement_is_settled_by_weight_and_flagged():
    players, ball = _scene(taker_team=0, ball_after="other_team")
    heavy_taker = _resolve(players, ball, w_taker=3.0, min_confidence=0.0)
    assert heavy_taker.team == 0 and heavy_taker.method == "taker>controller"
    heavy_ctrl = _resolve(players, ball, w_controller=3.0, min_confidence=0.0)
    assert heavy_ctrl.team == 1 and heavy_ctrl.method == "controller>taker"


def test_equal_weights_disagreement_is_unresolved_or_low_confidence():
    players, ball = _scene(taker_team=0, ball_after="other_team")
    res = _resolve(players, ball, min_confidence=0.5)
    assert not res.resolved and res.method == "unresolved"


def test_no_evidence_is_unresolved():
    players, ball = _scene(taker_on_line=False, ball_after="none", ball_visible=False)
    players[:, :, 1] = 30.0
    res = _resolve(players, ball)
    assert res == tr.ThrowInResolution(-1, T_STAR, -1, 0.0, "unresolved", -1, -1, -1)


def test_taker_is_the_most_isolated_candidate_near_the_line():
    players, ball = _scene(taker_team=0, taker_isolation=8.0, ball_visible=False)
    # a crowded opponent also walks along the touchline far from the ball, close to a home player
    players[:, 15] = (20.0, 1.0)
    players[:, 3] = (21.0, 1.2)
    res = _resolve(players, ball, taker_ball_radius_m=0.0)
    assert res.team == 0 and res.taker_slot == 0


def test_ball_radius_discards_a_far_but_more_isolated_candidate():
    players, ball = _scene(taker_team=0, taker_isolation=4.0)
    players[:, 12] = (10.0, 66.9)                    # far touchline, no marker around, away team
    players[:, 11:, :][:, :1] = players[:, 12:13]    # keep the array valid
    far = _resolve(players, ball, taker_ball_radius_m=0.0, w_controller=0.0)
    near = _resolve(players, ball, taker_ball_radius_m=3.0, w_controller=0.0)
    assert near.taker_team == 0 and near.taker_slot == 0
    assert far.taker_slot != 0


def test_resolve_throw_ins_skips_known_teams_and_other_types():
    players, ball = _scene(taker_team=1)

    def anchor(kind, team, t=T_STAR):
        return SetPieceAnchor(type=kind, team=team, t_star=t, confidence=1.0, method="ball", tag_start=t - 80,
                              tag_end=t + 20, window=(t - 100, t + 20), dead_onset=t - 50,
                              onset_method="dead-score")

    ar = AnchorResult(anchors=[anchor("throw-in", -1), anchor("throw-in", 0), anchor("corner kick", -1),
                               anchor("throw-in", -1, t=-1)],
                      restart_mask=np.zeros(N, bool), forced_dead_intervals=[],
                      forced_dead_mask=np.zeros(N, bool))
    res = tr.resolve_throw_ins(ar, players, TEAM, ball, params=tr.ThrowInParams())
    assert [r.anchor_index for r in res] == [0]
    res_all = tr.resolve_throw_ins(ar, players, TEAM, ball,
                                   params=dataclasses.replace(tr.ThrowInParams(), resolve_known_teams=True))
    assert [r.anchor_index for r in res_all] == [0, 1]
    new = tr.apply_resolutions(ar, res)
    assert new.anchors[0].team == 1 and new.anchors[1].team == 0 and new.anchors[2].team == -1
    assert ar.anchors[0].team == -1                       # input untouched


def test_emission_prior_biases_only_the_window_after_t_star():
    lp = np.round(np.full((N, 3), np.log(1 / 3)), 6)
    resolution = tr.ThrowInResolution(0, T_STAR, 1, 0.9, "taker+controller")
    unresolved = tr.ThrowInResolution(1, 30, -1, 0.0, "unresolved")
    p = dataclasses.replace(tr.ThrowInParams(), emission_bias=1.0, emission_window_s=1.0)
    out = tr.restart_emission_prior(lp, [resolution, unresolved], fps=FPS, params=p)
    w = slice(T_STAR, T_STAR + 25)
    assert np.allclose(out[w, 1] - lp[w, 1], 0.5) and np.allclose(out[w, 0] - lp[w, 0], -0.5)
    assert np.allclose(out[w, 2], lp[w, 2])
    assert np.array_equal(out[:T_STAR], lp[:T_STAR]) and np.array_equal(out[T_STAR + 25:], lp[T_STAR + 25:])
    zero = tr.restart_emission_prior(lp, [resolution], fps=FPS,
                                     params=dataclasses.replace(p, emission_bias=0.0))
    assert np.array_equal(zero, lp)


def test_accuracy_helper():
    rs = [tr.ThrowInResolution(0, 1, 0, 1.0, "taker"), tr.ThrowInResolution(1, 2, 1, 1.0, "taker"),
          tr.ThrowInResolution(2, 3, -1, 0.0, "unresolved")]
    acc = tr.accuracy_against_truth(rs, {0: 0, 1: 0, 2: 1})
    assert (acc["n"], acc["resolved"], acc["correct"]) == (3, 2, 1)
    assert acc["accuracy_resolved"] == 0.5 and acc["accuracy_overall"] == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------------- synthetic match
def _synthetic_resolutions(synthetic_kin):
    k = synthetic_kin
    kin, td = k["kin"], k["td"]
    spec = k["spec"]
    sts = k["st"]
    anchors = anchor_set_pieces(kin.ball_xy, kin.players_xy, td.team_of_slot, sts.set_pieces, sts.goals,
                                ball_speed=kin.ball_speed, players_speed=kin.players_speed,
                                params=AnchorParams())
    res = tr.resolve_throw_ins(anchors, kin.players_xy, td.team_of_slot, kin.ball_xy,
                               params=tr.ThrowInParams())
    labels = np.asarray(k["labels"])
    truth = {}
    for r in res:
        lab = labels[min(r.t_star + 5, len(labels) - 1)]
        if lab in ("H", "A"):
            truth[r.anchor_index] = 0 if lab == "H" else 1
    return anchors, res, truth, spec


def test_synthetic_match_throw_ins_are_resolved_correctly(synthetic_kin):
    anchors, res, truth, _ = _synthetic_resolutions(synthetic_kin)
    n_na = sum("throw" in a.type and a.team == -1 for a in anchors.anchors)
    assert n_na >= 1 and all(a.team == -1 for a in anchors.anchors if "throw" in a.type)
    acc = tr.accuracy_against_truth(res, truth)
    assert acc["n"] >= 1 and acc["resolved"] >= 1
    assert acc["correct"] == acc["resolved"]
    assert acc["accuracy_overall"] >= 0.8


def test_pipeline_feeds_the_resolved_teams(synthetic_dir, synthetic_spec, tmp_path):
    spec = synthetic_spec
    blocks = MatchConfig(n_video_frames=spec.n_frames, half1=spec.half1, half2_dead=spec.half2_dead,
                         half2_play=spec.half2_play)
    on = run_pipeline(synthetic_dir, blocks=blocks, log=lambda m: None)
    info = on.diagnostics["core"]["throw_ins"]
    assert info is not None and info["total"] >= 1 and info["resolved"] >= 1
    assert on.diagnostics["core"]["stages"]["throwin_resolver"]["status"] == "ok"

    from mtah2027.config import DEFAULT_CONFIG_PATH

    off_cfg = tmp_path / "off.yaml"
    off_cfg.write_text(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8").replace(
        "throwin_resolver:\n  enabled: true", "throwin_resolver:\n  enabled: false"), encoding="utf-8")
    off = run_pipeline(synthetic_dir, off_cfg, blocks, log=lambda m: None)
    assert off.diagnostics["core"]["throw_ins"] is None
    assert off.diagnostics["core"]["stages"]["throwin_resolver"]["status"].startswith("skipped")
    assert len(off.labels) == len(on.labels)
