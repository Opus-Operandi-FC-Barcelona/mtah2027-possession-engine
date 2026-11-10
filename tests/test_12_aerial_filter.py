"""Issue #12: aerial-ball mask from the ball track alone."""
from __future__ import annotations

import numpy as np
import pytest

from mtah2027.config import ConfigError, params_from_mapping
from mtah2027.features import ball_kinematics as bk
from mtah2027.vision import aerial_filter as af

FPS = 25.0
P = af.AerialParams()


def _line(n, speed, start=(10.0, 30.0), heading=0.0):
    t = np.arange(n) / FPS
    return np.stack([start[0] + speed * np.cos(heading) * t, start[1] + speed * np.sin(heading) * t], axis=1)


def _rolling(n, v0, decel):
    """Straight ball slowed down by a constant deceleration (rolling on grass)."""
    t = np.arange(n) / FPS
    s = v0 * t - 0.5 * decel * t ** 2
    return np.stack([10.0 + s, np.full(n, 30.0)], axis=1)


def test_yaml_section_matches_defaults_and_rejects_typos():
    assert af.AerialParams.from_config() == af.AerialParams()
    with pytest.raises(ConfigError):
        params_from_mapping(af.AerialParams, {"speed": 35}, "aerial_filter")


def test_apparent_speed_above_threshold_is_aerial_and_below_is_not():
    ball = np.concatenate([_line(30, 6.0, (10, 30)), _line(30, 40.0, (60, 30))])
    # 6 m/s then a jump then 40 m/s: only the 40 m/s stretch (and the jump frames) are flagged
    mask = af.compute_is_aerial(ball, fps=FPS, params=P)
    assert not mask[:25].any()
    assert mask[35:55].all()


def test_slow_and_short_tracks_are_ground():
    ball = _line(100, 8.0)
    assert not af.compute_is_aerial(ball, fps=FPS, params=P).any()
    assert not af.compute_is_aerial(ball[:2], fps=FPS, params=P).any()
    assert not af.compute_is_aerial(np.full((50, 2), np.nan), fps=FPS, params=P).any()


def test_long_straight_flight_without_deceleration_is_aerial_with_trimmed_edges():
    ball = _line(60, 20.0)                      # 20 m/s for 2.4 s: 48 m, constant speed
    mask = af.compute_is_aerial(ball, fps=FPS, params=P)
    assert mask[P.edge_trim_frames + 1:-P.edge_trim_frames - 1].all()
    assert not mask[:P.edge_trim_frames].any() and not mask[-P.edge_trim_frames:].any()


def test_same_path_with_grass_deceleration_is_ground():
    ball = _rolling(60, 22.0, 4.0)              # slows by 4 m/s every second: rolling on grass
    assert not af.compute_is_aerial(ball, fps=FPS, params=P).any()


def test_short_straight_pass_is_ground_even_without_deceleration():
    ball = _line(12, 20.0)                      # 20 m/s for 0.5 s -> 10 m < 18 m
    assert not af.compute_is_aerial(ball, fps=FPS, params=P).any()


def test_curved_fast_path_is_not_a_straight_flight():
    t = np.arange(60) / FPS
    ang = 3.0 * t                               # heading turns 6.9 deg per frame at 20 m/s
    ball = np.stack([10 + 20 * np.cumsum(np.cos(ang)) / FPS, 30 + 20 * np.cumsum(np.sin(ang)) / FPS], axis=1)
    assert not af.compute_is_aerial(ball, fps=FPS, params=af.AerialParams(straight_deg=3.0)).any()


def test_z_channel_marks_high_ball_and_is_found_by_name():
    ball = _line(40, 4.0)
    z = np.zeros(40)
    z[10:20] = 3.0
    mask = af.compute_is_aerial(ball, fps=FPS, ball_z=z, params=P)
    assert mask[10:20].all() and not mask[:10].any() and not mask[20:].any()
    assert af.find_z_channel({"Z": z, "speed": z}, P.z_channels, 40) is not None
    assert af.find_z_channel({"speed": z}, P.z_channels, 40) is None
    assert af.find_z_channel(None, P.z_channels, 40) is None


def test_reprojection_jump_needs_the_homography():
    ball = _line(40, 6.0)
    ball[20:] = _line(20, 6.0, (50.0, 30.0))    # 30 m teleport between frames 19 and 20
    H = np.tile(np.diag([1920.0, 1080.0, 1.0]), (40, 1, 1))     # 105 m -> ~18 px/m
    without = af.compute_is_aerial(ball, fps=FPS, params=af.AerialParams(speed_mps=1000.0))
    with_h = af.compute_is_aerial(ball, fps=FPS, H=H, params=af.AerialParams(speed_mps=1000.0))
    assert not without.any()
    assert with_h[18:22].all()


def test_flagged_frames_never_include_hidden_ones():
    ball = _line(60, 20.0)
    ball[30:33] = np.nan
    mask = af.compute_is_aerial(ball, fps=FPS, params=P)
    assert not mask[30:33].any()


def test_aerial_mask_suppresses_touches_in_ball_kinematics():
    # a ball with a sharp direction change next to a player is a touch ... unless the frame is aerial
    n = 40
    ball = np.zeros((n, 2), np.float32)
    ball[:20] = _line(20, 10.0, (10, 30))
    ball[20:] = ball[19] + np.stack([np.zeros(20), 10.0 * np.arange(1, 21) / FPS], axis=1)
    players = np.zeros((n, 2, 2), np.float32)
    players[:, 0] = ball[19]
    players[:, 1] = (80.0, 10.0)
    team = np.array([0, 1])
    base = bk.compute_ball_kinematics(players, team, ball, fps=FPS, params=bk.BallKinParams())
    assert base.touch_mask.any()
    hit = np.flatnonzero(base.touch_mask)
    aerial = np.zeros(n, bool)
    aerial[hit.min() - 3:hit.max() + 4] = True
    off = bk.compute_ball_kinematics(players, team, ball, fps=FPS, is_aerial=aerial,
                                     params=bk.BallKinParams())
    assert not off.touch_mask.any()


def test_pipeline_wires_the_filter_through_the_config_switch(tmp_path):
    from mtah2027.config import DEFAULT_CONFIG_PATH
    from mtah2027.pipeline import PipelineParams, run_core

    assert PipelineParams.from_config() == PipelineParams()
    text = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8").replace(
        "use_aerial_filter: false", "use_aerial_filter: true")
    on = tmp_path / "on.yaml"
    on.write_text(text, encoding="utf-8")
    n = 600
    rng = np.random.default_rng(0)
    players = (rng.uniform(5, 100, (1, 10, 2)) + np.zeros((n, 1, 1))).astype(np.float32)
    team = np.array([0] * 5 + [1] * 5)
    ball = np.zeros((n, 2), np.float32)
    ball[:] = _line(n, 3.0, (20, 30))
    ball[100:160] = _line(60, 25.0, (20, 40))             # a long constant-speed flight
    blocks = [(0, 299), (300, 599)]
    res_on = run_core(players, team, ball, play_blocks=blocks, config_path=on, log=lambda m: None)
    res_off = run_core(players, team, ball, play_blocks=blocks, log=lambda m: None)
    assert res_on.diagnostics["stages"]["aerial_filter"]["status"] == "ok"
    assert 0.0 < res_on.diagnostics["aerial_share"] < 0.2
    assert res_off.diagnostics["stages"]["aerial_filter"]["status"].startswith("skipped")
    assert res_off.diagnostics["aerial_share"] is None
