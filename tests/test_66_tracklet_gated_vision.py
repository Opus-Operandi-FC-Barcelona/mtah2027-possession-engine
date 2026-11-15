# ruff: noqa: E501
"""Issue #66: tracklet gate on the video detections and emission tempering on injected frames (both off)."""
from __future__ import annotations

import numpy as np

from mtah2027.vision.attention_crops import VisionParams
from mtah2027.vision.recovery import inject_points
from mtah2027.vision.sim_detector import (
    PERSIST_FP,
    SV_BALL640,
    TemporalModel,
    simulate_detections_temporal,
)
from mtah2027.vision.tracklet_gate import gate_points, temper_injected, tracklet_mask

FPS = 25.0


def _p(**kw) -> VisionParams:
    return VisionParams(tracklet_gate=True, **kw)


def _line(frames, x0=20.0, vx=5.0, y=30.0):
    f = np.asarray(frames, dtype=np.int64)
    return np.stack([x0 + vx * (f - f[0]) / FPS, np.full(len(f), y)], axis=1)


def test_defaults_are_off_and_change_nothing():
    p = VisionParams.from_config()
    assert p.tracklet_gate is False and p.injected_emission_scale == 1.0
    f = np.array([10, 15, 100])
    xy = np.array([[1.0, 2.0], [50.0, 3.0], [9.0, 9.0]])
    out_f, out_xy = gate_points(f, xy, fps=FPS, params=VisionParams())
    assert np.array_equal(out_f, f) and np.array_equal(out_xy, xy)


def test_isolated_detection_is_removed_smooth_run_is_kept():
    run = np.arange(100, 130, 5)
    xy_run = _line(run)
    f = np.concatenate([[20], run, [300]])
    xy = np.vstack([[[60.0, 10.0]], xy_run, [[5.0, 5.0]]])
    keep = tracklet_mask(f, xy, fps=FPS, params=_p())
    assert not keep[0] and not keep[-1]
    assert keep[1:-1].all()


def test_short_run_below_min_len_is_dropped():
    f = np.array([100, 105])
    keep = tracklet_mask(f, _line(f), fps=FPS, params=_p(tracklet_min_len=3))
    assert not keep.any()
    assert tracklet_mask(f, _line(f), fps=FPS, params=_p(tracklet_min_len=2)).all()


def test_speed_limit_breaks_the_run():
    f = np.array([100, 105, 110, 115])
    xy = np.array([[20.0, 30.0], [20.5, 30.0], [60.0, 30.0], [60.5, 30.0]])    # 40 m in 0.2 s = 200 m/s
    assert not tracklet_mask(f, xy, fps=FPS, params=_p(tracklet_min_len=3)).any()
    assert tracklet_mask(f, xy, fps=FPS, params=_p(tracklet_min_len=2)).all()


def test_outlier_inside_a_run_is_removed_and_the_rest_kept():
    f = np.arange(100, 140, 5)
    xy = _line(f)
    xy[3, 1] += 2.5                                           # off the smooth path by 2.5 m
    keep = tracklet_mask(f, xy, fps=FPS, params=_p(tracklet_min_len=3, tracklet_max_residual_m=1.0))
    assert not keep[3] and keep.sum() == len(f) - 1


def test_time_gap_between_crops_breaks_the_link():
    f = np.array([100, 105, 110, 200, 205, 210])
    xy = np.vstack([_line(f[:3]), _line(f[3:], x0=60.0)])
    assert tracklet_mask(f, xy, fps=FPS, params=_p()).all()
    f2 = np.array([100, 105, 200, 205])
    assert not tracklet_mask(f2, _line(f2, vx=0.0), fps=FPS, params=_p()).any()


def test_unsorted_input_and_determinism():
    f = np.array([110, 100, 105, 300])
    xy = np.vstack([_line([100, 105, 110])[[2, 0, 1]], [[1.0, 1.0]]])
    a = tracklet_mask(f, xy, fps=FPS, params=_p())
    assert list(a) == [True, True, True, False]
    assert np.array_equal(a, tracklet_mask(f, xy, fps=FPS, params=_p()))


def test_gate_points_filters_when_on():
    f = np.array([100, 105, 110, 300])
    xy = np.vstack([_line(f[:3]), [[5.0, 5.0]]])
    out_f, _ = gate_points(f, xy, fps=FPS, params=_p())
    assert list(out_f) == [100, 105, 110]


def test_temporal_simulator_is_deterministic_and_has_marginals():
    frames = np.arange(0, 5 * 6000, 5)
    truth = np.tile([50.0, 34.0], (len(frames), 1))
    a = simulate_detections_temporal(truth, frames, SV_BALL640, TemporalModel(), rate=1.0, seed=3)
    b = simulate_detections_temporal(truth, frames, SV_BALL640, TemporalModel(), rate=1.0, seed=3)
    assert all(np.array_equal(x, y, equal_nan=True) for x, y in zip(a, b))
    ret, xy, good = a
    assert abs(good.mean() - 0.757) < 0.03 and abs(ret.mean() - 0.95) < 0.02
    err = np.hypot(*(xy[good] - truth[good]).T)
    assert abs((err <= 1.0).mean() - 0.41 / 0.757) < 0.04


def test_persistent_fp_variant_has_longer_false_runs():
    frames = np.arange(0, 5 * 6000, 5)
    truth = np.tile([50.0, 34.0], (len(frames), 1))

    def far_pairs(model):
        ret, xy, good = simulate_detections_temporal(truth, frames, SV_BALL640, model, rate=1.0, seed=1)
        fp = ret & ~good
        return int((fp[1:] & fp[:-1]).sum()) / max(int(fp.sum()), 1)

    assert far_pairs(PERSIST_FP) > far_pairs(TemporalModel()) + 0.2


def test_gate_improves_injected_precision_on_simulated_gap():
    """On a simulated moving ball the gate removes most isolated false positives and keeps most good hits."""
    frames = np.arange(0, 5 * 3000, 5)
    t = frames / FPS
    truth = np.stack([50.0 + 8.0 * np.sin(t / 6.0), 34.0 + 5.0 * np.cos(t / 9.0)], axis=1)
    ret, xy, good = simulate_detections_temporal(truth, frames, SV_BALL640, TemporalModel(), rate=1.0, seed=5)
    f, q, g = frames[ret], xy[ret], good[ret]
    keep = tracklet_mask(f, q, fps=FPS, params=_p())
    fp_before = (~g).mean()
    fp_after = (~g[keep]).mean()
    assert fp_after < 0.4 * fp_before
    assert g[keep].sum() > 0.5 * g.sum()


def test_emission_scale_changes_only_injected_frames():
    lp = np.log(np.full((6, 3), 1 / 3.0))
    src = np.array([1, 2, 3, 0, 4, 2], dtype=np.uint8)
    assert temper_injected(lp, src, 1.0) is lp and temper_injected(lp, None, 0.5) is lp
    out = temper_injected(lp, src, 0.5)
    assert np.allclose(out[[1, 2, 5]], lp[[1, 2, 5]] * 0.5, atol=1e-6)
    assert np.array_equal(out[[0, 3, 4]], lp[[0, 3, 4]])
    assert temper_injected(lp, np.ones(6, np.uint8), 0.5) is lp


def test_run_core_default_scale_is_identical_with_ball_source():
    from mtah2027.pipeline import run_core

    rng = np.random.default_rng(0)
    n = 400
    players = rng.uniform(10, 90, (n, 22, 2)).astype(np.float32)
    team = np.array([0] * 11 + [1] * 11)
    ball = np.cumsum(rng.normal(0, 0.3, (n, 2)), axis=0).astype(np.float32) + 50.0
    ball[100:200] = np.nan
    src = np.where(np.isfinite(ball).all(axis=1), 1, 0).astype(np.uint8)
    src[120:140] = 2
    kw = dict(play_blocks=[(0, 199), (200, 399)], fps=25.0, log=lambda m: None)
    base = run_core(players, team, ball, **kw)
    same = run_core(players, team, ball, ball_source=src, **kw)
    assert np.array_equal(base.codes, same.codes)


def test_inject_points_source_flags_are_2_or_3():
    ball = np.full((60, 2), np.nan, dtype=np.float32)
    ball[0] = [10.0, 10.0]
    ball[30] = [11.0, 10.0]
    _, src = inject_points(ball, np.array([10]), np.array([[10.4, 10.0]]), fps=FPS,
                           params=VisionParams())
    assert set(np.unique(src)) <= {0, 1, 2, 3} and (src == 2).sum() == 1


def test_with_defaults_persist_model_constants():
    assert PERSIST_FP.persist_far > TemporalModel().persist_far
