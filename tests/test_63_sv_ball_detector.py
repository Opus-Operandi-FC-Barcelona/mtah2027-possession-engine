# ruff: noqa: E501
"""Issue #63: supervision + local ball-model detector (synthetic frames; skips without torch/ultralytics/supervision).

A tiny untrained single-class YOLO is not available offline, so the model is replaced by a fake: the tests cover
the supervision slicing wiring, the temporal filter and the guarded integration with the aggregate-only script.
"""
from __future__ import annotations

import numpy as np
import pytest
import real_video_ball_check as rvc

from mtah2027.vision import detectors as det
from mtah2027.vision import recovery as rec
from mtah2027.vision.attention_crops import VisionParams


def test_default_path_needs_no_torch_and_reports_missing_weights():
    """Without weights the sv_ball detectors are skipped (and reported), whatever is installed."""
    dets, skipped = rvc.make_detectors(["sv_ball", "sv_ball640"], VisionParams(detector="dryrun"), None, 0)
    assert dets == {} and set(skipped) == {"sv_ball", "sv_ball640"}


def test_motion_geometry_accepts_longer_clips():
    Hi = np.tile(np.eye(3), (5, 1, 1))
    assert det.motion_geometry((0, 0), Hi)["H_inv"].shape == (5, 3, 3)
    assert det.motion_geometry((0, 0), np.full((3, 3, 3), np.nan)) is None
    assert det.motion_geometry((0, 0), np.tile(np.eye(3), (2, 1, 1))) is None


def test_read_crops_clip_shape(tmp_path):
    cv2 = pytest.importorskip("cv2")
    path = tmp_path / "v.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (1920, 1080))
    for i in range(12):
        w.write(np.full((1080, 1920, 3), 10 * i, np.uint8))
    w.release()
    from mtah2027.vision.attention_crops import CropPlan

    params = VisionParams(detector="dryrun", batch_size=4)
    plan = CropPlan(frames=np.array([0, 5, 11]), search_xy=np.zeros((3, 2)), center_uv=np.full((3, 2), 960.0),
                    origin=np.zeros((3, 2), np.int64), gap_id=np.zeros(3, np.int64), n_hidden=0, n_gaps=0)
    got = [c for _, crops, _ in rec.read_crops(path, plan, params, context=2) for c in crops]
    assert len(got) == 3 and all(c.shape == (5, 640, 640, 3) for c in got)


def _fake(clip=5, cfg="sv_ball"):
    """SupervisionBallDetector without loading a model (the model call is replaced)."""
    pytest.importorskip("torch")
    pytest.importorskip("ultralytics")
    pytest.importorskip("supervision")
    from mtah2027.vision.sv_ball import SupervisionBallDetector

    obj = SupervisionBallDetector.__new__(SupervisionBallDetector)
    obj.max_speed_mps, obj.speed_slack_m = 38.0, 1.0
    obj.clip, obj.temporal, obj.context, obj.name, obj.config = clip, clip > 0, clip // 2, cfg, cfg
    obj._slicer = None
    return obj


def _geo():
    return {"origin": (0.0, 0.0), "H_inv": np.tile(np.diag([0.001, 0.001, 1.0]), (5, 1, 1)), "pitch": (105.3, 68.0),
            "fps": 25.0}


def test_temporal_filter_prefers_the_consistent_candidate():
    d = _fake()
    # true ball moves 4 px/frame (0.4 m); the distractor shows up only in the centre frame, farther than the gate
    per = [np.array([[100.0 + 4 * k, 200.0, 0.3]]) for k in range(5)]
    per[2] = np.array([[108.0, 200.0, 0.3], [900.0, 800.0, 0.35]])
    out = d.filter_clip(per, _geo())
    assert out.shape[0] == 2
    assert abs(out[0, 0] - 108.0) < 1e-9 and out[0, 2] > out[1, 2]          # consistent one ranks first
    assert out[1, 2] == pytest.approx(0.35 * 0.5)                             # unsupported: floor factor


def test_temporal_filter_without_geometry_returns_candidates():
    d = _fake()
    per = [np.array([[1.0, 2.0, 0.5]])] * 5
    assert np.allclose(d.filter_clip(per, None), per[2])


def test_sliced_detection_through_supervision_with_fake_model(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("ultralytics")
    sv = pytest.importorskip("supervision")
    d = _fake(clip=0)
    d._sv = sv

    def cb(tile):
        has = tile[..., 0].max() > 200
        if not has:
            return sv.Detections.empty()
        yy, xx = np.unravel_index(int(np.argmax(tile[..., 0])), tile.shape[:2])
        return sv.Detections(xyxy=np.array([[xx - 2, yy - 2, xx + 3, yy + 3]], dtype=float),
                             confidence=np.array([0.8]), class_id=np.array([0]))

    d._callback = cb
    d._slicer = sv.InferenceSlicer(callback=cb, slice_wh=320, overlap_wh=64, thread_workers=1)
    img = np.zeros((640, 640, 3), np.uint8)
    img[400:403, 500:503] = 255
    out = d.detect([img])[0]
    assert len(out) >= 1 and abs(out[0, 0] - 501.5) < 1.6 and abs(out[0, 1] - 401.5) < 1.6


def test_end_to_end_with_real_weights_if_provided(tmp_path_factory):
    """Optional: set MTAH_SV_BALL_WEIGHTS to a local football-ball .pt to run the real model on synthetic video."""
    import os

    weights = os.environ.get("MTAH_SV_BALL_WEIGHTS")
    if not weights or not os.path.isfile(weights):
        pytest.skip("MTAH_SV_BALL_WEIGHTS not set")
    pytest.importorskip("torch")
    pytest.importorskip("ultralytics")
    pytest.importorskip("supervision")
    cv2 = pytest.importorskip("cv2")
    import test_58_real_video_ball_check as t58

    world = t58.world.__wrapped__(tmp_path_factory) if hasattr(t58.world, "__wrapped__") else None
    if world is None:
        pytest.skip("fixture builder unavailable")
    ball = world["truth"].copy()
    rep = rvc.build_report(ball, world["hom"], [(0, t58.N - 1)], world["video"], t58.P, t58.PITCH, t58.FPS,
                           detectors=("sv_ball640",), n_sample=6, sv_weights=weights, sv_clip=3)
    m = rep["detectors"]["sv_ball640"]
    assert m["scanned"] == 6 and m["seconds_per_frame"] > 0 and cv2 is not None
    rvc.assert_aggregate(rep)
