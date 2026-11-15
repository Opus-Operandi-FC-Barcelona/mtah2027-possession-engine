"""Issue #58: aggregate-only ball-detector check on the tactical video (synthetic video, no real data)."""
from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest
import real_video_ball_check as rvc

from mtah2027.io.homography_loader import Homographies, invert_homographies, project_pitch_to_pixels
from mtah2027.vision import attention_crops as ac

cv2 = pytest.importorskip("cv2")

FPS, PITCH, N = 25.0, (105.3, 68.0), 60
P = ac.VisionParams(detector="hough", batch_size=8)


def _hom(n):
    dst = np.array([(380, 250), (1540, 250), (1900, 1020), (20, 1020)], dtype=np.float64)
    src = np.array([(0, 0), (1, 0), (1, 1), (0, 1)], dtype=np.float64)
    rows = []
    for (x, y), (u, v) in zip(src, dst):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    H = np.linalg.svd(np.array(rows))[2][-1].reshape(3, 3)
    Hs = np.tile(H / H[2, 2], (n, 1, 1))
    inv, ok = invert_homographies(Hs, np.ones(n, bool))
    return Homographies(H=Hs, H_inv=inv, reprojection_error=np.full(n, 0.5), valid=ok, pitch=PITCH)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    t = np.arange(N) / FPS
    truth = np.stack([40.0 + 3.0 * t, 25.0 + 2.0 * t], axis=1)
    hom = _hom(N)
    path = tmp_path_factory.mktemp("v58") / "tactical.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (1920, 1080))
    assert w.isOpened()
    u, v = project_pitch_to_pixels(hom.H, truth[:, 0], truth[:, 1], PITCH)
    for i in range(N):
        img = np.full((1080, 1920, 3), (40, 120, 40), np.uint8)
        cv2.circle(img, (int(round(u[i])), int(round(v[i]))), 7, (255, 255, 255), -1)
        w.write(img)
    w.release()
    return {"video": path, "truth": truth, "hom": hom}


def _report(world, ball, detectors=("hough",), n_sample=20, **kw):
    return rvc.build_report(ball, world["hom"], [(0, N - 1)], world["video"], P, PITCH, FPS,
                            detectors=detectors, n_sample=n_sample, **kw)


def test_finds_the_drawn_ball_and_reports_only_aggregates(world):
    ball = world["truth"].copy()
    ball[:5] = np.nan                                   # hidden frames are never sampled
    rep = _report(world, ball)
    assert rep["sample"]["sampled"] == 20 and rep["sample"]["eligible"] == N - 5
    m = rep["detectors"]["hough"]
    assert m["scanned"] == 20 and m["read_failures"] == 0
    assert m["detection_rate_pct"]["1m"] >= 90.0
    assert m["detection_rate_pct"]["5m"] >= m["detection_rate_pct"]["1m"]
    assert m["error_m_median_p90_of_nearest"][0] < 0.3
    assert m["false_positives_per_frame"] < 0.2
    assert 1.0 < rep["ball_size_px_median"] < 6.0       # 0.22 m through the synthetic homography
    assert len(json.dumps(rep)) < rvc.MAX_JSON_CHARS


def test_a_ball_that_is_not_there_is_a_miss(world):
    ball = world["truth"].copy()
    ball[:, 0] += 30.0                                   # ATD ball far from the drawn one
    m = _report(world, ball)["detectors"]["hough"]
    assert m["detection_rate_pct"]["1m"] <= 5.0


def test_sampling_is_deterministic_and_skips_invalid_homographies(world):
    hom = world["hom"]
    bad = dataclasses.replace(hom, reprojection_error=np.where(np.arange(N) < 30, 50.0, 0.5))
    a, ca = rvc.sample_frames(world["truth"], bad, [(0, N - 1)], P, 10, 3, PITCH)
    b, _ = rvc.sample_frames(world["truth"], bad, [(0, N - 1)], P, 10, 3, PITCH)
    assert np.array_equal(a, b) and (a >= 30).all()
    assert ca["skipped_invalid_homography_or_offimage"] == 30


def test_unavailable_detector_is_skipped_gracefully(world):
    rep = _report(world, world["truth"], detectors=("nonexistent",), n_sample=5)
    assert rep["detectors"] == {} and "nonexistent" in rep["skipped_detectors"]


def test_motion_detector_runs_without_crashing(world):
    rep = _report(world, world["truth"], detectors=("motion",), n_sample=10)
    assert rep["detectors"]["motion"]["scanned"] == 10


def test_assert_aggregate_rejects_per_frame_data():
    rvc.assert_aggregate({"a": 1, "b": [1.0, 2.0], "c": {"d": None}})
    with pytest.raises(ValueError):
        rvc.assert_aggregate({"frames": list(range(100))})
    with pytest.raises(ValueError):
        rvc.assert_aggregate({"img": np.zeros((4, 4))})
    with pytest.raises(ValueError):
        rvc.assert_aggregate({"x": [[1, 2]]})
