"""Issue #11: attention crops, detector interface, video reading, reprojection and injection.

Everything runs on synthetic data: no real video is ever needed.  The tests that write / read an mp4 are
skipped when OpenCV is not installed (``pip install -e .[vision]``).
"""
from __future__ import annotations

import dataclasses
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from mtah2027.config import ConfigError, params_from_mapping
from mtah2027.io.homography_loader import Homographies, invert_homographies, project_pitch_to_pixels
from mtah2027.vision import attention_crops as ac
from mtah2027.vision import detectors as det
from mtah2027.vision import recovery as rec

FPS = 25.0
PITCH = (105.3, 68.0)
P = ac.VisionParams(every_k=3, pressure_weight=0.0, interp_max_frames=25)
needs_cv2 = pytest.mark.skipif(
    importlib.util.find_spec("cv2") is None, reason="OpenCV (extra 'vision') not installed")


def synthetic_H(n: int, perspective: bool = True) -> Homographies:
    """Normalised pitch -> 1920x1080 pixels through a trapezoid (far side narrower), same H every frame."""
    dst = np.array([(380, 250), (1540, 250), (1900, 1020), (20, 1020)] if perspective
                   else [(0, 0), (1920, 0), (1920, 1080), (0, 1080)], dtype=np.float64)
    src = np.array([(0, 0), (1, 0), (1, 1), (0, 1)], dtype=np.float64)
    rows = []
    for (x, y), (u, v) in zip(src, dst):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    _, _, vt = np.linalg.svd(np.array(rows))
    H = vt[-1].reshape(3, 3)
    H = H / H[2, 2]
    Hs = np.tile(H, (n, 1, 1))
    Hinv, ok = invert_homographies(Hs, np.ones(n, bool))
    return Homographies(H=Hs, H_inv=Hinv, reprojection_error=np.full(n, 0.5), valid=ok, pitch=PITCH)


def _track(n=200, gap=(60, 140)):
    """Ball moving at 4 m/s along x with a hidden gap, and 6 players standing around it."""
    t = np.arange(n) / FPS
    ball = np.stack([20.0 + 4.0 * t, np.full(n, 34.0)], axis=1).astype(np.float32)
    truth = ball.copy()
    ball[gap[0]:gap[1]] = np.nan
    players = np.zeros((n, 6, 2), np.float32)
    for i in range(6):
        players[:, i] = truth + np.array([(-1) ** i * (3 + i), (i - 3) * 2.0])
    return ball, truth, players


# ----------------------------------------------------------------------------- geometry
def test_yaml_section_defaults_off_and_typos_rejected():
    p = ac.VisionParams.from_config()
    assert p == ac.VisionParams() and p.enabled is False
    with pytest.raises(ConfigError):
        params_from_mapping(ac.VisionParams, {"crop": 640}, "vision")


def test_crop_window_is_clamped_inside_the_image_and_centred_when_possible():
    org = ac.crop_window([960.0, 5.0, 1915.0, 960.0], [540.0, 5.0, 1075.0, -50.0], 640, 1920, 1080)
    assert org.tolist() == [[640, 220], [0, 0], [1280, 440], [640, 0]]
    assert ((org >= 0) & (org[:, 0:1] + 640 <= 1920) & (org[:, 1:2] + 640 <= 1080)).all()
    with pytest.raises(ValueError):
        ac.crop_window(0, 0, 2000, 1920, 1080)


def test_projection_round_trip_with_a_synthetic_homography():
    hom = synthetic_H(4)
    pts = np.array([[5.0, 3.0], [52.6, 34.0], [100.0, 60.0], [30.0, 10.0]])
    u, v = project_pitch_to_pixels(hom.H, pts[:, 0], pts[:, 1], PITCH)
    assert (u > 0).all() and (u < 1920).all() and (v > 240).all() and (v < 1030).all()
    org = ac.crop_window(u, v)
    local = np.stack([u, v], 1) - org
    back = ac.crop_points_to_pitch(local, org, hom.H_inv, PITCH)         # one matrix per point
    assert np.abs(back - pts).max() < 1e-6
    back1 = ac.crop_points_to_pitch(local, org, hom.H_inv[0], PITCH)     # a single shared matrix
    assert np.abs(back1 - pts).max() < 1e-6


def test_hidden_frames_and_subsampling():
    ball, _, _ = _track()
    hidden = ac.hidden_gap_frames(ball, [(0, 99), (100, 199)])
    assert hidden.sum() == 80 and hidden[60] and hidden[139] and not hidden[59]
    assert not ac.hidden_gap_frames(ball, [(0, 50)]).any()
    frames, gid = ac.subsample_gap_frames(hidden, 10)
    assert frames.tolist() == list(range(65, 140, 10)) and (gid == 0).all()
    short = np.zeros(30, bool)
    short[10:12] = True
    f2, _ = ac.subsample_gap_frames(short, 25)
    assert f2.tolist() == [10]                       # a short run still gets a sample


def test_interpolated_search_point_between_gap_edges():
    ball, truth, _ = _track()
    pts = ac.interpolated_search_points(np.array([60, 100, 139]), ball, FPS, 20.0)
    exp = np.array([ball[59] + (ball[140] - ball[59]) * w for w in (1 / 81, 41 / 81, 80 / 81)])
    assert np.abs(pts - exp).max() < 1e-4
    far = ac.interpolated_search_points(np.array([70]), ball, FPS, 1.0)   # gap too long: nearest edge
    assert np.abs(far[0] - ball[59]).max() < 1e-6
    # ball never seen after the frame: the last seen position is held
    tail = ball.copy()
    tail[150:] = np.nan
    assert np.allclose(ac.interpolated_search_points(np.array([170]), tail, FPS, 20.0)[0], tail[149])


def test_pressure_centroid_pulls_the_search_point_towards_converging_players():
    n = 40
    base = np.tile([[50.0, 30.0]], (n, 1))
    players = np.full((n, 2, 2), np.nan, np.float32)
    players[:, 0] = [50.0, 30.0]                              # standing at the point
    t = np.arange(n)[:, None] / FPS
    players[:, 1] = np.array([56.0, 30.0]) + np.array([-5.0, 0.0]) * t      # running towards it
    p = dataclasses.replace(P, pressure_weight=1.0)
    out = ac.pressure_search_points(np.array([10, 20]), base[:2], players, FPS, p)
    assert np.all(out[:, 0] > 50.0)                           # pulled towards the approaching player
    far = np.full((n, 1, 2), 100.0, np.float32)
    kept = ac.pressure_search_points(np.array([10]), base[:1], far, FPS, p)
    assert np.allclose(kept, base[:1])                        # nobody in range: unchanged
    half = ac.pressure_search_points(np.array([10]), base[:1], players,
                                     FPS, dataclasses.replace(p, pressure_weight=0.0))
    assert np.allclose(half, base[:1])


def test_plan_crops_only_hidden_graded_frames_inside_the_image():
    ball, truth, players = _track()
    hom = synthetic_H(len(ball))
    plan = ac.plan_crops(ball, players, hom, [(0, 99), (100, 199)], fps=FPS, pitch=PITCH, params=P)
    assert len(plan) == len(range(61, 140, 3)) and plan.n_hidden == 80 and plan.n_gaps == 1
    assert plan.frames.min() >= 60 and plan.frames.max() < 140
    assert (np.diff(plan.frames) > 0).all()
    assert ((plan.origin >= 0).all() and (plan.origin[:, 0] + 640 <= 1920).all()
            and (plan.origin[:, 1] + 640 <= 1080).all())
    # the search point is within a few metres of the true ball (it moves at 4 m/s along a straight line)
    err = np.hypot(*(plan.search_xy - truth[plan.frames]).T)
    assert err.max() < 0.5
    # reprojecting the true ball's pixel position gives it back through the crop
    u, v = project_pitch_to_pixels(hom.H[plan.frames], truth[plan.frames, 0], truth[plan.frames, 1], PITCH)
    local = np.stack([u, v], 1) - plan.origin
    assert ((local >= 0) & (local < 640)).all()
    assert plan.head(100).frames.max() < 100 and plan.head(0).frames.size == 0


def test_plan_respects_every_k_max_crops_and_unusable_homographies():
    ball, _, players = _track()
    hom = synthetic_H(len(ball))
    many = ac.plan_crops(ball, players, hom, [(0, 199)], fps=FPS, pitch=PITCH, params=P, every_k=1)
    assert len(many) == 80
    capped = ac.plan_crops(ball, players, hom, [(0, 199)], fps=FPS, pitch=PITCH,
                           params=dataclasses.replace(P, max_crops=10), every_k=1)
    assert len(capped) == 10
    bad = dataclasses.replace(hom, reprojection_error=np.where(np.arange(len(ball)) < 100, 99.0, 0.5))
    part = ac.plan_crops(ball, players, bad, [(0, 199)], fps=FPS, pitch=PITCH, params=P, every_k=1)
    assert part.frames.min() >= 100
    nothing = ac.plan_crops(np.zeros((50, 2), np.float32), players[:50], hom, [(0, 49)], fps=FPS, pitch=PITCH,
                            params=P)
    assert len(nothing) == 0 and nothing.n_hidden == 0


# ----------------------------------------------------------------------------- selection / injection
def _ds(frames, xy, conf, search=None):
    frames = np.asarray(frames, np.int64)
    pf = np.unique(frames)
    if search is None:
        search = np.tile([[40.0, 34.0]], (len(pf), 1))
    return rec.DetectionSet(plan_frames=pf, plan_search_xy=search, cand_frame=frames,
                            cand_xy=np.asarray(xy, float), cand_conf=np.asarray(conf, float))


def test_selection_filters_confidence_pitch_distance_and_speed():
    ball, truth, _ = _track()
    p = dataclasses.replace(P, confidence=0.3)
    ds = _ds([70, 70, 80, 90, 100, 100, 110],
             [[40, 34], [41, 34], [41, 34], [-20, 34], [90, 34], [43.5, 34.0], [44, 34]],
             [0.9, 0.8, 0.2, 0.9, 0.9, 0.5, 0.9])
    f, xy, cf = rec.select_detections(ds, ball, fps=FPS, pitch=PITCH, params=p)
    # 80: low confidence; 90: outside the pitch; 100: (90, 34) is too far from the search point (40, 34)
    assert f.tolist() == [70, 100, 110]
    assert xy[0].tolist() == [40.0, 34.0] and abs(xy[1][0] - 43.5) < 1e-9 and cf[0] == 0.9


def test_selection_rejects_implausible_speed_against_the_seen_ball_and_the_next_point():
    ball, truth, _ = _track()                     # last seen ball (frame 59) near x = 21.4
    p = dataclasses.replace(P, max_search_dist_m=100.0, max_speed_mps=10.0)
    # 0.4 s after the last seen ball a point 30 m away is impossible; a point 2 m away is fine
    ds = _ds([70, 75], [[52.0, 34.0], [23.0, 34.0]], [0.9, 0.9])
    f, xy, _ = rec.select_detections(ds, ball, fps=FPS, pitch=PITCH, params=p)
    assert f.tolist() == [75]
    # fine for the previous seen ball but not for the next one (frame 140): dropped by the backward pass
    ds2 = _ds([138], [[60.0, 34.0]], [0.9])
    assert rec.select_detections(ds2, ball, fps=FPS, pitch=PITCH, params=p)[0].size == 0
    assert rec.select_detections(_ds([], np.zeros((0, 2)), []), ball, params=p)[0].size == 0


def test_injection_only_fills_missing_frames_flags_the_source_and_interpolates_short_gaps():
    ball, truth, _ = _track()
    new, src = rec.inject_points(ball, np.array([70, 73, 100]), truth[[70, 73, 100]], fps=FPS, params=P)
    assert np.array_equal(new[:60], ball[:60]) and np.array_equal(new[140:], ball[140:])
    assert (src[:60] == rec.SRC_TRACKING).all() and (src[140:] == rec.SRC_TRACKING).all()
    assert src[70] == src[73] == src[100] == rec.SRC_VISION
    assert (src[71:73] == rec.SRC_VISION_INTERP).all()                  # two detections 3 frames apart
    assert np.abs(new[71:73] - truth[71:73]).max() < 1e-3
    assert (src[60:70] == rec.SRC_VISION_INTERP).all()                  # seen ball (59) -> detection (70)
    assert src[90] == rec.SRC_MISSING and np.isnan(new[90]).all()       # 73 -> 100 is 27 frames > 25
    # a detection cannot overwrite a tracked frame
    new2, src2 = rec.inject_points(ball, np.array([10]), np.array([[99.0, 1.0]]), fps=FPS, params=P)
    assert np.array_equal(new2, ball, equal_nan=True) and (src2[10] == rec.SRC_TRACKING)


def test_injection_interpolates_from_the_seen_ball_to_the_first_detection():
    ball, truth, _ = _track()
    new, src = rec.inject_points(ball, np.array([66]), truth[[66]], fps=FPS, params=P)
    assert (src[60:66] == rec.SRC_VISION_INTERP).all() and src[66] == rec.SRC_VISION
    assert np.abs(new[60:66] - truth[60:66]).max() < 0.01
    # an implausibly fast jump is not interpolated
    jump = rec.inject_points(ball, np.array([62]), np.array([[90.0, 34.0]]), fps=FPS, params=P)[1]
    assert jump[62] == rec.SRC_VISION and (jump[60:62] == rec.SRC_MISSING).all()


# ----------------------------------------------------------------------------- detectors and cache
def test_dry_run_detector_returns_nothing_and_satisfies_the_protocol():
    d = det.DryRunDetector()
    assert isinstance(d, det.BallDetector)
    out = d.detect([np.zeros((640, 640, 3), np.uint8)] * 2)
    assert len(out) == 2 and all(o.shape == (0, 3) for o in out)
    assert d.fingerprint() == det.detector_fingerprint(dataclasses.replace(P, detector="dryrun"))
    with pytest.raises(ValueError):
        det.make_detector(dataclasses.replace(P, detector="nope"))
    assert isinstance(det.make_detector(dataclasses.replace(P, detector="dryrun")), det.DryRunDetector)


def _fake_ultralytics(monkeypatch, calls):
    class Boxes:
        def __init__(self, xywh, conf):
            self.xywh, self.conf = np.array(xywh, float), np.array(conf, float)

        def __len__(self):
            return len(self.conf)

    class Result:
        def __init__(self, boxes):
            self.boxes = boxes

    class YOLO:
        def __init__(self, weights):
            calls["weights"] = weights

        def predict(self, imgs, **kw):
            calls["kw"] = kw
            return [Result(Boxes([[10, 20, 5, 5], [30, 40, 6, 6]], [0.4, 0.9])) if i == 0
                    else Result(None) for i in range(len(imgs))]

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=YOLO))


def test_ultralytics_detector_asks_for_the_sports_ball_class_and_sorts_by_confidence(monkeypatch):
    calls: dict = {}
    _fake_ultralytics(monkeypatch, calls)
    d = det.make_detector(dataclasses.replace(P, detector="yolo", weights="yolo11n.pt", confidence=0.3))
    out = d.detect([np.zeros((640, 640, 3), np.uint8)] * 2)
    assert calls["weights"] == "yolo11n.pt" and calls["kw"]["classes"] == [32]
    assert calls["kw"]["conf"] == 0.3 and calls["kw"]["imgsz"] == 640
    assert out[0].tolist() == [[30.0, 40.0, 0.9], [10.0, 20.0, 0.4]] and out[1].shape == (0, 3)
    assert "yolo11n.pt" in d.fingerprint() and isinstance(d, det.BallDetector)


def test_ultralytics_missing_gives_a_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "ultralytics", None)
    with pytest.raises(ImportError, match="yolo"):
        det.UltralyticsYoloDetector(P)


def test_detection_cache_round_trip_and_key_sensitivity(tmp_path):
    video = tmp_path / "match.mp4"
    video.write_bytes(b"0123456789")
    ball, _, players = _track()
    hom = synthetic_H(len(ball))
    plan = ac.plan_crops(ball, players, hom, [(0, 199)], fps=FPS, pitch=PITCH, params=P)
    key = rec.cache_key(video, plan, "dryrun", P)
    ds = _ds([70, 73], [[40, 34], [41, 34]], [0.9, 0.8])
    path = rec.cache_path(dataclasses.replace(P, cache_dir=str(tmp_path / "c")), video, key)
    rec.save_detections(path, ds, key)
    back = rec.load_detections(path, key)
    assert back is not None and back.from_cache and np.array_equal(back.cand_frame, ds.cand_frame)
    assert rec.load_detections(path, "other") is None
    assert rec.load_detections(tmp_path / "missing.npz", key) is None
    assert rec.cache_key(video, plan, "yolo|x", P) != key                       # detector
    assert rec.cache_key(video, plan.head(100), "dryrun", P) != key             # plan
    video.write_bytes(b"0123456789ab")
    assert rec.cache_key(video, plan, "dryrun", P) != key                       # video size / mtime


def test_recover_ball_without_video_or_cache_returns_the_tracking(tmp_path):
    ball, _, players = _track()
    hom = synthetic_H(len(ball))
    p = dataclasses.replace(P, cache_dir=str(tmp_path / "c"))
    res = rec.recover_ball(ball, players, hom, [(0, 199)], tmp_path / "none.mp4", detector=None, params=p,
                           fps=FPS, pitch=PITCH)
    assert res.stats["status"] == "video not found" and np.array_equal(res.ball_xy, ball, equal_nan=True)
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    res = rec.recover_ball(ball, players, hom, [(0, 199)], video, detector=None, params=p, fps=FPS,
                           pitch=PITCH)
    assert res.stats["status"] == "no cache" and res.stats["frames_filled"] == 0


# ----------------------------------------------------------------------------- synthetic video end to end
def _write_video(path: Path, truth: np.ndarray, hom: Homographies, n: int, radius: int = 7):
    import cv2

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (1920, 1080))
    assert writer.isOpened()
    u, v = project_pitch_to_pixels(hom.H[:n], truth[:n, 0], truth[:n, 1], PITCH)
    for i in range(n):
        img = np.zeros((1080, 1920, 3), np.uint8)
        img[:] = (40, 120, 40)                                         # green pitch (BGR)
        cv2.circle(img, (int(round(u[i])), int(round(v[i]))), radius, (255, 255, 255), -1)
        writer.write(img)
    writer.release()


@pytest.fixture(scope="module")
def synthetic_video(tmp_path_factory):
    pytest.importorskip("cv2")
    n = 80
    t = np.arange(n) / FPS
    truth = np.stack([30.0 + 5.0 * t, 20.0 + 8.0 * t], axis=1).astype(np.float32)     # a diagonal run
    ball = truth.copy()
    ball[20:60] = np.nan
    players = np.zeros((n, 4, 2), np.float32)
    for i in range(4):
        players[:, i] = truth + np.array([(i - 1.5) * 4.0, 3.0])
    hom = synthetic_H(n)
    path = tmp_path_factory.mktemp("video") / "synthetic_tactical.mp4"
    _write_video(path, truth, hom, n)
    return {"path": path, "truth": truth, "ball": ball, "players": players, "hom": hom, "n": n}


@needs_cv2
def test_sequential_reader_returns_the_planned_frames_in_order_with_and_without_seeking(synthetic_video):
    sv = synthetic_video
    plan = ac.plan_crops(sv["ball"], sv["players"], sv["hom"], [(0, sv["n"] - 1)], fps=FPS, pitch=PITCH,
                         params=dataclasses.replace(P, every_k=4))
    for seek in (1000, 3):
        p = dataclasses.replace(P, seek_threshold_frames=seek, batch_size=4)
        got, shapes = [], set()
        for idx, crops, _ in rec.read_crops(sv["path"], plan, p):
            got += idx
            shapes |= {c.shape for c in crops}
        assert got == list(range(len(plan))) and shapes == {(640, 640, 3)}


@needs_cv2
def test_hough_detector_finds_the_synthetic_ball_in_a_crop(synthetic_video):
    sv = synthetic_video
    plan = ac.plan_crops(sv["ball"], sv["players"], sv["hom"], [(0, sv["n"] - 1)], fps=FPS, pitch=PITCH,
                         params=P, every_k=10)
    crops = next(rec.read_crops(sv["path"], plan, dataclasses.replace(P, batch_size=1)))[1]
    out = det.HoughCircleDetector(P).detect(crops)[0]
    assert len(out) >= 1 and out[0][2] > 0.6
    f = plan.frames[0]
    u, v = project_pitch_to_pixels(sv["hom"].H[f], sv["truth"][f, 0], sv["truth"][f, 1], PITCH)
    assert abs(out[0][0] + plan.origin[0][0] - u) < 3 and abs(out[0][1] + plan.origin[0][1] - v) < 3
    blank = np.full((640, 640, 3), (40, 120, 40), np.uint8)
    assert det.HoughCircleDetector(P).detect([blank])[0].shape == (0, 3)


@needs_cv2
def test_full_chain_crop_detect_reproject_inject_recovers_the_ball_within_a_metre(synthetic_video, tmp_path):
    sv = synthetic_video
    p = dataclasses.replace(P, every_k=3, cache_dir=str(tmp_path / "vision"), batch_size=8)
    detector = det.HoughCircleDetector(p)
    res = rec.recover_ball(sv["ball"], sv["players"], sv["hom"], [(0, sv["n"] - 1)], sv["path"],
                           detector=detector, params=p, fps=FPS, pitch=PITCH)
    s = res.stats
    assert s["status"] == "ok" and not s["from_cache"] and s["crops_scanned"] == s["crops_planned"] > 10
    filled = (res.ball_source == rec.SRC_VISION) | (res.ball_source == rec.SRC_VISION_INTERP)
    assert s["frames_filled"] == int(filled.sum()) > 30 and s["fill_rate"] > 0.7 and s["gap_fill_rate"] == 1.0
    err = np.hypot(*(res.ball_xy[filled] - sv["truth"][filled]).T)
    assert err.max() < 1.0 and np.median(err) < 0.3                        # within ~1 m of the known path
    assert np.array_equal(res.ball_xy[:20], sv["ball"][:20])                # tracking untouched
    assert (res.ball_source[:20] == rec.SRC_TRACKING).all()
    # second run: cache only (no detector, no video decoding)
    again = rec.recover_ball(sv["ball"], sv["players"], sv["hom"], [(0, sv["n"] - 1)], sv["path"],
                             detector=None, params=p, fps=FPS, pitch=PITCH)
    assert again.stats["from_cache"] and np.array_equal(again.ball_xy, res.ball_xy, equal_nan=True)
    assert any((tmp_path / "vision").glob("*.npz"))
    # a looser plan (different k) is a different cache entry: nothing cached yet
    other = rec.recover_ball(sv["ball"], sv["players"], sv["hom"], [(0, sv["n"] - 1)], sv["path"],
                             detector=None, params=p, fps=FPS, pitch=PITCH, every_k=7)
    assert other.stats["status"] == "no cache"


@needs_cv2
def test_dry_run_chain_changes_nothing_but_reports_the_scan(synthetic_video, tmp_path):
    sv = synthetic_video
    p = dataclasses.replace(P, detector="dryrun", cache_dir=str(tmp_path / "v"))
    res = rec.recover_ball(sv["ball"], sv["players"], sv["hom"], [(0, sv["n"] - 1)], sv["path"],
                           detector=det.DryRunDetector(), params=p, fps=FPS, pitch=PITCH, max_frames=50)
    assert np.array_equal(res.ball_xy, sv["ball"], equal_nan=True)
    assert res.stats["crops_scanned"] > 0 and res.stats["detections_kept"] == 0
    assert res.stats["fill_rate"] == 0.0 and res.stats["video_frames_decoded"] <= 50


@needs_cv2
def test_reader_rejects_a_video_of_the_wrong_size(tmp_path):
    import cv2

    path = tmp_path / "small.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (320, 240))
    for _ in range(5):
        w.write(np.zeros((240, 320, 3), np.uint8))
    w.release()
    plan = ac.CropPlan(np.array([1]), np.zeros((1, 2)), np.zeros((1, 2)), np.zeros((1, 2), np.int64),
                       np.zeros(1, np.int64), 1, 1)
    with pytest.raises(rec.VisionUnavailableError, match="320x240"):
        list(rec.read_crops(path, plan, P))


# ----------------------------------------------------------------------------- pipeline / dashboard / CLI
def _cfg_with(tmp_path, **vision):
    import yaml

    from mtah2027.config import DEFAULT_CONFIG_PATH

    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["vision"].update(vision)
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def _synthetic_blocks(spec):
    from mtah2027.config import MatchConfig

    return MatchConfig(n_video_frames=spec.n_frames, half1=spec.half1, half2_dead=spec.half2_dead,
                       half2_play=spec.half2_play)


def test_vision_is_off_by_default_and_the_pipeline_ignores_it(synthetic_dir, synthetic_spec):
    from mtah2027.pipeline import run_pipeline

    res = run_pipeline(synthetic_dir, blocks=_synthetic_blocks(synthetic_spec), log=lambda m: None)
    assert res.diagnostics["vision"] is None
    assert res.diagnostics["loader_stages"]["vision_recovery"]["status"].startswith("skipped")


def test_pipeline_merges_cached_detections_only_when_enabled(synthetic_dir, synthetic_spec, tmp_path):
    import games
    from mtah2027.io.atd_loader import load_atd
    from mtah2027.io.homography_loader import load_homography
    from mtah2027.pipeline import _fit_length, run_pipeline

    blocks = _synthetic_blocks(synthetic_spec)
    n = blocks.n_video_frames
    video = Path(games.asset_paths(data_dir=str(synthetic_dir))["video"])
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"not a real video: only its size and mtime key the cache")
    cfg = _cfg_with(tmp_path, enabled=True, cache_dir=str(tmp_path / "vcache"), detector="dryrun")
    # enabled but nothing cached: the ball is unchanged and the status says so
    none = run_pipeline(synthetic_dir, cfg, blocks, log=lambda m: None)
    assert none.diagnostics["vision"]["status"] == "no cache"
    assert none.diagnostics["vision"]["frames_filled"] == 0
    # fabricate the cache the CLI would write: a candidate right on the search point of every planned crop
    vp = ac.VisionParams.from_config(cfg)
    td = load_atd(synthetic_dir, n_frames=n)
    hom = load_homography(synthetic_dir)
    ball, players = _fit_length(td.ball_xy, n), _fit_length(td.players_xy, n)
    plan = ac.plan_crops(ball, players, hom, [blocks.half1, blocks.half2_play], fps=float(td.fps),
                         pitch=tuple(td.pitch), params=vp)
    assert len(plan) > 0
    key = rec.cache_key(video, plan, "dryrun", vp)
    ds = rec.DetectionSet(plan.frames, plan.search_xy, plan.frames, plan.search_xy, np.full(len(plan), 0.9))
    rec.save_detections(rec.cache_path(vp, video, key), ds, key)
    res = run_pipeline(synthetic_dir, cfg, blocks, log=lambda m: None)
    v = res.diagnostics["vision"]
    assert v["status"] == "ok" and v["from_cache"] and v["frames_filled"] > 0 and 0 < v["fill_rate"] <= 1
    assert len(res.labels) == n
    off = run_pipeline(synthetic_dir, _cfg_with(tmp_path, enabled=False), blocks, log=lambda m: None)
    assert off.diagnostics["vision"] is None


def test_dashboard_reports_the_vision_fill_rate_only_when_present():
    from mtah2027.validation import kpi_dashboard as kd

    src = Path(kd.__file__).read_text(encoding="utf-8")
    assert '"vision" in report' in src and 'report["vision"]' in src


@needs_cv2
def test_cli_benchmark_and_dry_run_print_only_aggregates(synthetic_dir, synthetic_spec, tmp_path, capsys):
    import cv2
    import run_vision_recovery as cli
    import yaml

    import games
    from mtah2027.config import DEFAULT_CONFIG_PATH

    spec = synthetic_spec
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["match"].update({"n_video_frames": spec.n_frames, "half1": list(spec.half1),
                         "half2_dead": list(spec.half2_dead), "half2_play": list(spec.half2_play)})
    raw["vision"].update({"cache_dir": str(tmp_path / "vc")})
    cfg = tmp_path / "cli.yaml"
    cfg.write_text(yaml.safe_dump(raw), encoding="utf-8")
    video = Path(games.asset_paths(data_dir=str(synthetic_dir))["video"])
    video.parent.mkdir(parents=True, exist_ok=True)
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (1920, 1080))
    for _ in range(40):
        w.write(np.full((1080, 1920, 3), (40, 120, 40), np.uint8))
    w.release()
    assert cli.main(["--data-dir", str(synthetic_dir), "--config", str(cfg), "--detector", "hough",
                     "--benchmark", "30"]) == 0
    out = capsys.readouterr().out
    assert "benchmark detector=hough" in out and "estimated full run" in out
    assert cli.main(["--data-dir", str(synthetic_dir), "--config", str(cfg), "--detector", "dryrun",
                     "--max-frames", "40"]) == 0
    out = capsys.readouterr().out
    assert "frames scanned" in out and "fill" in out.lower() and "detections kept: 0" in out
    assert len(out.splitlines()) <= 6                       # aggregates, never per-frame rows
    video.unlink()
