"""Issue #36: multi-frame motion ball detector and audit-video overlay (all synthetic, no real data)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from test_11_attention_crops import FPS, PITCH, synthetic_H

from mtah2027.config import MatchConfig
from mtah2027.vision import attention_crops as ac
from mtah2027.vision import detectors as det
from mtah2027.vision import overlay_renderer as ovl
from mtah2027.vision import recovery as rec

needs_cv2 = pytest.mark.skipif(importlib.util.find_spec("cv2") is None, reason="OpenCV not installed")
BALL_R = 6
# the synthetic homography covers ~0.07 m per pixel: a 13 px ball is 0.9 m "wide" there
P = ac.VisionParams(detector="motion", ball_diameter_m=0.9)


# ----------------------------------------------------------------------------- synthetic crops
def _scene(pan: int, ball=None, distractor=None, size: int = 640, noise_seed: int = 0):
    """Green pitch, static white lines + centre circle + static 'boots' (all shifted by the camera pan)."""
    import cv2

    rng = np.random.default_rng(noise_seed)
    img = np.full((size, size, 3), (40, 120, 40), np.uint8)
    for x in (100, 330, 560):
        cv2.line(img, (x + pan, 0), (x + pan, size - 1), (255, 255, 255), 3)
    cv2.circle(img, (320 + pan, 320), 60, (255, 255, 255), 3)
    for bx, by in ((100, 200), (400, 300), (250, 500), (470, 120)):
        cv2.circle(img, (bx + pan, by), 6, (235, 235, 235), -1)         # static boots, ball-sized and bright
    if distractor is not None:                                             # a big moving player blob
        cv2.rectangle(img, (int(distractor[0]) - 14, int(distractor[1]) - 30),
                      (int(distractor[0]) + 14, int(distractor[1]) + 30), (200, 60, 60), -1)
    if ball is not None:
        cv2.circle(img, (int(round(ball[0])), int(round(ball[1]))), BALL_R, (255, 255, 255), -1)
    noise = rng.integers(-2, 3, img.shape)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _triplet(ball, vel, pan_step=3, distractor=None, distractor_vel=(0.0, 0.0)):
    b = np.asarray(ball, dtype=float)
    v = np.asarray(vel, dtype=float)
    dv = np.asarray(distractor_vel, dtype=float)
    frames = []
    for k, t in enumerate((-1, 0, 1)):
        d = None if distractor is None else np.asarray(distractor, float) + t * dv
        frames.append(_scene(pan_step * t, b + t * v, d, noise_seed=k))
    return np.stack(frames)


@needs_cv2
def test_motion_detector_localises_the_ball_and_rejects_static_distractors_under_a_camera_pan():
    d = det.MultiFrameMotionBallDetector(P)
    assert d.temporal and d.name == "motion" and isinstance(d, det.BallDetector)
    errs = []
    spots = [(200.3, 150.6), (450.7, 200.2), (250.5, 450.4), (480.2, 420.9), (150.8, 330.1), (600.4, 520.7),
             (420.6, 570.3), (230.1, 60.8), (500.9, 60.2), (400.4, 500.5), (60.7, 300.6), (215.5, 205.5)]
    for k, spot in enumerate(spots):
        ball = np.array(spot)
        out = d.detect([_triplet(ball, (9.0, 6.0))])[0]
        assert len(out) >= 1, k
        errs.append(np.hypot(*(out[0, :2] - ball)))
        assert out[0, 2] >= 0.25                                      # above the selection confidence
        assert len(out) == 1                                          # boots / lines / circle did not survive
    assert max(errs) < 1.5 and float(np.mean(errs)) < 1.0            # sub-pixel refinement, rounded drawing


@needs_cv2
def test_motion_detector_finds_nothing_in_a_static_scene_or_a_single_frame_or_a_big_mover():
    d = det.MultiFrameMotionBallDetector(P)
    static = _triplet((0, 0), (0, 0))                                 # ball drawn at (0,0), nothing moves
    static[:] = np.stack([_scene(0, None, noise_seed=k) for k in range(3)])
    assert len(d.detect([static])[0]) == 0
    panned = np.stack([_scene(3 * t, None, noise_seed=k) for k, t in enumerate((-1, 0, 1))])
    assert len(d.detect([panned])[0]) == 0                            # pure camera pan: no motion left
    assert len(d.detect([_scene(0, (200, 200))])[0]) == 0             # one frame has no temporal cue
    big = np.stack([_scene(3 * t, None, (300 + 12 * t, 300 + 8 * t), noise_seed=k)
                    for k, t in enumerate((-1, 0, 1))])
    assert len(d.detect([big])[0]) == 0                               # a player-sized blob is not a ball


@needs_cv2
def test_motion_detector_with_the_ball_among_a_moving_player_and_a_subpixel_pan():
    import cv2

    d = det.MultiFrameMotionBallDetector(P)
    ball = np.array([250.0, 400.0])
    trip = _triplet(ball, (10.0, -7.0), distractor=(450, 180), distractor_vel=(8, 5))
    # sub-pixel camera pan: shift frames t-1 and t+1 by +-0.6 px more with an interpolating warp
    for k, dx in ((0, -0.6), (2, 0.6)):
        M = np.array([[1, 0, dx], [0, 1, 0.3 * dx]], dtype=np.float64)
        trip[k] = cv2.warpAffine(trip[k], M, (640, 640), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
    out = d.detect([trip])[0]
    assert len(out) >= 1 and np.hypot(*(out[0, :2] - ball)) < 2.5


@needs_cv2
def test_geometry_gives_the_apparent_size_and_a_metric_speed_check():
    hom = synthetic_H(3)
    geo = det.motion_geometry((640, 220), hom.H_inv[:3], pitch=PITCH, fps=FPS)
    d = det.MultiFrameMotionBallDetector(P)
    d_px, mpp = d._ball_scale(geo, 640)
    assert 0.03 < mpp < 0.2 and abs(d_px - 0.9 / mpp) < 1e-6
    assert d._speed_ok(geo, (300.0, 300.0), (312.0, 306.0), 0, FPS)          # ~1 m in 1/25 s: 25 m/s
    assert not d._speed_ok(geo, (300.0, 300.0), (600.0, 600.0), 2, FPS)      # tens of metres: impossible
    assert det.motion_geometry((0, 0), np.full((3, 3, 3), np.nan)) is None
    out = d.detect([_triplet((300.0, 250.0), (10.0, 5.0))], [geo])[0]
    assert len(out) == 1 and np.hypot(out[0, 0] - 300.0, out[0, 1] - 250.0) < 1.5


def test_make_detector_motion_fingerprint_and_default_off():
    assert ac.VisionParams().enabled is False and ac.VisionParams().detector == "hough"
    if importlib.util.find_spec("cv2") is not None:
        assert isinstance(det.make_detector(P), det.MultiFrameMotionBallDetector)
    fp = det.detector_fingerprint(P)
    assert fp.startswith("motion|") and fp != det.detector_fingerprint(ac.VisionParams(detector="hough"))
    assert fp != det.detector_fingerprint(ac.VisionParams(detector="motion", motion_threshold=30.0))
    with pytest.raises(ValueError):
        det.detector_fingerprint(ac.VisionParams(detector="nope"))


# ------------------------------------------------------------------- triplet reading + recovery chain
@pytest.fixture(scope="module")
def pan_video(tmp_path_factory):
    """80 frames 1920x1080: a ball (20 m/s, ~11 px/frame) over static lines and boots (constant H)."""
    import cv2

    from mtah2027.io.homography_loader import project_pitch_to_pixels

    n = 80
    t = np.arange(n) / FPS
    truth = np.stack([15.0 + 16.0 * t, 10.0 + 12.0 * t], axis=1).astype(np.float32)
    hom = synthetic_H(n)
    u, v = project_pitch_to_pixels(hom.H[:n], truth[:, 0], truth[:, 1], PITCH)
    path = tmp_path_factory.mktemp("mvideo") / "motion_tactical.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (1920, 1080))
    assert writer.isOpened()
    base = np.full((1080, 1920, 3), (40, 120, 40), np.uint8)
    for x in range(200, 1800, 220):
        cv2.line(base, (x, 250), (x, 1050), (255, 255, 255), 3)
    for bx, by in ((700, 500), (900, 600), (1100, 450), (800, 700), (1000, 800)):
        cv2.circle(base, (bx, by), 6, (235, 235, 235), -1)                       # static boots
    for i in range(n):
        img = base.copy()
        cv2.circle(img, (int(round(u[i])), int(round(v[i]))), 3, (255, 255, 255), -1)      # ~0.45 m
        writer.write(img)
    writer.release()
    return {"path": path, "truth": truth, "hom": hom, "n": n}


@needs_cv2
def test_reader_gives_frame_triplets_with_the_same_window_and_clamps_at_the_video_ends(pan_video):
    import cv2

    n = pan_video["n"]
    frames = np.array([0, 10, 40, n - 1], dtype=np.int64)
    org = np.tile([[600, 300]], (len(frames), 1)).astype(np.int64)
    plan = ac.CropPlan(frames, np.zeros((4, 2)), np.zeros((4, 2)), org, np.zeros(4, np.int64), 0, 0)
    got = [c for _, crops, _ in rec.read_crops(pan_video["path"], plan, P, context=1) for c in crops]
    assert len(got) == 4 and all(c.shape == (3, 640, 640, 3) for c in got)
    single = [c for _, crops, _ in rec.read_crops(pan_video["path"], plan, P) for c in crops]
    assert all(s.shape == (640, 640, 3) for s in single)
    for trip, cur in zip(got, single):
        assert np.array_equal(trip[1], cur)                          # the middle frame is the planned one
    assert np.array_equal(got[0][0], got[0][1])                      # frame 0: no previous frame, repeated
    assert np.array_equal(got[3][2], got[3][1])                      # last frame: no next frame, repeated
    cap = cv2.VideoCapture(str(pan_video["path"]))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 9)
    ok, f9 = cap.read()
    cap.release()
    assert ok and np.array_equal(got[1][0], f9[300:940, 600:1240])   # t-1 really is the previous frame


@needs_cv2
def test_recovery_chain_with_the_motion_detector_recovers_the_ball_within_a_metre(pan_video, tmp_path):
    truth, hom, n = pan_video["truth"], pan_video["hom"], pan_video["n"]
    ball = truth.copy()
    ball[20:60] = np.nan
    players = np.zeros((n, 4, 2), np.float32)
    for i in range(4):
        players[:, i] = truth + np.array([(i - 1.5) * 4.0, 3.0])
    p = ac.VisionParams(detector="motion", ball_diameter_m=0.45, every_k=4, pressure_weight=0.0,
                        cache_dir=str(tmp_path / "vc"))
    detector = det.make_detector(p)
    plan = ac.plan_crops(ball, players, hom, [(0, n - 1)], fps=FPS, pitch=PITCH, params=p)
    assert len(plan) >= 8
    ds = rec.run_detection(pan_video["path"], plan, detector, hom, p, pitch=PITCH, fps=FPS)
    assert ds.n_scanned == len(plan) and len(ds.cand_frame) >= len(plan) // 2
    frames, xy, _ = rec.select_detections(ds, ball, fps=FPS, pitch=PITCH, params=p)
    assert len(frames) >= len(plan) // 2
    err = np.hypot(*(xy - truth[frames]).T)
    assert float(np.median(err)) < 0.5 and float(err.max()) < 1.0
    res = rec.recover_ball(ball, players, hom, [(0, n - 1)], pan_video["path"], detector=detector, params=p,
                           fps=FPS, pitch=PITCH, every_k=4, cache_root=tmp_path)
    assert res.stats["status"] == "ok" and res.stats["frames_filled"] > 0
    again = rec.recover_ball(ball, players, hom, [(0, n - 1)], pan_video["path"], detector=None, params=p,
                             fps=FPS, pitch=PITCH, every_k=4, cache_root=tmp_path)
    assert again.stats["from_cache"] and np.array_equal(again.ball_xy, res.ball_xy, equal_nan=True)


# ----------------------------------------------------------------------------- overlay: pure NumPy parts
def test_probabilities_are_a_softmax_of_the_log_emissions():
    lp = np.log(np.array([[0.2, 0.5, 0.3], [0.9, 0.05, 0.05]]))
    pr = ovl.probabilities_from_log(lp)
    assert np.allclose(pr.sum(axis=1), 1.0, atol=1e-6) and np.allclose(pr[0], [0.2, 0.5, 0.3], atol=1e-6)
    odd = ovl.probabilities_from_log(np.array([[np.nan, 0.0, -np.inf]]))
    assert np.isfinite(odd).all() and odd[0, 1] > 0.99


def test_pitch_polylines_cover_the_standard_markings():
    lines = ovl.pitch_polylines(PITCH)
    assert len(lines) == 2 + 1 + 2 * 4
    allpts = np.concatenate(lines)
    assert allpts[:, 0].min() >= -1e-9 and allpts[:, 0].max() <= PITCH[0] + 1e-9
    assert allpts[:, 1].min() >= -1e-9 and allpts[:, 1].max() <= PITCH[1] + 1e-9
    assert np.allclose(lines[0][0], lines[0][-1])                    # closed outline


def _toy(n=60, labels=None):
    s = 6
    team = np.array([0, 0, 0, 1, 1, 1])
    t = np.arange(n)[:, None] / FPS
    players = np.zeros((n, s, 2), np.float32)
    for i in range(3):                                                # home runs together towards +x
        players[:, i] = np.stack([30 + 5 * t[:, 0] + i, np.full(n, 20 + 8.0 * i)], axis=1)
    for i in range(3, 6):                                             # away stands still
        players[:, i] = [70.0 + i, 25.0 + 4.0 * (i - 3)]
    ball = players[:, 1].copy() + np.array([0.5, 0.0], np.float32)
    ball[40:50] = np.nan
    lab = np.zeros(n, np.int64) if labels is None else labels
    return players, team, ball, lab


def test_possessor_is_the_nearest_player_of_the_predicted_team_only_with_visible_ball():
    players, team, ball, lab = _toy()
    poss = ovl.possessor_slots(players, team, ball, lab)
    assert (poss[:40] == 1).all() and (poss[40:50] == -1).all() and (poss[50:] == 1).all()
    lab_a = np.ones(len(lab), np.int64)                               # predicted team A: nearest AWAY player
    pa = ovl.possessor_slots(players, team, ball, lab_a, max_dist_m=60.0)
    assert set(pa[:40]) <= {3, 4, 5} and pa[0] >= 3
    assert (ovl.possessor_slots(players, team, ball, np.full(len(lab), 2)) == -1).all()    # dead ball: nobody


def test_team_coherence_is_one_for_a_team_running_together_and_zero_when_standing():
    players, team, ball, lab = _toy()
    r, vec, cen = ovl.team_coherence(players, team, FPS)
    assert (r[5:-5, 0] > 0.99).all() and (r[:, 1] < 1e-6).all()
    assert (vec[10, 0, 0] > 0.99) and abs(vec[10, 0, 1]) < 1e-3
    assert np.allclose(cen[0, 1], players[0, 3:].mean(axis=0), atol=1e-4)


def test_audit_windows_merge_overlaps_and_clip_to_the_match():
    assert ovl.audit_windows([100, 120, 500, 3], 600, 25) == [(0, 28), (75, 145), (475, 525)]
    assert ovl.audit_windows([590], 600, 25) == [(565, 599)]
    assert ovl.audit_windows([], 600, 25) == []


# ----------------------------------------------------------------------------- overlay: drawing and script
@needs_cv2
def test_render_frame_draws_lines_possessor_halo_minimap_and_hud():
    players, team, ball, lab = _toy()
    hom = synthetic_H(len(lab))
    probs = np.tile([0.7, 0.2, 0.1], (len(lab), 1)).astype(np.float32)
    data = ovl.AuditData(players, team, ball, lab, H=hom.H, probs=probs, pitch=PITCH, fps=FPS)
    img = ovl.render_frame(data, 30)
    assert img.shape == (1080, 1920, 3) and img.dtype == np.uint8
    bg = ovl.blank_background((1920, 1080))
    assert (img != bg).any(axis=2).mean() > 0.02                      # something was drawn
    halo = np.all(img == ovl.COL_HALO, axis=2)
    assert halo.sum() > 50                                            # halo ring of the possessor
    white = np.all(img == ovl.COL_LINES, axis=2).sum()
    assert white > 2000                                               # projected pitch lines
    hidden = ovl.render_frame(data, 45)                               # ball hidden: no halo, still renders
    assert np.all(hidden == ovl.COL_HALO, axis=2).sum() < halo.sum() // 4
    no_h = ovl.AuditData(players, team, ball, lab, H=None, probs=None, pitch=PITCH, fps=FPS)
    assert ovl.render_frame(no_h, 10).shape == (1080, 1920, 3)        # without H or P: minimap + HUD only
    assert np.array_equal(ovl.render_frame(data, 30), img)            # deterministic


@needs_cv2
def test_render_video_writes_the_requested_frames_over_a_real_background(pan_video, tmp_path):
    import cv2

    players, team, ball, lab = _toy(n=pan_video["n"])
    data = ovl.AuditData(players, team, ball, lab, H=pan_video["hom"].H, pitch=PITCH, fps=FPS)
    out = tmp_path / "sub" / "audit.mp4"
    stats = ovl.render_video(data, [(5, 14), (30, 34)], out, video_path=pan_video["path"], scale=0.25)
    assert stats["frames_written"] == 15 and stats["clips"] == 2 and stats["background"] == "video"
    assert (stats["width"], stats["height"]) == (480, 270)
    cap = cv2.VideoCapture(str(out))
    assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 15
    cap.release()
    capped = ovl.render_video(data, [(0, 40)], tmp_path / "cap.mp4", scale=0.25, max_frames=7)
    assert capped["frames_written"] == 7 and capped["background"] == "synthetic pitch"


@pytest.fixture(scope="module")
def audit_run(synthetic_dir, synthetic_spec):
    import render_audit_video as script

    match = MatchConfig(n_video_frames=synthetic_spec.n_frames, half1=synthetic_spec.half1,
                        half2_dead=synthetic_spec.half2_dead, half2_play=synthetic_spec.half2_play)
    data, video = script.build_audit_data(synthetic_dir, match)
    return {"script": script, "match": match, "data": data, "video": video}


def test_pipeline_keep_features_exposes_probabilities_without_changing_the_labels(
        audit_run, synthetic_dir):
    from mtah2027.pipeline import run_pipeline

    data, match = audit_run["data"], audit_run["match"]
    plain = run_pipeline(synthetic_dir, blocks=match, log=lambda m: None)
    assert plain.features == {}
    assert np.array_equal(plain.labels, data.labels)                 # the audit shows the real submission
    assert data.probs is not None and data.probs.shape == (match.n_video_frames, 3)
    assert np.allclose(data.probs.sum(axis=1), 1.0, atol=1e-5)
    assert data.H is not None and data.H.shape == (match.n_video_frames, 3, 3)
    g = match.half1[0] + 500
    assert data.possessor[g] >= 0 or data.labels[g] == 2 or not data.ball_visible[g]


@needs_cv2
def test_cli_renders_a_window_and_the_audit_csv_and_prints_aggregates_only(
        audit_run, synthetic_dir, synthetic_spec, tmp_path, capsys, monkeypatch):
    script = audit_run["script"]
    # the pipeline run itself is covered by the test above: reuse its result to keep this test fast
    monkeypatch.setattr(script, "build_audit_data", lambda *a, **k: (audit_run["data"], audit_run["video"]))
    a = synthetic_spec.half1[0] + 200
    out = tmp_path / "window.mp4"
    rc = script.main(["--data-dir", str(synthetic_dir), "--start-frame", str(a), "--end-frame", str(a + 9),
                      "--out", str(out), "--scale", "0.25"])
    text = capsys.readouterr().out
    assert rc == 0 and out.is_file() and "10 frames" in text
    assert "git-ignored" in text and "P(H)" not in text              # no per-frame rows on stdout
    lab = audit_run["data"].labels
    changes = np.flatnonzero(lab[1:] != lab[:-1])[:3] + 1
    csv_path = tmp_path / "audit_transitions.csv"
    csv_path.write_text("frame,time_s,clock,from,to\n" + "".join(
        f"{int(f)},0,00:00,H,A\n" for f in changes), encoding="utf-8")
    assert script.read_audit_csv(csv_path) == [int(f) for f in changes]
    out2 = tmp_path / "audit.mp4"
    rc = script.main(["--data-dir", str(synthetic_dir), "--from-audit-csv", str(csv_path), "--pad-seconds",
                      "0.2", "--out", str(out2), "--scale", "0.25"])
    assert rc == 0 and out2.is_file()
    assert "clip(s)" in capsys.readouterr().out


def test_gitignore_blocks_videos_and_outputs():
    root = Path(__file__).resolve().parents[1]
    text = (root / ".gitignore").read_text(encoding="utf-8")
    assert "*.mp4" in text and "outputs/" in text
