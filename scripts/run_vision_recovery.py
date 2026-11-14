#!/usr/bin/env python
"""Recover the ball in the tactical video where the tracking lost it (Issue #11).  Prints aggregates only.

Reads the tracking, the tactical homographies and the tactical mp4 of ``--data-dir``, plans one 640x640 crop
every ``--every-k`` frames of each ball-missing gap, detects the ball in the crops, reprojects the
detections to pitch metres, and stores the raw detections in ``cache/vision/*.npz`` (git-ignored).  Afterwards
``run_pipeline`` merges them when ``vision.enabled`` is true in the configuration:

    pip install -e ".[vision]"                       # OpenCV (+ ultralytics for --detector yolo)
    python scripts/run_vision_recovery.py --data-dir data --benchmark 2000      # time the first 2000 frames
    python scripts/run_vision_recovery.py --data-dir data --detector yolo --weights yolov8n.pt
    # then set  vision.enabled: true  in configs/default.yaml and run generate_submission.py as usual

``--max-frames N`` only scans the crops of the first N video frames (a partial run; its cache is a separate
entry).  ``--detector dryrun`` runs the whole chain without any model (tracking-only).  The numbers
printed are counts, rates and timings: no frame, position or crop is ever printed or written (only the
cache file).
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT / "src", REPO_ROOT):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from mtah2027.config import MatchConfig, load_section, params_from_mapping  # noqa: E402
from mtah2027.io.homography_loader import project_pitch_to_pixels  # noqa: E402
from mtah2027.pipeline import _fit_length, resolve_config  # noqa: E402
from mtah2027.vision import attention_crops as ac  # noqa: E402
from mtah2027.vision import recovery as rec  # noqa: E402
from mtah2027.vision.detectors import make_detector  # noqa: E402


def load_inputs(data_dir, cfg_path):
    """Tracking, homographies and video path of the tactical source, with the arrays fitted to the match."""
    import games
    from mtah2027.io.atd_loader import load_atd
    from mtah2027.io.homography_loader import load_homography

    match = params_from_mapping(MatchConfig, load_section("match", cfg_path), "match")
    n = match.n_video_frames
    td = load_atd(data_dir, n_frames=n)
    hom = load_homography(data_dir)
    video = Path(games.asset_paths(data_dir=str(data_dir) if data_dir else None)["video"])
    blocks = [match.half1, match.half2_play]
    return (_fit_length(td.ball_xy, n), _fit_length(td.players_xy, n), hom, blocks, float(td.fps),
            tuple(td.pitch), video)


def benchmark(ball, players, hom, blocks, video, detector, params, fps, pitch, n_frames: int) -> dict:
    """Time reading + detection on the first ``n_frames`` video frames (crop around the search point if the
    frame is ball-missing, around the pitch centre otherwise) and extrapolate to the full plan."""
    full = ac.plan_crops(ball, players, hom, blocks, fps=fps, pitch=pitch, params=params)
    frames = np.arange(min(n_frames, len(ball)), dtype=np.int64)
    centre = np.array([[pitch[0] / 2.0, pitch[1] / 2.0]])
    search = np.tile(centre, (len(frames), 1))
    at = np.searchsorted(full.frames, frames)
    hit = (at < len(full)) & (full.frames[np.clip(at, 0, max(len(full) - 1, 0))] == frames)
    if hit.any():
        search[hit] = full.search_xy[at[hit]]
    u, v = project_pitch_to_pixels(hom.H[frames], search[:, 0], search[:, 1], pitch)
    ok = np.isfinite(u) & np.isfinite(v)
    frames, u, v, search = frames[ok], u[ok], v[ok], search[ok]
    plan = ac.CropPlan(frames=frames, search_xy=search, center_uv=np.stack([u, v], 1),
                       origin=ac.crop_window(u, v, params.crop_size, params.frame_width, params.frame_height),
                       gap_id=np.zeros(len(frames), np.int64), n_hidden=0, n_gaps=0)
    t0 = time.perf_counter()
    ds = rec.run_detection(video, plan, detector, hom, params, pitch=pitch, fps=fps)
    wall = time.perf_counter() - t0
    scanned = max(ds.n_scanned, 1)
    per_crop = ds.seconds_detect / scanned
    decode_fps = ds.video_frames_decoded / max(ds.seconds_read, 1e-9)
    last = int(full.frames.max()) + 1 if len(full) else 0
    est_s = len(full) * per_crop + last / max(decode_fps, 1e-9)
    return {"frames_scanned": ds.n_scanned, "candidates": int(len(ds.cand_frame)),
            "seconds_wall": round(wall, 2), "seconds_read": round(ds.seconds_read, 2),
            "seconds_detect": round(ds.seconds_detect, 2),
            "decode_fps": round(decode_fps, 1), "detect_crops_per_s": round(1.0 / max(per_crop, 1e-9), 1),
            "overall_fps": round(ds.n_scanned / max(wall, 1e-9), 1),
            "full_plan_crops": int(len(full)), "full_plan_last_frame": last,
            "estimated_full_run_min": round(est_s / 60.0, 1)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default=None, help="folder with tactical/ (default: the repo data folder)")
    ap.add_argument("--config", default=None, help="YAML configuration (default configs/default.yaml)")
    ap.add_argument("--detector", choices=("yolo", "hough", "motion", "dryrun"), default=None,
                    help="default: vision.detector of the configuration")
    ap.add_argument("--weights", default=None, help="ultralytics weights (default vision.weights)")
    ap.add_argument("--device", default=None, help="ultralytics device, e.g. cpu, 0, mps")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--every-k", type=int, default=None, help="one crop every k frames of each gap")
    ap.add_argument("--max-frames", type=int, default=None, help="only the crops of the first N video frames")
    ap.add_argument("--no-cache", action="store_true", help="neither read nor write cache/vision")
    ap.add_argument("--benchmark", type=int, nargs="?", const=2000, default=None, metavar="N",
                    help="time reading + detection on the first N video frames (default 2000), no cache")
    args = ap.parse_args(argv)

    cfg_path, _ = resolve_config(args.config)
    params = ac.VisionParams.from_config(cfg_path)
    over = {k: v for k, v in (("detector", args.detector), ("weights", args.weights), ("device", args.device),
                              ("batch_size", args.batch_size)) if v is not None}
    params = dataclasses.replace(params, **over)
    try:
        ball, players, hom, blocks, fps, pitch, video = load_inputs(args.data_dir, cfg_path)
        if not video.is_file():
            print(f"video not found: {video.name} (expected under <data-dir>/tactical/)")
            return 1
        detector = make_detector(params)
        if args.benchmark is not None:
            res = benchmark(ball, players, hom, blocks, video, detector, params, fps, pitch, args.benchmark)
            print(f"benchmark detector={params.detector} frames={res['frames_scanned']} "
                  f"candidates={res['candidates']}")
            print(f"read: {res['seconds_read']} s ({res['decode_fps']} frames/s decoded)   "
                  f"detect: {res['seconds_detect']} s ({res['detect_crops_per_s']} crops/s)   "
                  f"overall {res['overall_fps']} frames/s")
            print(f"full plan: {res['full_plan_crops']} crops up to frame {res['full_plan_last_frame']} "
                  f"-> estimated full run {res['estimated_full_run_min']} min")
            return 0
        t0 = time.perf_counter()
        out = rec.recover_ball(ball, players, hom, blocks, video, detector=detector, params=params, fps=fps,
                               pitch=pitch, max_frames=args.max_frames, every_k=args.every_k,
                               use_cache=not args.no_cache)
    except (rec.VisionUnavailableError, ImportError, FileNotFoundError) as exc:
        print(f"vision recovery unavailable: {exc}")
        return 2
    s = out.stats
    wall = time.perf_counter() - t0
    print(f"vision recovery detector={params.detector} status={s['status']} from_cache={s['from_cache']}")
    print(f"frames scanned (crops): {s['crops_scanned']} of {s['crops_planned']} planned   "
          f"candidates: {s['candidates']}   detections kept: {s['detections_kept']}")
    print(f"hidden graded frames: {s['hidden_graded_frames']} in {s['gaps']} gaps   "
          f"frames filled: {s['frames_filled']} ({100 * s['fill_rate']:.1f}%)   "
          f"gaps with a recovery: {s['gaps_touched']} ({100 * s['gap_fill_rate']:.1f}%)")
    if s["from_cache"]:
        print(f"runtime: {wall:.1f} s (cache)")
    else:
        print(f"runtime: {wall:.1f} s (read {s['seconds_read']} s, detect {s['seconds_detect']} s)   "
              f"{s['video_frames_decoded'] / max(wall, 1e-9):.1f} video frames/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
