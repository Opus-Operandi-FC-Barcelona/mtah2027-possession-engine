#!/usr/bin/env python
"""Label-free, aggregate-only check of the ball detectors on the real tactical video (Issue #58).

Counterpart of ``real_broadcast_check.py`` (Issue #39) for the vision route: it samples frames of the evaluated
halves where the ATD ball is visible, cuts the 640x640 attention crop around the ATD ball projected with the
homography, runs the available detectors on it (motion #36, hough, yolo only if installed; otherwise skipped
and reported), reprojects the detections to the pitch with H^-1 and compares them with the ATD ball.  It uses NO
possession labels (H/A/D) and the submission path is untouched.

Printed (and written with ``--json``) are aggregates only: detection rate within 1/2/5 m of the ATD ball, median
and p90 error of the nearest detection, false positives per frame (detections farther than 5 m), skipped frames,
median apparent ball size in px, runtime.  By design the report cannot contain per-frame rows or images:
``assert_aggregate`` rejects anything that is not a scalar / short list before it is printed or saved, and no
crop is ever written.

    python scripts/real_video_ball_check.py --data-dir data --sample 1000 --json reports/real_video_ball_check.json
    python scripts/real_video_ball_check.py --data-dir data --detectors motion,yolo --search-offset-m 3
    python scripts/real_video_ball_check.py --data-dir data --detectors sv_ball,sv_ball640 --sv-weights W.pt --sv-clip 5

``sv_ball`` / ``sv_ball640`` (Issue #63) need torch + ultralytics + supervision in a separate environment and a
LOCAL football-ball weights file; they are import-guarded, so the default path and CI need none of that.

Caveat: the crop is centred on the ATD ball (``--search-offset-m`` shifts it by a seeded random offset to mimic
the uncertainty of a real search point), so the detection rates are an upper bound of the gap-recovery use.
"""
# ruff: noqa: E501
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for _extra in (REPO_ROOT / "src", REPO_ROOT, REPO_ROOT / "scripts"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

MAX_LIST = 8                 # longest list allowed in a report
MAX_JSON_CHARS = 6000


def assert_aggregate(obj, _depth: int = 0) -> None:
    """Raise unless ``obj`` is made of scalars, dicts and short lists of scalars (no per-frame data)."""
    if _depth > 6:
        raise ValueError("report nested too deeply")
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(k, str):
                raise ValueError("report keys must be strings")
            assert_aggregate(v, _depth + 1)
        return
    if isinstance(obj, (list, tuple)):
        if len(obj) > MAX_LIST:
            raise ValueError(f"list of {len(obj)} entries in a report: per-frame data is not allowed")
        for v in obj:
            if isinstance(v, (list, tuple, dict)):
                raise ValueError("nested containers in lists are not allowed")
            assert_aggregate(v, _depth + 1)
        return
    raise ValueError(f"type {type(obj).__name__} is not allowed in a report")


def _q(values, qs=(0.5, 0.9), nd=3):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return [round(float(np.quantile(v, q)), nd) for q in qs] if len(v) else None


def sample_frames(ball_xy, hom, blocks, params, n_sample: int, seed: int, pitch, margin_px: float = 40.0):
    """Seeded sample of video frames inside ``blocks`` with a visible ATD ball and a usable homography whose
    projection falls inside the image.  Returns ``(frames, counts)``; labels are never consulted.

    Frame indices are video frames (``frame_id - 1`` convention of the ATD loader), the same index the
    homographies and the video reader use."""
    from mtah2027.io.homography_loader import project_pitch_to_pixels

    n = min(len(ball_xy), hom.n_frames)
    inplay = np.zeros(n, bool)
    for a, b in blocks:
        inplay[int(a):min(int(b), n - 1) + 1] = True
    visible = inplay & np.isfinite(ball_xy[:n]).all(axis=1)
    usable = hom.usable(params.max_homography_error)[:n]
    idx = np.flatnonzero(visible & usable)
    u, v = project_pitch_to_pixels(hom.H[idx], ball_xy[idx, 0], ball_xy[idx, 1], pitch)
    inside = (np.isfinite(u) & np.isfinite(v) & (u >= margin_px) & (u <= params.frame_width - margin_px) &
              (v >= margin_px) & (v <= params.frame_height - margin_px))
    eligible = idx[inside]
    rng = np.random.default_rng(seed)
    take = eligible if len(eligible) <= n_sample else np.sort(rng.choice(eligible, n_sample, replace=False))
    counts = {"graded_frames_ball_visible": int(visible.sum()), "eligible": int(len(eligible)),
              "skipped_invalid_homography_or_offimage": int(visible.sum() - len(eligible)),
              "sampled": int(len(take))}
    return take.astype(np.int64), counts


def build_plan(frames, ball_xy, hom, params, pitch, offset_m: float = 0.0, seed: int = 0):
    from mtah2027.io.homography_loader import project_pitch_to_pixels
    from mtah2027.vision import attention_crops as ac

    search = ball_xy[frames].astype(np.float64).copy()
    if offset_m > 0:
        ang = np.random.default_rng(seed + 1).uniform(0.0, 2.0 * np.pi, len(frames))
        search += offset_m * np.stack([np.cos(ang), np.sin(ang)], axis=1)
    u, v = project_pitch_to_pixels(hom.H[frames], search[:, 0], search[:, 1], pitch)
    ok = np.isfinite(u) & np.isfinite(v)
    frames, search, u, v = frames[ok], search[ok], u[ok], v[ok]
    origin = ac.crop_window(u, v, params.crop_size, params.frame_width, params.frame_height)
    return ac.CropPlan(frames=frames, search_xy=search, center_uv=np.stack([u, v], 1), origin=origin,
                       gap_id=np.zeros(len(frames), np.int64), n_hidden=0, n_gaps=0)


def ball_size_px(frames, ball_xy, hom, pitch, diameter_m: float):
    """Median apparent ball diameter (px) from the homography scale at the ATD ball."""
    from mtah2027.io.homography_loader import project_pitch_to_pixels

    r = diameter_m / 2.0
    x, y = ball_xy[frames, 0], ball_xy[frames, 1]
    u0, v0 = project_pitch_to_pixels(hom.H[frames], x - r, y, pitch)
    u1, v1 = project_pitch_to_pixels(hom.H[frames], x + r, y, pitch)
    u2, v2 = project_pitch_to_pixels(hom.H[frames], x, y - r, pitch)
    u3, v3 = project_pitch_to_pixels(hom.H[frames], x, y + r, pitch)
    d = _q(0.5 * (np.hypot(u1 - u0, v1 - v0) + np.hypot(u3 - u2, v3 - v2)), (0.5,), 2)
    return d[0] if d else None


def detection_metrics(ds, ball_xy, radii, fp_radius: float, n_planned: int) -> dict:
    """Aggregates of one DetectionSet against the ATD ball (nothing per frame leaves this function)."""
    frames = ds.plan_frames
    n = int(len(frames))
    out: dict = {"planned": int(n_planned), "scanned": n, "read_failures": int(n_planned - n)}
    if n == 0:
        return out
    truth = ball_xy[frames].astype(np.float64)
    nearest = np.full(n, np.inf)
    top1 = np.full(n, np.inf)
    n_det = np.zeros(n, np.int64)
    fp = 0
    if len(ds.cand_frame):
        at = np.searchsorted(frames, ds.cand_frame)
        d = np.hypot(*(ds.cand_xy - truth[at]).T)
        np.minimum.at(nearest, at, d)
        np.add.at(n_det, at, 1)
        fp = int((d > fp_radius).sum())
        order = np.lexsort((-ds.cand_conf, at))
        first = np.r_[True, at[order][1:] != at[order][:-1]]
        top1[at[order][first]] = d[order][first]
    out["detection_rate_pct"] = {f"{r:g}m": round(100.0 * float((nearest <= r).mean()), 2) for r in radii}
    out["top1_confidence_rate_pct"] = {f"{r:g}m": round(100.0 * float((top1 <= r).mean()), 2) for r in radii}
    out["frames_with_any_detection_pct"] = round(100.0 * float((n_det > 0).mean()), 2)
    out["error_m_median_p90_of_nearest"] = _q(nearest[np.isfinite(nearest)])
    out["false_positives_per_frame"] = round(fp / n, 4)
    out["false_positive_radius_m"] = float(fp_radius)
    out["detections_per_frame"] = round(float(n_det.mean()), 4)
    return out


def make_detectors(names, params, sv_weights=None, sv_clip: int = 0):
    """``({name: detector}, {name: reason})`` for the ones that can / cannot be constructed.

    ``sv_ball`` / ``sv_ball640`` (Issue #63: local football-ball YOLO + supervision InferenceSlicer) are built
    lazily so that torch / ultralytics / supervision are never needed by the default path."""
    from mtah2027.vision.detectors import make_detector

    dets, skipped = {}, {}
    for name in names:
        try:
            if name in ("sv_ball", "sv_ball640"):
                from mtah2027.vision.sv_ball import SupervisionBallDetector

                if not sv_weights or not Path(sv_weights).is_file():
                    raise FileNotFoundError("sv_ball weights (--sv-weights)")
                dets[name] = SupervisionBallDetector(str(sv_weights), name, clip=sv_clip)
                continue
            dets[name] = make_detector(dataclasses.replace(params, detector=name))
        except Exception as exc:                                  # missing ultralytics / weights / OpenCV
            skipped[name] = type(exc).__name__
    return dets, skipped


def build_report(ball_xy, hom, blocks, video, params, pitch, fps, *, detectors=("motion", "yolo"),
                 n_sample: int = 1000, seed: int = 0, radii=(1.0, 2.0, 5.0), fp_radius: float = 5.0,
                 offset_m: float = 0.0, sv_weights=None, sv_clip: int = 0) -> dict:
    """Aggregate report (the only thing this module ever returns)."""
    from mtah2027.vision import recovery as rec

    t0 = time.perf_counter()
    frames, counts = sample_frames(ball_xy, hom, blocks, params, n_sample, seed, pitch)
    rep: dict = {"sample": counts, "seed": int(seed), "crop_search_offset_m": float(offset_m),
                 "ball_size_px_median": ball_size_px(frames, ball_xy, hom, pitch, params.ball_diameter_m)
                 if len(frames) else None}
    dets, skipped = make_detectors(detectors, params, sv_weights, sv_clip)
    rep["skipped_detectors"] = skipped
    rep["detectors"] = {}
    plan = build_plan(frames, ball_xy, hom, params, pitch, offset_m, seed) if len(frames) else None
    for name, det in dets.items():
        if plan is None or not len(plan):
            break
        t1 = time.perf_counter()
        ds = rec.run_detection(video, plan, det, hom, params, pitch=pitch, fps=fps)
        m = detection_metrics(ds, ball_xy, radii, fp_radius, len(plan))
        m["seconds"] = round(time.perf_counter() - t1, 2)
        m["seconds_per_frame"] = round(m["seconds"] / max(1, len(plan)), 3)
        if name.startswith("sv_ball"):
            m["sv_clip_frames"] = int(sv_clip) if sv_clip > 1 else 1
        rep["detectors"][name] = m
    rep["seconds_total"] = round(time.perf_counter() - t0, 2)
    assert_aggregate(rep)
    if len(json.dumps(rep)) > MAX_JSON_CHARS:
        raise ValueError("report too large: only aggregates are allowed")
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--sample", type=int, default=1000, help="sampled frames (default 1000)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--detectors", default="motion,yolo", help="comma list of motion|yolo|hough|sv_ball|sv_ball640")
    ap.add_argument("--sv-weights", default=os.environ.get("MTAH_SV_BALL_WEIGHTS"),
                    help="local football-ball-detection.pt for sv_ball (Issue #63; torch/ultralytics/supervision)")
    ap.add_argument("--sv-clip", type=int, default=0,
                    help="sv_ball temporal filter over this many consecutive frames (odd, e.g. 5; 0 = off)")
    ap.add_argument("--radii", default="1,2,5", help="detection radii in metres")
    ap.add_argument("--fp-radius", type=float, default=5.0)
    ap.add_argument("--search-offset-m", type=float, default=0.0)
    ap.add_argument("--weights", default=None, help="yolo weights (default: from the configuration)")
    ap.add_argument("--json", default=None, help="write the aggregate report to this path")
    args = ap.parse_args(argv)

    from run_vision_recovery import load_inputs

    from mtah2027.config import set_global_seeds
    from mtah2027.pipeline import resolve_config
    from mtah2027.vision.attention_crops import VisionParams

    set_global_seeds(0)
    cfg_path, _ = resolve_config(args.config)
    params = VisionParams.from_config(cfg_path)
    if args.weights:
        params = dataclasses.replace(params, weights=args.weights)
    data_dir = args.data_dir or os.environ.get("MTAH_DATA_DIR") or str(REPO_ROOT / "data")
    ball, _players, hom, blocks, fps, pitch, video = load_inputs(data_dir, cfg_path)
    names = [s.strip() for s in args.detectors.split(",") if s.strip()]
    radii = tuple(float(s) for s in args.radii.split(",") if s.strip())
    rep = build_report(ball, hom, blocks, video, params, pitch, fps, detectors=names, n_sample=args.sample,
                       seed=args.seed, radii=radii, fp_radius=args.fp_radius, offset_m=args.search_offset_m,
                       sv_weights=args.sv_weights, sv_clip=args.sv_clip)
    text = json.dumps(rep, indent=1)
    print(text)
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
