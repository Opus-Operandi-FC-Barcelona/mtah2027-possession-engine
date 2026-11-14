"""Ball recovery from the tactical video (Issue #11): crops -> detector -> pitch metres -> injection.

Chain (every step but the video reading is pure NumPy and unit-tested without a video):

1. :func:`~mtah2027.vision.attention_crops.plan_crops` -- where to look, per ball-missing frame.
2. :func:`read_crops` -- the planned frames are read **sequentially** with OpenCV (``grab`` over short jumps,
   a seek over long ones) and the ``crop_size`` windows cut out.
3. a :class:`~mtah2027.vision.detectors.BallDetector` finds ball candidates in each crop; they are
   reprojected to pitch metres with ``H^-1`` and stored in a :class:`DetectionSet`.  The set is cached in
   ``cache/vision/*.npz`` keyed by the video file (size, mtime), the plan and the detector settings, so it is
   computed once; the selection thresholds below are applied after loading and need no new detection run.
4. :func:`select_detections` -- keeps one candidate per frame: confidence, inside the pitch, close enough
   to the search point, and a plausible speed with respect to the neighbouring accepted / seen ball points.
5. :func:`inject_points` -- writes the kept points into ``ball_xy`` where the tracking is NaN and, between two
   points of which at least one is a detection, interpolates the frames in between (short gaps only).
   ``ball_source`` flags every frame: ``0`` missing, ``1`` tracking, ``2`` vision detection, ``3`` vision
   interpolation.

Aggregates only: nothing here prints or stores per-frame data other than the cache file.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from mtah2027.config import FPS, PITCH, REPO_ROOT
from mtah2027.features._utils import mask_runs
from mtah2027.io.homography_loader import Homographies
from mtah2027.vision.attention_crops import (
    CropPlan,
    VisionParams,
    crop_points_to_pitch,
    plan_crops,
)
from mtah2027.vision.detectors import BallDetector, detector_fingerprint, motion_geometry

SRC_MISSING, SRC_TRACKING, SRC_VISION, SRC_VISION_INTERP = 0, 1, 2, 3
CACHE_VERSION = 1


class VisionUnavailableError(RuntimeError):
    """OpenCV is not installed or the video cannot be opened."""


# --------------------------------------------------------------------------------------------
# Video reading
# --------------------------------------------------------------------------------------------
def require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise VisionUnavailableError(
            "OpenCV is required to read the video: pip install -e .[vision]") from exc
    return cv2


def read_crops(video_path: str | os.PathLike, plan: CropPlan, params: VisionParams, *,
               batch_size: int | None = None, context: int = 0):
    """Yield ``(plan_indices, crops, n_decoded)`` batches, reading the planned frames in order.

    ``crops`` is a list of ``crop_size x crop_size`` BGR arrays.  Frames that cannot be decoded are skipped
    (their plan index is simply absent).  ``n_decoded`` counts the frames decoded/skipped so far, for rates.

    With ``context=1`` (temporal detectors, Issue #36) each crop is the frame triplet ``[3, size, size, 3]``
    of the frames ``t-1, t, t+1`` cut with the SAME window (clamped at the video ends: the missing neighbour
    repeats frame ``t``).
    """
    cv2 = require_cv2()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VisionUnavailableError(f"cannot open the video '{Path(video_path).name}'")
    try:
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (width, height) != (params.frame_width, params.frame_height):
            raise VisionUnavailableError(
                f"video is {width}x{height}, the plan assumes {params.frame_width}x{params.frame_height}")
        size, bs = int(params.crop_size), int(batch_size or params.batch_size)
        ctx = int(context)
        pos = 0
        idx_batch: list[int] = []
        crops: list[np.ndarray] = []
        for i, f in enumerate(plan.frames):
            f = int(f)
            first = max(f - ctx, 0)
            if first < pos and first < f:
                pos = -1                              # the previous read overlapped: force a seek
            if pos < 0 or first - pos > params.seek_threshold_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, first)
            elif first < pos:
                continue                              # f already behind the cursor (unsorted plan)
            else:
                for _ in range(first - pos):
                    if not cap.grab():
                        break
            pos = first
            x0, y0 = (int(v) for v in plan.origin[i])
            wanted = list(range(first, f + ctx + 1))
            imgs: list[np.ndarray | None] = []
            for _ in wanted:
                ok, img = cap.read()
                pos += 1
                imgs.append(np.ascontiguousarray(img[y0:y0 + size, x0:x0 + size]) if ok and img is not None
                            else None)
            cur = imgs[f - first]
            if cur is None:
                continue
            idx_batch.append(i)
            if ctx:
                prev = imgs[0] if first < f and imgs[0] is not None else cur
                nxt = imgs[-1] if imgs[-1] is not None and len(imgs) > f - first + 1 else cur
                crops.append(np.stack([prev, cur, nxt]))
            else:
                crops.append(cur)
            if len(crops) >= bs:
                yield idx_batch, crops, pos
                idx_batch, crops = [], []
        if crops:
            yield idx_batch, crops, pos
    finally:
        cap.release()


# --------------------------------------------------------------------------------------------
# Detection and cache
# --------------------------------------------------------------------------------------------
@dataclass
class DetectionSet:
    """Raw candidates (pitch metres) of the scanned crops plus the plan data needed to filter them."""

    plan_frames: np.ndarray            # [M] frames whose crop was scanned
    plan_search_xy: np.ndarray         # [M, 2] their search point
    cand_frame: np.ndarray             # [K] frame of each candidate
    cand_xy: np.ndarray                # [K, 2] pitch metres
    cand_conf: np.ndarray              # [K]
    seconds_read: float = 0.0
    seconds_detect: float = 0.0
    video_frames_decoded: int = 0
    from_cache: bool = False

    @property
    def n_scanned(self) -> int:
        return int(len(self.plan_frames))


def video_fingerprint(video_path: str | os.PathLike) -> str:
    st = os.stat(video_path)
    return f"{Path(video_path).name}|{st.st_size}|{st.st_mtime_ns}"


def cache_key(video_path, plan: CropPlan, detector_fp: str, params: VisionParams) -> str:
    h = hashlib.sha256()
    h.update(f"v{CACHE_VERSION}|{video_fingerprint(video_path)}|{detector_fp}|{params.crop_size}|"
             f"{params.frame_width}x{params.frame_height}".encode())
    h.update(np.ascontiguousarray(plan.frames, dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(plan.origin, dtype=np.int64).tobytes())
    return h.hexdigest()[:20]


def cache_path(params: VisionParams, video_path, key: str, root: Path | None = None) -> Path:
    base = Path(params.cache_dir)
    if not base.is_absolute():
        base = (root or REPO_ROOT) / base
    return base / f"{Path(video_path).stem}_{key}.npz"


def save_detections(path: Path, ds: DetectionSet, key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, plan_frames=ds.plan_frames, plan_search_xy=ds.plan_search_xy,
                        cand_frame=ds.cand_frame, cand_xy=ds.cand_xy, cand_conf=ds.cand_conf,
                        meta=np.array(json.dumps({"key": key, "version": CACHE_VERSION,
                                                  "seconds_read": ds.seconds_read,
                                                  "seconds_detect": ds.seconds_detect,
                                                  "video_frames_decoded": ds.video_frames_decoded})))
    os.replace(tmp, path)


def load_detections(path: Path, key: str) -> DetectionSet | None:
    if not path.is_file():
        return None
    try:
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        if meta.get("key") != key or meta.get("version") != CACHE_VERSION:
            return None
        return DetectionSet(z["plan_frames"], z["plan_search_xy"], z["cand_frame"], z["cand_xy"],
                            z["cand_conf"], float(meta["seconds_read"]), float(meta["seconds_detect"]),
                            int(meta["video_frames_decoded"]), from_cache=True)
    except (OSError, ValueError, KeyError):
        return None


def run_detection(video_path, plan: CropPlan, detector: BallDetector, homographies: Homographies,
                  params: VisionParams, *, pitch: tuple[float, float] = PITCH, fps: float = FPS,
                  progress: Callable[[int, int], None] | None = None) -> DetectionSet:
    """Read the planned crops, detect, reproject the candidates to pitch metres.

    A detector with ``temporal = True`` (Issue #36) receives frame triplets and, as second argument, one
    geometry record per crop (crop origin + the three homography inverses) for its metric checks."""
    context = int(getattr(detector, "context", 1)) if getattr(detector, "temporal", False) else 0
    frames_out: list[np.ndarray] = []
    xy_out: list[np.ndarray] = []
    conf_out: list[np.ndarray] = []
    scanned = np.zeros(len(plan), dtype=bool)
    t_read = t_det = 0.0
    decoded = 0
    t_prev = time.perf_counter()
    for idx, crops, pos in read_crops(video_path, plan, params, context=context):
        t_now = time.perf_counter()
        t_read += t_now - t_prev
        if context:
            n_h = len(homographies.H_inv)
            geo = [motion_geometry(plan.origin[i], homographies.H_inv[np.clip(
                plan.frames[i] + np.arange(-context, context + 1), 0, n_h - 1)], pitch=pitch, fps=fps)
                for i in idx]
            results = detector.detect(crops, geo)
        else:
            results = detector.detect(crops)
        t_prev = time.perf_counter()
        t_det += t_prev - t_now
        decoded = pos
        scanned[idx] = True
        pts, org, fr, cf = [], [], [], []
        for i, res in zip(idx, results):
            if len(res):
                pts.append(res[:, :2])
                org.append(np.repeat(plan.origin[i][None], len(res), axis=0))
                fr.append(np.full(len(res), plan.frames[i], dtype=np.int64))
                cf.append(res[:, 2])
        if pts:
            f = np.concatenate(fr)
            xy = crop_points_to_pitch(np.concatenate(pts), np.concatenate(org), homographies.H_inv[f], pitch)
            frames_out.append(f)
            xy_out.append(xy)
            conf_out.append(np.concatenate(cf))
        if progress is not None:
            progress(int(scanned.sum()), len(plan))
    return DetectionSet(
        plan_frames=plan.frames[scanned], plan_search_xy=plan.search_xy[scanned],
        cand_frame=np.concatenate(frames_out) if frames_out else np.zeros(0, np.int64),
        cand_xy=np.concatenate(xy_out) if xy_out else np.zeros((0, 2)),
        cand_conf=np.concatenate(conf_out) if conf_out else np.zeros(0),
        seconds_read=t_read, seconds_detect=t_det, video_frames_decoded=decoded)


# --------------------------------------------------------------------------------------------
# Selection and injection
# --------------------------------------------------------------------------------------------
def _within_speed(a: np.ndarray, fa: int, b: np.ndarray, fb: int, fps: float, p: VisionParams) -> bool:
    dt = abs(fb - fa) / fps
    return float(np.hypot(*(a - b))) <= p.max_speed_mps * dt + p.speed_slack_m


def select_detections(ds: DetectionSet, ball_xy: np.ndarray, *, fps: float = FPS,
                      pitch: tuple[float, float] = PITCH,
                      params: VisionParams | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One plausible candidate per frame: ``(frames, xy[m], confidence)`` sorted by frame.

    Static filters first (confidence, inside the pitch +- margin, within ``max_search_dist_m`` of the search
    point); then the candidates of each frame are tried by decreasing confidence against the previous
    accepted or seen ball point (speed), and a backward pass checks every kept point against the next one.
    """
    p = params or VisionParams.from_config()
    ball = np.asarray(ball_xy, dtype=np.float64)
    n = len(ball)
    empty = (np.zeros(0, np.int64), np.zeros((0, 2)), np.zeros(0))
    if not len(ds.cand_frame):
        return empty
    f, xy, cf = ds.cand_frame, ds.cand_xy, ds.cand_conf
    pi = np.searchsorted(ds.plan_frames, f)
    search = ds.plan_search_xy[np.clip(pi, 0, max(len(ds.plan_frames) - 1, 0))]
    L, W = pitch
    m = p.pitch_margin_m
    keep = (cf >= p.confidence) & np.isfinite(xy).all(axis=1) & (xy[:, 0] >= -m) & (xy[:, 0] <= L + m) & \
        (xy[:, 1] >= -m) & (xy[:, 1] <= W + m) & \
        (np.hypot(*(xy - search).T) <= p.max_search_dist_m)
    f, xy, cf = f[keep], xy[keep], cf[keep]
    if not len(f):
        return empty
    order = np.lexsort((-cf, f))
    f, xy, cf = f[order], xy[order], cf[order]

    visible = np.isfinite(ball).all(axis=1)
    idx = np.arange(n)
    last = np.maximum.accumulate(np.where(visible, idx, -1))
    nxt = np.minimum.accumulate(np.where(visible, idx, n)[::-1])[::-1]

    acc_f: list[int] = []
    acc_xy: list[np.ndarray] = []
    acc_cf: list[float] = []
    bounds = np.flatnonzero(np.concatenate([[True], f[1:] != f[:-1], [True]]))
    for s, e in zip(bounds[:-1], bounds[1:]):
        fr = int(f[s])
        ref_f, ref_xy = None, None
        if acc_f:
            ref_f, ref_xy = acc_f[-1], acc_xy[-1]
        lf = int(last[fr]) if fr < n else -1
        if lf >= 0 and (ref_f is None or lf > ref_f):
            ref_f, ref_xy = lf, ball[lf]
        for j in range(s, e):                      # best confidence first
            if ref_f is None or _within_speed(xy[j], fr, ref_xy, ref_f, fps, p):
                acc_f.append(fr)
                acc_xy.append(xy[j])
                acc_cf.append(float(cf[j]))
                break
    # backward pass: every kept point must also be consistent with the next kept / seen point
    out_f: list[int] = []
    out_xy: list[np.ndarray] = []
    out_cf: list[float] = []
    for fr, pt, c in zip(reversed(acc_f), reversed(acc_xy), reversed(acc_cf)):
        ref_f, ref_xy = (out_f[-1], out_xy[-1]) if out_f else (None, None)
        nf = int(nxt[fr]) if fr < n else n
        if nf < n and (ref_f is None or nf < ref_f):
            ref_f, ref_xy = nf, ball[nf]
        if ref_f is None or _within_speed(pt, fr, ref_xy, ref_f, fps, p):
            out_f.append(fr)
            out_xy.append(pt)
            out_cf.append(c)
    out_f.reverse()
    out_xy.reverse()
    out_cf.reverse()
    if not out_f:
        return empty
    return np.array(out_f, np.int64), np.array(out_xy), np.array(out_cf)


def inject_points(ball_xy: np.ndarray, frames: np.ndarray, xy: np.ndarray, *, fps: float = FPS,
                  params: VisionParams | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Write recovered points into the NaN frames of ``ball_xy``; returns ``(ball_xy, ball_source)``.

    Frames that already carry a tracked ball are never touched.  Between two consecutive anchors (a recovered
    point or a seen ball) at most ``interp_max_frames`` apart, with at least one recovered point among them,
    plausible speed and only missing frames in between, the missing frames are linearly interpolated.
    """
    p = params or VisionParams.from_config()
    ball = np.asarray(ball_xy, dtype=np.float32).copy()
    n = len(ball)
    visible = np.isfinite(ball).all(axis=1)
    source = np.where(visible, SRC_TRACKING, SRC_MISSING).astype(np.uint8)
    frames = np.asarray(frames, dtype=np.int64)
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    ok = (frames >= 0) & (frames < n)
    ok[ok] = ~visible[frames[ok]]
    frames, xy = frames[ok], xy[ok]
    if not len(frames):
        return ball, source
    order = np.argsort(frames, kind="stable")
    frames, xy = frames[order], xy[order]
    ball[frames] = xy.astype(np.float32)
    source[frames] = SRC_VISION

    # anchors = tracking + recovered points; interpolate between consecutive anchors if worthwhile
    anchor = np.isfinite(ball).all(axis=1)
    a_idx = np.flatnonzero(anchor)
    nxt_pos = np.searchsorted(a_idx, frames, side="right")     # position in a_idx of the next anchor
    prv_pos = np.searchsorted(a_idx, frames, side="left") - 1  # ... of the previous anchor
    pairs = set()
    for k in range(len(frames)):
        if nxt_pos[k] < len(a_idx):
            pairs.add((int(frames[k]), int(a_idx[nxt_pos[k]])))
        if prv_pos[k] >= 0:
            pairs.add((int(a_idx[prv_pos[k]]), int(frames[k])))
    maxf = int(p.interp_max_frames)
    for fa, fb in sorted(pairs):
        gap = fb - fa
        if gap <= 1 or gap > maxf:
            continue
        if visible[fa + 1:fb].any() or source[fa + 1:fb].max() != SRC_MISSING:
            continue
        a, b = ball[fa].astype(np.float64), ball[fb].astype(np.float64)
        if not _within_speed(a, fa, b, fb, fps, p):
            continue
        w = (np.arange(fa + 1, fb) - fa) / gap
        ball[fa + 1:fb] = (a[None] * (1 - w[:, None]) + b[None] * w[:, None]).astype(np.float32)
        source[fa + 1:fb] = SRC_VISION_INTERP
    return ball, source


# --------------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------------
@dataclass
class VisionResult:
    ball_xy: np.ndarray
    ball_source: np.ndarray
    stats: dict = field(default_factory=dict)


def vision_stats(ball_before: np.ndarray, ball_source: np.ndarray, plan: CropPlan, ds: DetectionSet | None,
                 play_blocks: Sequence[tuple[int, int]], n_kept: int, seconds: float,
                 limit_frames: int | None = None) -> dict:
    """Aggregate counters (no per-frame data)."""
    n = len(ball_before)
    graded = np.zeros(n, bool)
    for a, b in play_blocks:
        graded[max(int(a), 0):min(int(b), n - 1) + 1] = True
    if limit_frames is not None:
        graded[limit_frames:] = False
    hidden = graded & ~np.isfinite(ball_before).all(axis=1)
    vision = np.isin(ball_source, (SRC_VISION, SRC_VISION_INTERP))
    filled = hidden & vision
    starts, ends = mask_runs(hidden)
    gaps_touched = sum(bool(filled[a:b].any()) for a, b in zip(starts, ends))
    scanned = ds.n_scanned if ds is not None else 0
    return {
        "hidden_graded_frames": int(hidden.sum()), "gaps": int(len(starts)),
        "crops_planned": int(len(plan)), "crops_scanned": int(scanned),
        "candidates": int(len(ds.cand_frame)) if ds is not None else 0,
        "detections_kept": int(n_kept),
        "frames_filled": int(filled.sum()), "fill_rate": float(filled.sum() / max(hidden.sum(), 1)),
        "gaps_touched": int(gaps_touched), "gap_fill_rate": float(gaps_touched / max(len(starts), 1)),
        "from_cache": bool(ds.from_cache) if ds is not None else False,
        "seconds": round(float(seconds), 2),
        "seconds_read": round(float(ds.seconds_read), 2) if ds is not None else 0.0,
        "seconds_detect": round(float(ds.seconds_detect), 2) if ds is not None else 0.0,
        "video_frames_decoded": int(ds.video_frames_decoded) if ds is not None else 0,
    }


def recover_ball(ball_xy: np.ndarray, players_xy: np.ndarray, homographies: Homographies,
                 play_blocks: Sequence[tuple[int, int]], video_path, *, detector: BallDetector | None,
                 params: VisionParams | None = None, fps: float = FPS, pitch: tuple[float, float] = PITCH,
                 max_frames: int | None = None, every_k: int | None = None, use_cache: bool = True,
                 cache_root: Path | None = None,
                 progress: Callable[[int, int], None] | None = None) -> VisionResult:
    """Full recovery.  With ``detector=None`` only an existing cache is used (no video is decoded); a missing
    cache then returns the tracking unchanged with ``stats['status'] == 'no cache'``."""
    p = params or VisionParams.from_config()
    t0 = time.perf_counter()
    plan = plan_crops(ball_xy, players_xy, homographies, play_blocks, fps=fps, pitch=pitch, params=p,
                      every_k=every_k)
    if max_frames is not None:
        plan = plan.head(int(max_frames))
    fp = detector.fingerprint() if detector is not None else detector_fingerprint(p)
    ball_before = np.asarray(ball_xy, dtype=np.float32)
    if not Path(video_path).is_file():
        src = np.where(np.isfinite(ball_before).all(axis=1), SRC_TRACKING, SRC_MISSING).astype(np.uint8)
        st = vision_stats(ball_before, src, plan, None, play_blocks, 0, 0.0, max_frames)
        st["status"] = "video not found"
        return VisionResult(ball_before, src, st)
    key = cache_key(video_path, plan, fp, p)
    cpath = cache_path(p, video_path, key, cache_root)
    ds = load_detections(cpath, key) if use_cache else None
    if ds is None and detector is not None:
        ds = run_detection(video_path, plan, detector, homographies, p, pitch=pitch, fps=fps,
                           progress=progress)
        if use_cache:
            save_detections(cpath, ds, key)
    if ds is None:
        src = np.where(np.isfinite(ball_before).all(axis=1), SRC_TRACKING, SRC_MISSING).astype(np.uint8)
        st = vision_stats(ball_before, src, plan, None, play_blocks, 0, time.perf_counter() - t0, max_frames)
        st["status"] = "no cache"
        return VisionResult(ball_before, src, st)
    frames, xy, _ = select_detections(ds, ball_before, fps=fps, pitch=pitch, params=p)
    ball, source = inject_points(ball_before, frames, xy, fps=fps, params=p)
    st = vision_stats(ball_before, source, plan, ds, play_blocks, len(frames), time.perf_counter() - t0,
                      max_frames)
    st["status"] = "ok"
    return VisionResult(ball, source, st)
