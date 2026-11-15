"""Self-supervised ball dataset for the real tactical video (Issue #68).

Labels are ALGORITHMIC, never manual: the automatic ATD ball of the tracking, projected to pixels with the
frame homography, in the graded frames where the ball is visible and the homography is usable.  Nothing here
reads possession labels.

Temporal split by blocks: the video is cut in blocks of ``block_frames`` (2 min); every ``val_every``-th block
is validation, and training frames closer than ``guard_frames`` (10 s) to any validation block are dropped, so
no neighbouring frames are shared between the two sets.

For every selected frame the builder writes (YOLO layout, single class "ball"):

* train: one JPEG crop (``crop`` px square) whose position relative to the ball is random (the ball lands
  anywhere in the window), plus, with probability ``neg_prob``, a ball-free crop centred near a player;
* val: two lossless PNG crops per frame: ``atd`` (window centred on the ATD ball shifted by a random offset of
  at most ``atd_offset_m`` metres, the ceiling of #58/#63) and ``prior`` (window centred on a search point that
  does NOT use the true ball: interpolation between the last / next seen ball of a simulated gap whose length
  is drawn from the real tracking gaps, pulled to the pressure centroid exactly as
  :mod:`mtah2027.vision.attention_crops`).

Only aggregates are ever printed.  Images and metadata go to a cache directory outside git (and outside
OneDrive); the metadata arrays never leave it.
"""
# ruff: noqa: E501
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mtah2027.features._utils import mask_runs
from mtah2027.io.homography_loader import project_pitch_to_pixels
from mtah2027.vision import attention_crops as ac

BOX_PX = 10.0                      # side of the square label box (ball ~4.5 px; a tolerant box)


@dataclass(frozen=True)
class DatasetParams:
    block_frames: int = 3000       # 2 minutes at 25 fps
    val_every: int = 5             # every 5th block is validation
    guard_frames: int = 250        # 10 s guard band around validation blocks (train frames dropped)
    train_stride: int = 4          # one eligible train frame in k
    val_stride: int = 4            # one eligible val frame in k
    crop: int = 640
    margin_px: float = 40.0        # ball projected closer than this to the image border is not eligible
    edge_margin_px: float = 12.0   # a positive crop keeps the ball at least this far from its border
    neg_prob: float = 0.15
    atd_offset_m: float = 3.0
    near_gap_frames: int = 50      # "near a tracking gap" = within 2 s of a hidden-ball frame
    jpeg_quality: int = 97
    seed: int = 0


# --------------------------------------------------------------------------------------------
# Selection and split
# --------------------------------------------------------------------------------------------
def eligible_mask(ball_xy, hom, blocks, params_v: ac.VisionParams, pitch, margin_px: float) -> np.ndarray:
    """Graded frames with a visible ATD ball, a usable homography and the projection inside the image."""
    n = min(len(ball_xy), hom.n_frames)
    inplay = np.zeros(n, bool)
    for a, b in blocks:
        inplay[int(a):min(int(b), n - 1) + 1] = True
    ok = inplay & np.isfinite(ball_xy[:n]).all(axis=1) & hom.usable(params_v.max_homography_error)[:n]
    out = np.zeros(n, bool)
    idx = np.flatnonzero(ok)
    if len(idx):
        u, v = project_pitch_to_pixels(hom.H[idx], ball_xy[idx, 0], ball_xy[idx, 1], pitch)
        inside = (np.isfinite(u) & np.isfinite(v) & (u >= margin_px) & (u <= params_v.frame_width - margin_px) &
                  (v >= margin_px) & (v <= params_v.frame_height - margin_px))
        out[idx[inside]] = True
    return out


def split_labels(n: int, block_frames: int, val_every: int, guard_frames: int) -> np.ndarray:
    """``[n]`` int8: 0 train, 1 validation, -1 guard band (dropped)."""
    f = np.arange(n)
    is_val = (f // block_frames) % val_every == val_every - 1
    csum = np.concatenate([[0], np.cumsum(is_val)])
    lo, hi = np.clip(f - guard_frames, 0, n), np.clip(f + guard_frames + 1, 0, n)
    near = (csum[hi] - csum[lo]) > 0
    lab = np.zeros(n, np.int8)
    lab[near & ~is_val] = -1
    lab[is_val] = 1
    return lab


def near_gap_mask(ball_xy: np.ndarray, within: int) -> np.ndarray:
    """Frames with a hidden ball (non-finite ATD ball) within ``within`` frames, by tracking only."""
    hidden = ~np.isfinite(ball_xy).all(axis=1)
    n = len(hidden)
    csum = np.concatenate([[0], np.cumsum(hidden)])
    f = np.arange(n)
    return (csum[np.clip(f + within + 1, 0, n)] - csum[np.clip(f - within, 0, n)]) > 0


def select_frames(elig: np.ndarray, split: np.ndarray, p: DatasetParams) -> tuple[np.ndarray, np.ndarray]:
    """``(train_frames, val_frames)``: every k-th eligible frame of each side (deterministic)."""
    n = len(elig)
    tr = np.flatnonzero(elig & (split[:n] == 0))
    va = np.flatnonzero(elig & (split[:n] == 1))
    return tr[::max(1, p.train_stride)].astype(np.int64), va[::max(1, p.val_stride)].astype(np.int64)


# --------------------------------------------------------------------------------------------
# Crop geometry
# --------------------------------------------------------------------------------------------
def yolo_line(u: float, v: float, crop: int, box: float = BOX_PX) -> str:
    """YOLO label line (class 0) for a ball at crop pixel ``(u, v)``."""
    return f"0 {u / crop:.6f} {v / crop:.6f} {box / crop:.6f} {box / crop:.6f}"


def random_positive_windows(u, v, rng, p: DatasetParams, width: int, height: int):
    """Windows whose position relative to the ball is uniform: ``(origin [M,2], ball_in_crop [M,2])``.

    The window centre is the ball plus a uniform offset keeping the ball ``edge_margin_px`` inside the crop
    (before clamping to the image; the returned ball position is computed after clamping)."""
    half = p.crop / 2.0 - p.edge_margin_px
    off = rng.uniform(-half, half, size=(len(u), 2))
    origin = ac.crop_window(u + off[:, 0], v + off[:, 1], p.crop, width, height)
    bu, bv = u - origin[:, 0], v - origin[:, 1]
    return origin, np.stack([bu, bv], axis=1)


def negative_windows(centre_uv, ball_uv, rng, p: DatasetParams, width: int, height: int):
    """Windows centred near ``centre_uv`` that do not contain the ball (``keep`` mask, origin)."""
    jit = rng.uniform(-60.0, 60.0, size=centre_uv.shape)
    origin = ac.crop_window(centre_uv[:, 0] + jit[:, 0], centre_uv[:, 1] + jit[:, 1], p.crop, width, height)
    bu, bv = ball_uv[:, 0] - origin[:, 0], ball_uv[:, 1] - origin[:, 1]
    m = 24.0
    contains = (bu > -m) & (bu < p.crop + m) & (bv > -m) & (bv < p.crop + m)
    return ~contains, origin


def simulated_gap_prior(frames, ball_xy, players_xy, fps: float, params_v: ac.VisionParams,
                        rng: np.random.Generator, gap_lengths: np.ndarray) -> np.ndarray:
    """Search point for visible frames as if the ball were hidden around them (never uses the ball at ``t``).

    A gap of length ``L`` (drawn from the real tracking gaps) is placed around the frame at a uniform offset;
    the point is the interpolation between the last seen ball before it and the next seen after it (same rule
    as :func:`attention_crops.interpolated_search_points`), pulled to the pressure centroid."""
    n = len(ball_xy)
    visible = np.isfinite(ball_xy).all(axis=1)
    last, nxt = ac._edge_indices(visible)
    frames = np.asarray(frames, dtype=np.int64)
    lens = rng.choice(gap_lengths, size=len(frames)).astype(np.int64)
    lens = np.maximum(lens, 3)
    a = np.maximum((rng.uniform(0.0, 1.0, len(frames)) * lens).astype(np.int64), 1)   # frames before t in the gap
    b = np.maximum(lens - a, 1)
    lf = last[np.clip(frames - a - 1, 0, n - 1)]
    nf = nxt[np.clip(frames + b + 1, 0, n - 1)]
    has_l, has_n = lf >= 0, nf < n
    out = np.full((len(frames), 2), np.nan)
    both = has_l & has_n & ((nf - lf) / fps <= params_v.interp_max_s)
    span = np.maximum(nf - lf, 1)
    w = ((frames - lf) / span)[:, None]
    ball = np.asarray(ball_xy, dtype=np.float64)
    out[both] = (ball[np.clip(lf, 0, n - 1)] * (1 - w) + ball[np.clip(nf, 0, n - 1)] * w)[both]
    rest = ~both
    near_last = has_l & (~has_n | ((frames - lf) <= (nf - frames)))
    take_l, take_n = rest & near_last, rest & ~near_last & has_n
    out[take_l] = ball[np.clip(lf, 0, n - 1)][take_l]
    out[take_n] = ball[np.clip(nf, 0, n - 1)][take_n]
    fin = np.isfinite(out).all(axis=1)
    if fin.any():
        out[fin] = ac.pressure_search_points(frames[fin], out[fin], np.asarray(players_xy), fps, params_v)
    return out


def real_gap_lengths(ball_xy: np.ndarray, blocks) -> np.ndarray:
    """Lengths (frames) of the real ball-missing runs of the graded frames (tracking only)."""
    hidden = ac.hidden_gap_frames(ball_xy, blocks)
    s, e = mask_runs(hidden)
    return (e - s).astype(np.int64) if len(s) else np.array([50], np.int64)


# --------------------------------------------------------------------------------------------
# Builder
# --------------------------------------------------------------------------------------------
def _imwrite(cv2, path: Path, img, quality: int | None) -> None:
    if quality is None:
        cv2.imwrite(str(path), img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    else:
        cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])


def build_dataset(video, ball_xy, players_xy, hom, blocks, params_v: ac.VisionParams, out_dir, *,
                  fps: float = 25.0, pitch=ac.PITCH, p: DatasetParams | None = None,
                  max_frame: int | None = None) -> dict:
    """Write the YOLO dataset under ``out_dir`` and return AGGREGATE statistics only."""
    from mtah2027.vision.recovery import require_cv2

    cv2 = require_cv2()
    p = p or DatasetParams()
    rng = np.random.default_rng(p.seed)
    out = Path(out_dir)
    for sub in ("images/train", "labels/train", "images/val", "labels/val"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    n = min(len(ball_xy), hom.n_frames)
    ball = np.asarray(ball_xy, dtype=np.float64)[:n]
    elig = eligible_mask(ball, hom, blocks, params_v, pitch, p.margin_px)
    split = split_labels(n, p.block_frames, p.val_every, p.guard_frames)
    tr, va = select_frames(elig, split, p)
    if max_frame is not None:
        tr, va = tr[tr <= max_frame], va[va <= max_frame]
    near_gap = near_gap_mask(ball, p.near_gap_frames)
    W, Hh = params_v.frame_width, params_v.frame_height

    # ---- geometry of all crops (vectorised, before touching the video)
    u, v = project_pitch_to_pixels(hom.H[tr], ball[tr, 0], ball[tr, 1], pitch)
    t_origin, t_ball = random_positive_windows(u, v, rng, p, W, Hh)
    pl = np.asarray(players_xy)[tr].astype(np.float64)
    pu, pv = project_pitch_to_pixels(hom.H[tr], pl[..., 0], pl[..., 1], pitch)
    pu, pv = np.where(np.isfinite(pu), pu, np.nan), np.where(np.isfinite(pv), pv, np.nan)
    has_pl = np.isfinite(pu).any(axis=1) & (rng.uniform(size=len(tr)) < p.neg_prob)
    pick = np.array([rng.choice(np.flatnonzero(np.isfinite(pu[i]))) if has_pl[i] else 0 for i in range(len(tr))])
    cen = np.stack([pu[np.arange(len(tr)), pick], pv[np.arange(len(tr)), pick]], axis=1)
    neg_ok, n_origin = negative_windows(np.nan_to_num(cen), np.stack([u, v], 1), rng, p, W, Hh)
    neg_ok &= has_pl

    vu, vv = project_pitch_to_pixels(hom.H[va], ball[va, 0], ball[va, 1], pitch)
    ang = rng.uniform(0.0, 2.0 * np.pi, len(va))
    atd_search = ball[va] + p.atd_offset_m * rng.uniform(0.0, 1.0, len(va))[:, None] * np.stack(
        [np.cos(ang), np.sin(ang)], 1)
    gaps = real_gap_lengths(ball, blocks)
    prior = simulated_gap_prior(va, ball, np.asarray(players_xy)[:n], fps, params_v, rng, gaps)
    crops_meta: list[tuple] = []        # (frame, kind, x0, y0)
    for kind, search in ((0, atd_search), (1, prior)):
        su, sv = project_pitch_to_pixels(hom.H[va], search[:, 0], search[:, 1], pitch)
        okk = np.isfinite(su) & np.isfinite(sv)
        org = ac.crop_window(np.where(okk, su, W / 2), np.where(okk, sv, Hh / 2), p.crop, W, Hh)
        for i in np.flatnonzero(okk):
            crops_meta.append((int(va[i]), kind, int(org[i, 0]), int(org[i, 1])))

    # ---- jobs by frame, one sequential pass over the video
    jobs: dict[int, list[tuple]] = {}
    for i, f in enumerate(tr):
        bu, bv = t_ball[i]
        jobs.setdefault(int(f), []).append(("tp", int(t_origin[i, 0]), int(t_origin[i, 1]), float(bu), float(bv)))
        if neg_ok[i]:
            jobs.setdefault(int(f), []).append(("tn", int(n_origin[i, 0]), int(n_origin[i, 1]), -1.0, -1.0))
    vball = {int(f): (float(a), float(b)) for f, a, b in zip(va, vu, vv)}
    for f, kind, x0, y0 in crops_meta:
        jobs.setdefault(f, []).append(("v%d" % kind, x0, y0, vball[f][0] - x0, vball[f][1] - y0))

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError("cannot open the video")
    wanted = sorted(jobs)
    pos = 0
    n_tp = n_tn = 0
    val_rows: list[tuple] = []
    try:
        for f in wanted:
            while pos < f:
                if not cap.grab():
                    break
                pos += 1
            ok, img = cap.read()
            pos += 1
            if not ok or img is None:
                continue
            for kind, x0, y0, bu, bv in jobs[f]:
                crop = np.ascontiguousarray(img[y0:y0 + p.crop, x0:x0 + p.crop])
                if crop.shape[0] != p.crop or crop.shape[1] != p.crop:
                    continue
                if kind in ("tp", "tn"):
                    name = f"f{f:06d}_{'p' if kind == 'tp' else 'n'}"
                    _imwrite(cv2, out / "images/train" / f"{name}.jpg", crop, p.jpeg_quality)
                    line = yolo_line(bu, bv, p.crop) if kind == "tp" else ""
                    (out / "labels/train" / f"{name}.txt").write_text(line + ("\n" if line else ""))
                    n_tp += kind == "tp"
                    n_tn += kind == "tn"
                else:
                    k = int(kind[1])
                    name = f"f{f:06d}_{'atd' if k == 0 else 'prior'}"
                    _imwrite(cv2, out / "images/val" / f"{name}.png", crop, None)
                    inside = 0 <= bu < p.crop and 0 <= bv < p.crop
                    line = yolo_line(bu, bv, p.crop) if inside else ""
                    (out / "labels/val" / f"{name}.txt").write_text(line + ("\n" if line else ""))
                    val_rows.append((f, k, x0, y0, float(ball[f, 0]), float(ball[f, 1]), int(near_gap[f]),
                                     int(inside)))
    finally:
        cap.release()
    vm = np.array(val_rows, dtype=np.float64).reshape(-1, 8)
    np.savez_compressed(out / "val_meta.npz", frame=vm[:, 0].astype(np.int64), kind=vm[:, 1].astype(np.int8),
                        x0=vm[:, 2].astype(np.int64), y0=vm[:, 3].astype(np.int64), truth_xy=vm[:, 4:6],
                        near_gap=vm[:, 6].astype(bool), ball_inside=vm[:, 7].astype(bool))
    (out / "data.yaml").write_text(
        f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: ball\n")
    stats = {"eligible_frames": int(elig.sum()), "train_frames_selected": int(len(tr)),
             "val_frames_selected": int(len(va)), "train_positive_crops": int(n_tp),
             "train_negative_crops": int(n_tn), "val_crops": int(len(val_rows)),
             "val_crops_atd": int((vm[:, 1] == 0).sum()), "val_crops_prior": int((vm[:, 1] == 1).sum()),
             "val_prior_ball_inside_pct": round(100.0 * float(vm[vm[:, 1] == 1, 7].mean()), 2) if len(vm) else None,
             "val_near_gap_frames": int(np.unique(vm[(vm[:, 6] == 1), 0]).size) if len(vm) else 0,
             "block_frames": p.block_frames, "val_every": p.val_every, "guard_frames": p.guard_frames,
             "train_stride": p.train_stride, "val_stride": p.val_stride, "crop": p.crop, "seed": p.seed}
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    return stats


def default_cache_dir() -> Path:
    return Path(os.environ.get("MTAH_ISSUE68_CACHE", str(Path.home() / "issue68-cache")))
