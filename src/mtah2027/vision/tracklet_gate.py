"""Tracklet gate for the video ball detections (Issue #66; off by default).

A detection inside a tracking gap is trusted only if it belongs to a *tracklet*: a run of at least
``tracklet_min_len`` detections of consecutive planned crops (at most ``tracklet_max_link_frames`` apart),
whose
step speeds are plausible (``<= tracklet_max_speed_mps`` plus a position slack) and which lie within
``tracklet_max_residual_m`` of a smooth fit of the run (constant velocity for short runs, quadratic from 6
points).  Isolated detections, and points that do not follow the run, are dropped.  Pure NumPy, deterministic.
"""
from __future__ import annotations

import numpy as np

from mtah2027.vision.attention_crops import VisionParams

_QUADRATIC_FROM = 6


def _residuals(t: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Distance of every point to the least-squares polynomial fit of the run."""
    deg = 2 if len(t) >= _QUADRATIC_FROM else 1
    tc = (t - t.mean()) / max(float(np.ptp(t)), 1.0)
    a = np.vander(tc, deg + 1)
    coef, *_ = np.linalg.lstsq(a, xy, rcond=None)
    return np.hypot(*(xy - a @ coef).T)


def _links(f: np.ndarray, xy: np.ndarray, fps: float, p: VisionParams) -> np.ndarray:
    """``link[i]``: point i+1 continues the run of point i (``len(f) - 1`` booleans)."""
    df = np.diff(f).astype(np.float64)
    dist = np.hypot(*np.diff(xy, axis=0).T)
    return (df > 0) & (df <= p.tracklet_max_link_frames) & \
        (dist <= p.tracklet_max_speed_mps * df / fps + p.tracklet_speed_slack_m)


def tracklet_mask(frames: np.ndarray, xy: np.ndarray, *, fps: float, params: VisionParams | None = None
                  ) -> np.ndarray:
    """Boolean mask over the input rows: ``True`` where the detection belongs to a valid tracklet."""
    p = params or VisionParams.from_config()
    frames = np.asarray(frames, dtype=np.int64)
    pts = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    out = np.zeros(len(frames), dtype=bool)
    if not len(frames):
        return out
    order = np.argsort(frames, kind="stable")
    f, q = frames[order], pts[order]
    alive = np.isfinite(q).all(axis=1)
    min_len = max(int(p.tracklet_min_len), 1)
    for _ in range(len(f) + 1):
        idx = np.flatnonzero(alive)
        if not len(idx):
            break
        link = _links(f[idx], q[idx], fps, p) if len(idx) > 1 else np.zeros(0, bool)
        starts = np.flatnonzero(np.concatenate([[True], ~link]))
        ends = np.concatenate([starts[1:], [len(idx)]])
        changed = False
        for s, e in zip(starts, ends):
            run = idx[s:e]
            if len(run) < min_len:
                alive[run] = False
                changed = True
                continue
            if len(run) < 3:
                continue
            res = _residuals(f[run].astype(np.float64), q[run])
            j = int(np.argmax(res))
            if res[j] > p.tracklet_max_residual_m:
                alive[run[j]] = False          # drop the worst point, then re-link and re-check
                changed = True
        if not changed:
            break
    out[order] = alive
    return out


def gate_points(frames: np.ndarray, xy: np.ndarray, *, fps: float, params: VisionParams | None = None
                ) -> tuple[np.ndarray, np.ndarray]:
    """``(frames, xy)`` restricted to the points of valid tracklets (input unchanged when the gate is off)."""
    p = params or VisionParams.from_config()
    frames = np.asarray(frames)
    xy = np.asarray(xy).reshape(-1, 2)
    if not p.tracklet_gate:
        return frames, xy
    keep = tracklet_mask(frames, xy, fps=fps, params=p)
    return frames[keep], xy[keep]


def temper_injected(log_probs: np.ndarray, ball_source: np.ndarray | None, scale: float) -> np.ndarray:
    """Emissions with the rows whose ball came from video (``ball_source`` 2 or 3) multiplied by ``scale``.

    ``scale == 1`` (default) or no source returns the input object unchanged.
    """
    if ball_source is None or scale == 1.0:
        return log_probs
    inj = np.isin(np.asarray(ball_source), (2, 3))
    if not inj.any():
        return log_probs
    lp = np.array(log_probs, dtype=np.float64)
    lp[inj] = np.round(lp[inj] * float(scale), 6)
    return lp
