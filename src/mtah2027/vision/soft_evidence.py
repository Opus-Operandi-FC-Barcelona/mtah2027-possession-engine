# ruff: noqa: E501
"""Soft ball evidence (Issue #67; off by default).

Instead of injecting a detection as *the ball* (which feeds every ball feature of the specialists), the point is
used only to tilt the H / A emission towards the team whose nearest player is closer to the detection.  The ball
features stay empty, so a wrong detection can at most move the H-vs-A odds by ``weight * confidence`` nats.

``tilt = weight * conf * tanh((d_A - d_H) / scale_m)`` is added to ``log P(H)`` and subtracted from ``log P(A)``
(half each side); ``D`` is untouched.  A detection acts on the frames within ``hold_frames`` of it (nearest
detection wins).  Pure NumPy.
"""
from __future__ import annotations

import numpy as np

H_COL, A_COL = 0, 1


def nearest_team_distance(players_xy: np.ndarray, team_of_slot: np.ndarray, frames: np.ndarray,
                          xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distance (m) from each detection to the closest home / away player of its frame (NaN if none)."""
    frames = np.asarray(frames, dtype=np.int64)
    pts = np.asarray(players_xy, dtype=np.float64)[frames]                  # [M, S, 2]
    d = np.linalg.norm(pts - np.asarray(xy, dtype=np.float64)[:, None, :], axis=2)   # [M, S]
    team = np.asarray(team_of_slot)
    out = []
    for t in (0, 1):
        dt = d[:, team == t]
        dt = np.where(np.isfinite(dt), dt, np.inf)
        m = dt.min(axis=1) if dt.shape[1] else np.full(len(frames), np.inf)
        out.append(np.where(np.isfinite(m), m, np.nan))
    return out[0], out[1]


def soft_tilt(n: int, players_xy: np.ndarray, team_of_slot: np.ndarray, frames: np.ndarray, xy: np.ndarray,
              conf: np.ndarray | float, *, weight: float, scale_m: float, hold_frames: int) -> np.ndarray:
    """Per-frame tilt ``[n]`` (nats, + favours H) from detections ``frames`` / ``xy`` with ``conf``."""
    tilt = np.zeros(n, dtype=np.float64)
    frames = np.asarray(frames, dtype=np.int64)
    if weight == 0.0 or not len(frames):
        return tilt
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    conf = np.broadcast_to(np.asarray(conf, dtype=np.float64), (len(frames),))
    order = np.argsort(frames, kind="stable")
    frames, xy, conf = frames[order], xy[order], conf[order]
    d_h, d_a = nearest_team_distance(players_xy, team_of_slot, frames, xy)
    val = weight * conf * np.tanh((d_a - d_h) / float(scale_m))
    val = np.where(np.isfinite(val), val, 0.0)
    best = np.full(n, np.inf)
    hold = max(int(hold_frames), 0)
    for off in range(-hold, hold + 1):
        idx = frames + off
        ok = (idx >= 0) & (idx < n)
        closer = ok & (abs(off) < best[np.where(ok, idx, 0)])
        tilt[idx[closer]] = val[closer]
        best[idx[closer]] = abs(off)
    return tilt


def apply_soft_evidence(log_probs: np.ndarray, tilt: np.ndarray) -> np.ndarray:
    """``log_probs`` ``[N, 3]`` (H, A, D) with ``+tilt / 2`` on H and ``-tilt / 2`` on A, rounded to 1e-6."""
    if not np.any(tilt):
        return log_probs
    lp = np.array(log_probs, dtype=np.float64)
    lp[:, H_COL] += tilt / 2.0
    lp[:, A_COL] -= tilt / 2.0
    return np.round(lp, 6)
