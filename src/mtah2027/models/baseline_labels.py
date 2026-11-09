"""Nearest-player baseline on arrays (the official ``starter/baseline.py`` recipe) and its last-resort use.

Per frame the team of the player closest to the ball (nobody when the ball is missing), then a majority vote
over ``+-window`` frames, ``D`` where nobody had the ball within the window.  Ties go to the home side, like
the official ``smooth``.  It is used (a) as the reference the engine must beat on external matches and (b) as
the last fallback of the pipeline if the whole heuristic route fails.
"""
from __future__ import annotations

import numpy as np

H, A, D, X = 0, 1, 2, 3
NOBODY = -1


def nearest_player_team(players_xy: np.ndarray, team_of_slot: np.ndarray, ball_xy: np.ndarray) -> np.ndarray:
    """int8 ``[N]``: team (0 home, 1 away) of the slot closest to the ball (``-1``: no answer)."""
    n = len(ball_xy)
    out = np.full(n, NOBODY, dtype=np.int8)
    team = np.asarray(team_of_slot)
    chunk = 20000
    for c0 in range(0, n, chunk):
        sl = slice(c0, min(n, c0 + chunk))
        diff = players_xy[sl].astype(np.float64) - ball_xy[sl, None, :].astype(np.float64)
        dist = np.hypot(diff[..., 0], diff[..., 1])
        ok = np.isfinite(dist).any(axis=1) & np.isfinite(ball_xy[sl]).all(axis=1)
        nearest = np.argmin(np.where(np.isfinite(dist), dist, np.inf), axis=1)
        out[sl] = np.where(ok, team[nearest], NOBODY)
    return out


def smooth_majority(codes: np.ndarray, window: int) -> np.ndarray:
    """Majority vote over ``+-window`` frames: H / A, or D where nobody had the ball in the window."""
    codes = np.asarray(codes)
    n = len(codes)
    out = np.full(n, D, dtype=np.int8)
    if window <= 0:
        out[codes == H] = H
        out[codes == A] = A
        return out
    cum_h = np.concatenate([[0], np.cumsum(codes == H)])
    cum_a = np.concatenate([[0], np.cumsum(codes == A)])
    idx = np.arange(n)
    lo, hi = np.maximum(0, idx - window), np.minimum(n, idx + window + 1)
    home, away = cum_h[hi] - cum_h[lo], cum_a[hi] - cum_a[lo]
    seen = (home + away) > 0
    out[seen] = np.where(home[seen] >= away[seen], H, A)
    return out


def baseline_labels(players_xy: np.ndarray, team_of_slot: np.ndarray, ball_xy: np.ndarray,
                    window: int = 50) -> np.ndarray:
    """Nearest-player + ``+-window`` majority labels (codes H=0, A=1, D=2) for every frame."""
    return smooth_majority(nearest_player_team(players_xy, team_of_slot, ball_xy), window)
