"""Slot cleaning: off-pitch staff and duplicate tracks (Issue #8).

The ATD has 46 (tactical) / 51 (broadcast) *slots* that are tracklets, not people: substitutes and
staff on the touchline are tracked too, and one player is sometimes tracked twice at the same time
under two slots.  Two vectorised rules turn the raw slots into at most 11 active players per team:

1. **Bounds.**  A slot outside ``[-2, L + 2] x [-2, W + 2]`` metres (default ``[-2, 107.3] x
   [-2, 70.0]``) is masked as NaN in that frame.
2. **Duplicates.**  When two active slots of the *same team* are closer than ``0.6 m`` in the same
   frame, the one belonging to the shorter continuous track (length of the non-NaN run containing
   the frame, measured after the bounds rule) is deactivated; ties keep the lower slot index.  The
   rule is applied greedily from the best track downwards, so a chain of near slots keeps one
   representative and never drops a slot that is only near an already-removed duplicate.

Inputs are never modified; frames are processed in chunks so memory stays bounded.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mtah2027.config import PITCH

BOUNDS_MARGIN_M = 2.0
DUPLICATE_DISTANCE_M = 0.6
CHUNK_FRAMES = 8192


@dataclass
class CleanedTracks:
    """Cleaned slot arrays plus the masks that explain what was removed."""

    players_xy: np.ndarray              # [N, S, 2] float32 metres, NaN where removed
    players_speed: np.ndarray | None    # [N, S] or None
    extra: dict[str, np.ndarray]        # other per-slot channels masked the same way
    active: np.ndarray                  # [N, S] bool: slot has a position after cleaning
    n_active: np.ndarray                # [N] int16: active players per frame (both teams)
    n_active_by_team: np.ndarray        # [N, 2] int16: home, away
    removed_out_of_bounds: np.ndarray   # [N, S] bool
    removed_duplicate: np.ndarray       # [N, S] bool
    params: dict = field(default_factory=dict)


def run_lengths(valid: np.ndarray) -> np.ndarray:
    """Length of the contiguous True run containing each element along axis 0 (0 where False)."""
    n, s = valid.shape
    out = np.zeros((n, s), dtype=np.int32)
    padded = np.zeros((n + 2, s), dtype=np.int8)
    padded[1:-1] = valid
    edges = np.diff(padded, axis=0)                       # [n+1, s]: +1 run start, -1 run end
    for k in range(s):
        starts = np.flatnonzero(edges[:, k] == 1)
        ends = np.flatnonzero(edges[:, k] == -1)
        if len(starts):
            lengths = ends - starts
            out[valid[:, k], k] = np.repeat(lengths, lengths)
    return out


def _dedupe_team(xy: np.ndarray, active: np.ndarray, score: np.ndarray, threshold: float) -> np.ndarray:
    """Greedy duplicate removal inside one chunk and one team.

    ``xy`` [n, k, 2], ``active`` [n, k] bool, ``score`` [n, k] unique priorities (higher is kept).
    Returns the boolean mask of slots to remove.
    """
    n, k = active.shape
    if k < 2:
        return np.zeros_like(active)
    dx = xy[:, :, None, 0] - xy[:, None, :, 0]
    dy = xy[:, :, None, 1] - xy[:, None, :, 1]
    with np.errstate(invalid="ignore"):
        close = (dx * dx + dy * dy) < threshold * threshold
    close &= active[:, :, None] & active[:, None, :]
    close[:, np.arange(k), np.arange(k)] = False
    if not close.any():
        return np.zeros_like(active)
    keep = active.copy()
    order = np.argsort(-score, axis=1, kind="stable")       # best track first
    rows = np.arange(n)
    for r in range(k):
        idx = order[:, r]
        alive = keep[rows, idx]
        if not alive.any():
            continue
        kill = close[rows, idx] & keep & alive[:, None]
        keep &= ~kill
    return active & ~keep


def clean_tracks(players_xy: np.ndarray, team_of_slot: np.ndarray, players_speed: np.ndarray | None = None,
                 *, extra: dict[str, np.ndarray] | None = None, pitch: tuple[float, float] = PITCH,
                 margin: float = BOUNDS_MARGIN_M, duplicate_distance: float = DUPLICATE_DISTANCE_M,
                 chunk_frames: int = CHUNK_FRAMES) -> CleanedTracks:
    """Mask off-pitch slots and merge same-team duplicate tracks (see module docstring)."""
    xy = np.array(players_xy, dtype=np.float32, copy=True)
    n, s, _ = xy.shape
    team = np.asarray(team_of_slot)
    if team.shape != (s,):
        raise ValueError(f"team_of_slot has shape {team.shape}, expected ({s},)")
    length, width = pitch
    lo_x, hi_x, lo_y, hi_y = -margin, length + margin, -margin, width + margin

    x, y = xy[..., 0], xy[..., 1]
    finite = np.isfinite(x) & np.isfinite(y)
    with np.errstate(invalid="ignore"):
        inside = (x >= lo_x) & (x <= hi_x) & (y >= lo_y) & (y <= hi_y)
    oob = finite & ~inside
    active = finite & inside

    runs = run_lengths(active)
    # unique, higher = better: longer run first, then lower slot index
    score = runs.astype(np.int64) * (s + 1) + (s - np.arange(s))[None, :]
    dup = np.zeros((n, s), dtype=bool)
    for t in np.unique(team):
        cols = np.flatnonzero(team == t)
        for c0 in range(0, n, chunk_frames):
            sl = slice(c0, min(n, c0 + chunk_frames))
            dup[sl][:, cols] = _dedupe_team(xy[sl][:, cols], active[sl][:, cols], score[sl][:, cols],
                                            duplicate_distance)
    removed = oob | dup
    xy[removed] = np.nan
    active &= ~dup

    speed = None
    if players_speed is not None:
        speed = np.array(players_speed, dtype=np.float32, copy=True)
        speed[removed] = np.nan
    extras = {}
    for name, arr in (extra or {}).items():
        arr = np.array(arr, dtype=np.float32, copy=True)
        arr[removed] = np.nan
        extras[name] = arr
    by_team = np.stack([(active[:, team == 0]).sum(axis=1), (active[:, team == 1]).sum(axis=1)],
                       axis=1).astype(np.int16)
    return CleanedTracks(
        players_xy=xy, players_speed=speed, extra=extras, active=active,
        n_active=active.sum(axis=1).astype(np.int16), n_active_by_team=by_team,
        removed_out_of_bounds=oob, removed_duplicate=dup,
        params={"margin": margin, "duplicate_distance": duplicate_distance, "pitch": tuple(pitch)})
