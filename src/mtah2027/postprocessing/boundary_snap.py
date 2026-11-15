# ruff: noqa: E501
"""Post-Viterbi boundary refinement (Issue #61, off by default).

After the spell optimiser every boundary between two runs of the decoded block is looked at once, left to
right, and moved at most ``window_frames`` (N, fixed a priori) frames:

* ``H <-> A``: to the touch peak (``touch_score > 0``, ball visible and not aerial) of a player of the *gaining*
  team (``nearest_team`` of that frame) closest to the boundary (ties: the earlier one); no such peak: unchanged.
* ``D -> alive``: to the restart frame ``t*`` of a resolved set-piece anchor within ``+-N`` that no other
  boundary has used; no anchor: unchanged.
* ``alive -> D`` and block edges (``X``) are never moved.

A move is refused when it would relabel a frame of the forced dead mask, would leave a run empty or cross
the neighbouring boundary (the previous one, already refined, and the next original one).  Pure NumPy plus a
Python loop over *boundaries* (hundreds per half), deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from mtah2027.config import load_section, params_from_mapping

H, A, D, X = 0, 1, 2, 3


@dataclass(frozen=True)
class BoundarySnapParams:
    enabled: bool = False
    window_frames: int = 12
    snap_ha: bool = True
    snap_restart: bool = True

    @classmethod
    def from_config(cls, path=None) -> "BoundarySnapParams":
        return params_from_mapping(cls, load_section("boundary_snap", path), "boundary_snap")


def snap_boundaries(codes: np.ndarray, *, play_blocks: Sequence[tuple[int, int]], touch_score: np.ndarray | None,
                    nearest_team: np.ndarray | None, ball_visible: np.ndarray, is_aerial: np.ndarray | None = None,
                    restart_frames: np.ndarray | None = None, forced_dead_mask: np.ndarray | None = None,
                    params: BoundarySnapParams = BoundarySnapParams(enabled=True)) -> tuple[np.ndarray, dict]:
    """Return ``(codes, stats)``; ``codes`` is a refined copy (see the module docstring)."""
    out = np.asarray(codes).copy()
    stats = {"ha_moved": 0, "ha_candidates": 0, "restart_moved": 0, "restart_candidates": 0}
    n = len(out)
    nw = int(params.window_frames)
    if not params.enabled or nw <= 0:
        return out, stats
    touch = np.zeros(n, bool)
    if touch_score is not None and nearest_team is not None:
        touch = (np.asarray(touch_score) > 0) & np.asarray(ball_visible, bool)
        if is_aerial is not None:
            touch &= ~np.asarray(is_aerial, bool)
    team = np.asarray(nearest_team) if nearest_team is not None else np.full(n, -1)
    forced = np.asarray(forced_dead_mask, bool) if forced_dead_mask is not None else np.zeros(n, bool)
    used: set[int] = set()
    anchors = np.sort(np.asarray(restart_frames, np.int64)) if restart_frames is not None else np.zeros(0, np.int64)
    for a, b in play_blocks:
        a, b = max(0, int(a)), min(n - 1, int(b))
        if b <= a:
            continue
        seg = out[a:b + 1]
        starts = np.concatenate([[0], np.flatnonzero(seg[1:] != seg[:-1]) + 1]) + a    # first frame of each run
        bounds = starts[1:]
        prev = int(starts[0])                                                           # start of current left run
        for k, pos in enumerate(bounds):
            pos = int(pos)
            left, right = int(out[pos - 1]), int(out[pos])
            if X in (left, right):
                prev = pos
                continue
            nxt = int(bounds[k + 1]) if k + 1 < len(bounds) else b + 1
            lo, hi = prev + 1, nxt - 1               # new boundary range keeps both runs non-empty
            new = pos
            if left != D and right != D and params.snap_ha:
                stats["ha_candidates"] += 1
                w0, w1 = max(lo, pos - nw), min(hi, pos + nw)
                if w1 >= w0:
                    idx = np.arange(w0, w1 + 1)
                    ok = touch[idx] & (team[idx] == right)
                    if ok.any():
                        cand = idx[ok]
                        new = int(cand[np.argmin(np.abs(cand - pos) * 4 + (cand > pos))])   # nearest, earlier on ties
            elif left == D and right != D and params.snap_restart and len(anchors):
                stats["restart_candidates"] += 1
                inwin = anchors[(np.abs(anchors - pos) <= nw) & (anchors >= lo) & (anchors <= hi)]
                inwin = np.array([t for t in inwin if int(t) not in used], dtype=np.int64)
                if len(inwin):
                    new = int(inwin[np.argmin(np.abs(inwin - pos) * 4 + (inwin > pos))])
                    used.add(new)
            if new != pos:
                lo_f, hi_f = min(pos, new), max(pos, new)
                if forced[lo_f:hi_f].any():
                    new = pos
                else:
                    out[lo_f:hi_f] = right if new < pos else left
                    stats["ha_moved" if left != D and right != D else "restart_moved"] += 1
            prev = new
    return out, stats
