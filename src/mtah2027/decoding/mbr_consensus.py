"""Consensus of several decodings by matching their possession segments (Issue #26; wave-3 option).

Several runs/models give several label arrays for the same match.  Because spellF1 rewards the right
*number* of possessions and boundaries within a few frames, the consensus works on segments, not frames:

1. the **reference** run is the one whose segments agree best with all the others (largest sum of matched
   IoU; ties go to the lowest index);
2. every other run is matched to the reference segment by segment: same team only, greedy by decreasing IoU
   (ties: earlier start), IoU >= ``min_iou``, one-to-one -- the rule of the official metric;
3. a reference segment supported by at least ``min_support`` of the runs (itself included) is kept with the
   **median** start and end of its matches (half-integers round up);
4. overlaps between neighbouring consensus segments are split at the midpoint; frames not covered become
   ``D`` where at least half of the runs say ``D`` and otherwise continue the previous segment's team.

Everything is deterministic.  Frames outside ``blocks`` are taken from the reference run.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

H, A, D, X = 0, 1, 2, 3


def possession_segments(codes: np.ndarray, blocks: Sequence[tuple[int, int]]) -> np.ndarray:
    """H/A runs inside ``blocks`` as an int array ``[K, 3]`` of ``(team, start, end_inclusive)``."""
    codes = np.asarray(codes)
    rows = []
    for a, b in blocks:
        seg = codes[a:b + 1]
        if not len(seg):
            continue
        change = np.flatnonzero(np.diff(seg.astype(np.int16)) != 0) + 1
        starts = np.concatenate([[0], change])
        ends = np.concatenate([change, [len(seg)]]) - 1
        for s, e in zip(starts, ends):
            if seg[s] in (H, A):
                rows.append((int(seg[s]), a + int(s), a + int(e)))
    return np.array(rows, dtype=np.int64).reshape(-1, 3)


def match_segments(ref: np.ndarray, other: np.ndarray, min_iou: float) -> list[tuple[int, int, float]]:
    """Greedy one-to-one same-team matching ``[(ref_index, other_index, iou)]`` by decreasing IoU."""
    if not len(ref) or not len(other):
        return []
    inter = (np.minimum(ref[:, None, 2], other[None, :, 2]) - np.maximum(ref[:, None, 1], other[None, :, 1])
             + 1).clip(min=0)
    union = ((ref[:, 2] - ref[:, 1] + 1)[:, None] + (other[:, 2] - other[:, 1] + 1)[None, :] - inter)
    iou = inter / np.maximum(union, 1)
    iou = np.where(ref[:, None, 0] == other[None, :, 0], iou, 0.0)
    ri, oi = np.nonzero(iou >= max(min_iou, 1e-12))
    if not len(ri):
        return []
    order = np.lexsort((oi, ri, -iou[ri, oi]))
    used_r, used_o, out = set(), set(), []
    for k in order:
        r, o = int(ri[k]), int(oi[k])
        if r in used_r or o in used_o:
            continue
        used_r.add(r)
        used_o.add(o)
        out.append((r, o, float(iou[r, o])))
    return sorted(out)


def _median_int(values: Sequence[int]) -> int:
    return int(np.floor(np.median(np.asarray(values, dtype=np.float64)) + 0.5))


def consensus_labels(runs: Sequence[np.ndarray], blocks: Sequence[tuple[int, int]], *, min_iou: float = 0.3,
                     min_support: float = 0.5) -> np.ndarray:
    """Consensus label codes (int8) of several equally long label-code arrays (see the module docstring)."""
    arrays = [np.asarray(r, dtype=np.int8) for r in runs]
    if not arrays:
        raise ValueError("at least one run is required")
    n = len(arrays[0])
    if any(len(r) != n for r in arrays):
        raise ValueError("all runs must have the same length")
    if len(arrays) == 1:
        return arrays[0].copy()
    segs = [possession_segments(r, blocks) for r in arrays]
    k = len(arrays)
    pair: dict[tuple[int, int], list] = {}
    score = np.zeros(k)
    for i in range(k):
        for j in range(i + 1, k):
            m = match_segments(segs[i], segs[j], min_iou)
            pair[(i, j)] = m
            total = sum(x[2] for x in m)
            score[i] += total
            score[j] += total
    ref = int(np.argmax(score))                                  # first maximum -> lowest index
    out = arrays[ref].copy()
    matches = {}
    for j in range(k):
        if j != ref:
            if ref < j:
                matches[j] = {r: o for r, o, _ in pair[(ref, j)]}
            else:
                matches[j] = {r: o for o, r, _ in pair[(j, ref)]}
    for a, b in blocks:
        votes_dead = np.sum([r[a:b + 1] == D for r in arrays], axis=0) * 2 >= k
        cons = []
        for idx, (team, s, e) in enumerate(segs[ref]):
            if not (a <= s <= b):
                continue
            starts, ends = [int(s)], [int(e)]
            for j, mp in matches.items():
                if idx in mp:
                    starts.append(int(segs[j][mp[idx], 1]))
                    ends.append(int(segs[j][mp[idx], 2]))
            if len(starts) / k >= min_support:
                cons.append([int(team), _median_int(starts), _median_int(ends)])
        cons.sort(key=lambda c: (c[1], c[2]))
        for prev, cur in zip(cons, cons[1:]):
            if cur[1] <= prev[2]:
                mid = (prev[2] + cur[1]) // 2
                prev[2], cur[1] = mid, mid + 1
        cons = [c for c in cons if c[2] >= c[1]]
        block = np.where(votes_dead, D, -1).astype(np.int8)
        seg_team = np.full(b - a + 1, -1, dtype=np.int8)
        for team, s, e in cons:
            lo, hi = max(a, s) - a, min(b, e) - a + 1
            block[lo:hi] = team
            seg_team[lo:hi] = team
        if cons:
            # uncovered, non-dead frames continue the previous segment's team (before the first: the next)
            idx = np.arange(len(block))
            last = np.maximum.accumulate(np.where(seg_team >= 0, idx, -1))
            cont = np.where(last >= 0, seg_team[np.maximum(last, 0)], cons[0][0])
            hole = block < 0
            block[hole] = cont[hole]
        else:
            counts = np.stack([np.sum([r[a:b + 1] == c for r in arrays], axis=0) for c in range(4)])
            block = np.argmax(counts, axis=0).astype(np.int8)       # ties -> lowest code
        out[a:b + 1] = block
    return out
