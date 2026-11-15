# ruff: noqa: E501
"""Post-decoding boundary refinement with the TCN turnover head (Issue #79, ASRF-style, off by default).

Action-segmentation refinement (Ishikawa et al., WACV 2021) uses a boundary-probability branch *after* the
segmentation to cut segments.  Here: inside every decoded ``H`` / ``A`` possession (a maximal run of one team,
delimited by the other team, ``D``, ``X`` or the block edge) the turnover head of the sequence model
(``p[:, 1]`` = H -> A, ``p[:, 2]`` = A -> H, ``+-half_window`` frames around the change) is searched, left to right, for a
peak ``>= peak_threshold`` (a local maximum over ``+-peak_window_frames``) such that the team evidence of the
emissions favours the other team: the mean of ``log P(other) - log P(current)`` over the following
``window_s`` seconds (clipped to the end of the possession) is ``>= margin``.  The possession is split at the peak
and the remainder relabelled to the other team up to the next decoded boundary, or until the same windowed evidence
flips back (``<= -margin``), where the original team resumes.  The piece left of the split and the relabelled piece
must last at least ``min_piece_s`` (fixed a priori, not part of the grid).  Pure NumPy, one Python loop over the
decoded possessions, deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from mtah2027.config import load_section, params_from_mapping

H, A, D, X = 0, 1, 2, 3


@dataclass(frozen=True)
class BoundaryRefineParams:
    enabled: bool = False
    peak_threshold: float = 0.7      # minimum turnover-head probability of the split frame
    window_s: float = 2.0            # evidence window after the split (seconds)
    margin: float = 1.0              # minimum mean log-odds of the other team in the window
    min_piece_s: float = 0.5         # minimum length of the left piece and of the relabelled piece
    peak_window_frames: int = 5      # the split frame is a local maximum of the head over +- this many frames

    @classmethod
    def from_config(cls, path=None) -> "BoundaryRefineParams":
        return params_from_mapping(cls, load_section("boundary_refine", path), "boundary_refine")


def _windowed_mean(x: np.ndarray, w: int) -> np.ndarray:
    """``out[t] = mean(x[t:t + w])`` (the window is truncated at the end of ``x``)."""
    c = np.concatenate([[0.0], np.cumsum(x)])
    n = len(x)
    hi = np.minimum(np.arange(n) + w, n)
    return (c[hi] - c[np.arange(n)]) / np.maximum(hi - np.arange(n), 1)


def refine_boundaries(codes: np.ndarray, *, play_blocks: Sequence[tuple[int, int]], turnover_prob: np.ndarray | None,
                      log_probs: np.ndarray, fps: float = 25.0,
                      params: BoundaryRefineParams = BoundaryRefineParams(enabled=True)) -> tuple[np.ndarray, dict]:
    """Return ``(codes, stats)``; ``codes`` is a refined copy (see the module docstring)."""
    out = np.asarray(codes).copy()
    stats = {"possessions": 0, "candidates": 0, "splits": 0, "frames_relabelled": 0, "flip_back": 0}
    n = len(out)
    if not params.enabled or turnover_prob is None or n == 0:
        return out, stats
    tp = np.nan_to_num(np.asarray(turnover_prob, np.float64), nan=0.0)
    lp = np.asarray(log_probs, np.float64)
    w = max(int(round(params.window_s * fps)), 1)
    min_piece = max(int(round(params.min_piece_s * fps)), 1)
    pw = max(int(params.peak_window_frames), 0)
    # log-odds of team H over A per frame and its windowed means (A over H is the negation)
    odds = lp[:, H] - lp[:, A]
    thr, margin = float(params.peak_threshold), float(params.margin)
    for a, b in play_blocks:
        a, b = max(0, int(a)), min(n - 1, int(b))
        if b <= a:
            continue
        seg = out[a:b + 1]
        starts = np.concatenate([[0], np.flatnonzero(seg[1:] != seg[:-1]) + 1]) + a
        ends = np.concatenate([starts[1:] - 1, [b]])
        for s, e in zip(starts, ends):
            s, e = int(s), int(e)
            c = int(out[s])
            if c not in (H, A) or e - s + 1 < 2 * min_piece:
                continue
            stats["possessions"] += 1
            sign = 1.0 if c == A else -1.0         # evidence for the other team: A over H when c == H
            ev = _windowed_mean(sign * odds[s:e + 1], w)         # ev[k]: window starting at s + k, clipped at e
            head = tp[s:e + 1, 1 if c == H else 2]
            if pw > 0:
                pad = np.pad(head, pw, constant_values=-1.0)
                local_max = np.max(np.lib.stride_tricks.sliding_window_view(pad, 2 * pw + 1), axis=1)
            else:
                local_max = head
            cand = np.flatnonzero((head >= thr) & (head >= local_max))
            cand = cand[(cand >= min_piece) & (cand <= e - s + 1 - min_piece)]
            k = 0
            while k < len(cand):
                t = int(cand[k])
                k += 1
                stats["candidates"] += 1
                if ev[t] < margin:
                    continue
                # end of the relabelled piece: until the evidence flips back (original team favoured by the margin)
                back = np.flatnonzero(ev[t + w:] <= -margin) if t + w < len(ev) else np.zeros(0, np.int64)
                u = t + w + int(back[0]) if len(back) else e - s + 1
                if u - t < min_piece:
                    continue
                out[s + t:s + u] = A if c == H else H
                stats["splits"] += 1
                stats["frames_relabelled"] += u - t
                if len(back):
                    stats["flip_back"] += 1
                    while k < len(cand) and cand[k] < u:     # the original team resumes at u: later peaks are judged again
                        k += 1
                else:
                    break
    return out, stats
