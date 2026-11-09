"""Post-processing that targets spellF1 directly (Issue #25).

spellF1 matches same-team possession spells by temporal IoU, so what matters is the *number* of
possessions and where they start and end.  After Viterbi decoding this module applies, inside every graded
play block (never across a block boundary; ``X`` frames are left alone), in this order:

1. **Forced dead intervals** (optional ``forced_dead_mask``, Issue #14 restarts and goals) are set to ``D``.
2. **D bridge.**  Inside a possession of one team longer than ``bridge_min_poss_s`` a *dead-evidence valley*
   (smoothed ``dead_score >= bridge_dead_threshold``, gaps shorter than ``bridge_gap_close_s`` bridged) of at
   least ``bridge_min_dead_s`` that also has geometric support (touchline / corner / goal area / out of
   play / set-piece tag, ``dead_support``) becomes ``D``: ``H -> D -> H`` counts as two possessions.
3. **Clean-up to a fixed point**, always acting on the shortest offending run first (ties: earliest):

   * a micro-possession shorter than ``micro_poss_s`` between two runs of the *other* team (``H A H`` or
     ``A H A``) is absorbed, *unless* the ball is visible for at least ``keep_min_visible_frac`` of it and it
     holds at least ``keep_min_touches`` touch peaks (a genuine interception: gain and pass);
   * a ``D`` run shorter than ``min_dead_s`` that does not overlap a forced interval is absorbed: between runs
     of the same team it joins them, between two different teams it is split at its midpoint, at a block
     edge it joins its only neighbour;
   * an ``H``/``A`` run shorter than ``min_isolated_poss_s`` squeezed between two ``D`` runs becomes ``D``.

Pure NumPy plus Python loops over *runs* (hundreds per half), never over frames.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from mtah2027.config import FPS, HALF1, HALF2_PLAY, load_section, params_from_mapping
from mtah2027.features._utils import mask_runs, moving_mean

H, A, D, X = 0, 1, 2, 3


@dataclass(frozen=True)
class SpellParams:
    micro_poss_s: float = 1.2
    keep_min_visible_frac: float = 0.5
    keep_min_touches: int = 2
    touch_min_score: float = 0.5
    touch_pad_frames: int = 3
    min_dead_s: float = 0.6
    min_isolated_poss_s: float = 0.8
    bridge_min_poss_s: float = 25.0
    bridge_min_dead_s: float = 4.0
    bridge_dead_threshold: float = 0.6
    bridge_gap_close_s: float = 1.0
    bridge_requires_geometry: bool = True

    @classmethod
    def from_config(cls, path=None) -> "SpellParams":
        return params_from_mapping(cls, load_section("spell_optimizer", path), "spell_optimizer")


def _rle(codes: np.ndarray) -> list[list[int]]:
    """Runs ``[start, end_exclusive, state]`` of a 1-D code array."""
    if len(codes) == 0:
        return []
    change = np.flatnonzero(np.diff(codes.astype(np.int16)) != 0) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(codes)]])
    return [[int(s), int(e), int(codes[s])] for s, e in zip(starts, ends)]


def _coalesce(runs: list[list[int]]) -> list[list[int]]:
    out: list[list[int]] = []
    for s, e, c in runs:
        if e <= s:
            continue
        if out and out[-1][2] == c and out[-1][1] == s:
            out[-1][1] = e
        else:
            out.append([s, e, c])
    return out


def _cum(x: np.ndarray) -> np.ndarray:
    """Prefix sums with a leading zero (``_cum(x)[j] - _cum(x)[i]`` = sum of ``x[i:j]``)."""
    return np.concatenate([[0], np.cumsum(x, dtype=np.int64)])


def _close_short_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    """Bridge False gaps shorter than ``max_gap`` that lie between True runs."""
    out = mask.copy()
    starts, ends = mask_runs(~mask)
    for a, b in zip(starts, ends):
        if a > 0 and b < len(mask) and b - a < max_gap:
            out[a:b] = True
    return out


def _bridge_dead(runs, base, dead_ok, support_cum, p, fps, stats):
    """Step 2: insert ``D`` bridges inside long possessions (``base`` = block start frame)."""
    out: list[list[int]] = []
    min_poss = int(round(p.bridge_min_poss_s * fps))
    min_dead = int(round(p.bridge_min_dead_s * fps))
    for s, e, c in runs:
        if c in (H, A) and e - s > min_poss:
            seg = dead_ok[s - base:e - base]
            starts, ends = mask_runs(seg)
            cuts = []
            for a, b in zip(starts, ends):
                if b - a < min_dead:
                    continue
                if p.bridge_requires_geometry and support_cum[s - base + b] - support_cum[s - base + a] == 0:
                    continue
                cuts.append((s + int(a), s + int(b)))
            if cuts:
                cur = s
                for a, b in cuts:
                    out.append([cur, a, c])
                    out.append([a, b, D])
                    cur = b
                out.append([cur, e, c])
                stats["bridges"] += len(cuts)
                continue
        out.append([s, e, c])
    return _coalesce(out)


def _cleanup(runs, base, visible_cum, touch_cum, forced_cum, p, fps, stats):
    """Step 3: iterate the three absorption rules to a fixed point (shortest offender first)."""
    micro = int(round(p.micro_poss_s * fps))
    min_d = int(round(p.min_dead_s * fps))
    iso = int(round(p.min_isolated_poss_s * fps))
    pad = int(p.touch_pad_frames)
    nbase = len(touch_cum) - 1

    def keeps(s, e):
        n = e - s
        vis = (visible_cum[e - base] - visible_cum[s - base]) / max(n, 1)
        lo, hi = max(0, s - base - pad), min(nbase, e - base + pad)
        touches = touch_cum[hi] - touch_cum[lo]
        return vis >= p.keep_min_visible_frac and touches >= p.keep_min_touches

    runs = _coalesce(runs)
    while True:
        best = None
        for i, (s, e, c) in enumerate(runs):
            n = e - s
            prev = runs[i - 1] if i > 0 else None
            nxt = runs[i + 1] if i + 1 < len(runs) else None
            kind = None
            if c in (H, A):
                if (prev and nxt and prev[2] == nxt[2] and prev[2] in (H, A) and prev[2] != c
                        and n < micro and not keeps(s, e)):
                    kind = "micro_poss"
                elif prev and nxt and prev[2] == D and nxt[2] == D and n < iso:
                    kind = "isolated"
            elif c == D and n < min_d and (forced_cum[e - base] - forced_cum[s - base]) == 0:
                if prev or nxt:
                    kind = "micro_dead"
            if kind and (best is None or (n, s) < (best[0], best[1])):
                best = (n, s, i, kind)
        if best is None:
            return runs
        _, _, i, kind = best
        s, e, c = runs[i]
        prev = runs[i - 1] if i > 0 else None
        nxt = runs[i + 1] if i + 1 < len(runs) else None
        stats[kind] += 1
        if kind == "micro_poss":
            runs[i][2] = prev[2]
        elif kind == "isolated":
            runs[i][2] = D
        else:
            if prev and nxt and prev[2] != nxt[2]:
                mid = s + (e - s) // 2
                prev[1] = mid
                runs[i] = [mid, e, nxt[2]]
            elif prev and nxt:
                runs[i][2] = prev[2]
            else:
                runs[i][2] = prev[2] if prev else nxt[2]
        runs = _coalesce(runs)


def optimize_spells(
        codes: np.ndarray, *, play_blocks: Sequence[tuple[int, int]] = (HALF1, HALF2_PLAY),
        ball_visible: np.ndarray | None = None, touch_score: np.ndarray | None = None,
        dead_score: np.ndarray | None = None, dead_support: np.ndarray | None = None,
        forced_dead_mask: np.ndarray | None = None, fps: float = FPS,
        params: SpellParams | None = None) -> tuple[np.ndarray, dict]:
    """Optimise the label codes (H=0, A=1, D=2, X=3) for spellF1; returns ``(codes, stats)``.

    ``ball_visible[N]`` (bool), ``touch_score[N]``, ``dead_score[N]`` and ``dead_support[N]`` (bool:
    touchline, corner, goal area, out of play or set-piece tag) feed the rules above; a missing array
    simply disables the corresponding evidence (with ``ball_visible`` None no ball is assumed visible, so
    micro-possessions are always absorbed).  ``forced_dead_mask[N]`` marks frames that must be ``D``.
    """
    p = params or SpellParams.from_config()
    out = np.array(codes, dtype=np.int8, copy=True)
    n = len(out)
    vis = np.zeros(n, bool) if ball_visible is None else np.asarray(ball_visible, dtype=bool)
    touch = (np.zeros(n, bool) if touch_score is None
             else np.nan_to_num(np.asarray(touch_score, dtype=np.float64)) >= p.touch_min_score)
    forced = np.zeros(n, bool) if forced_dead_mask is None else np.asarray(forced_dead_mask, dtype=bool)
    support = np.zeros(n, bool) if dead_support is None else np.asarray(dead_support, dtype=bool)
    ds = None if dead_score is None else np.nan_to_num(np.asarray(dead_score, dtype=np.float64))
    stats = {"forced_dead_frames": 0, "bridges": 0, "micro_poss": 0, "micro_dead": 0, "isolated": 0}
    for arr in (vis, touch, forced, support) + (() if ds is None else (ds,)):
        if arr.shape != (n,):
            raise ValueError("all per-frame inputs must have the same length as codes")

    for a, b in play_blocks:
        a, b = max(0, int(a)), min(n - 1, int(b))
        if b < a:
            continue
        sl = slice(a, b + 1)
        block = out[sl].copy()
        playable = block != X
        # 1. forced dead
        f = forced[sl] & playable
        stats["forced_dead_frames"] += int((f & (block != D)).sum())
        block[f] = D
        runs = _rle(block)
        # 2. D bridge
        if ds is not None:
            smooth = moving_mean(ds[sl], max(1, int(round(0.4 * fps)))).astype(np.float64)
            ok = np.nan_to_num(smooth) >= p.bridge_dead_threshold
            ok = _close_short_gaps(ok, int(round(p.bridge_gap_close_s * fps)))
            runs = _bridge_dead(runs, a, ok, _cum(support[sl]), p, fps, stats)
        # 3. clean-up; X frames split the block into independent segments
        cums = (_cum(vis[sl]), _cum(touch[sl]), _cum(f))
        merged: list[list[int]] = []
        seg: list[list[int]] = []
        for r in runs + [[b + 1, b + 1, X]]:
            if r[2] == X:
                if seg:
                    merged.extend(_cleanup(seg, a, *cums, p, fps, stats))
                    seg = []
                merged.append(r)
            else:
                seg.append(r)
        runs = merged
        for s, e, c in runs:
            block[s - a:e - a] = c
        out[sl] = block
    return out, stats
