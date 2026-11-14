"""Viterbi decoding with a time-dependent transition matrix (Issue #24).

``viterbi_dynamic(log_probs[N, 3], log_trans[N, 3, 3], log_init[3])`` returns the single best state path
(H=0, A=1, D=2) maximising ``log_init[s_0] + log_probs[0, s_0] + sum_t (log_trans[t, s_{t-1}, s_t] +
log_probs[t, s_t])``.  ``log_trans[t, i, j]`` scores the move into frame ``t`` (row 0 is ignored).

Two implementations execute *the same sequence of IEEE-754 float64 operations* and therefore give
bit-identical paths:

* a ``numba.njit`` kernel (used when numba is importable and not disabled with ``MTAH_DISABLE_NUMBA=1``), and
* a pure-Python kernel over plain lists (the fallback).

Ties are broken deterministically: among equal predecessor scores the *lowest* state index wins, and the
final state is the lowest index among equal maxima.  Inputs are quantised with ``np.round(., 6)`` first,
as required for byte-exact reproducibility.

``decode_match_halves`` decodes every graded block independently (default: first half and second-half
play), forces ``D`` on the dead block of the second half and leaves ``X`` elsewhere.  Block boundaries are
parameters so that the short synthetic match (scaled boundaries) can use them too.
"""
from __future__ import annotations

import os
from typing import Sequence

import numpy as np

from mtah2027.config import HALF1, HALF2_DEAD, HALF2_PLAY, load_section

H, A, D, X = 0, 1, 2, 3
N_STATES = 3
ROUND_DECIMALS = 6
_NEG = -1.0e300         # replaces -inf so that sums stay ordered and finite in both kernels

try:  # numba is an optional accelerator
    import numba as _numba
except Exception:  # pragma: no cover - exercised only where numba is absent
    _numba = None


def numba_available() -> bool:
    """True when the njit kernel can be used (numba importable and not disabled by the environment)."""
    return _numba is not None and os.environ.get("MTAH_DISABLE_NUMBA", "") not in ("1", "true", "True")


# --------------------------------------------------------------------------------------------
# Kernels
# --------------------------------------------------------------------------------------------
def _viterbi_kernel(lp, tr, init, bp):
    """Array kernel (also the source of the njit version).  Fills ``bp[N, 3]``, returns the final path."""
    n = lp.shape[0]
    path = np.empty(n, dtype=np.int8)
    d0 = init[0] + lp[0, 0]
    d1 = init[1] + lp[0, 1]
    d2 = init[2] + lp[0, 2]
    for t in range(1, n):
        # j = 0
        best = d0 + tr[t, 0, 0]
        arg = 0
        v = d1 + tr[t, 1, 0]
        if v > best:
            best = v
            arg = 1
        v = d2 + tr[t, 2, 0]
        if v > best:
            best = v
            arg = 2
        n0 = best + lp[t, 0]
        bp[t, 0] = arg
        # j = 1
        best = d0 + tr[t, 0, 1]
        arg = 0
        v = d1 + tr[t, 1, 1]
        if v > best:
            best = v
            arg = 1
        v = d2 + tr[t, 2, 1]
        if v > best:
            best = v
            arg = 2
        n1 = best + lp[t, 1]
        bp[t, 1] = arg
        # j = 2
        best = d0 + tr[t, 0, 2]
        arg = 0
        v = d1 + tr[t, 1, 2]
        if v > best:
            best = v
            arg = 1
        v = d2 + tr[t, 2, 2]
        if v > best:
            best = v
            arg = 2
        n2 = best + lp[t, 2]
        bp[t, 2] = arg
        d0, d1, d2 = n0, n1, n2
    state = 0
    best = d0
    if d1 > best:
        best = d1
        state = 1
    if d2 > best:
        state = 2
    for t in range(n - 1, -1, -1):
        path[t] = state
        if t > 0:
            state = bp[t, state]
    return path


def _viterbi_python(lp, tr, init):
    """Pure-Python kernel over lists of floats (identical arithmetic to :func:`_viterbi_kernel`)."""
    n = len(lp)
    bp = [None] * n
    d0 = init[0] + lp[0][0]
    d1 = init[1] + lp[0][1]
    d2 = init[2] + lp[0][2]
    for t in range(1, n):
        row, cost = tr[t], lp[t]
        r0, r1, r2 = row
        cand = []
        for j in (0, 1, 2):
            best = d0 + r0[j]
            arg = 0
            v = d1 + r1[j]
            if v > best:
                best = v
                arg = 1
            v = d2 + r2[j]
            if v > best:
                best = v
                arg = 2
            cand.append((best + cost[j], arg))
        d0, d1, d2 = cand[0][0], cand[1][0], cand[2][0]
        bp[t] = (cand[0][1], cand[1][1], cand[2][1])
    state = 0
    best = d0
    if d1 > best:
        best = d1
        state = 1
    if d2 > best:
        state = 2
    path = [0] * n
    for t in range(n - 1, -1, -1):
        path[t] = state
        if t > 0:
            state = bp[t][state]
    return np.array(path, dtype=np.int8)


_njit_kernel = None


def _get_njit_kernel():
    global _njit_kernel
    if _njit_kernel is None:
        _njit_kernel = _numba.njit(cache=False)(_viterbi_kernel)
    return _njit_kernel


# --------------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------------
def default_log_init(path=None) -> np.ndarray:
    """``log_init`` (H, A, D) from the ``viterbi`` section of the configuration."""
    cfg = load_section("viterbi", path)
    return np.asarray(cfg.get("log_init", [-0.6931, -0.6931, -3.0]), dtype=np.float64)


def _prepare(log_probs, log_trans, log_init):
    lp = np.round(np.asarray(log_probs, dtype=np.float64), ROUND_DECIMALS)
    tr = np.round(np.asarray(log_trans, dtype=np.float64), ROUND_DECIMALS)
    if lp.ndim != 2 or lp.shape[1] != N_STATES:
        raise ValueError(f"log_probs must be [N, 3], got {lp.shape}")
    if tr.shape != (lp.shape[0], N_STATES, N_STATES):
        raise ValueError(f"log_trans must be [{lp.shape[0]}, 3, 3], got {tr.shape}")
    init = np.asarray(default_log_init() if log_init is None else log_init, dtype=np.float64)
    if init.shape != (N_STATES,):
        raise ValueError("log_init must have 3 entries")
    lp = np.nan_to_num(lp, nan=_NEG, neginf=_NEG, posinf=0.0)
    tr = np.nan_to_num(tr, nan=_NEG, neginf=_NEG, posinf=0.0)
    init = np.nan_to_num(init, nan=_NEG, neginf=_NEG, posinf=0.0)
    return np.ascontiguousarray(lp), np.ascontiguousarray(tr), np.ascontiguousarray(init)


def viterbi_dynamic(log_probs: np.ndarray, log_trans: np.ndarray, log_init: np.ndarray | None = None, *,
                    use_numba: bool | None = None) -> np.ndarray:
    """Best state path (int8 codes H=0, A=1, D=2) for ``log_probs[N, 3]`` and ``log_trans[N, 3, 3]``.

    ``use_numba=None`` picks the njit kernel when available; ``False`` forces the pure-Python fallback
    (identical output).
    """
    lp, tr, init = _prepare(log_probs, log_trans, log_init)
    n = lp.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    if use_numba is None:
        use_numba = numba_available()
    if use_numba:
        if _numba is None:
            raise RuntimeError("numba is not installed")
        bp = np.zeros((n, N_STATES), dtype=np.int8)
        return _get_njit_kernel()(lp, tr, init, bp)
    return _viterbi_python(lp.tolist(), tr.tolist(), init.tolist())


def decode_match_halves(
        log_probs: np.ndarray, log_trans: np.ndarray, *,
        play_blocks: Sequence[tuple[int, int]] = (HALF1, HALF2_PLAY),
        dead_blocks: Sequence[tuple[int, int]] = (HALF2_DEAD,),
        log_init: np.ndarray | None = None, use_numba: bool | None = None,
        duration=None) -> np.ndarray:
    """Decode each graded block independently into a full-length int8 code array.

    ``play_blocks`` (inclusive ``(first, last)``) are decoded separately with a fresh start; ``dead_blocks``
    are forced to ``D`` (the second-half period before the centre kick-off); every other frame is ``X`` (3).
    ``duration = (tables, weight)`` (Issue #34) switches every block to the semi-Markov decoder.
    """
    lp = np.asarray(log_probs)
    n = lp.shape[0]
    out = np.full(n, X, dtype=np.int8)
    for a, b in dead_blocks:
        a, b = max(0, int(a)), min(n - 1, int(b))
        if b >= a:
            out[a:b + 1] = D
    for a, b in play_blocks:
        a, b = max(0, int(a)), min(n - 1, int(b))
        if b >= a:
            if duration is not None:
                from mtah2027.decoding.semi_markov import viterbi_semi_markov

                init = default_log_init() if log_init is None else log_init
                out[a:b + 1] = viterbi_semi_markov(lp[a:b + 1], np.asarray(log_trans)[a:b + 1], init,
                                                   duration[0], duration[1], use_numba=use_numba)
            else:
                out[a:b + 1] = viterbi_dynamic(lp[a:b + 1], np.asarray(log_trans)[a:b + 1], log_init,
                                               use_numba=use_numba)
    return out
