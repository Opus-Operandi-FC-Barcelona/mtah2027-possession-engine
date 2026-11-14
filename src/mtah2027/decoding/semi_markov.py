"""Explicit-duration (semi-Markov) Viterbi decoder (Issue #34).

Best segmentation of a block into runs of H / A / D maximising

    sum over segments [u..t] of state s: entry(u, s' -> s) + w * log P_s(duration) + sum_k log_probs[k, s]

where ``entry`` is the time-dependent transition ``log_trans[u, s', s]`` of Issue #23 (touch / restart gates,
``s' != s``), the duration term comes from :class:`~mtah2027.decoding.duration_model.DurationModel` and the
first / last segment of the block are censored (their duration term is the survival probability P(D >= d)).
Durations up to ``cap`` frames are explicit; longer ones use a geometric tail handled with a running maximum,
so the cost is O(N * cap * 3) without any other bound.

Two kernels execute the same float64 operations in the same order and tie-break identically (shortest
duration, lowest state): a ``numba.njit`` loop and a NumPy fallback vectorised over the duration axis.
"""
from __future__ import annotations

import numpy as np

from mtah2027.decoding.dynamic_viterbi import _NEG, ROUND_DECIMALS, numba_available

try:
    import numba as _numba
except Exception:  # pragma: no cover
    _numba = None

N_STATES = 3


def _kernel(C, tr, init, wld, wls, ls_tail_base, w, lq, a_tail, cap, E, Eprev, Fstart, F):
    """Fill the tables and return ``(final_state, final_start)``.

    ``C[n + 1, 3]`` cumulative emissions, ``wld`` / ``wls`` = weight * log-pmf / log-survival on d = 0..cap,
    ``ls_tail_base[s]`` = log P(duration >= cap + 1) with a geometric tail beyond (``+ (d - cap - 1) * lq``).
    """
    n = C.shape[0] - 1
    M = np.full(3, -3.0e300)         # running max over u <= t - cap of E[u, s] - w * u * lq[s]
    Marg = np.zeros(3, dtype=np.int64)
    for t in range(n):
        if t > 0:
            for s in range(3):
                best = -3.0e300
                arg = -1
                for sp in range(3):
                    if sp == s:
                        continue
                    v = F[t - 1, sp] + tr[t, sp, s]
                    if arg < 0 or v > best:
                        best = v
                        arg = sp
                E[t, s] = best - C[t, s]
                Eprev[t, s] = arg
        else:
            for s in range(3):
                E[0, s] = init[s]
                Eprev[0, s] = -1
        u_new = t - cap
        if u_new >= 1:
            for s in range(3):
                v = E[u_new, s] - w * u_new * lq[s]
                if v > M[s]:
                    M[s] = v
                    Marg[s] = u_new
        for s in range(3):
            d0 = t + 1
            # censored first segment (u = 0)
            if d0 <= cap:
                best = E[0, s] + wls[s, d0]
            else:
                best = E[0, s] + w * (ls_tail_base[s] + (d0 - cap - 1) * lq[s])
            arg = 0
            lo = t - cap + 1
            if lo < 1:
                lo = 1
            for u in range(t, lo - 1, -1):            # d = 1, 2, ... (shortest first, strict > keeps it)
                v = E[u, s] + wld[s, t - u + 1]
                if v > best:
                    best = v
                    arg = u
            if u_new >= 1 and M[s] > -1.0e299:
                v = w * a_tail[s] + w * (t + 1) * lq[s] + M[s]
                if v > best:
                    best = v
                    arg = Marg[s]
            F[t, s] = C[t + 1, s] + best
            Fstart[t, s] = arg
    # termination: the last segment [u..n-1] is censored as well
    fbest = -3.0e300
    fs = 0
    fu = 0
    first = True
    for s in range(3):
        for u in range(n):
            d = n - u
            if u == 0:
                surv = 0.0
            elif d <= cap:
                surv = wls[s, d]
            else:
                surv = w * (ls_tail_base[s] + (d - cap - 1) * lq[s])
            v = E[u, s] + C[n, s] + surv
            if first or v > fbest:
                fbest = v
                fs = s
                fu = u
                first = False
    return fs, fu


_njit = None


def _get_njit():
    global _njit
    if _njit is None:
        _njit = _numba.njit(cache=False)(_kernel)
    return _njit


def _kernel_numpy(C, tr, init, wld, wls, ls_tail_base, w, lq, a_tail, cap, E, Eprev, Fstart, F):
    """Same arithmetic as :func:`_kernel`, the duration axis vectorised."""
    n = C.shape[0] - 1
    M = np.full(3, -3.0e300)
    Marg = np.zeros(3, dtype=np.int64)
    for t in range(n):
        if t > 0:
            for s in range(3):
                others = [sp for sp in range(3) if sp != s]
                v0 = F[t - 1, others[0]] + tr[t, others[0], s]
                v1 = F[t - 1, others[1]] + tr[t, others[1], s]
                if v1 > v0:
                    E[t, s], Eprev[t, s] = v1 - C[t, s], others[1]
                else:
                    E[t, s], Eprev[t, s] = v0 - C[t, s], others[0]
        else:
            E[0], Eprev[0] = init, -1
        u_new = t - cap
        if u_new >= 1:
            for s in range(3):
                v = E[u_new, s] - w * u_new * lq[s]
                if v > M[s]:
                    M[s], Marg[s] = v, u_new
        lo = max(t - cap + 1, 1)
        for s in range(3):
            d0 = t + 1
            if d0 <= cap:
                best = E[0, s] + wls[s, d0]
            else:
                best = E[0, s] + w * (ls_tail_base[s] + (d0 - cap - 1) * lq[s])
            arg = 0
            if t >= lo:
                us = np.arange(t, lo - 1, -1)
                vals = E[lo:t + 1, s][::-1] + wld[s, 1:t - lo + 2]
                k = int(np.argmax(vals))
                if vals[k] > best:
                    best, arg = vals[k], int(us[k])
            if u_new >= 1 and M[s] > -1.0e299:
                v = w * a_tail[s] + w * (t + 1) * lq[s] + M[s]
                if v > best:
                    best, arg = v, int(Marg[s])
            F[t, s] = C[t + 1, s] + best
            Fstart[t, s] = arg
    fbest, fs, fu, first = -3.0e300, 0, 0, True
    for s in range(3):
        for u in range(n):
            d = n - u
            if u == 0:
                surv = 0.0
            elif d <= cap:
                surv = wls[s, d]
            else:
                surv = w * (ls_tail_base[s] + (d - cap - 1) * lq[s])
            v = E[u, s] + C[n, s] + surv
            if first or v > fbest:
                fbest, fs, fu, first = v, s, u, False
    return fs, fu


def viterbi_semi_markov(log_probs: np.ndarray, log_trans: np.ndarray, log_init: np.ndarray, tables,
                        weight: float = 1.0, *, use_numba: bool | None = None) -> np.ndarray:
    """Best segmentation (int8 H=0, A=1, D=2) of one block; ``tables`` from ``DurationModel.tables(fps)``."""
    ld, ls, lq, a_tail, ls_tail, cap = tables
    lp = np.round(np.asarray(log_probs, dtype=np.float64), ROUND_DECIMALS)
    tr = np.round(np.asarray(log_trans, dtype=np.float64), ROUND_DECIMALS)
    n = lp.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    if lp.shape != (n, 3) or tr.shape != (n, 3, 3):
        raise ValueError("log_probs must be [N, 3] and log_trans [N, 3, 3]")
    init = np.nan_to_num(np.asarray(log_init, dtype=np.float64), nan=_NEG, neginf=_NEG, posinf=0.0)
    lp = np.nan_to_num(lp, nan=_NEG, neginf=_NEG, posinf=0.0)
    tr = np.ascontiguousarray(np.nan_to_num(tr, nan=_NEG, neginf=_NEG, posinf=0.0))
    C = np.zeros((n + 1, 3))
    C[1:] = np.cumsum(lp, axis=0)
    w = float(weight)
    wld = np.ascontiguousarray(w * ld)
    wls = np.ascontiguousarray(w * ls)
    ls_tail_base = np.ascontiguousarray(ls_tail)             # log P(duration >= cap + 1); ratio q beyond
    E = np.zeros((n, 3))
    Eprev = np.zeros((n, 3), dtype=np.int64)
    Fstart = np.zeros((n, 3), dtype=np.int64)
    F = np.zeros((n, 3))
    args = (C, tr, np.ascontiguousarray(init), wld, wls, ls_tail_base, w, np.ascontiguousarray(lq),
            np.ascontiguousarray(a_tail), int(cap), E, Eprev, Fstart, F)
    if use_numba is None:
        use_numba = numba_available()
    fs, fu = (_get_njit() if use_numba else _kernel_numpy)(*args)
    path = np.empty(n, dtype=np.int8)
    s, u, end = int(fs), int(fu), n - 1
    while True:
        path[u:end + 1] = s
        if u == 0:
            break
        sp = int(Eprev[u, s])
        end = u - 1
        s, u = sp, int(Fstart[end, sp])
    return path
