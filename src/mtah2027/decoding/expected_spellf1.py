"""Decoding by expected spellF1 (Issue #62): minimum Bayes risk over segmentations.

Viterbi returns the single most probable state path, but the ranking metric (spellF1) rewards the right number
of possessions and boundaries, not the joint probability of the path.  Per graded block this module

1. runs the forward pass in log space over the same quantised emissions / time-varying transitions as Viterbi,
2. draws ``k`` paths from the posterior by backward sampling (fixed seed; the uniform numbers come from
   ``numpy.random.default_rng`` and the conditional probabilities are pre-computed with NumPy, so the njit
   kernel and the pure-Python fallback only compare floats and agree),
3. adds the Viterbi path as candidate 0 and picks the candidate with the highest mean spellF1 against all the
   others (ties: lowest index, i.e. Viterbi first).

The spellF1 of a pair of candidates is :func:`pair_spell_f1`, a re-implementation of the official
``possession_metrics.spell_scores`` (same-team spells, greedy one-to-one matching by decreasing IoU with ties
by (gt index, pred index), thresholds 0.3 .. 0.7) over arrays; ``tests/test_62_expected_spellf1.py`` checks it
equal to the official scorer.  A temperature ``T`` divides the log-emissions and log-transitions of the
*sampling* distribution only (T > 1 flattens the posterior); the Viterbi candidate is untouched.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from mtah2027.config import HALF1, HALF2_DEAD, HALF2_PLAY, load_section, params_from_mapping
from mtah2027.decoding.dynamic_viterbi import (
    N_STATES,
    _prepare,
    default_log_init,
    numba_available,
    viterbi_dynamic,
)

H, A, D, X = 0, 1, 2, 3
IOUS = (0.3, 0.4, 0.5, 0.6, 0.7)

try:
    import numba as _numba
except Exception:  # pragma: no cover
    _numba = None


@dataclass(frozen=True)
class ExpectedSpellF1Params:
    enabled: bool = False
    k: int = 100                 # posterior samples per block (the Viterbi path is added to them)
    temperature: float = 1.0     # sampling temperature (> 1 flattens the posterior)
    seed: int = 0

    @classmethod
    def from_config(cls, path=None) -> "ExpectedSpellF1Params":
        return params_from_mapping(cls, load_section("expected_spellf1", path), "expected_spellf1")


# --------------------------------------------------------------------------------------------
# kernels (plain Python, optionally compiled)
# --------------------------------------------------------------------------------------------
def _forward_kernel(lp, tr, init, alpha):
    n = lp.shape[0]
    for j in range(3):
        alpha[0, j] = init[j] + lp[0, j]
    for t in range(1, n):
        for j in range(3):
            v0 = alpha[t - 1, 0] + tr[t, 0, j]
            v1 = alpha[t - 1, 1] + tr[t, 1, j]
            v2 = alpha[t - 1, 2] + tr[t, 2, j]
            m = max(v0, max(v1, v2))
            alpha[t, j] = m + math.log(math.exp(v0 - m) + math.exp(v1 - m) + math.exp(v2 - m)) + lp[t, j]


def _sample_kernel(cum, last_cum, u, out):
    """``cum[t, j, :]`` = cumulative P(s_{t-1} = 0, 1 | s_t = j, y_<t); fills ``out[k, t]``."""
    k_paths, n = out.shape
    for k in range(k_paths):
        r = u[k, n - 1]
        s = 0 if r < last_cum[0] else (1 if r < last_cum[1] else 2)
        out[k, n - 1] = s
        for t in range(n - 1, 0, -1):
            r = u[k, t - 1]
            if r < cum[t, s, 0]:
                s = 0
            elif r < cum[t, s, 1]:
                s = 1
            else:
                s = 2
            out[k, t - 1] = s


def _pair_kernel(gs, ge, gc, ps, pe, pc, ious, ia, ib, iou, flag_g, flag_p):
    """Mean F1 over ``ious`` of the greedy one-to-one same-team matching of gt vs pred spells."""
    ng, npred = gs.shape[0], ps.shape[0]
    if ng == 0 or npred == 0:
        return 0.0
    m = 0
    j0 = 0
    for i in range(ng):
        while j0 < npred and pe[j0] <= gs[i]:
            j0 += 1
        j = j0
        while j < npred and ps[j] < ge[i]:
            if pc[j] == gc[i] and pe[j] > gs[i]:
                inter = min(ge[i], pe[j]) - max(gs[i], ps[j])
                union = max(ge[i], pe[j]) - min(gs[i], ps[j])
                ia[m] = i
                ib[m] = j
                iou[m] = inter / float(union)
                m += 1
            j += 1
    if m == 0:
        return 0.0
    order = np.argsort(-iou[:m], kind="mergesort")      # stable: ties keep the (gt, pred) generation order
    total = 0.0
    for q in range(ious.shape[0]):
        thr = ious[q]
        flag_g[:ng] = False
        flag_p[:npred] = False
        tp = 0
        for r in range(m):
            o = order[r]
            if iou[o] < thr:
                break
            gi = ia[o]
            pj = ib[o]
            if flag_g[gi] or flag_p[pj]:
                continue
            flag_g[gi] = True
            flag_p[pj] = True
            tp += 1
        if tp > 0:
            total += 2.0 * tp / (ng + npred)
    return total / ious.shape[0]


_jit: dict = {}


def _compiled(name: str):
    fn = {"fwd": _forward_kernel, "smp": _sample_kernel, "pair": _pair_kernel}[name]
    if name not in _jit:
        _jit[name] = _numba.njit(cache=False)(fn)
    return _jit[name]


def _kernel(name: str, use_numba: bool):
    if use_numba and _numba is not None:
        return _compiled(name)
    return {"fwd": _forward_kernel, "smp": _sample_kernel, "pair": _pair_kernel}[name]


def _matrix_njit():
    if "matrix" not in _jit:
        pair = _compiled("pair")

        def matrix(starts, stops, labs, offs, ious, ia, ib, iou, flag_g, flag_p, out):
            c = offs.shape[0] - 1
            for a in range(c):
                for b in range(c):
                    if a != b:
                        out[a, b] = pair(starts[offs[a]:offs[a + 1]], stops[offs[a]:offs[a + 1]],
                                         labs[offs[a]:offs[a + 1]], starts[offs[b]:offs[b + 1]],
                                         stops[offs[b]:offs[b + 1]], labs[offs[b]:offs[b + 1]],
                                         ious, ia, ib, iou, flag_g, flag_p)

        _jit["matrix"] = _numba.njit(cache=False)(matrix)
    return _jit["matrix"]


# --------------------------------------------------------------------------------------------
# posterior sampling
# --------------------------------------------------------------------------------------------
def _softmax_last(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def sample_posterior_paths(log_probs, log_trans, log_init, k: int, seed: int, *, temperature: float = 1.0,
                           use_numba: bool | None = None) -> np.ndarray:
    """``k`` paths (int8 ``[k, N]``) drawn from the HMM posterior; deterministic for a given seed."""
    lp, tr, init = _prepare(log_probs, log_trans, log_init)
    n = lp.shape[0]
    if n == 0 or k <= 0:
        return np.zeros((max(k, 0), n), dtype=np.int8)
    if temperature != 1.0:
        lp, tr, init = lp / temperature, tr / temperature, init / temperature
    use_numba = numba_available() if use_numba is None else use_numba
    alpha = np.zeros((n, N_STATES), dtype=np.float64)
    _kernel("fwd", use_numba)(lp, tr, init, alpha)
    cum = np.zeros((n, N_STATES, 2), dtype=np.float64)
    if n > 1:
        score = alpha[:-1, :, None] + tr[1:]                  # [n-1, i, j]
        prob = _softmax_last(np.swapaxes(score, 1, 2))        # [n-1, j, i]
        cum[1:, :, 0] = prob[:, :, 0]
        cum[1:, :, 1] = prob[:, :, 0] + prob[:, :, 1]
    pl = _softmax_last(alpha[-1])
    last_cum = np.array([pl[0], pl[0] + pl[1]], dtype=np.float64)
    u = np.random.default_rng(seed).random((k, n))
    out = np.zeros((k, n), dtype=np.int8)
    _kernel("smp", use_numba)(np.ascontiguousarray(cum), last_cum, u, out)
    return out


# --------------------------------------------------------------------------------------------
# fast spellF1
# --------------------------------------------------------------------------------------------
def spells_of(codes: np.ndarray) -> np.ndarray:
    """Possession spells ``[m, 3]`` (start, stop exclusive, team) of one block, as the official scorer."""
    codes = np.asarray(codes)
    if codes.size == 0:
        return np.zeros((0, 3), dtype=np.int64)
    br = np.flatnonzero(codes[1:] != codes[:-1]) + 1
    starts = np.concatenate(([0], br))
    stops = np.concatenate((br, [codes.size]))
    lab = codes[starts]
    keep = (lab == H) | (lab == A)
    return np.stack([starts[keep], stops[keep], lab[keep]], axis=1).astype(np.int64)


def _buffers(max_spells: int):
    cap = (2 * max_spells + 1) * 4 + 64          # overlapping same-team pairs <= ng + np - 1 per team
    return (np.zeros(cap, np.int64), np.zeros(cap, np.int64), np.zeros(cap, np.float64),
            np.zeros(max(max_spells, 1), np.bool_), np.zeros(max(max_spells, 1), np.bool_))


def pair_spell_f1(gt_spells: np.ndarray, pred_spells: np.ndarray, *, use_numba: bool | None = None) -> float:
    """spellF1 of ``pred_spells`` against ``gt_spells`` (mean F1 over the official IoU thresholds)."""
    use_numba = numba_available() if use_numba is None else use_numba
    g, p = np.asarray(gt_spells, dtype=np.int64), np.asarray(pred_spells, dtype=np.int64)
    ia, ib, iou, fg, fp = _buffers(max(g.shape[0], p.shape[0]))
    cols = lambda s: tuple(np.ascontiguousarray(s[:, i]) for i in range(3))  # noqa: E731
    return float(_kernel("pair", use_numba)(*cols(g), *cols(p), np.asarray(IOUS, dtype=np.float64),
                                            ia, ib, iou, fg, fp))


def utility_matrix(paths: Sequence[np.ndarray], *, use_numba: bool | None = None) -> np.ndarray:
    """``U[a, b]`` = spellF1 of path ``b`` scored against path ``a`` as ground truth (diagonal 0)."""
    use_numba = numba_available() if use_numba is None else use_numba
    spells = [spells_of(p) for p in paths]
    c = len(paths)
    out = np.zeros((c, c), dtype=np.float64)
    if not use_numba or _numba is None:
        for a in range(c):
            for b in range(c):
                if a != b:
                    out[a, b] = pair_spell_f1(spells[a], spells[b], use_numba=False)
        return out
    offs = np.concatenate(([0], np.cumsum([s.shape[0] for s in spells]))).astype(np.int64)
    allsp = np.concatenate(spells, axis=0) if offs[-1] else np.zeros((0, 3), np.int64)
    starts, stops, labs = (np.ascontiguousarray(allsp[:, i]) for i in range(3))
    ia, ib, iou, fg, fp = _buffers(int(max(s.shape[0] for s in spells)))
    _matrix_njit()(starts, stops, labs, offs, np.asarray(IOUS, dtype=np.float64), ia, ib, iou, fg, fp, out)
    return out


def select_mbr(paths: Sequence[np.ndarray], *, use_numba: bool | None = None) -> int:
    """Index of the path with the highest mean spellF1 against the others (ties: lowest index)."""
    if len(paths) <= 1:
        return 0
    u = utility_matrix(paths, use_numba=use_numba)
    score = np.round(u.sum(axis=0) / (len(paths) - 1), 12)
    return int(np.argmax(score))


# --------------------------------------------------------------------------------------------
# block / match level
# --------------------------------------------------------------------------------------------
def decode_block_mbr(log_probs, log_trans, log_init, params: ExpectedSpellF1Params, *, block_index: int = 0,
                     use_numba: bool | None = None) -> np.ndarray:
    """Viterbi + ``k`` posterior samples of one block; returns the candidate of highest expected spellF1."""
    vit = viterbi_dynamic(log_probs, log_trans, log_init, use_numba=use_numba)
    if params.k <= 0 or vit.size == 0:
        return vit
    smp = sample_posterior_paths(log_probs, log_trans, log_init, params.k, params.seed + 1009 * block_index,
                                 temperature=params.temperature, use_numba=use_numba)
    cands = [vit] + [smp[i] for i in range(smp.shape[0])]
    return np.ascontiguousarray(cands[select_mbr(cands, use_numba=use_numba)]).astype(np.int8)


def decode_match_halves_mbr(
        log_probs: np.ndarray, log_trans: np.ndarray, *, params: ExpectedSpellF1Params,
        play_blocks: Sequence[tuple[int, int]] = (HALF1, HALF2_PLAY),
        dead_blocks: Sequence[tuple[int, int]] = (HALF2_DEAD,),
        log_init: np.ndarray | None = None, use_numba: bool | None = None) -> np.ndarray:
    """Same layout as ``decode_match_halves`` with the per-block expected-spellF1 decoder."""
    lp = np.asarray(log_probs)
    n = lp.shape[0]
    out = np.full(n, X, dtype=np.int8)
    for a, b in dead_blocks:
        a, b = max(0, int(a)), min(n - 1, int(b))
        if b >= a:
            out[a:b + 1] = D
    init = default_log_init() if log_init is None else log_init
    for bi, (a, b) in enumerate(play_blocks):
        a, b = max(0, int(a)), min(n - 1, int(b))
        if b >= a:
            out[a:b + 1] = decode_block_mbr(lp[a:b + 1], np.asarray(log_trans)[a:b + 1], init, params,
                                            block_index=bi, use_numba=use_numba)
    return out
