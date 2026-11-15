# ruff: noqa: E501
"""Linear-chain CRF helpers of the sequence model (Issue #78; OFF by default).

The TCN of Issue #72 can be trained with the sequence negative log-likelihood of a linear-chain CRF instead of
(or besides) the per-frame cross-entropy (``scripts/train_sequence.py --crf c1|c2``).  The CRF scores a state path
``y_0 .. y_{T-1}`` (H=0, A=1, D=2) as ``sum_t u_t[y_t] + sum_{t>=1} T_t[y_{t-1}, y_t]`` with the unaries ``u_t`` (the
emission logits of the TCN) and ``T_t[i, j]`` the score of the move into frame ``t`` (the convention of
``decoding.dynamic_viterbi``: ``log_trans[t]`` scores ``t-1 -> t``).

* Variant C1: ``T_t`` is a learned global 3x3 matrix (zero diagonal).  At inference only the unaries are used
  (log-softmax = the usual emission log-probabilities of the sequence model); the production transition
  matrix and Viterbi are unchanged.
* Variant C2: the H -> A and A -> H entries are a learned function of the TCN features at each frame,
  ``-softplus(w . h_t + b)`` (a log-probability-like score <= 0).  At inference those two columns of the
  production ``log_trans`` are replaced by them (``override_ha_transitions``); D transitions stay as in production.

Everything here is NumPy (float64) and mirrors the torch code of the training script, which is checked against it
in ``tests/test_78_sequence_crf.py``.
"""
from __future__ import annotations

import numpy as np

H, A, D = 0, 1, 2


def neg_softplus(z: np.ndarray) -> np.ndarray:
    """``-log(1 + exp(z))`` (stable): the C2 transition score, always <= 0."""
    return -np.logaddexp(0.0, np.asarray(z, dtype=np.float64))


def global_transitions(raw: np.ndarray) -> np.ndarray:
    """Learned 3x3 matrix with a zero diagonal (staying in a state costs nothing, like ``transition_matrix.stay``)."""
    g = np.asarray(raw, dtype=np.float64).copy()
    g[np.eye(3, dtype=bool)] = 0.0
    return g


def chain_transitions(raw_global: np.ndarray, n: int, ha: np.ndarray | None = None) -> np.ndarray:
    """``trans[n, 3, 3]``: the global matrix at every frame, with ``trans[:, H, A] = ha[:, 0]`` and
    ``trans[:, A, H] = ha[:, 1]`` when the per-frame H<->A scores of C2 are given."""
    tr = np.broadcast_to(global_transitions(raw_global), (n, 3, 3)).copy()
    if ha is not None:
        tr[:, H, A] = ha[:, 0]
        tr[:, A, H] = ha[:, 1]
    return tr


def chain_score(unary: np.ndarray, trans: np.ndarray, y: np.ndarray) -> float:
    """Score of the path ``y`` (frame 0 contributes its unary only)."""
    y = np.asarray(y, dtype=np.int64)
    s = float(unary[np.arange(len(y)), y].sum())
    if len(y) > 1:
        s += float(trans[np.arange(1, len(y)), y[:-1], y[1:]].sum())
    return s


def chain_log_partition(unary: np.ndarray, trans: np.ndarray) -> float:
    """``log sum_y exp(score(y))`` by the forward algorithm."""
    alpha = np.asarray(unary[0], dtype=np.float64)
    for t in range(1, len(unary)):
        z = alpha[:, None] + trans[t]
        m = z.max(axis=0)
        alpha = m + np.log(np.exp(z - m).sum(axis=0)) + unary[t]
    m = alpha.max()
    return float(m + np.log(np.exp(alpha - m).sum()))


def chain_nll(unary: np.ndarray, trans: np.ndarray, y: np.ndarray) -> float:
    return chain_log_partition(unary, trans) - chain_score(unary, trans, y)


def chain_viterbi(unary: np.ndarray, trans: np.ndarray) -> np.ndarray:
    """Best path of the chain, with the same decoder as the pipeline (``viterbi_dynamic``, zero initial scores)."""
    from mtah2027.decoding.dynamic_viterbi import viterbi_dynamic

    return viterbi_dynamic(unary, trans, np.zeros(3))


def override_ha_transitions(log_trans: np.ndarray, ha: np.ndarray, decimals: int = 6) -> np.ndarray:
    """Replace the H -> A (``ha[:, 0]``) and A -> H (``ha[:, 1]``) entries of ``log_trans[n, 3, 3]`` by the learned
    scores at the frames where they are finite (NaN rows, outside the play blocks, keep the production values)."""
    ha = np.asarray(ha, dtype=np.float64)
    ok = np.isfinite(ha).all(axis=1)
    out = np.array(log_trans, dtype=np.float64, copy=True)
    out[ok, H, A] = np.round(ha[ok, 0], decimals)
    out[ok, A, H] = np.round(ha[ok, 1], decimals)
    return out
