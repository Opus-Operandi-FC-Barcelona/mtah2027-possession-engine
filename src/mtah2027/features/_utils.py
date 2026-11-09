"""Small vectorised helpers shared by the feature modules (NaN-aware, no Python loops over frames)."""
from __future__ import annotations

import numpy as np


def velocity_from_xy(xy: np.ndarray, fps: float, half_span: int = 1) -> np.ndarray:
    """Velocity (m/s) of positions ``xy[N, ..., 2]`` by a central difference over ``+-half_span`` frames.

    Where one of the two neighbours is missing the available one-sided difference (over a single
    side) is used; NaN where neither side (or the sample itself) is available.
    """
    h = max(1, int(half_span))
    xy = np.asarray(xy, dtype=np.float64)
    n = xy.shape[0]
    nxt = np.full_like(xy, np.nan)
    prv = np.full_like(xy, np.nan)
    if n > h:
        nxt[:-h] = xy[h:]
        prv[h:] = xy[:-h]
    ok_n = np.isfinite(nxt).all(-1)
    ok_p = np.isfinite(prv).all(-1)
    both = ok_n & ok_p
    only_n = ok_n & ~both
    only_p = ok_p & ~both
    here = np.isfinite(xy).all(-1)
    d_both = (nxt - prv) / (2.0 * h)
    d_n = (nxt - xy) / h
    d_p = (xy - prv) / h
    diff = np.where(both[..., None], d_both,
                    np.where(only_n[..., None], d_n, np.where(only_p[..., None], d_p, np.nan)))
    diff = diff * fps
    diff[~here] = np.nan
    return diff.astype(np.float32)


def speed_from_xy(xy: np.ndarray, fps: float, half_span: int = 1) -> np.ndarray:
    """Speed (m/s): the norm of :func:`velocity_from_xy`."""
    return np.linalg.norm(velocity_from_xy(xy, fps, half_span), axis=-1).astype(np.float32)


def window_mean(x: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """NaN-aware mean of ``x[i+lo : i+hi+1]`` along axis 0 (offsets inclusive, clipped at the edges)."""
    x = np.asarray(x, dtype=np.float64)
    valid = np.isfinite(x)
    vals = np.where(valid, x, 0.0)
    shape = (1,) + x.shape[1:]
    cs = np.concatenate([np.zeros(shape), np.cumsum(vals, axis=0)], axis=0)
    cn = np.concatenate([np.zeros(shape), np.cumsum(valid, axis=0)], axis=0)
    n = x.shape[0]
    idx = np.arange(n)
    a = np.clip(idx + lo, 0, n)
    b = np.clip(idx + hi + 1, 0, n)
    s = cs[b] - cs[a]
    c = cn[b] - cn[a]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    return out.astype(np.float32)


def moving_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Centered NaN-aware moving mean along axis 0 over ``window`` samples (NaN if all are NaN)."""
    window = max(1, int(window))
    if window == 1:
        return np.asarray(x, dtype=np.float32)
    return window_mean(x, -(window // 2), window - 1 - window // 2)


def frames_since(mask: np.ndarray) -> np.ndarray:
    """Frames elapsed since the last True of ``mask`` (0 where True); -1 where there was none yet."""
    idx = np.arange(len(mask))
    last = np.maximum.accumulate(np.where(mask, idx, -1))
    return np.where(last >= 0, idx - last, -1)


def frames_until(mask: np.ndarray) -> np.ndarray:
    """Frames until the next True of ``mask`` (0 where True); -1 where there is none any more."""
    return frames_since(mask[::-1])[::-1]


def run_length_before(mask: np.ndarray) -> np.ndarray:
    """Length of the run of consecutive True ending at each index (0 where ``mask`` is False)."""
    idx = np.arange(len(mask))
    last_false = np.maximum.accumulate(np.where(~mask, idx, -1))
    return np.where(mask, idx - last_false, 0)


def row_percentile(values: np.ndarray, q: float, min_count: int = 1) -> np.ndarray:
    """Linear-interpolation percentile ``q`` in [0, 1] of the finite entries of each row.

    Equivalent to ``np.nanpercentile(values, 100 * q, axis=1)`` but vectorised through one sort;
    rows with fewer than ``min_count`` finite entries give NaN.
    """
    srt = np.sort(np.asarray(values, dtype=np.float64), axis=1)      # NaN sorts last
    n = np.isfinite(srt).sum(axis=1)
    pos = q * np.maximum(n - 1, 0)
    lo = np.floor(pos).astype(np.int64)
    hi = np.minimum(lo + 1, np.maximum(n - 1, 0))
    rows = np.arange(srt.shape[0])
    a, b = srt[rows, lo], srt[rows, hi]
    out = a + (b - a) * (pos - lo)
    out[n < max(1, min_count)] = np.nan
    return out.astype(np.float32)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


def mask_runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start and exclusive end indices of the maximal True runs of a 1-D boolean ``mask``."""
    edges = np.diff(np.concatenate([[0], np.asarray(mask, dtype=np.int8), [0]]))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
