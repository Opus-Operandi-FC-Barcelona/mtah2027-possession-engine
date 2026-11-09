"""Per-frame homographies ``H`` and ``H^-1``, vectorised (Issue #7).

``homography.csv`` has one row per video frame (``frame`` counting from 0)::

    frame,h00,h01,h02,h10,h11,h12,h20,h21,h22,reprojection_error,valid

The 3x3 matrix (row-major) maps **normalised pitch coordinates** ``[0, 1]^2`` (the ATD convention) to
image pixels.  Where ``valid`` is 0 the matrix is NaN -- no transform exists.  This module works with
**metres** at its interface: :func:`project_pitch_to_pixels` divides by the pitch size before applying
``H`` and :func:`project_pixels_to_pitch` multiplies after applying ``H^-1``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from mtah2027.config import PITCH, load_games_module

MATRIX_COLUMNS = tuple(f"h{r}{c}" for r in range(3) for c in range(3))
_COND_LIMIT = 1e12


class HomographyFormatError(ValueError):
    """Raised when the homography CSV does not have the documented columns / frames."""


@dataclass
class Homographies:
    """Matrices indexed by video frame; NaN rows wherever ``valid`` is False."""

    H: np.ndarray                  # [N, 3, 3] float64, normalised pitch -> pixels
    H_inv: np.ndarray              # [N, 3, 3] float64, pixels -> normalised pitch (NaN where invalid)
    reprojection_error: np.ndarray  # [N] float64, lower is better
    valid: np.ndarray              # [N] bool, matrix exists and is invertible
    pitch: tuple[float, float] = PITCH

    @property
    def n_frames(self) -> int:
        return int(self.H.shape[0])

    def usable(self, max_error: float = 5.0) -> np.ndarray:
        """``valid`` and ``reprojection_error < max_error`` (NaN errors are not usable)."""
        with np.errstate(invalid="ignore"):
            return self.valid & (self.reprojection_error < max_error)

    def project_pitch_to_pixels(self, x_m, y_m):
        """Metres -> pixels using ``H`` (broadcast over frames, see the module function)."""
        return project_pitch_to_pixels(self.H, x_m, y_m, self.pitch)

    def project_pixels_to_pitch(self, u_px, v_px):
        """Pixels -> metres using ``H_inv``."""
        return project_pixels_to_pitch(self.H_inv, u_px, v_px, self.pitch)


def invert_homographies(H: np.ndarray, valid: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised batch inverse; returns ``(H_inv, ok)`` with NaN inverses where not invertible.

    ``ok`` is True where ``valid`` holds, the matrix is finite and well conditioned.
    """
    H = np.asarray(H, dtype=np.float64)
    finite = np.isfinite(H).all(axis=(-2, -1))
    ok = finite if valid is None else finite & np.asarray(valid, dtype=bool)
    inv = np.full_like(H, np.nan)
    if ok.any():
        sub = H[ok]
        with np.errstate(all="ignore"):
            cond = np.linalg.cond(sub)
        good = np.isfinite(cond) & (cond < _COND_LIMIT)
        idx = np.flatnonzero(ok)
        inv_sub = np.full_like(sub, np.nan)
        if good.any():
            inv_sub[good] = np.linalg.inv(sub[good])
        inv[idx] = inv_sub
        ok = ok.copy()
        ok[idx[~good]] = False
    return inv, ok


def _apply(M: np.ndarray, a, b) -> tuple[np.ndarray, np.ndarray]:
    """Apply homographies ``M[..., 3, 3]`` to points ``(a, b)`` of shape ``(*batch, *extra)``."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    a, b = np.broadcast_arrays(a, b)
    extra = a.ndim - (M.ndim - 2)
    if extra < 0:
        raise ValueError(f"points of shape {a.shape} cannot be matched with matrices of shape {M.shape}")
    M = M.reshape(M.shape[:-2] + (1,) * extra + (3, 3))
    den = M[..., 2, 0] * a + M[..., 2, 1] * b + M[..., 2, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        return ((M[..., 0, 0] * a + M[..., 0, 1] * b + M[..., 0, 2]) / den,
                (M[..., 1, 0] * a + M[..., 1, 1] * b + M[..., 1, 2]) / den)


def project_pitch_to_pixels(H: np.ndarray, x_m, y_m, pitch: tuple[float, float] = PITCH):
    """Pitch positions in **metres** -> pixels ``(u, v)``.

    ``H`` is ``[3, 3]`` or ``[N, 3, 3]``; with ``[N, 3, 3]``, ``x_m``/``y_m`` have a leading axis ``N``
    (``[N]`` or ``[N, K]``).  NaN in, NaN out (invalid frames have NaN matrices).
    """
    length, width = pitch
    return _apply(np.asarray(H, dtype=np.float64), np.asarray(x_m, dtype=np.float64) / length,
                  np.asarray(y_m, dtype=np.float64) / width)


def project_pixels_to_pitch(H_inv: np.ndarray, u_px, v_px, pitch: tuple[float, float] = PITCH):
    """Pixels ``(u, v)`` -> pitch positions in **metres**, using the inverse matrices."""
    length, width = pitch
    x, y = _apply(np.asarray(H_inv, dtype=np.float64), u_px, v_px)
    return x * length, y * width


def parse_homography_csv(path: str | os.PathLike, pitch: tuple[float, float] = PITCH) -> Homographies:
    """Parse one ``*_homography.csv`` into :class:`Homographies` (indexed by the ``frame`` column)."""
    table = pd.read_csv(path)
    missing = [c for c in ("frame", "reprojection_error", "valid") + MATRIX_COLUMNS if c not in table.columns]
    if missing:
        raise HomographyFormatError(f"{path}: missing columns {missing}")
    frames = table["frame"].to_numpy(dtype=np.int64)
    if len(frames) == 0 or frames.min() < 0:
        raise HomographyFormatError(f"{path}: empty file or negative frame numbers")
    n = int(frames.max()) + 1
    H = np.full((n, 3, 3), np.nan)
    err = np.full(n, np.nan)
    flag = np.zeros(n, dtype=bool)
    H[frames] = table[list(MATRIX_COLUMNS)].to_numpy(dtype=np.float64).reshape(-1, 3, 3)
    err[frames] = table["reprojection_error"].to_numpy(dtype=np.float64)
    flag[frames] = table["valid"].to_numpy(dtype=np.float64) > 0.5
    H[~flag] = np.nan
    H_inv, ok = invert_homographies(H, flag)
    return Homographies(H=H, H_inv=H_inv, reprojection_error=err, valid=ok,
                        pitch=(float(pitch[0]), float(pitch[1])))


def load_homography(data_dir: str | os.PathLike | None = None, source: str = "tactical") -> Homographies:
    """Load the homographies of ``source`` ('tactical' or 'broadcast') from the data bundle."""
    games = load_games_module()
    if games is None:
        raise ImportError("the vendored games.py is required to locate the homography file")
    path = games.require(["homography"], games.GAME, data_dir, source)[0]
    return parse_homography_csv(path, tuple(games.GAMES[games.GAME]["pitch"]))
