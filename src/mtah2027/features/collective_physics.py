"""Collective physics of each team: dynamic goalkeepers, team shape areas and their derivatives (Issue #17).

Inputs are plain arrays in metres, *already oriented so that the home team attacks +x* (Issue #9) -- the
home team therefore defends ``x = 0`` and the away team ``x = pitch length``.  NaN marks missing slots.
Everything is vectorised over frames; the only per-frame loop is the numba-compiled convex hull.

Goalkeepers
-----------
The goalkeeper of a team is the player closest to its own goal line at every frame (home: minimum ``x``,
away: maximum ``x``), i.e. *dynamic* -- it survives identity cuts and substitutions of the slot.  Outputs:
slot, position and velocity (``gk_*``).  The goalkeeper is excluded from every area below.

Areas of the outfield players of each team (``min_players`` valid outfield players, else NaN)
--------------------------------------------------------------------------------------------
* **Spectral area**  ``pi * sqrt(lambda1 * lambda2)`` with ``lambda_i`` the eigenvalues of the per-frame 2x2
  covariance matrix of the positions (population covariance): the area of the one-sigma dispersion
  ellipse.  The eigenvalues are obtained in closed form, ``lambda = tr/2 +- sqrt((tr/2)^2 - det)``, for
  all frames at once.
* **Convex hull area**  of the same points with Andrew's monotone chain, compiled with ``numba.njit`` when
  numba is importable (and not disabled with ``MTAH_DISABLE_NUMBA=1``); the pure-Python fallback executes
  the very same code on the same float64 values and returns identical areas.  The sort of the (at most a
  few tens of) points is an insertion sort so that both versions share it.
* **Derivatives**  ``d_*`` = central difference over ``deriv_span_s`` seconds (default 2 s), per second
  (``(A[t + h] - A[t - h]) / deriv_span_s``, NaN if either end is missing).

Also per team: centroid (``centroid_*``), depth (``depth_*`` = x extent) and width (``width_*`` = y extent) of
the outfield block and the centroid velocity along x (``centroid_vx_*``, the derivative of the centroid
``x`` over the same span, m/s; positive = moving towards +x).
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass

import numpy as np

from mtah2027.config import FPS, load_section, params_from_mapping
from mtah2027.features._utils import velocity_from_xy

try:  # numba is an optional accelerator
    import numba as _numba
except Exception:  # pragma: no cover - exercised only where numba is absent
    _numba = None


@dataclass(frozen=True)
class CollectivePhysicsParams:
    min_players: int = 4             # fewer valid outfield players -> shape features are NaN
    deriv_span_s: float = 2.0        # total span of the central difference of the areas / centroid
    gk_vel_half_span: int = 1        # frames each side when the velocities are derived from positions

    @classmethod
    def from_config(cls, path=None) -> "CollectivePhysicsParams":
        return params_from_mapping(cls, load_section("collective_physics", path), "collective_physics")


def numba_available() -> bool:
    """True when the njit kernel can be used (numba importable and not disabled by the environment)."""
    return _numba is not None and os.environ.get("MTAH_DISABLE_NUMBA", "") not in ("1", "true", "True")


# --------------------------------------------------------------------------------------------
# Convex hull kernel (monotone chain)
# --------------------------------------------------------------------------------------------
def _hull_areas_kernel(px, py, out):
    """Convex-hull area of the finite points of every row of ``px[N, k]``, ``py[N, k]`` into ``out[N]``.

    ``out`` is NaN for rows with fewer than three finite points; collinear points give 0.
    """
    n, k = px.shape
    xs = np.empty(k)
    ys = np.empty(k)
    order = np.empty(k, dtype=np.int64)
    hull = np.empty(2 * k + 2, dtype=np.int64)
    for f in range(n):
        m = 0
        for j in range(k):
            x = px[f, j]
            y = py[f, j]
            if x == x and y == y:                      # finite (NaN != NaN)
                xs[m] = x
                ys[m] = y
                m += 1
        if m < 3:
            out[f] = np.nan
            continue
        # insertion sort of the indices by (x, y)
        for i in range(m):
            order[i] = i
        for i in range(1, m):
            cur = order[i]
            j = i - 1
            while j >= 0 and (xs[order[j]] > xs[cur] or (xs[order[j]] == xs[cur] and ys[order[j]] > ys[cur])):
                order[j + 1] = order[j]
                j -= 1
            order[j + 1] = cur
        h = 0
        for i in range(m):                              # lower hull
            c = order[i]
            while h >= 2:
                a = hull[h - 2]
                b = hull[h - 1]
                cross = (xs[b] - xs[a]) * (ys[c] - ys[a]) - (ys[b] - ys[a]) * (xs[c] - xs[a])
                if cross <= 0.0:
                    h -= 1
                else:
                    break
            hull[h] = c
            h += 1
        lower = h + 1
        for i in range(m - 2, -1, -1):                  # upper hull
            c = order[i]
            while h >= lower:
                a = hull[h - 2]
                b = hull[h - 1]
                cross = (xs[b] - xs[a]) * (ys[c] - ys[a]) - (ys[b] - ys[a]) * (xs[c] - xs[a])
                if cross <= 0.0:
                    h -= 1
                else:
                    break
            hull[h] = c
            h += 1
        h -= 1                                          # the last point repeats the first
        area = 0.0
        for i in range(h):
            a = hull[i]
            b = hull[(i + 1) % h]
            area += xs[a] * ys[b] - xs[b] * ys[a]
        out[f] = 0.5 * abs(area)


_hull_areas_njit = _numba.njit(cache=False)(_hull_areas_kernel) if _numba is not None else None


def convex_hull_areas(x: np.ndarray, y: np.ndarray, *, use_numba: bool | None = None) -> np.ndarray:
    """Convex hull area (m^2) per row of ``x[N, k]``, ``y[N, k]`` (NaN entries are ignored).

    NaN where fewer than three finite points; ``use_numba=None`` picks numba when available.
    """
    px = np.ascontiguousarray(x, dtype=np.float64)
    py = np.ascontiguousarray(y, dtype=np.float64)
    out = np.empty(px.shape[0], dtype=np.float64)
    use = numba_available() if use_numba is None else (bool(use_numba) and _hull_areas_njit is not None)
    (_hull_areas_njit if use else _hull_areas_kernel)(px, py, out)
    return out


def spectral_areas(x: np.ndarray, y: np.ndarray, min_points: int = 3) -> np.ndarray:
    """``pi * sqrt(lambda1 * lambda2)`` of the per-row 2x2 covariance of the finite points (NaN if fewer
    than ``min_points``); the eigenvalues are computed in closed form for every row at once."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    cnt = ok.sum(axis=1)
    safe = np.maximum(cnt, 1)
    xv, yv = np.where(ok, x, 0.0), np.where(ok, y, 0.0)
    mx, my = xv.sum(axis=1) / safe, yv.sum(axis=1) / safe
    dx, dy = np.where(ok, x - mx[:, None], 0.0), np.where(ok, y - my[:, None], 0.0)
    sxx, syy, sxy = (dx * dx).sum(axis=1) / safe, (dy * dy).sum(axis=1) / safe, (dx * dy).sum(axis=1) / safe
    half_tr = 0.5 * (sxx + syy)
    disc = np.sqrt(np.maximum(half_tr * half_tr - (sxx * syy - sxy * sxy), 0.0))
    lam1, lam2 = half_tr + disc, np.maximum(half_tr - disc, 0.0)
    area = np.pi * np.sqrt(lam1 * lam2)
    return np.where(cnt >= max(min_points, 1), area, np.nan)


def central_derivative(values: np.ndarray, span_frames: int, fps: float) -> np.ndarray:
    """``(v[t + h] - v[t - h]) / (2 h / fps)`` with ``2 h = span_frames`` (NaN where an end is missing)."""
    h = max(int(span_frames) // 2, 1)
    v = np.asarray(values, dtype=np.float64)
    out = np.full_like(v, np.nan)
    if len(v) > 2 * h:
        out[h:-h] = (v[2 * h:] - v[:-2 * h]) * (fps / (2.0 * h))
    return out


# --------------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------------
@dataclass
class CollectivePhysics:
    """Outputs of :func:`compute_collective_physics`; per frame, float32 unless stated otherwise."""

    gk_slot_home: np.ndarray        # int32, -1 when the team has no valid player
    gk_slot_away: np.ndarray
    gk_xy_home: np.ndarray          # [N, 2]
    gk_xy_away: np.ndarray
    gk_vel_home: np.ndarray         # [N, 2] m/s
    gk_vel_away: np.ndarray
    n_field_home: np.ndarray        # valid outfield players
    n_field_away: np.ndarray
    centroid_home: np.ndarray       # [N, 2]
    centroid_away: np.ndarray
    centroid_vx_home: np.ndarray    # m/s
    centroid_vx_away: np.ndarray
    depth_home: np.ndarray          # x extent of the outfield block
    depth_away: np.ndarray
    width_home: np.ndarray          # y extent
    width_away: np.ndarray
    spectral_area_home: np.ndarray
    spectral_area_away: np.ndarray
    hull_area_home: np.ndarray
    hull_area_away: np.ndarray
    d_spectral_home: np.ndarray     # m^2 / s
    d_spectral_away: np.ndarray
    d_hull_home: np.ndarray
    d_hull_away: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return dict(self.__dict__)

    def matrix(self, names: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
        """Stack the features into ``[N, F]`` float32; ``[N, 2]`` arrays are expanded as ``_x`` / ``_y``."""
        cols, keys = [], []
        for name in (names or list(self.__dict__)):
            arr = getattr(self, name)
            if arr.ndim == 1:
                cols.append(arr.astype(np.float32))
                keys.append(name)
            else:
                for j, ax in enumerate("xy"):
                    cols.append(arr[:, j].astype(np.float32))
                    keys.append(f"{name}_{ax}")
        return np.stack(cols, axis=1), keys


def dynamic_goalkeeper(players_xy: np.ndarray, team_of_slot: np.ndarray, team: int) -> np.ndarray:
    """Slot of the goalkeeper of ``team`` (0 home -> minimum ``x``; 1 away -> maximum ``x``) per frame; -1
    where the team has no valid player."""
    cols = np.flatnonzero(np.asarray(team_of_slot) == team)
    n = players_xy.shape[0]
    if not len(cols):
        return np.full(n, -1, dtype=np.int32)
    x = players_xy[:, cols, 0].astype(np.float64)
    ok = np.isfinite(x) & np.isfinite(players_xy[:, cols, 1])
    key = np.where(ok, x, np.inf if team == 0 else -np.inf)
    pick = np.argmin(key, axis=1) if team == 0 else np.argmax(key, axis=1)
    return np.where(ok.any(axis=1), cols[pick], -1).astype(np.int32)


def compute_collective_physics(
        players_xy: np.ndarray, team_of_slot: np.ndarray, *, players_vel: np.ndarray | None = None,
        fps: float = FPS, params: CollectivePhysicsParams | None = None,
        use_numba: bool | None = None) -> CollectivePhysics:
    """Goalkeepers, areas and derivatives for ``players_xy[N, S, 2]`` (metres, home attacks +x) and
    ``team_of_slot[S]`` (0 home, 1 away); see the module docstring.  ``players_vel[N, S, 2]`` (m/s) is
    derived from the positions when omitted."""
    p = params or CollectivePhysicsParams.from_config()
    pxy = np.asarray(players_xy, dtype=np.float32)
    if pxy.ndim != 3 or pxy.shape[2] != 2:
        raise ValueError(f"players_xy has shape {pxy.shape}, expected (N, S, 2)")
    n, s, _ = pxy.shape
    team = np.asarray(team_of_slot)
    if team.shape != (s,):
        raise ValueError("team_of_slot must have one entry per slot")
    vel = velocity_from_xy(pxy, fps, p.gk_vel_half_span) if players_vel is None else np.asarray(players_vel)
    span = max(int(round(p.deriv_span_s * fps)), 2)
    rows = np.arange(n)
    valid_all = np.isfinite(pxy).all(axis=2)
    out: dict[str, np.ndarray] = {}
    for code, name in ((0, "home"), (1, "away")):
        gk = dynamic_goalkeeper(pxy, team, code)
        has = gk >= 0
        g = np.maximum(gk, 0)
        out[f"gk_slot_{name}"] = gk
        out[f"gk_xy_{name}"] = np.where(has[:, None], pxy[rows, g], np.nan).astype(np.float32)
        out[f"gk_vel_{name}"] = np.where(has[:, None], vel[rows, g], np.nan).astype(np.float32)

        field = valid_all & (team == code)[None, :]
        field[rows[has], gk[has]] = False
        cnt = field.sum(axis=1)
        enough = cnt >= p.min_players
        fx = np.where(field, pxy[..., 0], np.nan).astype(np.float64)
        fy = np.where(field, pxy[..., 1], np.nan).astype(np.float64)
        keep = np.flatnonzero(team == code)
        fx, fy = fx[:, keep], fy[:, keep]
        with np.errstate(invalid="ignore", all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            cx, cy = np.nanmean(fx, axis=1), np.nanmean(fy, axis=1)
            depth = np.nanmax(fx, axis=1) - np.nanmin(fx, axis=1)
            width = np.nanmax(fy, axis=1) - np.nanmin(fy, axis=1)
        spectral = spectral_areas(fx, fy, p.min_players)
        hull = convex_hull_areas(fx, fy, use_numba=use_numba)
        for arr in (cx, cy, depth, width, spectral, hull):
            arr[~enough] = np.nan
        out[f"n_field_{name}"] = cnt.astype(np.float32)
        out[f"centroid_{name}"] = np.stack([cx, cy], axis=1).astype(np.float32)
        out[f"centroid_vx_{name}"] = central_derivative(cx, span, fps).astype(np.float32)
        out[f"depth_{name}"] = depth.astype(np.float32)
        out[f"width_{name}"] = width.astype(np.float32)
        out[f"spectral_area_{name}"] = spectral.astype(np.float32)
        out[f"hull_area_{name}"] = hull.astype(np.float32)
        out[f"d_spectral_{name}"] = central_derivative(spectral, span, fps).astype(np.float32)
        out[f"d_hull_{name}"] = central_derivative(hull, span, fps).astype(np.float32)
    return CollectivePhysics(**out)
