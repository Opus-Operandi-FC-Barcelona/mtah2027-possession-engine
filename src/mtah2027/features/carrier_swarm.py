"""Implicit ball carrier and 22-v-22 pressure micro-geometry (Issue #64).

The hidden-ball specialist cannot see the ball, but a team that has it is the one whose moving player is being
converged on by the rivals.  Per frame and per team ``T`` (outfield players only, goalkeepers excluded):

* ``pressure_i`` for every moving outfield player ``i`` of ``T`` (``|v_i| >= moving_speed``): the sum over the
  outfield rivals ``j`` within ``radius`` metres of ``exp(-d_ij / decay) * max(closing_speed_ij, 0)`` with
  ``closing_speed_ij = -(p_j - p_i) . (v_j - v_i) / d_ij`` (positive when they approach each other); players
  that do not move score 0;
* ``swarm`` = ``max_i pressure_i``; implicit carrier = the argmax player: ``carrier_x`` and ``carrier_vx``
  (orientation-normalised, home attacks +x);
* ``min_interteam_dist`` = smallest home-away outfield distance and ``duel_density`` = number of home-away
  outfield pairs closer than ``duel_m`` (both team-independent).

NaN marks frames where a team has no valid moving outfield player (swarm / carrier) or fewer than one valid
pair.  Vectorised in chunks of frames, deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mtah2027.features._utils import velocity_from_xy
from mtah2027.features.collective_physics import dynamic_goalkeeper

NAMES = ("swarm_home", "swarm_away", "carrier_vx_home", "carrier_vx_away", "carrier_x_home",
         "carrier_x_away", "min_interteam_dist", "duel_density_3m")


@dataclass(frozen=True)
class CarrierSwarmParams:
    decay_m: float = 2.0
    radius_m: float = 3.5
    moving_speed: float = 1.5
    duel_m: float = 3.0
    chunk: int = 4096


def carrier_swarm(pxy: np.ndarray, team: np.ndarray, pvel: np.ndarray | None = None, *, fps: float = 25.0,
                  params: CarrierSwarmParams | None = None) -> dict[str, np.ndarray]:
    """Features of ``pxy[N, S, 2]`` (m, home attacks +x), ``team[S]`` (0 home, 1 away), ``pvel`` (m/s,
    derived from the positions when omitted).  Returns one float32 array (length N) per name in ``NAMES``."""
    p = params or CarrierSwarmParams()
    pxy = np.asarray(pxy, np.float32)
    team = np.asarray(team)
    n = len(pxy)
    vel = velocity_from_xy(pxy, fps, 1) if pvel is None else np.asarray(pvel, np.float32)
    gk = {t: dynamic_goalkeeper(pxy, team, t) for t in (0, 1)}
    out = {k: np.full(n, np.nan, np.float32) for k in NAMES}
    cols = {t: np.flatnonzero(team == t) for t in (0, 1)}
    if min(len(cols[0]), len(cols[1])) == 0:
        return out
    for a in range(0, n, p.chunk):
        b = min(a + p.chunk, n)
        rows = np.arange(b - a)
        pos, v = {}, {}
        for t in (0, 1):
            xy = pxy[a:b][:, cols[t]].astype(np.float64)
            vv = vel[a:b][:, cols[t]].astype(np.float64)
            ok = np.isfinite(xy).all(axis=2) & np.isfinite(vv).all(axis=2)
            g = gk[t][a:b]
            has = g >= 0
            local = np.searchsorted(cols[t], g)           # slot -> column of this team
            ok[rows[has], local[has]] = False             # goalkeepers excluded
            pos[t] = (np.where(ok[..., None], xy, np.nan), ok)
            v[t] = np.where(ok[..., None], vv, 0.0)
        # pair geometry between the outfield players of both teams: d[f, i_home, j_away]
        (ph, okh), (pa, oka) = pos[0], pos[1]
        pair_ok = okh[:, :, None] & oka[:, None, :]
        dx = np.where(pair_ok, pa[:, None, :, 0] - ph[:, :, None, 0], 0.0)
        dy = np.where(pair_ok, pa[:, None, :, 1] - ph[:, :, None, 1], 0.0)
        d = np.sqrt(dx * dx + dy * dy)
        dsafe = np.maximum(d, 1e-3)
        rvx = v[1][:, None, :, 0] - v[0][:, :, None, 0]   # away velocity relative to the home player
        rvy = v[1][:, None, :, 1] - v[0][:, :, None, 1]
        closing = np.maximum(-(dx * rvx + dy * rvy) / dsafe, 0.0)
        w = np.where(pair_ok & (d <= p.radius_m), np.exp(-d / p.decay_m) * closing, 0.0)
        dmin = np.where(pair_ok, d, np.inf)
        npair = pair_ok.sum(axis=(1, 2))
        mind = dmin.min(axis=(1, 2))
        out["min_interteam_dist"][a:b] = np.where(npair > 0, mind, np.nan)
        duels = (pair_ok & (d < p.duel_m)).sum(axis=(1, 2))
        out["duel_density_3m"][a:b] = np.where(npair > 0, duels, np.nan)
        for t, name in ((0, "home"), (1, "away")):
            # pressure on player i of team t by the rivals; moving players only
            press = w.sum(axis=2) if t == 0 else w.sum(axis=1)
            speed = np.hypot(v[t][..., 0], v[t][..., 1])
            moving = pos[t][1] & (speed >= p.moving_speed)
            score = np.where(moving, press, -np.inf)
            best = np.argmax(score, axis=1)
            any_mv = moving.any(axis=1)
            out[f"swarm_{name}"][a:b] = np.where(any_mv, score[rows, best], np.nan)
            out[f"carrier_vx_{name}"][a:b] = np.where(any_mv, v[t][rows, best, 0], np.nan)
            out[f"carrier_x_{name}"][a:b] = np.where(any_mv, pos[t][0][rows, best, 0], np.nan)
    return out
