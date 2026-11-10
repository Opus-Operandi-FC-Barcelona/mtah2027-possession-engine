"""Velocity phase coherence and ball memory for the frames in which the ball is hidden (Issue #18).

Inputs are plain arrays in metres, *oriented so that the home team attacks +x* (Issue #9); NaN marks missing
slots / ball.  Everything is vectorised over frames.  Teams are coded 0 (home), 1 (away), -1 (unknown / none).

Phase coherence and line advance (per team, over the valid players)
------------------------------------------------------------------
``phase_R_*``  ``R = || mean_i  v_i / (|v_i| + 1e-3) ||`` in [0, 1] (the length of the mean unit velocity
    vector, i.e. a circular statistic immune to the discontinuity of the heading at +-pi).  A team moving
    as a block (pressing, counter-attack) has ``R`` close to 1; players that stand still contribute almost
    nothing because of the ``1e-3`` term.  NaN with fewer than ``min_players`` valid velocities.
``line_vx_*``  mean ``v_x`` of the team (m/s, smoothed over ``smooth_s``): positive = the block moves towards
    +x.  ``line_vx_diff = line_vx_home + line_vx_away``: home advance minus away advance, since the away team
    attacks -x (positive = the home block is gaining ground on the away block).

Ball memory (from the frames in which the ball is visible)
----------------------------------------------------------
``since_seen_s`` / ``until_seen_s``  seconds since the ball was last seen / until it is seen again (0 when
    visible, capped at ``cap_s``, NaN when it is never seen on that side); ``last_seen_xy`` / ``next_seen_xy``
    and ``last_seen_vel`` / ``next_seen_vel`` its position and velocity there; ``gap_frac`` the relative
    position inside the current hidden run (``since / (since + until)``, NaN outside or when open-ended);
    ``gap_dist_m`` the distance between the last and the next seen position.
``last_touch_team`` / ``next_touch_team`` (+ ``since_last_touch_s`` / ``until_next_touch_s``)  the team of the
    player that clearly touched the ball last / next: a touch peak with ``|delta_d| >= touch_margin_m``, the
    sign of ``delta_d`` (home closer if positive) giving the team.
``last_clear_team`` / ``next_clear_team`` (+ times)  same for any visible frame with ``|possession_evidence|
    >= clear_evidence`` (a denser, weaker cue than touches).
Pressure (players counted within ``pressure_radius_m`` at the *current* frame):
``press_last_home/away``, ``press_next_home/away``  around the last / next seen ball position;
``dist_last_home/away``, ``dist_next_home/away``  distance of the nearest player of each team to those points;
``adv_x_home/away``  ``x`` of each team's most advanced outfield player (the goalkeeper -- the extreme player
    on its own side -- is skipped), and ``adv_press_home/away``  opponents within the radius of that player.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mtah2027.config import FPS, load_section, params_from_mapping
from mtah2027.features._utils import moving_mean, velocity_from_xy

H_TEAM, A_TEAM, NO_TEAM = 0, 1, -1


@dataclass(frozen=True)
class BallMemoryParams:
    min_players: int = 4             # fewer valid players -> coherence / advance are NaN
    smooth_s: float = 1.0            # moving average of the team mean v_x
    cap_s: float = 120.0             # cap of the time since / until features
    ball_vel_half_span: int = 2      # frames each side when the ball velocity is derived from positions
    player_vel_half_span: int = 1
    touch_margin_m: float = 0.5      # |delta_d| at a touch needed to attribute it to a team
    clear_evidence: float = 0.35     # |possession_evidence| from which a visible frame is a clear possessor
    pressure_radius_m: float = 5.0   # radius of the pressure counts

    @classmethod
    def from_config(cls, path=None) -> "BallMemoryParams":
        return params_from_mapping(cls, load_section("ball_memory", path), "ball_memory")


@dataclass
class BallMemory:
    """Outputs of :func:`compute_ball_memory` (see the module docstring); ``[N]`` unless noted."""

    phase_R_home: np.ndarray
    phase_R_away: np.ndarray
    line_vx_home: np.ndarray
    line_vx_away: np.ndarray
    line_vx_diff: np.ndarray
    ball_visible: np.ndarray
    since_seen_s: np.ndarray
    until_seen_s: np.ndarray
    gap_frac: np.ndarray
    gap_dist_m: np.ndarray
    last_seen_xy: np.ndarray          # [N, 2]
    next_seen_xy: np.ndarray          # [N, 2]
    last_seen_vel: np.ndarray         # [N, 2]
    next_seen_vel: np.ndarray         # [N, 2]
    last_touch_team: np.ndarray       # int8
    next_touch_team: np.ndarray       # int8
    since_last_touch_s: np.ndarray
    until_next_touch_s: np.ndarray
    last_clear_team: np.ndarray       # int8
    next_clear_team: np.ndarray       # int8
    since_last_clear_s: np.ndarray
    until_next_clear_s: np.ndarray
    press_last_home: np.ndarray
    press_last_away: np.ndarray
    press_next_home: np.ndarray
    press_next_away: np.ndarray
    dist_last_home: np.ndarray
    dist_last_away: np.ndarray
    dist_next_home: np.ndarray
    dist_next_away: np.ndarray
    adv_x_home: np.ndarray
    adv_x_away: np.ndarray
    adv_press_home: np.ndarray
    adv_press_away: np.ndarray

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


# --------------------------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------------------------
def last_index(mask: np.ndarray) -> np.ndarray:
    """Index of the last True at or before each frame (-1 if none)."""
    idx = np.arange(len(mask))
    return np.maximum.accumulate(np.where(mask, idx, -1))


def next_index(mask: np.ndarray) -> np.ndarray:
    """Index of the first True at or after each frame (``len(mask)`` if none)."""
    n = len(mask)
    idx = np.arange(n)
    return np.minimum.accumulate(np.where(mask, idx, n)[::-1])[::-1]


def phase_coherence(vel: np.ndarray, team_of_slot: np.ndarray, team: int, min_players: int = 4,
                    eps: float = 1e-3) -> np.ndarray:
    """``|| mean_i v_i / (|v_i| + eps) ||`` over the valid players of ``team`` per frame (NaN if fewer than
    ``min_players``)."""
    v = np.asarray(vel, dtype=np.float64)[:, np.asarray(team_of_slot) == team]
    ok = np.isfinite(v).all(axis=2)
    vz = np.where(ok[..., None], v, 0.0)
    unit = vz / (np.linalg.norm(vz, axis=2)[..., None] + eps)
    cnt = ok.sum(axis=1)
    mean = unit.sum(axis=1) / np.maximum(cnt, 1)[:, None]
    return np.where(cnt >= max(min_players, 1), np.linalg.norm(mean, axis=1), np.nan).astype(np.float32)


def team_mean_vx(vel: np.ndarray, team_of_slot: np.ndarray, team: int, min_players: int = 4) -> np.ndarray:
    """Mean ``v_x`` of the valid players of ``team`` per frame (NaN if fewer than ``min_players``)."""
    v = np.asarray(vel, dtype=np.float64)[:, np.asarray(team_of_slot) == team]
    ok = np.isfinite(v).all(axis=2)
    cnt = ok.sum(axis=1)
    mean = np.where(ok, v[..., 0], 0.0).sum(axis=1) / np.maximum(cnt, 1)
    return np.where(cnt >= max(min_players, 1), mean, np.nan).astype(np.float32)


def nearest_and_count(players_xy: np.ndarray, team_of_slot: np.ndarray, point_xy: np.ndarray, radius_m: float,
                      team: int) -> tuple[np.ndarray, np.ndarray]:
    """Per frame: distance of the nearest ``team`` player to ``point_xy[N, 2]`` and how many are within
    ``radius_m`` (NaN / 0 where the point or all players are unknown)."""
    cols = np.flatnonzero(np.asarray(team_of_slot) == team)
    diff = players_xy[:, cols].astype(np.float32) - point_xy[:, None, :].astype(np.float32)
    dist = np.hypot(diff[..., 0], diff[..., 1])
    with np.errstate(invalid="ignore"):
        count = (dist <= radius_m).sum(axis=1).astype(np.float32)
    inf = np.where(np.isfinite(dist), dist, np.inf)
    nearest = inf.min(axis=1) if inf.shape[1] else np.full(len(players_xy), np.inf, dtype=np.float32)
    nearest = np.where(np.isfinite(nearest), nearest, np.nan).astype(np.float32)
    known = np.isfinite(point_xy).all(axis=1)
    return nearest, np.where(known, count, np.nan).astype(np.float32)


def advanced_outfield_xy(players_xy: np.ndarray, team_of_slot: np.ndarray, team: int) -> np.ndarray:
    """Position ``[N, 2]`` of the most advanced outfield player of ``team`` (home: max ``x``; away: min
    ``x``), skipping the goalkeeper (the extreme player on the own side); NaN with fewer than two valid
    players."""
    cols = np.flatnonzero(np.asarray(team_of_slot) == team)
    n = players_xy.shape[0]
    pts = players_xy[:, cols].astype(np.float64)
    ok = np.isfinite(pts).all(axis=2)
    x = np.where(ok, pts[..., 0], np.nan)
    rows = np.arange(n)
    if not len(cols):
        return np.full((n, 2), np.nan, dtype=np.float32)
    own = np.where(ok, x, np.inf if team == 0 else -np.inf)
    gk = np.argmin(own, axis=1) if team == 0 else np.argmax(own, axis=1)
    ok2 = ok.copy()
    ok2[rows, gk] = False
    key = np.where(ok2, x, -np.inf if team == 0 else np.inf)
    pick = np.argmax(key, axis=1) if team == 0 else np.argmin(key, axis=1)
    out = pts[rows, pick]
    return np.where((ok.sum(axis=1) >= 2)[:, None], out, np.nan).astype(np.float32)


def _team_at(team_code: np.ndarray, idx: np.ndarray, n: int) -> np.ndarray:
    """Team coded at event index ``idx`` (``NO_TEAM`` where ``idx`` is out of range)."""
    ok = (idx >= 0) & (idx < n)
    return np.where(ok, team_code[np.clip(idx, 0, n - 1)], NO_TEAM).astype(np.int8)


def _seconds(delta_frames: np.ndarray, exists: np.ndarray, fps: float, cap_s: float) -> np.ndarray:
    return np.where(exists, np.minimum(delta_frames / fps, cap_s), np.nan).astype(np.float32)


# --------------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------------
def compute_ball_memory(
        players_xy: np.ndarray, team_of_slot: np.ndarray, ball_xy: np.ndarray, *,
        players_vel: np.ndarray | None = None, ball_vel: np.ndarray | None = None,
        touch_mask: np.ndarray | None = None, delta_d: np.ndarray | None = None,
        possession_evidence: np.ndarray | None = None, fps: float = FPS,
        params: BallMemoryParams | None = None) -> BallMemory:
    """Phase coherence, line advance and ball memory for ``players_xy[N, S, 2]`` (metres, home attacks +x),
    ``team_of_slot[S]`` and ``ball_xy[N, 2]`` (NaN while hidden).

    ``touch_mask[N]`` / ``delta_d[N]`` (from Issue #16) give the touch events and the sign of the closer team;
    ``possession_evidence[N]`` (Issue #16) the clear-possessor frames.  Without them the corresponding
    features are ``NO_TEAM`` / NaN.  Velocities (m/s) are derived from positions when not supplied.
    """
    p = params or BallMemoryParams.from_config()
    pxy = np.asarray(players_xy, dtype=np.float32)
    bxy = np.asarray(ball_xy, dtype=np.float32)
    n, s, _ = pxy.shape
    team = np.asarray(team_of_slot)
    if bxy.shape != (n, 2) or team.shape != (s,):
        raise ValueError("ball_xy / team_of_slot are inconsistent with players_xy")
    pvel = (velocity_from_xy(pxy, fps, p.player_vel_half_span) if players_vel is None
            else np.asarray(players_vel))
    bvel = velocity_from_xy(bxy, fps, p.ball_vel_half_span) if ball_vel is None else np.asarray(ball_vel)
    visible = np.isfinite(bxy).all(axis=1)
    rows = np.arange(n)
    win = max(1, int(round(p.smooth_s * fps)))

    out: dict[str, np.ndarray] = {}
    # coherence and line advance ----------------------------------------------------------------
    for code, name in ((0, "home"), (1, "away")):
        out[f"phase_R_{name}"] = phase_coherence(pvel, team, code, p.min_players)
        out[f"line_vx_{name}"] = moving_mean(team_mean_vx(pvel, team, code, p.min_players), win)
    out["line_vx_diff"] = (out["line_vx_home"] + out["line_vx_away"]).astype(np.float32)

    # visibility memory ---------------------------------------------------------------------------
    last, nxt = last_index(visible), next_index(visible)
    has_last, has_next = last >= 0, nxt < n
    out["ball_visible"] = visible.astype(np.float32)
    out["since_seen_s"] = _seconds(rows - last, has_last, fps, p.cap_s)
    out["until_seen_s"] = _seconds(nxt - rows, has_next, fps, p.cap_s)
    both = has_last & has_next & ~visible
    span = (nxt - last).astype(np.float64)
    out["gap_frac"] = np.where(both, (rows - last) / np.maximum(span, 1.0), np.nan).astype(np.float32)
    last_xy = np.where(has_last[:, None], bxy[np.maximum(last, 0)], np.nan).astype(np.float32)
    next_xy = np.where(has_next[:, None], bxy[np.minimum(nxt, n - 1)], np.nan).astype(np.float32)
    out["last_seen_xy"], out["next_seen_xy"] = last_xy, next_xy
    out["last_seen_vel"] = np.where(has_last[:, None], bvel[np.maximum(last, 0)], np.nan).astype(np.float32)
    out["next_seen_vel"] = np.where(has_next[:, None], bvel[np.minimum(nxt, n - 1)],
                                    np.nan).astype(np.float32)
    out["gap_dist_m"] = np.where(both, np.linalg.norm(next_xy - last_xy, axis=1), np.nan).astype(np.float32)

    # touches and clear possessors ------------------------------------------------------------------
    def events(code: np.ndarray, mask: np.ndarray, prefix: str, last_name: str, next_name: str):
        ev = mask & (code != NO_TEAM)
        li, ni = last_index(ev), next_index(ev)
        out[f"last_{prefix}_team"] = _team_at(code, li, n)
        out[f"next_{prefix}_team"] = _team_at(code, ni, n)
        out[last_name] = _seconds(rows - li, li >= 0, fps, p.cap_s)
        out[next_name] = _seconds(ni - rows, ni < n, fps, p.cap_s)

    gap = np.full(n, np.nan) if delta_d is None else np.asarray(delta_d, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        touch_code = np.where(gap > p.touch_margin_m, H_TEAM,
                              np.where(gap < -p.touch_margin_m, A_TEAM, NO_TEAM))
    touches = np.zeros(n, bool) if touch_mask is None else np.asarray(touch_mask, dtype=bool)
    events(touch_code.astype(np.int8), touches, "touch", "since_last_touch_s", "until_next_touch_s")

    evid = (np.full(n, np.nan) if possession_evidence is None
            else np.asarray(possession_evidence, dtype=np.float64))
    with np.errstate(invalid="ignore"):
        clear = visible & (np.abs(evid) >= p.clear_evidence)
        clear_code = np.where(evid > 0, H_TEAM, A_TEAM).astype(np.int8)
    events(clear_code, clear, "clear", "since_last_clear_s", "until_next_clear_s")

    # pressure around the remembered ball positions and the most advanced players ------------------
    for tag, point in (("last", last_xy), ("next", next_xy)):
        for code, name in ((0, "home"), (1, "away")):
            dist, count = nearest_and_count(pxy, team, point, p.pressure_radius_m, code)
            out[f"press_{tag}_{name}"], out[f"dist_{tag}_{name}"] = count, dist
    for code, name, opp in ((0, "home", 1), (1, "away", 0)):
        adv = advanced_outfield_xy(pxy, team, code)
        out[f"adv_x_{name}"] = adv[:, 0].astype(np.float32)
        out[f"adv_press_{name}"] = nearest_and_count(pxy, team, adv, p.pressure_radius_m, opp)[1]
    return BallMemory(**out)
