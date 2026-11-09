"""Continuous dead-ball (``D``) signals per frame, with and without a visible ball (Issue #13).

Everything is vectorised NumPy over full-length arrays.  Inputs are plain arrays (metres, home
attacking ``+x`` -- the output of Issues #5/#8/#9) so the same function runs on the competition match
and on external benchmark matches.

Per-frame features (float32 ``[N]`` unless noted)
-------------------------------------------------
``ball_visible``      1 when the ball has a position.
``ball_out``          1 when the ball is beyond a touchline/goal line by more than ``out_margin_m``;
                      ``ball_out_dist`` is that distance in metres (0 inside, NaN without ball).
``ball_still_s``      cumulative seconds with ball speed < ``still_speed_mps``; reset to 0 as soon as the
                      ball moves; NaN while the ball (or its speed) is missing -- a gap restarts the count.
``speed_mean``, ``speed_p80``  mean / 80th percentile of the speed of the *active* players (finite
                      position and speed; NaN if fewer than ``min_active_players``).
``frac_walking``      share of active players slower than ``walk_speed_mps``.
``t_since_ball`` / ``t_to_ball``  seconds since the last / until the next frame with a visible ball
                      (0 when visible, capped at ``visibility_cap_s``, also when there is none).
``geom_corner``       ball within ``corner_radius_m`` of a corner flag.
``geom_goal_area``    ball inside a 6-yard box (goal kick spot).
``geom_kickoff``      ball within ``centre_radius_m`` of the centre spot *and* at least
                      ``kickoff_min_fraction`` of each team's active players in their own half (home
                      ``x <= L/2``, away ``x >= L/2``, with a tolerance) -- requires home attacking ``+x``.
``geom_touchline``    ball within ``touchline_margin_m`` of a touchline outside the corner zones.
``geom_any``          max of the four geometry indicators.
``status_dead``       1/0 from the ATD alive/dead ball channel if one is declared, else all NaN.
``dead_score``        in [0, 1], heuristic dead-ball probability (below).

``dead_score``
--------------
With ``z`` the logit::

    z = bias + w_out*ball_out + w_still*min(ball_still_s/still_ref_s, 1)
             + w_slow*clip(1 - speed_mean_sm/speed_ref, 0, 1) + w_walk*frac_walking_sm
             + w_geom*geom_any*min(ball_still_s/still_ref_s, 1) + w_status*(2*status_dead - 1)
    dead_score = 1 / (1 + exp(-z))

``*_sm`` are NaN-aware moving averages over ``speed_smooth_s``.  A term whose input is NaN
contributes 0, and without a ball ``bias_no_ball`` replaces ``bias`` (so the score then rests on the
collective speed terms).  Weights come from ``configs/default.yaml`` (section ``dead_ball_signals``)
and are the MVP emission for ``D`` before the learned specialists exist.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Mapping

import numpy as np

from mtah2027.config import FPS, PITCH, load_section, params_from_mapping
from mtah2027.features._utils import (
    frames_since,
    frames_until,
    moving_mean,
    row_percentile,
    run_length_before,
    sigmoid,
    speed_from_xy,
)

DEAD_WORDS = ("not alive", "notalive", "dead", "out", "stopp", "paus", "inactive", "off")
ALIVE_WORDS = ("alive", "live", "play", "active", "in")
_NON_STATUS = {"x", "y", "z", "speed"}


@dataclass(frozen=True)
class DeadBallParams:
    out_margin_m: float = 0.3
    still_speed_mps: float = 0.4
    still_ref_s: float = 2.0
    walk_speed_mps: float = 1.7
    min_active_players: int = 6
    speed_smooth_s: float = 1.0
    visibility_cap_s: float = 60.0
    ball_speed_half_span: int = 2
    player_speed_half_span: int = 1
    corner_radius_m: float = 2.0
    goal_area_depth_m: float = 6.0
    goal_area_half_width_m: float = 9.7
    centre_radius_m: float = 1.5
    kickoff_own_half_tol_m: float = 1.0
    kickoff_min_fraction: float = 0.85
    kickoff_min_players: int = 5
    touchline_margin_m: float = 1.5
    status_channels: tuple = ("status", "ball_status", "ballstatus", "alive", "ball_alive", "ballalive",
                              "dead", "ball_dead", "state")
    bias: float = -3.5
    bias_no_ball: float = -3.0
    w_out: float = 4.0
    w_still: float = 3.0
    w_slow: float = 2.0
    w_walk: float = 2.5
    w_geom: float = 1.5
    w_status: float = 6.0
    speed_ref_mps: float = 2.5

    @classmethod
    def from_config(cls, path=None) -> "DeadBallParams":
        return params_from_mapping(cls, load_section("dead_ball_signals", path), "dead_ball_signals")


@dataclass
class DeadBallSignals:
    """Per-frame dead-ball features (see the module docstring)."""

    ball_visible: np.ndarray
    ball_out: np.ndarray
    ball_out_dist: np.ndarray
    ball_still_s: np.ndarray
    speed_mean: np.ndarray
    speed_p80: np.ndarray
    frac_walking: np.ndarray
    n_active: np.ndarray
    t_since_ball: np.ndarray
    t_to_ball: np.ndarray
    geom_corner: np.ndarray
    geom_goal_area: np.ndarray
    geom_kickoff: np.ndarray
    geom_touchline: np.ndarray
    geom_any: np.ndarray
    status_dead: np.ndarray
    dead_logit: np.ndarray
    dead_score: np.ndarray
    status_channel: str | None = None
    meta: dict = field(default_factory=dict)

    @property
    def has_status(self) -> bool:
        return self.status_channel is not None

    def as_dict(self) -> dict[str, np.ndarray]:
        return {f.name: getattr(self, f.name) for f in fields(self)
                if isinstance(getattr(self, f.name), np.ndarray)}

    def matrix(self, names: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
        """Stack features into ``[N, F]`` float32 (feature names returned alongside)."""
        d = self.as_dict()
        keys = list(names) if names is not None else list(d)
        return np.stack([d[k].astype(np.float32) for k in keys], axis=1), keys


# --------------------------------------------------------------------------------------------
def resolve_status_dead(ball_channels: Mapping[str, np.ndarray] | None,
                        ball_channel_codes: Mapping[str, Mapping[str, int]] | None,
                        names: tuple, n_frames: int) -> tuple[np.ndarray | None, str | None]:
    """Dead (1) / alive (0) / unknown (NaN) per frame from an ATD ball status channel, if declared.

    A text channel is read through its code table (texts containing dead/out/stopped... vs
    alive/live/play...).  A purely numeric channel is only interpreted when its name says what 1
    means (``dead`` -> 1 = dead; ``alive``/``live`` -> 1 = alive); anything ambiguous is ignored.
    """
    if not ball_channels:
        return None, None
    wanted = {str(n).lower() for n in names}
    for key, arr in ball_channels.items():
        low = str(key).lower()
        if low in _NON_STATUS or low not in wanted:
            continue
        values = np.asarray(arr, dtype=np.float64)
        if values.shape != (n_frames,):
            continue
        table = (ball_channel_codes or {}).get(key)
        if table:
            dead_codes = [c for t, c in table.items() if any(w in str(t).lower() for w in DEAD_WORDS)]
            alive_codes = [c for t, c in table.items()
                           if c not in dead_codes and any(w in str(t).lower() for w in ALIVE_WORDS)]
            if not dead_codes and not alive_codes:
                continue
            out = np.full(n_frames, np.nan, dtype=np.float32)
            out[np.isin(values, alive_codes)] = 0.0
            out[np.isin(values, dead_codes)] = 1.0
            return out, key
        finite = np.isfinite(values)
        if "dead" in low:
            out = np.where(finite, (values > 0.5).astype(np.float32), np.nan)
        elif "alive" in low or "live" in low:
            out = np.where(finite, (values <= 0.5).astype(np.float32), np.nan)
        else:
            continue
        return out.astype(np.float32), key
    return None, None


def compute_dead_ball_signals(
        players_xy: np.ndarray, team_of_slot: np.ndarray, ball_xy: np.ndarray, *,
        players_speed: np.ndarray | None = None, ball_speed: np.ndarray | None = None,
        ball_channels: Mapping[str, np.ndarray] | None = None,
        ball_channel_codes: Mapping[str, Mapping[str, int]] | None = None,
        fps: float = FPS, pitch: tuple[float, float] = PITCH,
        params: DeadBallParams | None = None) -> DeadBallSignals:
    """All dead-ball features for ``players_xy[N, S, 2]``, ``team_of_slot[S]`` (0 home, 1 away) and
    ``ball_xy[N, 2]`` (metres, NaN when missing).  Speeds (m/s) are derived from positions when not given.
    """
    p = params or DeadBallParams.from_config()
    n, _s, _ = players_xy.shape
    if ball_xy.shape != (n, 2):
        raise ValueError(f"ball_xy has shape {ball_xy.shape}, expected ({n}, 2)")
    team = np.asarray(team_of_slot)
    length, width = pitch
    bx, by = ball_xy[:, 0].astype(np.float64), ball_xy[:, 1].astype(np.float64)
    visible = np.isfinite(bx) & np.isfinite(by)

    # (a) out of bounds -- distance beyond the rectangle
    dx = np.maximum(np.maximum(-bx, bx - length), 0.0)
    dy = np.maximum(np.maximum(-by, by - width), 0.0)
    out_dist = np.where(visible, np.hypot(dx, dy), np.nan)
    with np.errstate(invalid="ignore"):
        ball_out = np.where(visible, (out_dist > p.out_margin_m).astype(np.float32), 0.0)

    # (b) stationary time
    if ball_speed is None:
        bspeed = speed_from_xy(ball_xy, fps, p.ball_speed_half_span)
    else:
        bspeed = np.asarray(ball_speed, dtype=np.float32)
    known = np.isfinite(bspeed) & visible
    with np.errstate(invalid="ignore"):
        still = known & (bspeed < p.still_speed_mps)
    still_s = np.where(known, run_length_before(still) / fps, np.nan).astype(np.float32)

    # (c) collective speed over active players
    if players_speed is None:
        pspeed = speed_from_xy(players_xy, fps, p.player_speed_half_span)
    else:
        pspeed = np.asarray(players_speed, dtype=np.float32)
    active = np.isfinite(pspeed) & np.isfinite(players_xy).all(axis=2)
    spd = np.where(active, pspeed, np.nan)
    n_active = active.sum(axis=1).astype(np.int16)
    enough = n_active >= p.min_active_players
    with np.errstate(invalid="ignore"):
        mean = np.where(enough, np.nansum(spd, axis=1) / np.maximum(n_active, 1), np.nan).astype(np.float32)
        walking = np.where(active, (spd < p.walk_speed_mps).astype(np.float32), np.nan)
        frac_walk = np.where(enough, np.nansum(walking, axis=1) / np.maximum(n_active, 1),
                             np.nan).astype(np.float32)
    p80 = row_percentile(spd, 0.8, p.min_active_players)

    # visibility timing
    cap = p.visibility_cap_s
    since = frames_since(visible)
    until = frames_until(visible)
    t_since = np.where(since >= 0, np.minimum(since / fps, cap), cap).astype(np.float32)
    t_to = np.where(until >= 0, np.minimum(until / fps, cap), cap).astype(np.float32)

    # (d) set-piece geometry
    c1 = np.minimum(np.hypot(bx, by), np.hypot(length - bx, by))
    c2 = np.minimum(np.hypot(bx, width - by), np.hypot(length - bx, width - by))
    geom_corner = visible & (np.minimum(c1, c2) <= p.corner_radius_m)
    in_depth = (bx <= p.goal_area_depth_m) | (bx >= length - p.goal_area_depth_m)
    geom_goal = visible & in_depth & (np.abs(by - width / 2.0) <= p.goal_area_half_width_m)
    near_centre = visible & (np.hypot(bx - length / 2.0, by - width / 2.0) <= p.centre_radius_m)
    px = players_xy[..., 0]
    own = np.zeros(n, dtype=bool)
    if near_centre.any():
        own = np.ones(n, dtype=bool)
        for code, inside in ((0, px <= length / 2.0 + p.kickoff_own_half_tol_m),
                             (1, px >= length / 2.0 - p.kickoff_own_half_tol_m)):
            cols = team == code
            act = active[:, cols]
            cnt = act.sum(axis=1)
            frac = (inside[:, cols] & act).sum(axis=1) / np.maximum(cnt, 1)
            own &= (cnt >= p.kickoff_min_players) & (frac >= p.kickoff_min_fraction)
    geom_kick = near_centre & own
    near_line = visible & ((by <= p.touchline_margin_m) | (by >= width - p.touchline_margin_m))
    geom_touch = near_line & ~geom_corner & (bx >= 0) & (bx <= length)
    geom_any = geom_corner | geom_goal | geom_kick | geom_touch

    # status channel (alive / dead) from the ATD
    status, status_key = resolve_status_dead(ball_channels, ball_channel_codes, p.status_channels, n)
    status_arr = status if status is not None else np.full(n, np.nan, dtype=np.float32)

    # combined heuristic
    win = max(1, int(round(p.speed_smooth_s * fps)))
    mean_sm, walk_sm = moving_mean(mean, win), moving_mean(frac_walk, win)
    still_n = np.where(np.isfinite(still_s), np.minimum(still_s / p.still_ref_s, 1.0), 0.0)
    slow = np.where(np.isfinite(mean_sm), np.clip(1.0 - mean_sm / p.speed_ref_mps, 0.0, 1.0), 0.0)
    walk = np.where(np.isfinite(walk_sm), walk_sm, 0.0)
    z = np.where(visible, p.bias, p.bias_no_ball).astype(np.float64)
    z += p.w_out * ball_out + p.w_still * still_n + p.w_slow * slow + p.w_walk * walk
    z += p.w_geom * geom_any * still_n
    z += np.where(np.isfinite(status_arr), p.w_status * (2.0 * np.nan_to_num(status_arr) - 1.0), 0.0)

    f32 = np.float32
    return DeadBallSignals(
        ball_visible=visible.astype(f32), ball_out=ball_out.astype(f32), ball_out_dist=out_dist.astype(f32),
        ball_still_s=still_s, speed_mean=mean, speed_p80=p80, frac_walking=frac_walk, n_active=n_active,
        t_since_ball=t_since, t_to_ball=t_to, geom_corner=geom_corner.astype(f32),
        geom_goal_area=geom_goal.astype(f32), geom_kickoff=geom_kick.astype(f32),
        geom_touchline=geom_touch.astype(f32), geom_any=geom_any.astype(f32), status_dead=status_arr,
        dead_logit=z.astype(f32), dead_score=sigmoid(z).astype(f32), status_channel=status_key,
        meta={"fps": float(fps), "pitch": tuple(pitch)})
