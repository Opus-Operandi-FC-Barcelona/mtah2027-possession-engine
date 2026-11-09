"""Ball-to-player geometry, control weights and touch detection (Issue #16).

Fully vectorised over frames.  Inputs are plain arrays in metres (home attacking +x; NaN where a slot
or the ball is missing), so the same code serves the competition match and external benchmarks.

Per-frame outputs (``k = 3`` by default)
----------------------------------------
``d_home[N, k]``, ``d_away[N, k]``  distances (m) from the ball to the k nearest players of each team,
    ascending; NaN where fewer than k players (or no ball) are available.
``delta_d[N]``  ``d_away[:, 0] - d_home[:, 0]``; positive when the home team is closer.
``nearest_slot_home/away[N]``  slot index of the closest player of each team (-1 if none).
``rel_speed_home/away[N]``  ``|v_player - v_ball|`` of that closest player (m/s).
``closing_speed_home/away[N]``  rate at which that team's nearest distance shrinks (m/s, central
    difference; positive = approaching).
``nearest_team[N]``  0 (home) / 1 (away) for the team of the closest player to the ball, -1 when the
    ball is missing, a team has no valid player, or the two distances tie.

Ball-control weight (inspired by Link & Hoernig's ball-control/dominance models)
--------------------------------------------------------------------------------
Each of the k nearest players ``i`` of a team gets a *control weight* that decreases with distance and
with the speed of the player relative to the ball (a player running past the ball does not control it)::

    w_i = exp(-d_i^2 / (2 sigma^2)) * exp(-|v_i - v_ball| / tau)

with ``sigma = control_sigma_m`` and ``tau = control_tau_mps``;

and the team control is the probability that at least one of them controls the ball,
``C_team = 1 - prod_i (1 - w_i)``.  ``control_diff = (C_home - C_away) / (C_home + C_away)`` in [-1, 1]
(0 when both are 0).  An unknown relative speed counts as 0.

Touches
-------
With ``a = |a_ball|`` and ``theta`` the angle between the ball velocity ``touch_angle_half_span`` frames
before and after (defined when both speeds >= ``touch_min_speed_mps``), the frame strength is
``s = max(a / touch_accel_mps2, theta / touch_angle_deg)`` (so ``s >= 1`` means a threshold is crossed).  A
touch is a local maximum of ``s`` (within ``touch_peak_halfwidth``) with ``s >= 1``, a player within
``touch_distance_m`` of the ball (within ``touch_gate_halfwidth`` frames) and ``not is_aerial``.
``touch_score = clip(s / 2, 0, 1)`` at touch frames (>= 0.5) and 0 elsewhere.

Possession evidence
-------------------
``raw = w * control_diff + (1 - w) * tanh(delta_d / evidence_dist_scale_m)`` with
``w = evidence_control_weight``, smoothed with a NaN-aware moving average over ``evidence_smooth_s``;
``possession_evidence`` in [-1, 1] (positive = home), 0 where there is no information
(``evidence_valid`` tells which frames carry evidence).  It is the MVP emission for H vs A.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d

from mtah2027.config import FPS, load_section, params_from_mapping
from mtah2027.features._utils import moving_mean, velocity_from_xy
from mtah2027.preprocessing.kinematics_and_orientation import savgol_kinematics

TIE_EPS_M = 1e-6


@dataclass(frozen=True)
class BallKinParams:
    k_nearest: int = 3
    control_sigma_m: float = 2.0
    control_tau_mps: float = 10.0
    touch_accel_mps2: float = 12.0
    touch_angle_deg: float = 35.0
    touch_angle_half_span: int = 2
    touch_min_speed_mps: float = 1.0
    touch_distance_m: float = 1.8
    touch_gate_halfwidth: int = 2
    touch_peak_halfwidth: int = 3
    evidence_smooth_s: float = 0.6
    evidence_dist_scale_m: float = 2.0
    evidence_control_weight: float = 0.5

    @classmethod
    def from_config(cls, path=None) -> "BallKinParams":
        return params_from_mapping(cls, load_section("ball_kinematics", path), "ball_kinematics")


@dataclass
class BallKinematics:
    """Outputs of :func:`compute_ball_kinematics` (see the module docstring)."""

    ball_visible: np.ndarray
    d_home: np.ndarray
    d_away: np.ndarray
    delta_d: np.ndarray
    nearest_slot_home: np.ndarray
    nearest_slot_away: np.ndarray
    rel_speed_home: np.ndarray
    rel_speed_away: np.ndarray
    closing_speed_home: np.ndarray
    closing_speed_away: np.ndarray
    nearest_team: np.ndarray
    control_home: np.ndarray
    control_away: np.ndarray
    control_diff: np.ndarray
    ball_acc_mag: np.ndarray
    turn_angle_deg: np.ndarray
    touch_score: np.ndarray
    touch_mask: np.ndarray
    possession_evidence: np.ndarray
    evidence_valid: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return dict(self.__dict__)

    def matrix(self, names: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
        """Stack the features into ``[N, F]`` float32; ``[N, k]`` arrays are expanded column-wise."""
        cols, keys = [], []
        for name in (names or list(self.__dict__)):
            arr = getattr(self, name)
            if arr.ndim == 1:
                cols.append(arr.astype(np.float32))
                keys.append(name)
            else:
                for j in range(arr.shape[1]):
                    cols.append(arr[:, j].astype(np.float32))
                    keys.append(f"{name}_{j + 1}")
        return np.stack(cols, axis=1), keys


def team_nearest(dist: np.ndarray, cols: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """k smallest distances of ``dist[N, S]`` among columns ``cols``: ``(values[N, k], slots[N, k])``.

    Distances are ``inf`` (slot ``-1``) where fewer than ``k`` finite distances exist.
    """
    n = dist.shape[0]
    values = np.full((n, k), np.inf, dtype=np.float32)
    slots = np.full((n, k), -1, dtype=np.int64)
    sub = dist[:, cols]
    kk = min(k, sub.shape[1])
    if kk == 0:
        return values, slots
    part = np.argpartition(sub, kk - 1, axis=1)[:, :kk] if sub.shape[1] > kk else np.tile(
        np.arange(kk), (n, 1))
    vals = np.take_along_axis(sub, part, axis=1)
    order = np.argsort(vals, axis=1, kind="stable")
    part = np.take_along_axis(part, order, axis=1)
    vals = np.take_along_axis(vals, order, axis=1)
    values[:, :kk] = vals
    slots[:, :kk] = np.where(np.isfinite(vals), cols[part], -1)
    return values, slots


def _shift(x: np.ndarray, by: int) -> np.ndarray:
    """``out[i] = x[i + by]`` with NaN padding."""
    out = np.full_like(x, np.nan)
    if by > 0:
        out[:-by] = x[by:]
    elif by < 0:
        out[-by:] = x[:by]
    else:
        out[:] = x
    return out


def compute_ball_kinematics(
        players_xy: np.ndarray, team_of_slot: np.ndarray, ball_xy: np.ndarray, *,
        players_vel: np.ndarray | None = None, ball_vel: np.ndarray | None = None,
        ball_acc: np.ndarray | None = None, is_aerial: np.ndarray | None = None, fps: float = FPS,
        params: BallKinParams | None = None) -> BallKinematics:
    """Distances, control weights, touches and possession evidence for ``players_xy[N, S, 2]`` (metres),
    ``team_of_slot[S]`` (0 home, 1 away) and ``ball_xy[N, 2]`` (NaN when missing).

    Velocities/accelerations (m/s, m/s^2) are taken from Issue #9 when supplied; the ball's are otherwise
    computed with the same Savitzky-Golay filter, the players' by central differences.  ``is_aerial[N]``
    (bool, default all False) suppresses touches.
    """
    p = params or BallKinParams.from_config()
    n, s, _ = players_xy.shape
    if ball_xy.shape != (n, 2):
        raise ValueError(f"ball_xy has shape {ball_xy.shape}, expected ({n}, 2)")
    team = np.asarray(team_of_slot)
    if team.shape != (s,):
        raise ValueError("team_of_slot must have one entry per slot")
    aerial = np.zeros(n, dtype=bool) if is_aerial is None else np.asarray(is_aerial, dtype=bool)
    if aerial.shape != (n,):
        raise ValueError("is_aerial must have one entry per frame")
    k = int(p.k_nearest)
    visible = np.isfinite(ball_xy).all(axis=1)

    if ball_vel is None or ball_acc is None:
        v_sg, a_sg = savgol_kinematics(ball_xy[:, None, :].astype(np.float32), fps)
        ball_vel = v_sg[:, 0] if ball_vel is None else ball_vel
        ball_acc = a_sg[:, 0] if ball_acc is None else ball_acc
    ball_vel = np.asarray(ball_vel, dtype=np.float32)
    ball_acc = np.asarray(ball_acc, dtype=np.float32)
    if players_vel is None:
        players_vel = velocity_from_xy(players_xy, fps, 1)
    players_vel = np.asarray(players_vel, dtype=np.float32)

    # distances ball -> every slot (inf where unknown so that partitioning ignores them)
    diff = players_xy.astype(np.float32) - ball_xy[:, None, :].astype(np.float32)
    dist = np.hypot(diff[..., 0], diff[..., 1])
    dist = np.where(np.isfinite(dist), dist, np.inf).astype(np.float32)
    rows = np.arange(n)[:, None]

    out_d, out_slots, out_rel, out_ctrl = {}, {}, {}, {}
    for code, name in ((0, "home"), (1, "away")):
        cols = np.flatnonzero(team == code)
        vals, slots = team_nearest(dist, cols, k)
        out_d[name], out_slots[name] = vals, slots
        vel = players_vel[rows, np.maximum(slots, 0)]                          # [N, k, 2]
        rel = np.linalg.norm(vel - ball_vel[:, None, :], axis=-1)              # NaN if unknown
        rel = np.where(slots >= 0, rel, np.nan)
        out_rel[name] = rel
        w = np.exp(-(np.where(np.isfinite(vals), vals, 0.0) ** 2) / (2.0 * p.control_sigma_m ** 2))
        w = w * np.exp(-np.nan_to_num(rel, nan=0.0) / p.control_tau_mps)
        w = np.where(np.isfinite(vals), w, 0.0)
        out_ctrl[name] = 1.0 - np.prod(1.0 - w, axis=1)

    d_home = np.where(np.isfinite(out_d["home"]), out_d["home"], np.nan).astype(np.float32)
    d_away = np.where(np.isfinite(out_d["away"]), out_d["away"], np.nan).astype(np.float32)
    dh1, da1 = d_home[:, 0], d_away[:, 0]
    both = np.isfinite(dh1) & np.isfinite(da1)
    delta_d = np.where(both, da1 - dh1, np.nan).astype(np.float32)
    with np.errstate(invalid="ignore"):
        nearest_team = np.where(both & (delta_d > TIE_EPS_M), 0,
                                np.where(both & (delta_d < -TIE_EPS_M), 1, -1)).astype(np.int8)

    def closing(d1: np.ndarray) -> np.ndarray:
        return ((_shift(d1, -1) - _shift(d1, 1)) * (fps / 2.0)).astype(np.float32)

    c_home, c_away = out_ctrl["home"], out_ctrl["away"]
    tot = c_home + c_away
    control_diff = np.where(tot > 1e-9, (c_home - c_away) / np.maximum(tot, 1e-9), 0.0).astype(np.float32)

    # touches: acceleration peak or direction change, with a player close by and not aerial
    a_mag = np.linalg.norm(ball_acc, axis=-1)
    h = int(p.touch_angle_half_span)
    v_before, v_after = _shift(ball_vel, -h), _shift(ball_vel, h)
    cross = v_before[:, 0] * v_after[:, 1] - v_before[:, 1] * v_after[:, 0]
    dot = v_before[:, 0] * v_after[:, 0] + v_before[:, 1] * v_after[:, 1]
    fast = (np.linalg.norm(v_before, axis=-1) >= p.touch_min_speed_mps) & (
        np.linalg.norm(v_after, axis=-1) >= p.touch_min_speed_mps)
    theta = np.where(fast, np.degrees(np.arctan2(np.abs(cross), dot)), np.nan).astype(np.float32)
    strength = np.maximum(np.nan_to_num(a_mag / p.touch_accel_mps2, nan=0.0),
                          np.nan_to_num(theta / p.touch_angle_deg, nan=0.0)).astype(np.float32)
    strength = np.where(visible, strength, 0.0).astype(np.float32)
    pk = int(p.touch_peak_halfwidth)
    local_max = strength >= maximum_filter1d(strength, 2 * pk + 1, mode="nearest")
    first_of_plateau = strength > np.concatenate([[-1.0], strength[:-1]])
    dmin = np.minimum(np.nan_to_num(dh1, nan=np.inf), np.nan_to_num(da1, nan=np.inf))
    near = minimum_filter1d(dmin, 2 * int(p.touch_gate_halfwidth) + 1, mode="nearest") <= p.touch_distance_m
    touch = (strength >= 1.0) & local_max & first_of_plateau & near & ~aerial & visible
    touch_score = np.where(touch, np.clip(strength / 2.0, 0.0, 1.0), 0.0).astype(np.float32)

    # possession evidence in [-1, 1], positive = home
    wc = p.evidence_control_weight
    valid = visible & both
    raw = np.where(valid, wc * control_diff + (1.0 - wc) * np.tanh(np.nan_to_num(delta_d)
                                                                   / p.evidence_dist_scale_m), np.nan)
    smooth = moving_mean(raw, max(1, int(round(p.evidence_smooth_s * fps))))
    evidence = np.clip(np.where(valid, np.nan_to_num(smooth, nan=0.0), 0.0), -1.0, 1.0).astype(np.float32)

    return BallKinematics(
        ball_visible=visible.astype(np.float32), d_home=d_home, d_away=d_away, delta_d=delta_d,
        nearest_slot_home=out_slots["home"][:, 0].astype(np.int32),
        nearest_slot_away=out_slots["away"][:, 0].astype(np.int32),
        rel_speed_home=out_rel["home"][:, 0].astype(np.float32),
        rel_speed_away=out_rel["away"][:, 0].astype(np.float32),
        closing_speed_home=closing(dh1), closing_speed_away=closing(da1), nearest_team=nearest_team,
        control_home=c_home.astype(np.float32), control_away=c_away.astype(np.float32),
        control_diff=control_diff, ball_acc_mag=a_mag.astype(np.float32), turn_angle_deg=theta,
        touch_score=touch_score, touch_mask=touch, possession_evidence=evidence,
        evidence_valid=valid)
