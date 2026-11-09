"""Time-dependent log transition matrix ``log_trans[N, 3, 3]`` (Issue #23).

States are ``H = 0``, ``A = 1``, ``D = 2``.  ``log_trans[t, i, j]`` is the log-score of moving from
state ``i`` at frame ``t - 1`` to state ``j`` at frame ``t`` (row ``0`` is never used).  Staying costs
``stay`` (0).  Everything else encodes *when a change of state is plausible*; the emissions then say
which state the evidence prefers, so the decoder produces few, well-placed possession changes.

H <-> A (turnovers)
    Closed (``ha_closed``, -11.5) at rest.  Open (``ha_open``, -1.4) within ``touch_halfwidth`` frames of
    a touch peak (``touch_score >= touch_min_score``) of the visible, non-aerial ball when the *gaining*
    team has a player within ``touch_both_dist_m`` of it (within ``touch_gate_halfwidth`` frames of the peak):
    a team can only win the ball through a touch of one of its own players, so H -> A opens when an away
    player is at the ball and A -> H when a home player is.  With ``require_both_teams`` the stricter
    reading "a contested touch, both teams within the distance" is used for both directions.  Moderately
    open (``ha_occluded``, -6) during and just around a missing-ball run of at least ``occlusion_min_s``
    seconds, so that possession can flip across an occlusion only with strong evidence.
H, A -> D (ball becomes dead)
    ``dead_closed`` .. ``dead_open`` according to the smoothed dead-ball score (linear between the two
    ``dead_ramp`` values); fully open while the ball is out of bounds.
D -> H, A (ball becomes alive again)
    Closed (``d_exit_closed``) while the dead evidence persists, ramping to ``d_exit_open`` as the smoothed
    ``1 - dead_score`` rises through ``alive_ramp``; additionally opened to ``restart_open`` within
    ``restart_halfwidth`` frames of each restart frame ``t*`` (Issue #14).  If the restart team is known
    the other team receives the extra ``restart_other_team`` penalty there.

All values come from ``configs/default.yaml`` (section ``transition_matrix``), are float64 and rounded
to 1e-6 so that decoding is bit-reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d

from mtah2027.config import FPS, load_section, params_from_mapping
from mtah2027.features._utils import mask_runs, moving_mean

H, A, D = 0, 1, 2
ROUND_DECIMALS = 6


@dataclass(frozen=True)
class TransitionParams:
    stay: float = 0.0
    ha_closed: float = -11.5
    ha_open: float = -1.4
    touch_halfwidth: int = 5
    touch_min_score: float = 0.5
    touch_both_dist_m: float = 2.0
    require_both_teams: bool = False
    touch_gate_halfwidth: int = 2
    ha_occluded: float = -6.0
    occlusion_min_s: float = 2.0
    occlusion_margin_frames: int = 5
    dead_closed: float = -8.0
    dead_open: float = -1.5
    dead_ramp: tuple = (0.3, 0.7)
    dead_smooth_s: float = 0.4
    d_exit_closed: float = -9.0
    d_exit_open: float = -2.0
    alive_ramp: tuple = (0.5, 0.9)
    restart_open: float = -1.0
    restart_halfwidth: int = 3
    restart_other_team: float = -3.0

    @classmethod
    def from_config(cls, path=None) -> "TransitionParams":
        return params_from_mapping(cls, load_section("transition_matrix", path), "transition_matrix")


def _ramp(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((x - lo) / max(hi - lo, 1e-9), 0.0, 1.0)


def occlusion_mask(ball_visible: np.ndarray, min_frames: int, margin: int) -> np.ndarray:
    """True inside (and ``margin`` frames around) every missing-ball run of at least ``min_frames``."""
    missing = ~np.asarray(ball_visible, dtype=bool)
    out = np.zeros(len(missing), dtype=bool)
    starts, ends = mask_runs(missing)
    for a, b in zip(starts, ends):
        if b - a >= min_frames:
            out[max(0, a - margin):min(len(missing), b + margin)] = True
    return out


def touch_window_masks(touch_score: np.ndarray, d_home: np.ndarray, d_away: np.ndarray,
                       is_aerial: np.ndarray | None, p: TransitionParams) -> tuple[np.ndarray, np.ndarray]:
    """``(to_home, to_away)``: True within ``touch_halfwidth`` of a touch where that team can win the ball."""
    n = len(touch_score)
    peaks = np.nan_to_num(np.asarray(touch_score, dtype=np.float64), nan=0.0) >= p.touch_min_score
    if is_aerial is not None:
        peaks &= ~np.asarray(is_aerial, dtype=bool)
    gate = 2 * int(p.touch_gate_halfwidth) + 1
    dh = minimum_filter1d(np.where(np.isfinite(d_home), d_home, np.inf), gate, mode="nearest")
    da = minimum_filter1d(np.where(np.isfinite(d_away), d_away, np.inf), gate, mode="nearest")
    near_h, near_a = dh <= p.touch_both_dist_m, da <= p.touch_both_dist_m
    if p.require_both_teams:
        near_h = near_a = near_h & near_a
    width = 2 * int(p.touch_halfwidth) + 1

    def spread(mask: np.ndarray) -> np.ndarray:
        if not mask.any():
            return np.zeros(n, dtype=bool)
        return maximum_filter1d(mask.astype(np.uint8), width, mode="constant") > 0

    return spread(peaks & near_h), spread(peaks & near_a)


def touch_window_mask(touch_score: np.ndarray, d_home: np.ndarray, d_away: np.ndarray,
                      is_aerial: np.ndarray | None, p: TransitionParams) -> np.ndarray:
    """True where a possession change is possible in either direction (union of both masks)."""
    to_home, to_away = touch_window_masks(touch_score, d_home, d_away, is_aerial, p)
    return to_home | to_away


def build_log_transitions(
        n: int, *, touch_score: np.ndarray | None = None, d_home: np.ndarray | None = None,
        d_away: np.ndarray | None = None, ball_visible: np.ndarray | None = None,
        is_aerial: np.ndarray | None = None, dead_score: np.ndarray | None = None,
        ball_out: np.ndarray | None = None, restart_frames: np.ndarray | None = None,
        restart_teams: np.ndarray | None = None, fps: float = FPS,
        params: TransitionParams | None = None) -> np.ndarray:
    """Build ``log_trans[n, 3, 3]`` (float64, rounded to 1e-6) from the per-frame evidence.

    ``d_home`` / ``d_away`` are the distances (m) of the closest player of each team to the ball (NaN
    when unknown).  Every input is optional: a missing one just removes the corresponding opening, so
    the matrix degrades towards "closed" transitions.  ``restart_frames`` are the frames ``t*``;
    ``restart_teams`` (same length, 0 / 1 / -1 unknown) the team that restarts.
    """
    p = params or TransitionParams.from_config()
    nan = np.full(n, np.nan)
    touch_score = nan if touch_score is None else np.asarray(touch_score, dtype=np.float64)
    d_home = nan if d_home is None else np.asarray(d_home, dtype=np.float64)
    d_away = nan if d_away is None else np.asarray(d_away, dtype=np.float64)
    visible = np.ones(n, bool) if ball_visible is None else np.asarray(ball_visible, dtype=bool)
    for name, arr in (("touch_score", touch_score), ("d_home", d_home), ("d_away", d_away),
                      ("ball_visible", visible)):
        if arr.shape != (n,):
            raise ValueError(f"{name} must have shape ({n},), got {arr.shape}")

    log_trans = np.full((n, 3, 3), -np.inf, dtype=np.float64)
    for s in (H, A, D):
        log_trans[:, s, s] = p.stay

    # --- H <-> A ---------------------------------------------------------------------------
    ha = np.full(n, p.ha_closed, dtype=np.float64)
    occ = occlusion_mask(visible, max(1, int(round(p.occlusion_min_s * fps))), int(p.occlusion_margin_frames))
    ha = np.where(occ, np.maximum(ha, p.ha_occluded), ha)
    to_home, to_away = touch_window_masks(touch_score, d_home, d_away, is_aerial, p)
    log_trans[:, H, A] = np.where(to_away, np.maximum(ha, p.ha_open), ha)      # away team wins the ball
    log_trans[:, A, H] = np.where(to_home, np.maximum(ha, p.ha_open), ha)      # home team wins the ball

    # --- H, A -> D and D -> H, A -------------------------------------------------------------
    ds = np.zeros(n)
    if dead_score is not None:
        ds = np.nan_to_num(np.asarray(dead_score, dtype=np.float64), nan=0.0)
    win = max(1, int(round(p.dead_smooth_s * fps)))
    ds_sm = np.nan_to_num(moving_mean(ds, win).astype(np.float64), nan=0.0)
    open_in = _ramp(ds_sm, *p.dead_ramp)
    if ball_out is not None:
        open_in = np.maximum(open_in, (np.nan_to_num(np.asarray(ball_out, dtype=np.float64)) > 0.5)
                             .astype(np.float64))
    to_dead = p.dead_closed + (p.dead_open - p.dead_closed) * open_in
    log_trans[:, H, D] = to_dead
    log_trans[:, A, D] = to_dead

    open_out = _ramp(1.0 - ds_sm, *p.alive_ramp)
    leave = p.d_exit_closed + (p.d_exit_open - p.d_exit_closed) * open_out
    leave_h, leave_a = leave.copy(), leave.copy()
    if restart_frames is not None and len(restart_frames):
        frames = np.asarray(restart_frames, dtype=np.int64)
        teams = (np.full(len(frames), -1, dtype=np.int64) if restart_teams is None
                 else np.asarray(restart_teams, dtype=np.int64))
        if teams.shape != frames.shape:
            raise ValueError("restart_teams must have the same length as restart_frames")
        hw = int(p.restart_halfwidth)
        for code, target in ((H, leave_h), (A, leave_a)):
            # unknown team opens both columns; a known team opens its own fully, the other one less
            own = np.zeros(n, dtype=bool)
            other = np.zeros(n, dtype=bool)
            for f, tm in zip(frames, teams):
                lo, hi = max(0, int(f) - hw), min(n, int(f) + hw + 1)
                if tm < 0 or tm == code:
                    own[lo:hi] = True
                else:
                    other[lo:hi] = True
            target[own] = np.maximum(target[own], p.restart_open)
            soft = other & ~own
            target[soft] = np.maximum(target[soft], p.restart_open + p.restart_other_team)
    log_trans[:, D, H] = leave_h
    log_trans[:, D, A] = leave_a

    log_trans[0] = np.where(np.eye(3) > 0, p.stay, -np.inf)
    return np.round(log_trans, ROUND_DECIMALS)
