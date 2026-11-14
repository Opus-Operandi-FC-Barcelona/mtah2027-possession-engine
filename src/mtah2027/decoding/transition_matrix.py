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
    restart_team_before_s: float = 0.0   # Issue #49: the other-team penalty also holds from t* - this ...
    restart_team_after_s: float = 0.0    # ... to t* + this (0 = only t* +- restart_halfwidth, as before)
    # Specialist-driven openings inside occlusions (learned route only; off by default).  H<->A opens to
    # ``spec_ha_open`` around the frames where the specialists' H-vs-A logit changes sign (hysteresis
    # ``spec_hyst``), only towards the team the new sign names; H/A->D is opened to ``spec_dead_open`` where
    # P(D) is a local peak >= ``spec_dead_min``; collective-speed accelerations above ``spec_accel_min`` m/s
    # (change of the mean player speed over +-``spec_accel_half`` frames) open both to ``spec_accel_open``.
    # Issue #43: H<->A opened around the start of a Smart Tagging Attacking / Defensive Transition (towards
    # the team that gains the ball), only inside occlusions when ``phase_cue_hidden_only``.  Off by default.
    phase_cue: bool = False
    phase_cue_open: float = -4.0
    phase_cue_before_s: float = 4.0
    phase_cue_after_s: float = 8.0
    phase_cue_hidden_only: bool = True
    spec_open: bool = False
    spec_ha_open: float = -4.0
    spec_hyst: float = 0.5
    spec_halfwidth: int = 12
    spec_dead_open: float = -4.0
    spec_dead_min: float = 0.5
    spec_accel_open: float = -5.0
    spec_accel_min: float = 1.0
    spec_accel_half: int = 12

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


def logit_flips(z: np.ndarray, hyst: float) -> tuple[np.ndarray, np.ndarray]:
    """Frames where the sign of ``z`` (H-vs-A logit) switches with hysteresis: ``(to_home, to_away)``.

    The state becomes +1 once ``z > hyst`` and -1 once ``z < -hyst`` (NaN and the dead band keep the last
    state); a flip is a frame where the state differs from the previous non-zero one."""
    z = np.nan_to_num(np.asarray(z, dtype=np.float64), nan=0.0)
    st = np.where(z > hyst, 1, np.where(z < -hyst, -1, 0))
    idx = np.arange(len(st))
    last = np.maximum.accumulate(np.where(st != 0, idx, -1))
    ff = np.where(last >= 0, st[np.maximum(last, 0)], 0)
    prev = np.concatenate([[0], ff[:-1]])
    prev_nz = np.maximum.accumulate(np.where(prev != 0, idx, -1))
    prev_state = np.where(prev_nz >= 0, prev[np.maximum(prev_nz, 0)], 0)
    flip = (st != 0) & (prev_state != 0) & (st != prev_state)
    return flip & (st > 0), flip & (st < 0)


def specialist_openings(n: int, occ: np.ndarray, p: TransitionParams, *, home_logit, p_dead,
                        speed) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(to_home, to_away, to_dead, accel)`` boolean masks, True only inside ``occ`` (see the params)."""
    hw = 2 * int(p.spec_halfwidth) + 1

    def spread(mask):
        if not mask.any():
            return np.zeros(n, bool)
        return maximum_filter1d(mask.astype(np.uint8), hw, mode="constant") > 0

    zero = np.zeros(n, bool)
    to_home = to_away = to_dead = accel = zero
    if home_logit is not None:
        fh, fa = logit_flips(home_logit, p.spec_hyst)
        to_home, to_away = spread(fh) & occ, spread(fa) & occ
    if p_dead is not None:
        pd = np.nan_to_num(np.asarray(p_dead, dtype=np.float64), nan=0.0)
        peak = (pd >= p.spec_dead_min) & (pd >= maximum_filter1d(pd, hw, mode="nearest") - 1e-12)
        to_dead = spread(peak) & occ
    if speed is not None:
        sp = np.nan_to_num(np.asarray(speed, dtype=np.float64), nan=0.0)
        h = max(1, int(p.spec_accel_half))
        fwd = np.concatenate([sp[h:], np.repeat(sp[-1:], h)])
        bwd = np.concatenate([np.repeat(sp[:1], h), sp[:-h]])
        accel = (np.abs(fwd - bwd) >= p.spec_accel_min) & occ
    return to_home, to_away, to_dead, accel


def phase_cue_masks(n: int, transitions: list, fps: float,
                    p: TransitionParams) -> tuple[np.ndarray, np.ndarray]:
    """``(to_home, to_away)``: True within ``[start - before, start + after]`` of a Transition won by it."""
    pre, post = int(round(p.phase_cue_before_s * fps)), int(round(p.phase_cue_after_s * fps))
    to_home, to_away = np.zeros(n, bool), np.zeros(n, bool)
    for start, side in transitions:
        target = to_home if int(side) == H else to_away
        target[max(0, int(start) - pre):min(n, int(start) + post + 1)] = True
    return to_home, to_away


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
        restart_teams: np.ndarray | None = None, restart_weights: np.ndarray | None = None,
        fps: float = FPS,
        params: TransitionParams | None = None, spec_home_logit: np.ndarray | None = None,
        spec_p_dead: np.ndarray | None = None, spec_speed: np.ndarray | None = None,
        transitions: list | None = None) -> np.ndarray:
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
    ha_ha = np.where(to_away, np.maximum(ha, p.ha_open), ha)
    ha_ah = np.where(to_home, np.maximum(ha, p.ha_open), ha)
    s_dead = s_acc = np.zeros(n, bool)
    if p.spec_open and (spec_home_logit is not None or spec_p_dead is not None or spec_speed is not None):
        s_home, s_away, s_dead, s_acc = specialist_openings(
            n, occ, p, home_logit=spec_home_logit, p_dead=spec_p_dead, speed=spec_speed)
        ha_ha = np.where(s_away, np.maximum(ha_ha, p.spec_ha_open), ha_ha)
        ha_ah = np.where(s_home, np.maximum(ha_ah, p.spec_ha_open), ha_ah)
        ha_ha = np.where(s_acc, np.maximum(ha_ha, p.spec_accel_open), ha_ha)
        ha_ah = np.where(s_acc, np.maximum(ha_ah, p.spec_accel_open), ha_ah)
    if p.phase_cue and transitions:
        cue_h, cue_a = phase_cue_masks(n, transitions, fps, p)
        if p.phase_cue_hidden_only:
            cue_h, cue_a = cue_h & occ, cue_a & occ
        ha_ha = np.where(cue_a, np.maximum(ha_ha, p.phase_cue_open), ha_ha)
        ha_ah = np.where(cue_h, np.maximum(ha_ah, p.phase_cue_open), ha_ah)
    log_trans[:, H, A] = ha_ha      # away team wins the ball
    log_trans[:, A, H] = ha_ah      # home team wins the ball

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
    to_dead = np.where(s_dead, np.maximum(to_dead, p.spec_dead_open), to_dead)
    to_dead = np.where(s_acc, np.maximum(to_dead, p.spec_accel_open), to_dead)
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
        wts = (np.ones(len(frames)) if restart_weights is None
               else np.clip(np.asarray(restart_weights, dtype=np.float64), 0.0, 1.0))
        if wts.shape != frames.shape:
            raise ValueError("restart_weights must have the same length as restart_frames")
        hw = int(p.restart_halfwidth)
        wide_b = int(round(p.restart_team_before_s * fps))
        wide_a = int(round(p.restart_team_after_s * fps))
        for code, target in ((H, leave_h), (A, leave_a)):
            # unknown team opens both columns; a known team opens its own fully, the other one less
            own = np.zeros(n, dtype=bool)
            other_w = np.full(n, -np.inf)          # strongest team weight of the 'other' windows per frame
            wide_w = np.full(n, -np.inf)
            for f, tm, wt in zip(frames, teams, wts):
                lo, hi = max(0, int(f) - hw), min(n, int(f) + hw + 1)
                if tm < 0 or tm == code:
                    own[lo:hi] = True
                else:
                    other_w[lo:hi] = np.maximum(other_w[lo:hi], wt)
                    if wide_b > 0 or wide_a > 0:
                        wlo, whi = max(0, int(f) - max(wide_b, hw)), min(n, int(f) + max(wide_a, hw) + 1)
                        wide_w[wlo:whi] = np.maximum(wide_w[wlo:whi], wt)
            target[own] = np.maximum(target[own], p.restart_open)
            soft = np.isfinite(other_w) & ~own
            target[soft] = np.maximum(target[soft], p.restart_open + p.restart_other_team * other_w[soft])
            wide = np.isfinite(wide_w) & ~own & ~np.isfinite(other_w)    # outside the core windows
            target[wide] = target[wide] + p.restart_other_team * wide_w[wide]
    log_trans[:, D, H] = leave_h
    log_trans[:, D, A] = leave_a

    log_trans[0] = np.where(np.eye(3) > 0, p.stay, -np.inf)
    return np.round(log_trans, ROUND_DECIMALS)
