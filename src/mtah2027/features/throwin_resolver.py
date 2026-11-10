"""Deterministic resolution of the restarting team of ``Throw-in`` set pieces with ``Team = N/A`` (Issue #15).

Smart Tagging names no team for throw-ins, yet the possession that follows belongs to the thrower's team and
the decoder opens ``D -> H`` or ``D -> A`` at the restart frame ``t*`` (Issue #14, Issue #23).  The team is
recovered from the tracking alone, by two independent cues that vote:

1. **Taker cue.**  In the window ``[max(tag_start, t* - taker_window_s), t*]`` a *candidate* is a player slot
   that stays within ``touchline_band_m`` of a touchline (``y ~ 0`` or ``y ~ pitch width``; the arrays are
   metres, orientation-normalised -- both touchlines are symmetric under the reflection) in at least
   ``min_band_frac`` of its valid frames and, when the ball is visible and ``taker_ball_radius_m > 0``, is
   within that distance of the ball on median (the taker holds the ball; the radius is dropped if no candidate
   passes it).  The taker is the candidate *most isolated from opponents*: the median over its frames in the
   band of the distance to the nearest opponent (capped at ``isolation_cap_m``).  The vote strength is the
   isolation gap to the best candidate of the other team (0 if none), saturating at ``isolation_sat_m``.
2. **Controller cue.**  In ``[t*, t* + confirm_window_frames]`` the first frame at which a player of one
   team is within ``control_radius_m`` of the (visible) ball and the nearest player of the other team is at
   least ``control_margin_m`` farther: the first *clear* controller.  Strength = gap/``control_gap_sat_m``.

``score_team = w_taker * strength_taker + w_controller * strength_controller`` (each only for the team its cue
names).  The team with the larger score wins; the confidence is ``|score_H - score_A|`` over the total
weight of the cues that exist; below ``min_confidence`` (or on a tie) the throw-in stays unresolved and
nothing is invented.  ``method`` is ``'taker+controller'`` (both agree), ``'taker>controller'`` /
``'controller>taker'``
(they disagree, the heavier one wins), ``'taker'``, ``'controller'`` or ``'unresolved'``.

Wiring: :func:`apply_resolutions` returns an :class:`AnchorResult` whose resolved throw-in anchors carry the
recovered team (so the existing ``restart_teams`` of the transition matrix biases ``D -> team`` at ``t*``) and
:func:`restart_emission_prior` adds a small log-bias towards the restart team in the emissions for
``emission_window_s`` after ``t*``.  Everything is configured in ``configs/default.yaml`` (section
``throwin_resolver``); ``enabled: false`` or a failing resolver leaves the pipeline exactly as before.
"""
from __future__ import annotations

import dataclasses
import warnings
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from mtah2027.config import FPS, PITCH, load_section, params_from_mapping
from mtah2027.features.setpiece_anchor import AnchorResult

HOME, AWAY, NO_TEAM = 0, 1, -1
EMISSION_DECIMALS = 6


@dataclass(frozen=True)
class ThrowInParams:
    enabled: bool = True
    touchline_band_m: float = 1.5
    taker_window_s: float = 3.0
    min_band_frac: float = 0.5
    isolation_cap_m: float = 15.0
    isolation_sat_m: float = 6.0
    taker_ball_radius_m: float = 3.0
    confirm_window_frames: int = 35
    control_radius_m: float = 2.0
    control_margin_m: float = 1.0
    control_gap_sat_m: float = 4.0
    w_taker: float = 1.0
    w_controller: float = 1.0
    min_confidence: float = 0.3
    resolve_known_teams: bool = False
    emission_bias: float = 0.0
    emission_window_s: float = 2.0

    @classmethod
    def from_config(cls, path=None) -> "ThrowInParams":
        return params_from_mapping(cls, load_section("throwin_resolver", path), "throwin_resolver")


@dataclass(frozen=True)
class ThrowInResolution:
    """Outcome for one throw-in anchor (``team == -1`` when unresolved)."""

    anchor_index: int
    t_star: int
    team: int
    confidence: float
    method: str
    taker_team: int = NO_TEAM
    taker_slot: int = -1
    controller_team: int = NO_TEAM

    @property
    def resolved(self) -> bool:
        return self.team != NO_TEAM


def _is_throw_in(kind: str) -> bool:
    return "throw" in kind.lower()


def _taker_cue(players: np.ndarray, team: np.ndarray, ball: np.ndarray, p: ThrowInParams, lo: int, hi: int,
               width: float) -> tuple[int, int, float]:
    """``(team, slot, strength)`` of the taker in frames ``lo..hi`` or ``(-1, -1, 0)``."""
    seg = np.asarray(players[lo:hi + 1], dtype=np.float64)
    n_f, n_s = seg.shape[:2]
    if n_f == 0:
        return NO_TEAM, -1, 0.0
    finite = np.isfinite(seg).all(axis=2)
    y = seg[..., 1]
    line = np.minimum(np.abs(y), np.abs(width - y))
    in_band = finite & (line <= p.touchline_band_m)
    frac = in_band.sum(axis=0) / np.maximum(finite.sum(axis=0), 1)
    cand = (frac >= p.min_band_frac) & (in_band.sum(axis=0) > 0)
    if not cand.any():
        return NO_TEAM, -1, 0.0

    # distance to the nearest opponent, per frame and slot (capped)
    dist = np.linalg.norm(seg[:, :, None, :] - seg[:, None, :, :], axis=3)           # [F, S, S]
    opp = (team[:, None] != team[None, :])[None]
    dist = np.where(opp & finite[:, None, :] & finite[:, :, None], dist, np.inf)
    iso = np.minimum(dist.min(axis=2), p.isolation_cap_m)                              # [F, S]
    iso = np.where(np.isfinite(iso), iso, p.isolation_cap_m)

    if p.taker_ball_radius_m > 0:
        b = np.asarray(ball[lo:hi + 1], dtype=np.float64)
        bvis = np.isfinite(b).all(axis=1)
        if bvis.any():
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                d_ball = np.linalg.norm(seg - b[:, None, :], axis=2)
                d_ball = np.where(bvis[:, None] & finite, d_ball, np.nan)
                med = np.nanmedian(d_ball, axis=0)
            near = cand & np.isfinite(med) & (med <= p.taker_ball_radius_m)
            if near.any():
                cand = near

    scores = np.full(n_s, -np.inf)
    for s in np.flatnonzero(cand):
        scores[s] = float(np.median(iso[in_band[:, s], s]))
    best = int(np.argmax(scores))
    best_team = int(team[best])
    other = scores[team != best_team]
    other_best = float(other.max()) if len(other) and np.isfinite(other.max()) else 0.0
    gap = scores[best] - other_best
    return best_team, best, float(np.clip(gap / p.isolation_sat_m, 0.0, 1.0))


def _controller_cue(players: np.ndarray, team: np.ndarray, ball: np.ndarray, p: ThrowInParams, t_star: int
                    ) -> tuple[int, float]:
    """``(team, strength)`` of the first clear controller in ``[t*, t* + confirm_window_frames]``."""
    lo, hi = t_star, min(len(ball) - 1, t_star + int(p.confirm_window_frames))
    if hi < lo:
        return NO_TEAM, 0.0
    b = np.asarray(ball[lo:hi + 1], dtype=np.float64)
    seg = np.asarray(players[lo:hi + 1], dtype=np.float64)
    d = np.linalg.norm(seg - b[:, None, :], axis=2)
    d = np.where(np.isfinite(d), d, np.inf)
    d_h = d[:, team == HOME].min(axis=1) if (team == HOME).any() else np.full(len(b), np.inf)
    d_a = d[:, team == AWAY].min(axis=1) if (team == AWAY).any() else np.full(len(b), np.inf)
    with np.errstate(invalid="ignore"):
        gap = np.where(np.isfinite(d_h) | np.isfinite(d_a), np.abs(d_h - d_a), 0.0)
    gap = np.where(np.isfinite(gap), gap, p.control_gap_sat_m)       # one team has nobody tracked
    near = np.minimum(d_h, d_a)
    clear = np.isfinite(b).all(axis=1) & (near <= p.control_radius_m) & (gap >= p.control_margin_m)
    if not clear.any():
        return NO_TEAM, 0.0
    k = int(np.flatnonzero(clear)[0])
    return (HOME if d_h[k] < d_a[k] else AWAY), float(np.clip(gap[k] / p.control_gap_sat_m, 0.0, 1.0))


def resolve_throw_in(players_xy: np.ndarray, team_of_slot: np.ndarray, ball_xy: np.ndarray, *,
                     tag_start: int, t_star: int, pitch: tuple[float, float] = PITCH, fps: float = FPS,
                     params: ThrowInParams | None = None, anchor_index: int = -1) -> ThrowInResolution:
    """Restart team of one throw-in (see the module docstring); arrays are metres, home attacking +x."""
    p = params or ThrowInParams.from_config()
    team = np.asarray(team_of_slot)
    n = len(ball_xy)
    if t_star < 0 or t_star >= n:
        return ThrowInResolution(anchor_index, int(t_star), NO_TEAM, 0.0, "unresolved")
    lo = max(0, int(tag_start), t_star - int(round(p.taker_window_s * fps)))
    t_team, t_slot, t_str = _taker_cue(players_xy, team, ball_xy, p, min(lo, t_star), t_star, float(pitch[1]))
    c_team, c_str = _controller_cue(players_xy, team, ball_xy, p, t_star)

    score = np.zeros(2)
    total = 0.0
    if t_team != NO_TEAM:
        score[t_team] += p.w_taker * t_str
        total += p.w_taker
    if c_team != NO_TEAM:
        score[c_team] += p.w_controller * c_str
        total += p.w_controller
    if total <= 0 or score[HOME] == score[AWAY]:
        return ThrowInResolution(anchor_index, int(t_star), NO_TEAM, 0.0, "unresolved", t_team, t_slot,
                                 c_team)
    winner = HOME if score[HOME] > score[AWAY] else AWAY
    conf = float(abs(score[HOME] - score[AWAY]) / total)
    if t_team != NO_TEAM and c_team != NO_TEAM:
        method = "taker+controller" if t_team == c_team else (
            "taker>controller" if winner == t_team else "controller>taker")
    else:
        method = "taker" if t_team != NO_TEAM else "controller"
    if conf < p.min_confidence:
        return ThrowInResolution(anchor_index, int(t_star), NO_TEAM, conf, "unresolved", t_team, t_slot,
                                 c_team)
    return ThrowInResolution(anchor_index, int(t_star), winner, conf, method, t_team, t_slot, c_team)


def resolve_throw_ins(anchors: AnchorResult, players_xy: np.ndarray, team_of_slot: np.ndarray,
                      ball_xy: np.ndarray, *, pitch: tuple[float, float] = PITCH, fps: float = FPS,
                      params: ThrowInParams | None = None) -> list[ThrowInResolution]:
    """Resolutions for throw-in anchors with a located ``t*`` and no team (all if ``resolve_known_teams``)."""
    p = params or ThrowInParams.from_config()
    out: list[ThrowInResolution] = []
    for i, a in enumerate(anchors.anchors):
        if not _is_throw_in(a.type) or not a.resolved:
            continue
        if a.team != NO_TEAM and not p.resolve_known_teams:
            continue
        out.append(resolve_throw_in(players_xy, team_of_slot, ball_xy, tag_start=a.tag_start, t_star=a.t_star,
                                    pitch=pitch, fps=fps, params=p, anchor_index=i))
    return out


def apply_resolutions(anchors: AnchorResult, resolutions: Sequence[ThrowInResolution]) -> AnchorResult:
    """Copy of ``anchors`` whose throw-in anchors carry the recovered team (unresolved ones are untouched)."""
    new = list(anchors.anchors)
    for r in resolutions:
        if r.resolved:
            new[r.anchor_index] = dataclasses.replace(new[r.anchor_index], team=int(r.team))
    return dataclasses.replace(anchors, anchors=new)


def restart_emission_prior(log_probs: np.ndarray, resolutions: Sequence[ThrowInResolution], *,
                           fps: float = FPS, params: ThrowInParams | None = None) -> np.ndarray:
    """``log_probs[N, 3]`` with ``+bias/2`` on the restart team and ``-bias/2`` on the other for the frames
    ``[t*, t* + emission_window_s)`` of every resolved throw-in (rounded to 1e-6, input untouched)."""
    p = params or ThrowInParams.from_config()
    out = np.array(log_probs, dtype=np.float64, copy=True)
    if p.emission_bias == 0.0:
        return out
    n = len(out)
    width = max(1, int(round(p.emission_window_s * fps)))
    for r in resolutions:
        if not r.resolved:
            continue
        lo, hi = max(0, r.t_star), min(n, r.t_star + width)
        out[lo:hi, r.team] += 0.5 * p.emission_bias
        out[lo:hi, 1 - r.team] -= 0.5 * p.emission_bias
    return np.round(out, EMISSION_DECIMALS)


def accuracy_against_truth(resolutions: Sequence[ThrowInResolution], truth: dict[int, int]) -> dict:
    """Aggregate accuracy of ``resolutions`` against ``{anchor_index: true_team}`` (tests / diagnostics)."""
    rows = [r for r in resolutions if r.anchor_index in truth]
    done = [r for r in rows if r.resolved]
    correct = sum(1 for r in done if r.team == truth[r.anchor_index])
    return {"n": len(rows), "resolved": len(done), "correct": int(correct),
            "accuracy_resolved": float(correct / len(done)) if done else float("nan"),
            "accuracy_overall": float(correct / len(rows)) if rows else float("nan")}


__all__ = ["ThrowInParams", "ThrowInResolution", "resolve_throw_in", "resolve_throw_ins",
           "apply_resolutions", "restart_emission_prior", "accuracy_against_truth"]
