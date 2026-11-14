"""Wave-1 heuristic emissions ``log_probs[N, 3]`` for (H, A, D) (Issue #28).

The learned specialists of wave 2 (#20-#22) are replaced here by a transparent, calibrated-ish model built
from the continuous features of Issues #13 and #16:

``p_D``  dead-ball probability: ``sigmoid(s * logit(dead_score) + w_status * (2*status_dead - 1))``
         where ``s`` is ``dead_scale`` with a visible ball and ``dead_scale_no_ball`` without one (then the
         score rests on collective player speeds).  ``status_dead`` is the ATD alive/dead channel, if any.
``z``    H-vs-A logit:

         * **last toucher** (when touch events exist): possession belongs to the team of the player that last
           touched the ball (``touch_signed``: sign of the distance gap at every touch peak, weight
           ``clip(|gap| / touch_margin_m, 0, 1)``), ``w_touch * s * exp(-dt / touch_tau_s)``.  On external
           matches the team of the last toucher agrees with the labelled possession on ~92% of the alive
           frames against ~78% for the instantaneously nearest player, and 99% of the labelled turnovers
           happen at a touch;
         * ball with evidence: a weak ``w_evidence * possession_evidence`` (Issue #16, in [-1, 1], + = home);
         * ball missing or without evidence: the **carried possession** -- the last toucher (or, without any
           touch event, the last confident team, ``|evidence| >= carry_conf_threshold`` decaying as
           ``exp(-dt / carry_tau_s)``), plus (``carry_backward``) the *next* one carried backwards over the
           gap.  Inside an occlusion this keeps possession (what spellF1 rewards) and, when the team before
           and after the gap differ, puts the change near the middle of the gap, where the unknown flip is
           least costly;
         * ``w_phase * phase_prior_home`` -- the weak Smart Tagging prior (Issue #6);
         * ``w_owner * (1 - 2*owner)`` -- a strong term when the ATD ball group declares an owning team.

``p_H = (1 - p_D) sigmoid(z)``, ``p_A = (1 - p_D) sigmoid(-z)``; probabilities are floored at ``prob_floor``
and ``log_probs = round(log p, 6)``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from mtah2027.config import FPS, load_section, params_from_mapping
from mtah2027.features._utils import sigmoid

ROUND_DECIMALS = 6
_TEAM_WORDS = {"home": 0, "local": 0, "away": 1, "visitor": 1, "visitante": 1}


@dataclass(frozen=True)
class EmissionParams:
    w_evidence: float = 0.5
    w_phase: float = 0.3
    w_phase_alive: float = 0.0       # Issue #43: D logit lowered where the ball is hidden in a live phase
    w_owner: float = 4.0
    owner_channels: tuple = ("owner", "owning_team", "ball_owner", "possession", "possessor", "team",
                             "teamid", "team_id", "ballowner")
    dead_scale: float = 1.0
    dead_scale_no_ball: float = 0.6
    w_status: float = 3.0
    prob_floor: float = 0.001
    carry_conf_threshold: float = 0.35
    carry_weight: float = 1.5
    carry_tau_s: float = 30.0
    carry_backward: bool = True
    w_touch: float = 3.0
    touch_tau_s: float = 60.0
    touch_margin_m: float = 0.5

    @classmethod
    def from_config(cls, path=None) -> "EmissionParams":
        return params_from_mapping(cls, load_section("heuristic_emissions", path), "heuristic_emissions")


@dataclass
class EmissionResult:
    log_probs: np.ndarray                 # [N, 3] float64, rounded to 1e-6
    p_dead: np.ndarray                    # [N] float64
    home_logit: np.ndarray                # [N] float64 (H vs A)
    carried: np.ndarray                   # [N] float64, carried possession in [-1, 1] (0 where not used)
    meta: dict = field(default_factory=dict)


def resolve_owner_team(ball_channels: Mapping[str, np.ndarray] | None,
                       ball_channel_codes: Mapping[str, Mapping[str, int]] | None,
                       names: tuple, team_ids: tuple = (), team_names: tuple = (),
                       n_frames: int | None = None) -> tuple[np.ndarray | None, str | None]:
    """Owning team per frame (0 home, 1 away, NaN unknown) from an ATD ball channel, if one is declared.

    A text channel is read through its code table (texts equal to a team id / name, or ``home`` / ``away``);
    a numeric channel is only used when its values are team ids (numbers equal to the ids of the two teams).
    Anything ambiguous is ignored: a wrong owner would be a strong wrong emission.
    """
    if not ball_channels:
        return None, None
    wanted = {str(n).lower() for n in names}
    def keys(i: int, word: str) -> set[str]:
        found = {word}
        for seq in (team_ids, team_names):
            if len(seq) == 2:
                found.add(str(seq[i]).strip().lower())
        return found

    home_keys, away_keys = keys(0, "home"), keys(1, "away")
    for key, arr in ball_channels.items():
        low = str(key).lower()
        if low in ("x", "y", "z", "speed") or low not in wanted:
            continue
        values = np.asarray(arr, dtype=np.float64)
        if n_frames is not None and values.shape != (n_frames,):
            continue
        table = (ball_channel_codes or {}).get(key)
        out = np.full(values.shape, np.nan)
        if table:
            for text, code in table.items():
                t = str(text).strip().lower()
                if t in home_keys:
                    out[values == code] = 0.0
                elif t in away_keys:
                    out[values == code] = 1.0
        elif len(team_ids) == 2:
            try:
                ids = [float(t) for t in team_ids]
            except ValueError:
                continue
            out[values == ids[0]] = 0.0
            out[values == ids[1]] = 1.0
        if np.isfinite(out).any():
            return out, key
    return None, None


def touch_signal(touch_mask: np.ndarray, delta_d: np.ndarray, margin_m: float) -> np.ndarray:
    """Signed touch events ``[N]``: ``clip(delta_d / margin_m, -1, 1)`` at touch frames (+ = home player
    closer to the ball), NaN elsewhere and where the gap is unknown or exactly zero."""
    gap = np.asarray(delta_d, dtype=np.float64)
    out = np.full(len(gap), np.nan)
    ok = np.asarray(touch_mask, dtype=bool) & np.isfinite(gap) & (gap != 0.0)
    out[ok] = np.clip(gap[ok] / max(margin_m, 1e-9), -1.0, 1.0)
    return out


def _carry_events(values: np.ndarray, tau_frames: float, backward: bool) -> tuple[np.ndarray, np.ndarray]:
    """``(forward, backward)`` carries of the event values ``[N]`` (NaN = no event), each decaying as
    ``exp(-age / tau_frames)``; the second is zeros unless ``backward``."""
    n = len(values)
    idx = np.arange(n)
    events = np.isfinite(values)
    vals = np.where(events, values, 0.0)
    last = np.maximum.accumulate(np.where(events, idx, -1))
    fwd = np.where(last >= 0, vals[np.maximum(last, 0)] * np.exp(-(idx - last) / tau_frames), 0.0)
    if not backward:
        return fwd, np.zeros(n)
    nxt = np.minimum.accumulate(np.where(events, idx, n)[::-1])[::-1]
    bwd = np.where(nxt < n, vals[np.minimum(nxt, n - 1)] * np.exp(-(nxt - idx) / tau_frames), 0.0)
    return fwd, bwd


def alive_prior_mask(phase_alive: np.ndarray, ball_visible: np.ndarray) -> np.ndarray:
    """1.0 where the ball is hidden and a live Smart Tagging phase covers the frame (Issue #43), else 0.0."""
    return (np.asarray(phase_alive, dtype=bool) & ~np.asarray(ball_visible, dtype=bool)).astype(np.float64)


def _carry(evidence: np.ndarray, valid: np.ndarray, p: EmissionParams, fps: float) -> np.ndarray:
    """Carried possession in [-1, 1] from confident evidence samples (used when there are no touch events)."""
    confident = valid & (np.abs(evidence) >= p.carry_conf_threshold)
    fwd, bwd = _carry_events(np.where(confident, evidence, np.nan), max(p.carry_tau_s * fps, 1.0),
                             p.carry_backward)
    return np.clip(fwd + bwd, -1.0, 1.0)


def compute_emissions(
        dead_score: np.ndarray, possession_evidence: np.ndarray, evidence_valid: np.ndarray,
        ball_visible: np.ndarray, *, status_dead: np.ndarray | None = None,
        phase_prior_home: np.ndarray | None = None, owner_team: np.ndarray | None = None,
        touch_signed: np.ndarray | None = None, phase_alive: np.ndarray | None = None, fps: float = FPS,
        params: EmissionParams | None = None) -> EmissionResult:
    """``log_probs[N, 3]`` (H, A, D) from the per-frame features (see the module docstring)."""
    p = params or EmissionParams.from_config()
    n = len(dead_score)
    ds = np.clip(np.nan_to_num(np.asarray(dead_score, dtype=np.float64), nan=0.0), 1e-4, 1 - 1e-4)
    ev = np.nan_to_num(np.asarray(possession_evidence, dtype=np.float64), nan=0.0)
    valid = np.asarray(evidence_valid, dtype=bool)
    vis = np.asarray(ball_visible, dtype=bool)
    for name, arr in (("possession_evidence", ev), ("evidence_valid", valid), ("ball_visible", vis)):
        if arr.shape != (n,):
            raise ValueError(f"{name} must have shape ({n},), got {arr.shape}")

    zd = np.log(ds) - np.log1p(-ds)
    zd = zd * np.where(vis, p.dead_scale, p.dead_scale_no_ball)
    if status_dead is not None:
        st = np.asarray(status_dead, dtype=np.float64)
        zd = zd + np.where(np.isfinite(st), p.w_status * (2.0 * np.nan_to_num(st) - 1.0), 0.0)
    if phase_alive is not None and p.w_phase_alive != 0.0:      # Issue #43
        zd = zd - p.w_phase_alive * alive_prior_mask(phase_alive, vis)
    p_dead = sigmoid(zd)

    has_ev = valid & vis
    use_touch = touch_signed is not None and bool(np.isfinite(np.asarray(touch_signed)).any())
    if use_touch:
        fwd, bwd = _carry_events(np.asarray(touch_signed, dtype=np.float64), max(p.touch_tau_s * fps, 1.0),
                                 p.carry_backward)
        carried = np.clip(np.where(has_ev, fwd, fwd + bwd), -1.0, 1.0)
        z = np.where(has_ev, p.w_evidence * ev, 0.0) + p.w_touch * carried
    else:
        carried = np.where(has_ev, 0.0, _carry(ev, has_ev, p, fps))
        z = np.where(has_ev, p.w_evidence * ev, p.carry_weight * carried)
    if phase_prior_home is not None:
        z = z + p.w_phase * np.nan_to_num(np.asarray(phase_prior_home, dtype=np.float64))
    if owner_team is not None:
        ow = np.asarray(owner_team, dtype=np.float64)
        z = z + np.where(np.isfinite(ow), p.w_owner * (1.0 - 2.0 * np.nan_to_num(ow)), 0.0)

    alive = 1.0 - p_dead
    probs = np.stack([alive * sigmoid(z), alive * sigmoid(-z), p_dead], axis=1)
    log_probs = np.round(np.log(np.maximum(probs, p.prob_floor)), ROUND_DECIMALS)
    return EmissionResult(log_probs=log_probs, p_dead=p_dead, home_logit=z, carried=carried,
                          meta={"with_owner": owner_team is not None, "with_status": status_dead is not None,
                                "with_phase": phase_prior_home is not None, "with_touches": use_touch,
                                "frac_with_evidence": float(has_ev.mean()) if n else 0.0})
