# ruff: noqa: E501
"""Per-player possession zones with duel and control rules (Issue #80).

Rule-based, fully vectorised signal on ball-visible frames (literature: decision-tree possession detection from
tracking data, per-player possession vectors, FIFA EFI distance/speed rules).  The challenge counts a possession
change only when the opponent gains *controlled* possession, so touches without control, deflections and aerial
duels do not count.

Per frame ``t`` (ball visible and not aerial):

* **zone**: a team is *in the zone* when its closest player is within ``radius_m`` of the ball
  (``d_home[:, 0]`` / ``d_away[:, 0]`` of Issue #16, i.e. the possession zone of every player of the team);
* **duel rule**: both teams in the zone -> nobody is gaining control (the frame belongs to no team);
* **control confirmed** for team ``x`` when (a) ``x`` alone keeps the zone for at least ``confirm_s`` seconds
  (runs with gaps of at most ``bridge_frames`` frames bridged, never over frames of the other team), or (b) two
  consecutive touches (``touch_score >= touch_min_score``, Issue #16) belong to the same team, each with a player of
  that team alone in the zone and at most ``touch_pair_max_s`` seconds apart; control then holds between the touches.

Output ``control[N]``: 0 = home controls, 1 = away controls, -1 = none / unknown.  ``change_to[N]``: frames where
confirmed control passes to the other team than the previous confirmed controller (0 = to home, 1 = to away,
-1 = no change).  Usage (Issue #80): ``emission_tilt`` (log-odds of the controlling team added to the H/A
emissions) and ``opening_masks`` (windows where H<->A opens to the touch-gate value ``ha_open``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d

from mtah2027.config import FPS, load_section, params_from_mapping
from mtah2027.features._utils import mask_runs

USAGES = ("emission", "transition", "both")


@dataclass(frozen=True)
class PlayerPossessionParams:
    enabled: bool = False
    usage: str = "emission"            # emission | transition | both
    radius_m: float = 1.5              # possession zone radius R
    confirm_s: float = 0.8             # control confirmed after C seconds of the same team alone in the zone
    emission_weight: float = 0.5       # weight of the log-odds term (preregistered)
    confidence: float = 0.8            # P(controlling team = possessing team) behind the log-odds (fixed, not tuned)
    touch_min_score: float = 0.7       # touches of the two-consecutive-touches rule
    touch_pair_max_s: float = 3.0      # maximum separation of two consecutive same-team touches
    bridge_frames: int = 3             # gaps inside a zone run that are bridged
    opening_halfwidth: int = 7         # frames around a change of confirmed control where H<->A opens

    @classmethod
    def from_config(cls, path=None) -> "PlayerPossessionParams":
        p = params_from_mapping(cls, load_section("player_possession", path), "player_possession")
        if p.usage not in USAGES:
            raise ValueError(f"player_possession.usage must be one of {USAGES}")
        return p


@dataclass
class PlayerPossession:
    control: np.ndarray        # int8[N]: 0 home, 1 away, -1 none
    change_to: np.ndarray      # int8[N]: team that gains confirmed control at this frame, -1 none
    n_changes: int


def _paint(n: int, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Boolean mask True on every ``[start, end)`` interval (difference array, no Python loop over frames)."""
    diff = np.zeros(n + 1, dtype=np.int32)
    np.add.at(diff, starts, 1)
    np.add.at(diff, ends, -1)
    return np.cumsum(diff[:n]) > 0


def _long_runs(mask: np.ndarray, min_frames: int) -> np.ndarray:
    starts, ends = mask_runs(mask)
    keep = (ends - starts) >= min_frames
    return _paint(len(mask), starts[keep], ends[keep])


def compute_player_possession(d_home: np.ndarray, d_away: np.ndarray, touch_score: np.ndarray,
                              visible: np.ndarray, is_aerial: np.ndarray | None = None, *,
                              fps: float = FPS, params: PlayerPossessionParams | None = None) -> PlayerPossession:
    """Controlling team per frame from the nearest-player distances of each team (see the module docstring)."""
    p = params or PlayerPossessionParams()
    dh, da = np.asarray(d_home, np.float64), np.asarray(d_away, np.float64)
    dh = dh[:, 0] if dh.ndim == 2 else dh
    da = da[:, 0] if da.ndim == 2 else da
    n = len(dh)
    ok = np.asarray(visible, bool).copy()
    if is_aerial is not None:
        ok &= ~np.asarray(is_aerial, bool)
    in_h = ok & np.isfinite(dh) & (dh <= p.radius_m)
    in_a = ok & np.isfinite(da) & (da <= p.radius_m)
    sole_h, sole_a = in_h & ~in_a, in_a & ~in_h        # duel rule: both inside -> neither

    k = 2 * int(p.bridge_frames) + 1
    need = max(1, int(round(p.confirm_s * fps)))

    def closed(mask: np.ndarray, other: np.ndarray) -> np.ndarray:
        c = minimum_filter1d(maximum_filter1d(mask.astype(np.uint8), k, mode="constant"), k, mode="constant", cval=1) > 0
        return c & ~other & ok

    run_h = _long_runs(closed(sole_h, sole_a), need)
    run_a = _long_runs(closed(sole_a, sole_h), need)

    # two consecutive touches of the same team, each with a player of that team alone in the zone
    ev = np.flatnonzero(ok & (np.nan_to_num(np.asarray(touch_score, np.float64), nan=0.0) >= p.touch_min_score))
    tm = np.where(in_h[ev] & ~in_a[ev], 0, np.where(in_a[ev] & ~in_h[ev], 1, -1))
    pair = (tm[:-1] == tm[1:]) & (tm[:-1] >= 0) & ((ev[1:] - ev[:-1]) <= p.touch_pair_max_s * fps)
    i = np.flatnonzero(pair)
    for_h, for_a = i[tm[i] == 0], i[tm[i] == 1]
    touch_h = _paint(n, ev[for_h], ev[for_h + 1] + 1) if len(for_h) else np.zeros(n, bool)
    touch_a = _paint(n, ev[for_a], ev[for_a + 1] + 1) if len(for_a) else np.zeros(n, bool)

    ch, ca = run_h | touch_h, run_a | touch_a
    control = np.where(ch & ~ca, 0, np.where(ca & ~ch, 1, -1)).astype(np.int8)

    # a change: confirmed control of a team whose previous confirmed controller was the other team
    idx = np.arange(n)
    last = np.maximum.accumulate(np.where(control >= 0, idx, -1))
    prev_last = np.concatenate([[-1], last[:-1]])
    prev_team = np.where(prev_last >= 0, control[np.maximum(prev_last, 0)], -1)
    chg = (control >= 0) & (prev_team >= 0) & (prev_team != control)
    change_to = np.where(chg, control, -1).astype(np.int8)
    return PlayerPossession(control=control, change_to=change_to, n_changes=int(chg.sum()))


def emission_tilt(pp: PlayerPossession, params: PlayerPossessionParams) -> np.ndarray:
    """Additive H-vs-A log-odds ``w * logit(confidence)`` (positive = home) on frames with confirmed control."""
    c = float(params.emission_weight) * float(np.log(params.confidence / (1.0 - params.confidence)))
    return np.where(pp.control == 0, c, np.where(pp.control == 1, -c, 0.0))


def apply_emission_tilt(log_probs: np.ndarray, tilt: np.ndarray) -> np.ndarray:
    """Split the tilt symmetrically between H and A (the H-A log-odds moves by ``tilt``); D untouched."""
    if not np.any(tilt):
        return log_probs
    out = np.asarray(log_probs, np.float64).copy()
    fin = np.isfinite(out).all(axis=1)
    out[:, 0] += np.where(fin, tilt / 2.0, 0.0)
    out[:, 1] -= np.where(fin, tilt / 2.0, 0.0)
    return np.round(out, 6)


def opening_masks(pp: PlayerPossession, params: PlayerPossessionParams) -> tuple[np.ndarray, np.ndarray]:
    """``(to_home, to_away)``: windows of ``opening_halfwidth`` frames around a change of confirmed control."""
    width = 2 * int(params.opening_halfwidth) + 1

    def spread(mask: np.ndarray) -> np.ndarray:
        if not mask.any():
            return np.zeros(len(mask), bool)
        return maximum_filter1d(mask.astype(np.uint8), width, mode="constant") > 0

    return spread(pp.change_to == 0), spread(pp.change_to == 1)
