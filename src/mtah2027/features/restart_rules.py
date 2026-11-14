# ruff: noqa: E501
"""Restart-team rules of Smart Tagging and tag-derived occlusion flags (Issue #49).

Smart Tagging names the team that performs a set piece (``Team``), and the possession that follows ``t*``
belongs to it.  The names can be checked against rules of the game and against the tracking itself, all of it
label-free:

* ``Kick Off (After Goal)`` is taken by the team that **conceded** the preceding goal (``GOALS`` tag, whose ``Team``
  is the scorer);
* the second-half ``Kick Off (Start)`` is taken by the **opposite** team of the first-half one;
* the team of the first *clear controller* of the ball just after ``t*`` (a player within ``ctrl_radius_m`` of the
  visible ball, ``ctrl_margin_m`` closer than any opponent) is a tracking measurement of the same fact.

For every restart the available sources (tag, rule, tracking) vote; the winner becomes ``team`` and the share of
agreeing sources the reliability ``team_conf`` that scales the restart-team penalty of the transition matrix.
No frame of any match is hard-coded: the first-half / second-half pairing is taken from the order of the
``Kick Off (Start)`` tags and the rules are properties of the game.

``calibrate_team_weight`` turns the agreement between the tagged teams and the first confident state of Specialist
A after each ``t*`` into a global weight in [0, 1] (never the ground truth).  ``long_ball_mask`` marks the frames
covered by ``Long Ball`` / ``Ball in the Box`` phases: a long ball in flight and a crowded box are exactly the
occlusions the engine reads as stoppage, so over hidden-ball frames they are weak evidence of an alive ball
(``heuristic_emissions.w_long_ball`` / ``specialist_fusion``; ``auto_long_ball_weight`` calibrates the
weight on the visible frames where Specialist A is confident).

Everything is configured in ``configs/default.yaml`` (section ``restart_rules``) and is off by default.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from mtah2027.config import FPS, load_section, params_from_mapping
from mtah2027.features.setpiece_anchor import AnchorResult, SetPieceAnchor

HOME, AWAY, NO_TEAM = 0, 1, -1


@dataclass(frozen=True)
class RestartRuleParams:
    enabled: bool = False
    kickoff_after_goal: bool = True       # rule: Kick Off (After Goal) = team that conceded
    second_half_opposite: bool = True     # rule: second-half Kick Off (Start) = opposite of the first-half one
    tracking_vote: bool = True            # the first clear controller after t* votes
    ctrl_window_s: float = 1.4
    ctrl_radius_m: float = 3.0
    ctrl_margin_m: float = 1.0
    goal_link_s: float = 150.0            # a Kick Off (After Goal) is linked to a goal at most this long before
    unreliable_conf: float = 0.5          # team_conf of a restart whose sources disagree / stand alone
    fill_unnamed: bool = True             # a restart with Team = N/A takes the voted team (rule / tracking)
    calibrate_weight: bool = False        # scale the team penalty by the agreement with Specialist A
    calib_window_s: float = 3.0
    calib_prior: float = 1.0              # pseudo-count of the Laplace estimate
    long_ball_codes: tuple = ("long ball", "ball in the box")
    auto_long_ball_weight: bool = False   # w_long_ball from Specialist A's confident visible frames
    long_ball_cap: float = 1.5            # ceiling of the calibrated weight
    long_ball_min_frames: int = 250       # confident frames needed inside the mask to calibrate

    @classmethod
    def from_config(cls, path=None) -> "RestartRuleParams":
        return params_from_mapping(cls, load_section("restart_rules", path), "restart_rules")


def tracking_team(ball_xy: np.ndarray, players_xy: np.ndarray, team_of_slot: np.ndarray, t: int, *,
                  fps: float = FPS, params: RestartRuleParams | None = None) -> int:
    """Team of the first clear controller of the ball in ``[t, t + ctrl_window_s]`` (``-1`` if none)."""
    p = params or RestartRuleParams()
    n = len(ball_xy)
    lo, hi = max(0, int(t)), min(n, int(t) + max(1, int(round(p.ctrl_window_s * fps))) + 1)
    if hi <= lo:
        return NO_TEAM
    b = np.asarray(ball_xy[lo:hi], dtype=np.float64)
    pl = np.asarray(players_xy[lo:hi], dtype=np.float64)
    d = np.linalg.norm(pl - b[:, None, :], axis=2)                      # [W, S]
    d = np.where(np.isfinite(d), d, np.inf)
    team = np.asarray(team_of_slot)
    dh = np.where(team[None, :] == HOME, d, np.inf).min(axis=1)
    da = np.where(team[None, :] == AWAY, d, np.inf).min(axis=1)
    near = np.minimum(dh, da)
    with np.errstate(invalid="ignore"):
        clear = (near <= p.ctrl_radius_m) & (np.abs(dh - da) >= p.ctrl_margin_m) & np.isfinite(near)
    if not clear.any():
        return NO_TEAM
    k = int(np.argmax(clear))
    return HOME if dh[k] < da[k] else AWAY


def _vote(sources: Sequence[int]) -> tuple[int, int, int]:
    """``(team, n_for, n_named)`` of the majority of the named sources (ties give ``-1``)."""
    named = [int(s) for s in sources if s in (HOME, AWAY)]
    h, a = named.count(HOME), named.count(AWAY)
    if h == a:
        return NO_TEAM, max(h, a), len(named)
    return (HOME if h > a else AWAY), max(h, a), len(named)


def _is_kickoff_start(a: SetPieceAnchor) -> bool:
    return "kick off" in a.type and "start" in a.type


def apply_team_rules(anchors: AnchorResult, goals: Sequence, ball_xy: np.ndarray, players_xy: np.ndarray,
                     team_of_slot: np.ndarray, *, fps: float = FPS,
                     params: RestartRuleParams | None = None) -> tuple[AnchorResult, dict]:
    """Anchors with the voted restart team and its reliability, plus an aggregate report."""
    p = params or RestartRuleParams.from_config()
    items = list(anchors.anchors)
    report = {"n_resolved": 0, "changed": 0, "filled": 0, "unreliable": 0, "rule_goal": 0, "rule_half": 0,
              "tracking_agree": 0, "tracking_named": 0}
    track = [NO_TEAM] * len(items)
    if p.tracking_vote:
        for i, a in enumerate(items):
            if a.resolved:
                track[i] = tracking_team(ball_xy, players_xy, team_of_slot, a.t_star, fps=fps, params=p)
    rule = [NO_TEAM] * len(items)
    if p.kickoff_after_goal and goals:
        link = int(round(p.goal_link_s * fps))
        for i, a in enumerate(items):
            if "kick off" in a.type and "after goal" in a.type:
                prior = [g for g in goals if int(g.start_frame) <= a.tag_end
                         and a.tag_start - int(g.start_frame) <= link and int(g.team) in (HOME, AWAY)]
                if prior:
                    g = max(prior, key=lambda x: int(x.start_frame))
                    rule[i] = 1 - int(g.team)
                    report["rule_goal"] += 1
    starts = sorted((i for i, a in enumerate(items) if _is_kickoff_start(a)), key=lambda i: items[i].tag_start)
    half_src: dict[int, list[int]] = {}
    if p.second_half_opposite and len(starts) >= 2:
        i1, i2 = starts[0], starts[1]
        inv = lambda x: (1 - x) if x in (HOME, AWAY) else NO_TEAM       # noqa: E731
        half_src[i1] = [inv(items[i2].team), inv(track[i2])]
        half_src[i2] = [inv(items[i1].team), inv(track[i1])]
        report["rule_half"] = 2

    out = list(items)
    for i, a in enumerate(items):
        if not a.resolved:
            continue
        report["n_resolved"] += 1
        src = [a.team, track[i], rule[i], *half_src.get(i, [])]
        if track[i] in (HOME, AWAY) and a.team in (HOME, AWAY):
            report["tracking_named"] += 1
            report["tracking_agree"] += int(track[i] == a.team)
        team, n_for, n_named = _vote(src)
        if a.team in (HOME, AWAY):
            if team == NO_TEAM:                       # tie: keep the tag but trust it less
                out[i] = dataclasses.replace(a, team_conf=p.unreliable_conf, team_source="tag")
                report["unreliable"] += 1
                continue
            conf = 1.0 if n_for == n_named else p.unreliable_conf     # any dissent lowers it
            if team != a.team:
                report["changed"] += 1
            if conf < 1.0:
                report["unreliable"] += 1
            out[i] = dataclasses.replace(a, team=int(team), team_conf=float(conf),
                                         team_source="tag" if team == a.team else "vote")
        elif p.fill_unnamed and team != NO_TEAM:
            out[i] = dataclasses.replace(a, team=int(team), team_conf=1.0 if n_for >= 2 else p.unreliable_conf,
                                         team_source="rule" if (rule[i] == team or i in half_src) else "tracking")
            report["filled"] += 1
    return dataclasses.replace(anchors, anchors=out), report


def calibrate_team_weight(anchors: AnchorResult, a_label: np.ndarray, *, fps: float = FPS,
                          params: RestartRuleParams | None = None) -> tuple[float, int]:
    """``(weight, n)``: reliability of the tagged restart teams measured against Specialist A's confident states.

    ``a_label[N]`` holds the winning class on the frames where Specialist A is ultra-confident and ``-1`` elsewhere.
    For every named, resolved restart the first H/A state in ``[t*, t* + calib_window_s]`` is compared with the
    tagged team.  ``weight = clip(2 r - 1, 0, 1)`` with ``r`` the Laplace-smoothed agreement rate; with no
    comparable restart the weight is 1 (the tags are taken as they are)."""
    p = params or RestartRuleParams.from_config()
    lab = np.asarray(a_label)
    n = len(lab)
    win = max(1, int(round(p.calib_window_s * fps)))
    agree = total = 0
    for a in anchors.anchors:
        if not a.resolved or a.team not in (HOME, AWAY):
            continue
        seg = lab[max(0, a.t_star):min(n, a.t_star + win + 1)]
        hit = np.flatnonzero((seg == HOME) | (seg == AWAY))
        if not len(hit):
            continue
        total += 1
        agree += int(seg[hit[0]] == a.team)
    if total == 0:
        return 1.0, 0
    r = (agree + p.calib_prior) / (total + 2.0 * p.calib_prior)
    return float(np.clip(2.0 * r - 1.0, 0.0, 1.0)), total


def long_ball_mask(smart_tagging, n: int, codes: Sequence[str] = ("long ball", "ball in the box")) -> np.ndarray:
    """[N] bool: frames covered by a phase whose code is in ``codes`` (case-insensitive)."""
    want = {str(c).strip().lower() for c in codes}
    diff = np.zeros(n + 1, np.int32)
    for e in getattr(smart_tagging, "events", []) or []:
        if str(e.code).strip().lower() in want:
            lo, hi = max(0, min(int(e.start_frame), n)), max(0, min(int(e.end_frame), n))
            if hi > lo:
                diff[lo] += 1
                diff[hi] -= 1
    return np.cumsum(diff[:n]) > 0


def auto_long_ball_weight(mask: np.ndarray, a_label: np.ndarray, visible: np.ndarray, *,
                          params: RestartRuleParams | None = None) -> tuple[float, dict]:
    """D-logit reduction for the long-ball mask from Specialist A's confident visible frames.

    ``w = clip(logit(alive | mask) - logit(alive | ~mask), 0, cap)``: how much more alive the confident visible
    frames inside the phases are than outside.  Fewer than ``long_ball_min_frames`` confident frames inside the
    mask give 0 (nothing is invented)."""
    p = params or RestartRuleParams.from_config()
    lab = np.asarray(a_label)
    ok = np.asarray(visible, bool) & (lab >= 0)
    inside, outside = ok & np.asarray(mask, bool), ok & ~np.asarray(mask, bool)
    info = {"n_inside": int(inside.sum()), "n_outside": int(outside.sum())}
    if inside.sum() < p.long_ball_min_frames or outside.sum() < p.long_ball_min_frames:
        return 0.0, info

    def logit_alive(m: np.ndarray) -> float:
        k = float((lab[m] != 2).sum()) + 0.5
        q = k / (float(m.sum()) + 1.0)
        return float(np.log(q) - np.log1p(-q))

    w = float(np.clip(logit_alive(inside) - logit_alive(outside), 0.0, p.long_ball_cap))
    info["alive_inside"] = float((lab[inside] != 2).mean())
    info["alive_outside"] = float((lab[outside] != 2).mean())
    return w, info
