"""Exact restart-frame anchoring ``t*`` for Smart Tagging SET PIECES and after GOALS (Issue #14).

Smart Tagging intervals are *windows*, not instants (a ``Kick Off (Start)`` opens ~3.5 s before the
kick), so ``start_frame`` is never used as the restart.  For every tagged set piece the restart frame
``t*`` -- the first frame of the new possession -- is located in the tracking inside the search window
``[start_frame - pre_margin, end_frame - 1 + post_margin]``:

1. **Ball method** (``method='ball'``).  A *launch* is a frame where the ball speed is >=
   ``launch_speed_mps`` (4 m/s) for ``launch_sustain_frames`` consecutive frames and was not already so
   the frame before; it must be preceded by a *rest* (speed <= ``rest_speed_mps`` = 1 m/s) of at least
   ``min_rest_s``.  The onset ``t*`` is the first frame after that rest (at most ``max_ramp_s`` before the
   launch crossing).  If several launches qualify, the one with the longest rest (capped at
   ``rest_cap_s``) wins, ties going to the earliest -- the long rest is the dead-ball period.
   ``confidence = 0.3 + 0.7 * min(rest_s, 1 s) * min(peak_speed / (2 * launch), 1)``.
2. **Collective method** (``method='collective'``), when the ball is missing or no rest->launch exists.
   With ``V`` the mean speed of the active players, ``J(t) = mean(V[t, t+w)) - mean(V[t-w, t))`` and
   ``t* = argmax J`` in the window, provided ``J >= collective_jump_min_mps``.
   ``confidence = collective_conf_max * min(J / (2 * jump_min), 1)``.
3. Otherwise the anchor is kept as ``method='unresolved'`` with ``t_star = -1``: nothing is invented.

**Forced dead interval before the restart.**  With the heuristic ``dead_score`` of Issue #13 (computed
here from the arrays when not supplied), dead evidence is ``dead_score >= dead_threshold`` with gaps
shorter than ``dead_gap_close_s`` bridged.  The interval is ``[onset, t* - 1]`` where ``onset`` is the
start of the evidence run that touches ``t* - 1`` (or ends at most ``dead_tail_s`` before it), searched
no farther back than ``max_dead_back_s``.  With no evidence the tag window's own start is used
(``onset_method='tag-window'``), if it precedes ``t*``.

**Goals.**  For each GOALS tag the dead onset is the first start of an evidence run inside the goal
window (``tag-window`` start otherwise); ``D`` is forced from there to the ``t* - 1`` of the next
resolved ``Kick Off (After Goal)`` anchor (at most ``max_goal_to_kickoff_s`` later), or -- without such a
kick-off -- to the end of the evidence run, never beyond that.

Half boundaries and restart frames of the competition match are not used anywhere; only tags and
tracking.  Pure NumPy over the window (the per-frame work is vectorised once for the whole match).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
from scipy.ndimage import binary_closing

from mtah2027.config import FPS, PITCH, load_section, params_from_mapping
from mtah2027.features._utils import run_length_before, speed_from_xy, window_mean
from mtah2027.features.dead_ball_signals import DeadBallParams, compute_dead_ball_signals


@dataclass(frozen=True)
class AnchorParams:
    pre_margin_s: float = 1.0
    post_margin_s: float = 2.0
    rest_speed_mps: float = 1.0
    launch_speed_mps: float = 4.0
    launch_sustain_frames: int = 3
    min_rest_s: float = 0.4
    rest_cap_s: float = 5.0
    max_ramp_s: float = 0.5
    ball_speed_half_span: int = 2
    player_speed_half_span: int = 1
    collective_window_s: float = 1.0
    collective_jump_min_mps: float = 0.4
    collective_conf_max: float = 0.5
    collective_dead_lag_s: float = 1.0   # >0: the jump must follow dead evidence within this lag (0 = off)
    dead_threshold: float = 0.5
    dead_gap_close_s: float = 1.0
    dead_tail_s: float = 1.0
    max_dead_back_s: float = 40.0
    goal_search_pre_s: float = 3.0
    goal_search_post_s: float = 6.0
    max_goal_to_kickoff_s: float = 150.0
    restart_mask_halfwidth: int = 0

    @classmethod
    def from_config(cls, path=None) -> "AnchorParams":
        return params_from_mapping(cls, load_section("setpiece_anchor", path), "setpiece_anchor")


@dataclass(frozen=True)
class SetPieceAnchor:
    """One tagged set piece anchored in the tracking (``t_star == -1`` when unresolved)."""

    type: str
    team: int                  # 0 home, 1 away, -1 N/A
    t_star: int
    confidence: float
    method: str                # 'ball' | 'collective' | 'unresolved'
    tag_start: int
    tag_end: int               # exclusive
    window: tuple[int, int]    # inclusive search window
    dead_onset: int            # first frame of the forced dead interval, -1 if none
    onset_method: str          # 'dead-score' | 'tag-window' | 'none'
    rest_s: float = 0.0

    @property
    def resolved(self) -> bool:
        return self.t_star >= 0


@dataclass(frozen=True)
class ForcedDead:
    """Inclusive interval of frames that must be ``D`` (up to the restart)."""

    start: int
    end: int
    kind: str                  # 'pre-restart' | 'goal'
    anchor_index: int          # index into ``anchors`` (the restart anchor), -1 for a goal without one
    goal_index: int = -1


@dataclass
class AnchorResult:
    anchors: list[SetPieceAnchor]
    restart_mask: np.ndarray               # [N] bool, True on t* (+- restart_mask_halfwidth)
    forced_dead_intervals: list[ForcedDead]
    forced_dead_mask: np.ndarray           # [N] bool, union of the intervals
    t_star: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))   # resolved restarts

    @property
    def n_resolved(self) -> int:
        return int(sum(a.resolved for a in self.anchors))


def _intervals_to_mask(n: int, intervals: Iterable[tuple[int, int]]) -> np.ndarray:
    diff = np.zeros(n + 1, dtype=np.int32)
    for a, b in intervals:
        a, b = max(0, int(a)), min(n - 1, int(b))
        if b >= a:
            diff[a] += 1
            diff[b + 1] -= 1
    return np.cumsum(diff[:n]) > 0


def _close_gaps(evidence: np.ndarray, gap_frames: int) -> np.ndarray:
    """Bridge False gaps shorter than ``gap_frames`` between True runs (edges are left untouched)."""
    if gap_frames <= 1 or not evidence.any():
        return evidence
    pad = gap_frames
    padded = np.concatenate([np.zeros(pad, bool), evidence, np.zeros(pad, bool)])
    closed = binary_closing(padded, structure=np.ones(gap_frames, dtype=bool))
    return closed[pad:-pad] | evidence


def _ball_restart(c_lo: int, c_hi: int, g: dict, p: AnchorParams, fps: float):
    """Best rest->launch inside the window: ``(t_star, confidence, rest_s)`` or None."""
    cands = np.flatnonzero(g["launch_start"][c_lo:c_hi + 1]) + c_lo
    if not len(cands):
        return None
    last_rest = g["last_rest"][cands]
    keep = last_rest >= 0
    cands, last_rest = cands[keep], last_rest[keep]
    if not len(cands):
        return None
    rest_len = g["rest_run"][last_rest]
    ok = rest_len >= max(1, int(round(p.min_rest_s * fps)))
    cands, last_rest, rest_len = cands[ok], last_rest[ok], rest_len[ok]
    if not len(cands):
        return None
    score = np.minimum(rest_len, int(round(p.rest_cap_s * fps)))
    best = int(np.argmax(score))                      # first maximum -> earliest
    c, onset = int(cands[best]), int(last_rest[best]) + 1
    t_star = max(onset, c - int(round(p.max_ramp_s * fps)))
    rest_s = float(rest_len[best]) / fps
    peak = float(np.nanmax(g["bspeed"][c:c + p.launch_sustain_frames + 10]))
    conf = 0.3 + 0.7 * min(rest_s, 1.0) * min(peak / (2.0 * p.launch_speed_mps), 1.0)
    return t_star, float(np.clip(conf, 0.0, 1.0)), rest_s


def _collective_restart(lo: int, hi: int, jump: np.ndarray, p: AnchorParams,
                        after_dead: np.ndarray | None = None):
    seg = jump[lo:hi + 1]
    if after_dead is not None:
        seg = np.where(after_dead[lo:hi + 1], seg, np.nan)
    if not len(seg) or not np.isfinite(seg).any():
        return None
    k = int(np.nanargmax(seg))
    if seg[k] < p.collective_jump_min_mps:
        return None
    conf = p.collective_conf_max * min(float(seg[k]) / (2.0 * p.collective_jump_min_mps), 1.0)
    return lo + k, float(conf)


def _dead_onset(t_star: int, tag_start: int, closed: np.ndarray, run_before: np.ndarray, p: AnchorParams,
                fps: float) -> tuple[int, str]:
    """First frame of the dead interval ending at ``t_star - 1`` (see the module docstring)."""
    j = t_star - 1
    if j < 0:
        return -1, "none"
    back = int(round(p.max_dead_back_s * fps))
    tail = int(round(p.dead_tail_s * fps))
    lo = max(0, t_star - tail)
    tail_hits = np.flatnonzero(closed[lo:j + 1])
    if len(tail_hits):
        jj = lo + int(tail_hits[-1])
        start = jj - int(run_before[jj]) + 1
        return max(start, t_star - back), "dead-score"
    if tag_start < t_star:
        return max(tag_start, t_star - back, 0), "tag-window"
    return -1, "none"


def anchor_set_pieces(
        ball_xy: np.ndarray, players_xy: np.ndarray, team_of_slot: np.ndarray,
        set_pieces: Sequence, goals: Sequence = (), *,
        ball_speed: np.ndarray | None = None, players_speed: np.ndarray | None = None,
        dead_score: np.ndarray | None = None, fps: float = FPS, pitch: tuple[float, float] = PITCH,
        params: AnchorParams | None = None, dead_params: DeadBallParams | None = None) -> AnchorResult:
    """Anchor every tagged set piece and goal in the tracking.

    ``ball_xy[N, 2]`` (NaN when missing), ``players_xy[N, S, 2]`` and ``team_of_slot[S]`` are metres with
    home attacking +x.  ``set_pieces`` items need ``type, team, start_frame, end_frame`` (end exclusive;
    :class:`~mtah2027.io.smart_tagging_loader.SetPiece`), ``goals`` items ``team, start_frame, end_frame``.
    Speeds (m/s) are derived from the positions when not supplied; ``dead_score`` (Issue #13) is computed
    when not supplied.  Missing tags are never invented: the result only holds what was tagged.
    """
    p = params or AnchorParams.from_config()
    n = ball_xy.shape[0]
    if players_xy.shape[0] != n:
        raise ValueError("players_xy and ball_xy must have the same number of frames")
    bspeed = (speed_from_xy(ball_xy, fps, p.ball_speed_half_span) if ball_speed is None
              else np.asarray(ball_speed, dtype=np.float32))
    pspeed = (speed_from_xy(players_xy, fps, p.player_speed_half_span) if players_speed is None
              else np.asarray(players_speed, dtype=np.float32))
    if dead_score is None:
        dead_score = compute_dead_ball_signals(
            players_xy, team_of_slot, ball_xy, players_speed=pspeed, ball_speed=bspeed, fps=fps, pitch=pitch,
            params=dead_params).dead_score
    idx = np.arange(n)

    finite = np.isfinite(bspeed)
    with np.errstate(invalid="ignore"):
        rest = finite & (bspeed <= p.rest_speed_mps)
        fast = finite & (bspeed >= p.launch_speed_mps)
    k = max(1, int(p.launch_sustain_frames))
    fast_run = run_length_before(fast)
    sustained = np.zeros(n, dtype=bool)
    if n >= k:
        sustained[:n - k + 1] = fast_run[k - 1:] >= k
    launch_start = sustained & ~np.concatenate([[False], sustained[:-1]]) & ~np.concatenate(
        [[False], fast[:-1]])
    ball_g = {"launch_start": launch_start, "rest_run": run_length_before(rest),
              "last_rest": np.maximum.accumulate(np.where(rest, idx, -1)), "bspeed": bspeed}

    active = np.isfinite(pspeed) & np.isfinite(players_xy).all(axis=2)
    cnt = active.sum(axis=1)
    mean_v = np.where(cnt >= 3, np.nansum(np.where(active, pspeed, 0.0), axis=1) / np.maximum(cnt, 1), np.nan)
    w = max(1, int(round(p.collective_window_s * fps)))
    jump = window_mean(mean_v, 0, w - 1) - window_mean(mean_v, -w, -1)

    ev = np.nan_to_num(np.asarray(dead_score, dtype=np.float64), nan=0.0) >= p.dead_threshold
    closed = _close_gaps(ev, int(round(p.dead_gap_close_s * fps)))
    run_before = run_length_before(closed)
    after_dead = None
    if p.collective_dead_lag_s > 0:      # a restart follows a dead run: gate the jump on recent dead evidence
        idx_all = np.arange(n)
        last_dead = np.maximum.accumulate(np.where(closed, idx_all, -10 ** 9))
        prev = np.concatenate([[-10 ** 9], last_dead[:-1]])
        after_dead = (idx_all - prev) <= max(1, int(round(p.collective_dead_lag_s * fps)))

    pre, post = int(round(p.pre_margin_s * fps)), int(round(p.post_margin_s * fps))
    anchors: list[SetPieceAnchor] = []
    intervals: list[ForcedDead] = []
    for sp in set_pieces:
        a, b = int(sp.start_frame), int(sp.end_frame)
        lo, hi = max(0, a - pre), min(n - 1, b - 1 + post)
        found = _ball_restart(lo, hi, ball_g, p, fps) if hi >= lo else None
        method, rest_s = "ball", 0.0
        if found is not None:
            t_star, conf, rest_s = found
        else:
            col = _collective_restart(lo, hi, jump, p, after_dead) if hi >= lo else None
            if col is not None:
                (t_star, conf), method = col, "collective"
            else:
                t_star, conf, method = -1, 0.0, "unresolved"
        onset, onset_method = (-1, "none")
        if t_star >= 0:
            onset, onset_method = _dead_onset(t_star, a, closed, run_before, p, fps)
            if onset >= 0:
                intervals.append(ForcedDead(onset, t_star - 1, "pre-restart", len(anchors)))
        anchors.append(SetPieceAnchor(
            type=str(sp.type), team=int(sp.team), t_star=int(t_star), confidence=float(conf), method=method,
            tag_start=a, tag_end=b, window=(lo, hi), dead_onset=int(onset), onset_method=onset_method,
            rest_s=rest_s))

    # goals: dead from the goal to the t* of the next Kick Off (After Goal)
    kickoffs = sorted(((a.t_star, i) for i, a in enumerate(anchors)
                       if a.resolved and "kick off" in a.type and "after goal" in a.type))
    g_pre, g_post = int(round(p.goal_search_pre_s * fps)), int(round(p.goal_search_post_s * fps))
    max_link = int(round(p.max_goal_to_kickoff_s * fps))
    for gi, goal in enumerate(goals):
        a, b = int(goal.start_frame), int(goal.end_frame)
        lo, hi = max(0, a - g_pre), min(n - 1, b - 1 + g_post)
        if hi < lo:
            continue
        starts = np.flatnonzero(closed[lo:hi + 1] & ~np.concatenate([[False], closed[lo:hi]])) + lo
        if len(starts):
            g_on = int(starts[0])
        elif closed[lo]:
            g_on = int(lo - run_before[lo] + 1)
        else:
            g_on = max(0, a)
        link = next(((t, i) for t, i in kickoffs if g_on < t <= g_on + max_link), None)
        if link is not None:
            intervals.append(ForcedDead(g_on, link[0] - 1, "goal", link[1], gi))
        elif closed[g_on]:
            end = g_on
            while end + 1 < min(n, g_on + max_link) and closed[end + 1]:
                end += 1
            intervals.append(ForcedDead(g_on, end, "goal", -1, gi))

    intervals.sort(key=lambda d: (d.start, d.end))
    restart_mask = np.zeros(n, dtype=bool)
    h = int(p.restart_mask_halfwidth)
    for an in anchors:
        if an.resolved:
            restart_mask[max(0, an.t_star - h):min(n, an.t_star + h + 1)] = True
    forced = _intervals_to_mask(n, [(d.start, d.end) for d in intervals])
    t_stars = np.array(sorted(a.t_star for a in anchors if a.resolved), dtype=np.int64)
    return AnchorResult(anchors=anchors, restart_mask=restart_mask, forced_dead_intervals=intervals,
                        forced_dead_mask=forced, t_star=t_stars)
