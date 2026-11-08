#!/usr/bin/env python
"""Deterministic synthetic BAR-ATH match in the same file layout as the real bundle (Issue #3).

Writes, under ``<out>``::

    tactical/BAR-ATH_tactical_atd.xml          FIFA/EPTS metadata (Metrica flavour, kloppy readable)
    tactical/BAR-ATH_tactical_atd.txt          "<frame_id>:<46 x 'x,y,speed' ;-separated>:<ball>"
    tactical/BAR-ATH_tactical_homography.csv   frame,h00..h22,reprojection_error,valid
    tactical/BAR-ATH_tactical_smart_tagging.xml  SportsCode XML (SET PIECES, GOALS, phases)
    broadcast/BAR-ATH_broadcast_atd.{xml,txt}  offset / cropped copy with cuts (51 slots)
    broadcast/BAR-ATH_broadcast_homography.csv
    synthetic_gt.csv                           frame,label with the true H/A/D/X labels

so that ``games.asset_paths(data_dir=out)`` finds every file.  The default is a short match
(12,000 tactical frames, official half boundaries scaled proportionally); ``full_length=True``
produces exactly 199,319 tactical frames with the official half boundaries.

Conventions: x,y are normalised pitch coordinates (0-1, pitch 105.3 x 68.0 m); speed is in m/s
(the XML declares the unit).  The first ``<Team>`` is the home side (``H``), which attacks +x in
the first half and -x in the second.  Output is byte-identical for a given (seed, size).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

GAME = "BAR-ATH"
FPS = 25
PITCH_L, PITCH_W = 105.3, 68.0
FULL_FRAMES = 199319
FULL_BROADCAST_FRAMES = 164488
FULL_HALF1 = (11185, 90651)
FULL_HALF2_DEAD = (110808, 113741)
FULL_HALF2_PLAY = (113742, 190197)
TACTICAL_SLOTS = (23, 23)
BROADCAST_SLOTS = (25, 26)
HOME_NAME, AWAY_NAME = "Barcelona", "Atletic Club"
TEAM_NAMES = (HOME_NAME, AWAY_NAME)
TEAM_IDS = ("FIFATMA", "FIFATMB")
HIDDEN_BALL_FRACTION = 0.41
P_DEAD = 0.55
IMG_W, IMG_H = 1920, 1080

# Formation in own-attack coordinates: u (0 = own goal line, 1 = opponent goal line), y (0..1).
FORM_U = np.array([0.0, 0.22, 0.22, 0.22, 0.22, 0.45, 0.45, 0.45, 0.70, 0.70, 0.70])
FORM_Y = np.array([0.5, 0.15, 0.38, 0.62, 0.85, 0.25, 0.50, 0.75, 0.20, 0.50, 0.80])

DEAD_TYPES = ("Throw-in", "Free Kick", "Goal Kick", "Corner Kick", "Goal")
DEAD_WEIGHTS = np.array([0.40, 0.25, 0.15, 0.17, 0.03])
DEAD_LEN_S = {
    "Throw-in": (4.0, 12.0),
    "Free Kick": (6.0, 25.0),
    "Goal Kick": (8.0, 20.0),
    "Corner Kick": (10.0, 25.0),
    "Goal": (45.0, 70.0),
}
PHASES = ("BUILD UP", "PROGRESSION", "FINAL THIRD", "TRANSITION")


# --------------------------------------------------------------------------------------------
# Specification
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class MatchSpec:
    n_frames: int
    n_broadcast: int
    half1: tuple
    half2_dead: tuple
    half2_play: tuple

    @property
    def half2(self):
        return (self.half2_dead[0], self.half2_play[1])

    @property
    def full_length(self):
        return self.n_frames == FULL_FRAMES


def make_spec(n_frames: int = 12000, full_length: bool = False) -> MatchSpec:
    """Match layout: official boundaries for the full length, proportionally scaled otherwise."""
    if full_length or n_frames == FULL_FRAMES:
        return MatchSpec(FULL_FRAMES, FULL_BROADCAST_FRAMES, FULL_HALF1, FULL_HALF2_DEAD, FULL_HALF2_PLAY)
    if n_frames < 2000 or n_frames > FULL_FRAMES:
        raise ValueError("n_frames must be in [2000, 199319]")
    s = n_frames / FULL_FRAMES

    def sc(v):
        return int(round(v * s))

    a, b = sc(FULL_HALF1[0]), sc(FULL_HALF1[1])
    c0, c1 = sc(FULL_HALF2_DEAD[0]), sc(FULL_HALF2_DEAD[1])
    d1 = sc(FULL_HALF2_PLAY[1])
    return MatchSpec(n_frames, int(round(n_frames * FULL_BROADCAST_FRAMES / FULL_FRAMES)),
                     (a, b), (c0, c1), (c1 + 1, d1))


# --------------------------------------------------------------------------------------------
# Script of play: possession / dead-ball segments and the ball-focus control points
# --------------------------------------------------------------------------------------------
@dataclass
class Seg:
    start: int
    end: int
    kind: str                 # 'poss' | 'dead'
    team: int                 # possessing team, or restart team for a dead segment
    half: int
    origin: str = ""          # poss: 'kickoff' | 'restart' | 'turnover'
    typ: str = ""             # dead: Throw-in, Free Kick, Goal Kick, Corner Kick, Goal, half_start
    exit_team: int = -1       # dead: team that had the ball when it stopped
    has_restart: bool = False
    events: list = field(default_factory=list)   # poss: (receive, pass_start|None, Fx, Fy)
    goal_frame: int = -1      # dead 'Goal': frame at which the ball enters the net
    E: tuple = (0.0, 0.0)     # dead: exit point
    R: tuple = (0.0, 0.0)     # dead: restart point


def attack_dir(team: int, half: int) -> int:
    """+1 when the team attacks towards +x. Home attacks +x in the first half."""
    return 1 if (team == 0) == (half == 1) else -1


class _Ctrl:
    """Monotone list of (frame, x, y) control points for the ball-focus path."""

    def __init__(self):
        self.pts = []

    def add(self, frame, x, y):
        frame = int(frame)
        if self.pts and frame <= self.pts[-1][0]:
            frame = self.pts[-1][0] + 1
        self.pts.append((frame, float(x), float(y)))


def _poss_duration(rng, team):
    sec = float(np.clip(rng.lognormal(np.log(11.5 * (1.15 if team == 0 else 1.0)), 0.55), 5.0, 30.0))
    return int(round(sec * FPS))


def _poss_events(rng, seg, f_start, ctrl, restart_start):
    """Pass chain of one possession; also appends focus control points."""
    d = attack_dir(seg.team, seg.half)
    t, f = seg.start, np.array(f_start, dtype=float)
    ctrl.add(t, f[0], f[1])
    events = []
    first = True
    while True:
        hold = rng.integers(2, 9) if (first and restart_start) else int(round(rng.uniform(0.5, 2.8) * FPS))
        s = t + hold
        if s >= seg.end - 6:
            events.append((t, None, f[0], f[1]))
            break
        step = np.array([d * rng.uniform(-5.0, 13.0), rng.normal(0.0, 8.0)])
        fn = np.array([np.clip(f[0] + step[0], 3.0, PITCH_L - 3.0),
                       np.clip(f[1] + step[1], 3.0, PITCH_W - 3.0)])
        flight = float(np.clip(np.linalg.norm(fn - f) / rng.uniform(9.0, 18.0), 0.45, 2.2))
        r = s + int(round(flight * FPS))
        if r >= seg.end - 3:
            events.append((t, None, f[0], f[1]))
            break
        events.append((t, s, f[0], f[1]))
        ctrl.add(s, f[0], f[1])
        ctrl.add(r, fn[0], fn[1])
        t, f, first = r, fn, False
    ctrl.add(seg.end, f[0], f[1])
    seg.events = events
    return f


def _dead_geometry(rng, seg, f_exit, last_team_dir):
    """Choose restart team, exit point E and restart point R for a dead segment."""
    typ, t_ = seg.typ, seg.exit_team
    goal_x = PITCH_L if last_team_dir > 0 else 0.0
    if typ == "Throw-in":
        seg.team = 1 - t_ if rng.random() < 0.8 else t_
        x = float(np.clip(f_exit[0] + rng.normal(0, 5), 3.0, PITCH_L - 3.0))
        y = -0.4 if f_exit[1] < PITCH_W / 2 else PITCH_W + 0.4
        seg.E = seg.R = (x, y)
    elif typ == "Goal Kick":
        seg.team = 1 - t_
        seg.E = (goal_x + (0.8 if goal_x > 0 else -0.8), PITCH_W / 2 + rng.uniform(-6, 6))
        seg.R = (goal_x - (5.5 if goal_x > 0 else -5.5), PITCH_W / 2 + rng.uniform(-8, 8))
    elif typ == "Corner Kick":
        seg.team = t_
        y = 0.3 if f_exit[1] < PITCH_W / 2 else PITCH_W - 0.3
        x = goal_x - (0.3 if goal_x > 0 else -0.3)
        seg.E = seg.R = (x, y)
    elif typ == "Goal":
        seg.team = 1 - t_
        seg.E = (goal_x + (0.5 if goal_x > 0 else -0.5), PITCH_W / 2 + rng.normal(0, 1.5))
        seg.R = (PITCH_L / 2, PITCH_W / 2)
    else:  # Free Kick: the ball stops where it was
        seg.team = t_ if rng.random() < 0.5 else 1 - t_
        seg.E = seg.R = (float(f_exit[0]), float(f_exit[1]))


def simulate_script(rng, spec: MatchSpec):
    """Segments (possession / dead ball) for both halves and the focus control points."""
    centre = (PITCH_L / 2, PITCH_W / 2)
    ctrl = _Ctrl()
    ctrl.add(0, *centre)
    segs: list[Seg] = []
    goals = 0
    kickoff_team = int(rng.integers(0, 2))
    halves = ((1, spec.half1[0], spec.half1[1]), (2, spec.half2_play[0], spec.half2_play[1]))
    for half, start, end in halves:
        team = kickoff_team if half == 1 else 1 - kickoff_team
        if half == 2:
            segs.append(Seg(spec.half2_dead[0], spec.half2_dead[1], "dead", team, 2, typ="half_start",
                            exit_team=-1, has_restart=True, E=centre, R=centre))
            ctrl.add(spec.half2_dead[0], *centre)
            ctrl.add(spec.half2_dead[1], *centre)
        t = start
        f = np.array(centre)
        origin = "kickoff"
        while t <= end:
            seg = Seg(t, min(t + _poss_duration(rng, team) - 1, end), "poss", team, half, origin=origin)
            f = _poss_events(rng, seg, f, ctrl, origin != "turnover")
            segs.append(seg)
            t = seg.end + 1
            if t > end:
                break
            forced_goal = half == 1 and goals == 0 and t > start + 0.4 * (end - start)
            if forced_goal or rng.random() < P_DEAD:
                typ = "Goal" if forced_goal else str(rng.choice(DEAD_TYPES, p=DEAD_WEIGHTS))
                goals += typ == "Goal"
                lo, hi = (18.0, 30.0) if (typ == "Goal" and not spec.full_length) else DEAD_LEN_S[typ]
                length = int(round(rng.uniform(lo, hi) * FPS))
                dseg = Seg(t, min(t + length - 1, end), "dead", team, half, typ=typ, exit_team=team)
                dseg.has_restart = dseg.end < end
                _dead_geometry(rng, dseg, f, attack_dir(team, half))
                segs.append(dseg)
                ctrl.add(dseg.start, f[0], f[1])
                ctrl.add(min(dseg.start + 8 * FPS, dseg.end), *dseg.R)
                ctrl.add(dseg.end, *dseg.R)
                t = dseg.end + 1
                team, f, origin = dseg.team, np.array(dseg.R), "restart"
                if t > end:
                    break
            else:
                team, origin = 1 - team, "turnover"
        ctrl.add(end + 1, *centre)
    ctrl.add(spec.n_frames - 1, *centre)
    return segs, ctrl.pts


# --------------------------------------------------------------------------------------------
# Players, ball, tracks
# --------------------------------------------------------------------------------------------
def labels_from_segments(spec: MatchSpec, segs) -> np.ndarray:
    lab = np.full(spec.n_frames, "X", dtype="<U1")
    for s in segs:
        lab[s.start:s.end + 1] = "D" if s.kind == "dead" else ("H" if s.team == 0 else "A")
    lab[spec.half2_dead[0]:spec.half2_dead[1] + 1] = "D"
    return lab


def tracked_mask(spec: MatchSpec) -> np.ndarray:
    m = np.zeros(spec.n_frames, dtype=bool)
    m[max(0, spec.half1[0] - 1500):spec.half1[1] + 251] = True
    m[max(0, spec.half2_dead[0] - 250):min(spec.n_frames, spec.half2_play[1] + 501)] = True
    return m


def _smooth(a, sigma):
    return gaussian_filter1d(a, sigma, axis=0, mode="nearest")


def build_positions(rng, spec, segs, ctrl_pts):
    """Player positions P[T, team, 11, 2] in metres, ball focus F[T, 2], takers and flags."""
    T = spec.n_frames
    frames = np.arange(T)
    cf = np.array(ctrl_pts)
    F = np.stack([np.interp(frames, cf[:, 0], cf[:, 1]), np.interp(frames, cf[:, 0], cf[:, 2])], 1)
    F = _smooth(F, 1.2 * FPS)

    poss_team = np.full(T, -1, dtype=int)
    for s in segs:
        if s.kind == "poss":
            poss_team[s.start:s.end + 1] = s.team
    graded = np.zeros(T, dtype=bool)
    for s in segs:
        graded[s.start:s.end + 1] = True
    alive = (poss_team >= 0).astype(float)
    act = _smooth(0.4 + 0.6 * alive, 1.0 * FPS)

    mid = (spec.half1[1] + spec.half2_dead[0]) // 2
    half_of = np.where(frames <= mid, 1, 2)
    dsign = np.stack([np.where(half_of == 1, 1, -1), np.where(half_of == 1, -1, 1)], 1)

    P = np.zeros((T, 2, 11, 2))
    tt = frames / FPS
    for k in (0, 1):
        d = dsign[:, k]
        uf = np.where(d > 0, F[:, 0] / PITCH_L, 1 - F[:, 0] / PITCH_L)
        u = np.clip(0.10 + 0.78 * FORM_U[None, :] + 0.45 * (uf[:, None] - 0.5), 0.05, 0.97)
        u[:, 0] = 0.025 + 0.07 * uf
        x = np.where(d[:, None] > 0, u * PITCH_L, (1 - u) * PITCH_L)
        y = PITCH_W * (0.5 + (FORM_Y[None, :] - 0.5) * 0.9 + 0.25 * (F[:, 1:2] / PITCH_W - 0.5))
        x, y = np.broadcast_arrays(x, y)
        x, y = x.copy(), y.copy()
        per = rng.uniform(7.0, 22.0, (11, 2, 2))
        ph = rng.uniform(0.0, 2 * np.pi, (11, 2, 2))
        amp = rng.uniform(1.0, 2.2, (11, 2, 2))
        for c in range(2):
            arg_x = 2 * np.pi * tt[:, None] / per[None, :, 0, c] + ph[None, :, 0, c]
            arg_y = 2 * np.pi * tt[:, None] / per[None, :, 1, c] + ph[None, :, 1, c]
            x += act[:, None] * amp[None, :, 0, c] * np.sin(arg_x)
            y += act[:, None] * amp[None, :, 1, c] * np.sin(arg_y)
        ip = _smooth((poss_team == k).astype(float), 1.0 * FPS)
        dist = np.hypot(x - F[:, 0:1], y - F[:, 1:2])
        w = (0.18 + 0.27 * (1 - ip))[:, None] * np.exp(-(dist / 14.0) ** 2) * (0.4 + 0.6 * act)[:, None]
        x = x + w * (F[:, 0:1] - x)
        y = y + w * (F[:, 1:2] - y)
        P[:, k, :, 0] = np.clip(x, 0.5, PITCH_L - 0.5)
        P[:, k, :, 1] = np.clip(y, 0.5, PITCH_W - 0.5)

    # restart takers walk to the ball and stand on it until the restart
    takers = {}
    for i, s in enumerate(segs):
        if s.kind != "poss" or s.origin == "turnover":
            continue
        R = np.array((PITCH_L / 2, PITCH_W / 2)) if s.origin == "kickoff" else np.array(segs[i - 1].R)
        typ = "Kick Off" if s.origin == "kickoff" else segs[i - 1].typ
        t0 = s.start
        ref = max(0, t0 - 110)
        if typ == "Goal Kick":
            p = 0
        else:
            p = int(np.argmin(np.hypot(*(P[ref, s.team] - R).T)))
        takers[t0] = (s.team, p)
        lo = max(0, t0 - 110)
        up_end, hold_end, down_end = t0 - 10, min(T, t0 + 6), min(T, t0 + 56)
        w = np.zeros(T)
        if up_end > lo:
            w[lo:up_end] = np.linspace(0.0, 1.0, up_end - lo, endpoint=False)
        w[max(up_end, 0):hold_end] = 1.0
        if down_end > hold_end:
            w[hold_end:down_end] = np.linspace(1.0, 0.0, down_end - hold_end, endpoint=False)
        sl = slice(lo, down_end)
        P[sl, s.team, p, :] += w[sl, None] * (R[None, :] - P[sl, s.team, p, :])
    return P, F, takers, dsign, poss_team, graded


def _flight(B, p0, p1, t0, t1):
    n = t1 - t0
    if n <= 0:
        return
    w = (np.arange(n) + 1.0) / n
    B[t0:t1] = np.asarray(p0)[None, :] + (np.asarray(p1) - np.asarray(p0))[None, :] * w[:, None]


def build_ball(rng, spec, segs, P, takers, dsign):
    """True (fully visible) ball path B[T, 2] in metres; NaN where there is no ball."""
    T = spec.n_frames
    centre = np.array((PITCH_L / 2, PITCH_W / 2))
    B = np.full((T, 2), np.nan)
    a = spec.half1[0]
    B[max(0, a - 150):a] = centre
    cur = centre.copy()

    def hold(k, c, t0, t1, off):
        if t1 <= t0:
            return
        pos = P[t0:t1, k, c, :].copy()
        pos[:, 0] += off * dsign[t0:t1, k]
        B[t0:t1] = pos

    for seg in segs:
        s, e = seg.start, seg.end
        n = e - s + 1
        if seg.kind == "poss":
            k = seg.team
            first_hold = s
            if seg.origin == "turnover":
                c = int(np.argmin(np.hypot(*(P[s, k] - cur).T)))
                dist = float(np.linalg.norm(P[s, k, c] - cur))
                n0 = max(0, min(int(np.clip(dist / 14.0 * FPS, 3, 45)), n - 1))
                tgt = P[min(s + n0, T - 1), k, c].copy()
                tgt[0] += 0.35 * dsign[min(s + n0, T - 1), k]
                _flight(B, cur, tgt, s, s + n0)
                first_hold = s + n0
            else:
                c = takers[s][1]
            t_recv = first_hold
            for idx, (r_plan, sp_plan, _fx, _fy) in enumerate(seg.events):
                off = 0.0 if (idx == 0 and seg.origin != "turnover") else 0.35
                if sp_plan is None or idx + 1 >= len(seg.events):
                    hold(k, c, t_recv, e + 1, off)
                    break
                sp = t_recv + max(2, sp_plan - r_plan)
                if sp >= e - 12:
                    hold(k, c, t_recv, e + 1, off)
                    break
                hold(k, c, t_recv, sp, off)
                fx, fy = seg.events[idx + 1][2], seg.events[idx + 1][3]
                pp = B[sp - 1].copy()
                probe = min(sp + 12, T - 1)
                d_focus = np.hypot(*(P[probe, k] - np.array([fx, fy])).T)
                d_pass = np.hypot(*(P[probe, k] - pp).T)
                ranked = [int(i) for i in np.argsort(d_focus) if int(i) != c]
                order = [i for i in ranked if 2.0 <= d_pass[i] <= 38.0][:3] or [
                    min(ranked, key=lambda i: d_pass[i])]
                pw = np.array([0.5, 0.3, 0.2])[:len(order)]
                cn = int(rng.choice(order, p=pw / pw.sum()))
                v_pass = rng.uniform(10.0, 20.0)
                fl = int(np.clip(d_pass[cn] / v_pass * FPS, 10, 90))
                for _ in range(4):          # settle the flight time on the receiver's actual position
                    tgt = P[min(sp + fl, T - 1), k, cn].copy()
                    tgt[0] += 0.35 * dsign[min(sp + fl, T - 1), k]
                    fl_new = int(np.clip(np.ceil(np.linalg.norm(tgt - pp) / min(v_pass, 22.0) * FPS), 10, 90))
                    if fl_new == fl:
                        break
                    fl = fl_new
                r_next = sp + fl
                if r_next >= e - 3:
                    hold(k, c, sp, e + 1, off)
                    break
                tgt = P[r_next, k, cn].copy()
                tgt[0] += 0.35 * dsign[r_next, k]
                _flight(B, pp, tgt, sp, r_next)
                c, t_recv = cn, r_next
            cur = B[e].copy()
        else:
            if seg.typ == "half_start":
                B[s:e + 1] = centre
                cur = centre.copy()
            else:
                E, R = np.array(seg.E), np.array(seg.R)
                if seg.typ == "Free Kick":      # the ball stops where it was and is spotted afterwards
                    E = cur.copy()
                    fe = 0
                else:
                    fe = min(int(np.clip(np.linalg.norm(E - cur) / rng.uniform(10.0, 18.0) * FPS, 8, 90)), n)
                _flight(B, cur, E, s, s + fe)
                seg.goal_frame = s + fe
                if np.linalg.norm(E - R) < 1e-6:
                    B[s + fe:e + 1] = E
                else:
                    wait = int(rng.uniform(1.5, 5.0) * FPS)
                    rd = int(np.clip(np.linalg.norm(E - R) / 4.0 * FPS, 8, 400))
                    avail = (n - fe) - 20
                    if wait + rd > avail:
                        wait = max(0, int(avail * 0.3))
                        rd = max(1, avail - wait)
                    wait = min(wait, n - fe)
                    B[s + fe:s + fe + wait] = E
                    t1 = min(s + fe + wait + rd, e + 1)
                    _flight(B, E, R, s + fe + wait, t1)
                    B[t1:e + 1] = R
                cur = B[e].copy()
    return B


def hidden_ball_mask(rng, alive: np.ndarray) -> np.ndarray:
    """Contiguous hidden blocks (5-60 s) covering HIDDEN_BALL_FRACTION of the alive frames."""
    T = len(alive)
    target = int(round(HIDDEN_BALL_FRACTION * alive.sum()))
    hidden = np.zeros(T, dtype=bool)
    idx = np.flatnonzero(alive)
    covered, attempts = 0, 0
    while covered < target - 125 and attempts < 5000:
        attempts += 1
        u = rng.random()
        sec = rng.uniform(5.0, 20.0) if u < 0.40 else rng.uniform(20.0, 60.0)
        length = int(sec * FPS)
        start = int(idx[rng.integers(0, len(idx))])
        stop = min(start + length, T)
        if hidden[max(0, start - FPS):min(T, stop + FPS)].any():
            continue
        gain = int(alive[start:stop].sum())
        if covered + gain > target:
            # trim the block so the total hits the target
            need = target - covered
            cum = np.cumsum(alive[start:stop])
            stop = start + int(np.searchsorted(cum, need)) + 1
            gain = int(alive[start:stop].sum())
        hidden[start:stop] = True
        covered += gain
    return hidden


def _speed(pos):
    return np.linalg.norm(np.gradient(pos, axis=0), axis=-1) * FPS


def build_tactical_tracks(rng, spec, P, B_obs, ball_speed, tracked, alive):
    """Slot table X[T, 46, 3] (x, y, speed) with id breaks, swaps, duplicates and sideline staff."""
    T = spec.n_frames
    n_slots = sum(TACTICAL_SLOTS)
    speed = _speed(P)
    X = np.full((T, n_slots, 3), np.nan, dtype=np.float32)
    alive_idx = np.flatnonzero(alive & tracked)
    graded_s = max(1.0, alive.sum() / FPS)
    stats = {"id_breaks": 0, "id_swaps": 0, "duplicates": 0}
    track_slots = []
    dup_jobs = []
    for k in (0, 1):
        ts = np.tile(np.arange(11, dtype=np.int16), (T, 1))
        free = list(range(11, 20))
        n_events = max(2, int(round(graded_s / 400.0)))
        frames_ev = np.sort(rng.choice(alive_idx, size=n_events, replace=False))
        for f in frames_ev:
            f = int(f)
            if rng.random() < 0.6:
                p = int(rng.integers(0, 11))
                if ts[f, p] < 0 or not free:
                    continue
                old = int(ts[f, p])
                new = free.pop(int(rng.integers(0, len(free))))
                gap = int(rng.integers(0, 31))
                ts[f:, p] = new
                ts[f:f + gap, p] = -1
                free.append(old)
                stats["id_breaks"] += 1
            else:
                d = np.hypot(*(P[f, k, :, None, :] - P[f, k, None, :, :]).transpose(2, 0, 1))
                d[np.arange(11), np.arange(11)] = np.inf
                d[0, :] = d[:, 0] = np.inf
                p, q = np.unravel_index(int(np.argmin(d)), d.shape)
                tmp = ts[f:, p].copy()
                ts[f:, p] = ts[f:, q]
                ts[f:, q] = tmp
                stats["id_swaps"] += 1
        track_slots.append(ts)
        base = TACTICAL_SLOTS[0] * k
        for p in range(11):
            s = ts[:, p]
            m = (s >= 0) & tracked
            tt = np.flatnonzero(m)
            X[tt, base + s[tt], 0] = P[tt, k, p, 0] / PITCH_L
            X[tt, base + s[tt], 1] = P[tt, k, p, 1] / PITCH_W
            X[tt, base + s[tt], 2] = speed[tt, k, p]
        # sideline staff / substitutes (slots 20, 21): outside the pitch
        tsec = np.arange(T) / FPS
        for j, slot in enumerate((20, 21)):
            x0 = rng.uniform(0.2, 0.8) * PITCH_L
            y0 = -3.2 if j == 0 else PITCH_W + 3.2
            sx = x0 + 4.0 * np.sin(2 * np.pi * tsec / rng.uniform(60, 120))
            sy = y0 + 0.2 * np.sin(2 * np.pi * tsec / rng.uniform(20, 40))
            sp = np.hypot(np.gradient(sx), np.gradient(sy)) * FPS
            X[tracked, base + slot, 0] = (sx / PITCH_L)[tracked]
            X[tracked, base + slot, 1] = (sy / PITCH_W)[tracked]
            X[tracked, base + slot, 2] = sp[tracked]
        # duplicate tracks on free spare slots
        n_dup = max(2, int(round(graded_s / 500.0)))
        for _ in range(n_dup):
            f0 = int(rng.choice(alive_idx))
            f1 = min(T, f0 + int(rng.integers(40, 301)))
            p = int(rng.integers(1, 11))
            for slot in rng.permutation(np.arange(11, 20)):
                if not (ts[f0:f1] == slot).any() and not any(
                        j[0] == k and j[1] == int(slot) and j[2] < f1 and f0 < j[3] for j in dup_jobs):
                    dup_jobs.append((k, int(slot), f0, f1, p, rng.normal(0, 0.15, (f1 - f0, 2))))
                    break
    for k, slot, f0, f1, p, noise in dup_jobs:
        base = TACTICAL_SLOTS[0] * k
        xy = P[f0:f1, k, p, :] + noise
        m = tracked[f0:f1]
        X[f0:f1, base + slot, 0] = np.where(m, xy[:, 0] / PITCH_L, np.nan)
        X[f0:f1, base + slot, 1] = np.where(m, xy[:, 1] / PITCH_W, np.nan)
        X[f0:f1, base + slot, 2] = np.where(m, speed[f0:f1, k, p], np.nan)
        stats["duplicates"] += 1
    ball = np.full((T, 3), np.nan, dtype=np.float32)
    ball[:, 0] = B_obs[:, 0] / PITCH_L
    ball[:, 1] = B_obs[:, 1] / PITCH_W
    ball[:, 2] = ball_speed
    return X, ball, stats


# --------------------------------------------------------------------------------------------
# Broadcast feed: offset, cropped, with cuts
# --------------------------------------------------------------------------------------------
def build_broadcast(rng, spec, P, B_true, Bspeed, F, tracked):
    n_b, T = spec.n_broadcast, spec.n_frames
    a, b = spec.half1
    c0, d1 = spec.half2[0], spec.half2[1]
    rem = max(0, n_b - (b - a + 1) - (d1 - c0 + 1))
    pre1, post1, pre2 = int(round(0.30 * rem)), int(round(0.15 * rem)), int(round(0.25 * rem))
    post2 = rem - pre1 - post1 - pre2
    p1 = (max(0, a - pre1), min(T - 1, b + post1))
    p2 = (max(p1[1] + 1, c0 - pre2), min(T - 1, d1 + post2))
    base = np.concatenate([np.arange(p1[0], p1[1] + 1), np.arange(p2[0], p2[1] + 1)])
    base = base[:n_b] if len(base) >= n_b else np.concatenate([base, np.full(n_b - len(base), base[-1])])

    tau = np.zeros(n_b, dtype=np.int64)
    live = np.zeros(n_b, dtype=bool)
    block_id = np.full(n_b, -1, dtype=np.int64)
    j, bid = 0, 0
    max_live = max(150, n_b // 8)
    while j < n_b:
        length = int(min(rng.integers(150, max_live + 1), n_b - j))
        delta = int(rng.integers(-60, 61) if bid == 0 else rng.integers(-150, 151))
        sl = slice(j, j + length)
        tau[sl] = np.clip(base[sl] + delta, 0, T - 1)
        live[sl] = True
        block_id[sl] = bid
        j += length
        bid += 1
        cut = int(min(rng.integers(40, 251), n_b - j))
        if cut > 0:
            tau[j:j + cut] = -1
            j += cut
    live &= tracked[np.maximum(tau, 0)]

    n_slots = sum(BROADCAST_SLOTS)
    X = np.full((n_b, n_slots, 3), np.nan, dtype=np.float32)
    ball = np.full((n_b, 3), np.nan, dtype=np.float32)
    Hmat = np.full((n_b, 3, 3), np.nan)
    err = np.zeros(n_b)
    speed = _speed(P)
    t_safe = np.maximum(tau, 0)
    xc = F[t_safe, 0] / PITCH_L
    wv = 0.40 + 0.04 * np.sin(np.arange(n_b) / 400.0)
    for bid_ in range(bid):
        idx = np.flatnonzero((block_id == bid_) & live)
        if len(idx) == 0:
            continue
        tt = tau[idx]
        for k in (0, 1):
            slots = rng.permutation(BROADCAST_SLOTS[k])[:11] + sum(BROADCAST_SLOTS[:k])
            for p in range(11):
                pos = P[tt, k, p, :] + rng.normal(0, 0.12, (len(idx), 2))
                vis = np.abs(pos[:, 0] / PITCH_L - xc[idx]) <= wv[idx] / 2
                X[idx, slots[p], 0] = np.where(vis, pos[:, 0] / PITCH_L, np.nan)
                X[idx, slots[p], 1] = np.where(vis, pos[:, 1] / PITCH_W, np.nan)
                X[idx, slots[p], 2] = np.where(vis, speed[tt, k, p], np.nan)
        bp = B_true[tt]
        vis = np.abs(bp[:, 0] / PITCH_L - xc[idx]) <= wv[idx] / 2
        if rng.random() < 0.35 and len(idx) > 100:
            s0 = int(rng.integers(0, len(idx) - 50))
            vis[s0:s0 + int(rng.integers(25, 200))] = False
        ball[idx, 0] = np.where(vis, bp[:, 0] / PITCH_L, np.nan)
        ball[idx, 1] = np.where(vis, bp[:, 1] / PITCH_W, np.nan)
        ball[idx, 2] = np.where(vis, Bspeed[tt], np.nan)

    p0 = homography_from_corners(
        [(120, 260), (1800, 260), (1950, 1000), (-30, 1000)])
    x0 = xc - wv / 2
    A = np.zeros((n_b, 3, 3))
    A[:, 0, 0] = 1.0 / wv
    A[:, 0, 2] = -x0 / wv
    A[:, 1, 1] = 1.0
    A[:, 2, 2] = 1.0
    Hb = np.einsum("ij,njk->nik", p0, A)
    Hmat[live] = Hb[live]
    err[live] = rng.lognormal(np.log(0.8), 0.5, live.sum())
    err[~live] = rng.uniform(40.0, 304.0, (~live).sum())
    return X, ball, Hmat, err, live


def homography_from_corners(px):
    """3x3 matrix mapping the unit square corners (0,0),(1,0),(1,1),(0,1) to pixels ``px``."""
    src = [(0, 0), (1, 0), (1, 1), (0, 1)]
    rows, rhs = [], []
    for (x, y), (u, v) in zip(src, px):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        rhs.append(u)
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        rhs.append(v)
    h = np.linalg.solve(np.array(rows, dtype=float), np.array(rhs, dtype=float))
    return np.append(h, 1.0).reshape(3, 3)


def build_tactical_homography(rng, spec, tracked):
    T = spec.n_frames
    h0 = homography_from_corners([(330, 210), (1590, 210), (1900, 1020), (20, 1020)])
    t = np.arange(T)
    du = 0.8 * np.sin(2 * np.pi * t / 900.0)
    dv = 0.5 * np.sin(2 * np.pi * t / 1300.0)
    H = np.tile(h0, (T, 1, 1))
    H[:, 0, :] += du[:, None] * h0[2][None, :]
    H[:, 1, :] += dv[:, None] * h0[2][None, :]
    err = np.where(tracked, rng.lognormal(np.log(0.24), 0.35, T), rng.lognormal(np.log(1.5), 0.3, T))
    return H, err, np.ones(T, dtype=bool)


# --------------------------------------------------------------------------------------------
# Writers
# --------------------------------------------------------------------------------------------
_XY_TAB = np.array([f"{v / 100000:.5f}" for v in range(-10000, 110001)] + ["NaN"], dtype=object)
_SP_TAB = np.array([f"{v / 100:.2f}" for v in range(0, 6001)] + ["NaN"], dtype=object)


def _xy_idx(v):
    idx = np.rint(np.clip(np.nan_to_num(v, nan=0.0), -0.1, 1.1) * 100000).astype(np.int64) + 10000
    return np.where(np.isnan(v), len(_XY_TAB) - 1, idx)


def _sp_idx(v):
    idx = np.rint(np.clip(np.nan_to_num(v, nan=0.0), 0.0, 60.0) * 100).astype(np.int64)
    return np.where(np.isnan(v), len(_SP_TAB) - 1, idx)


def _group_strings(arr):
    arr = arr.astype(np.float64)
    x, y, sp = _XY_TAB[_xy_idx(arr[..., 0])], _XY_TAB[_xy_idx(arr[..., 1])], _SP_TAB[_sp_idx(arr[..., 2])]
    return x + "," + y + "," + sp


def write_atd_txt(path: Path, X: np.ndarray, ball: np.ndarray, chunk: int = 4000):
    T = len(X)
    with open(path, "w", encoding="ascii", newline="\n") as fh:
        for c0 in range(0, T, chunk):
            c1 = min(T, c0 + chunk)
            groups = _group_strings(X[c0:c1])
            balls = _group_strings(ball[c0:c1])
            fh.write("".join(
                f"{c0 + i + 1}:{';'.join(groups[i])}:{balls[i]}\n" for i in range(c1 - c0)))


def _xml_header():
    return '<?xml version="1.0" encoding="UTF-8"?>\n'


def atd_xml(spec_frames, slots, scores, periods=None):
    """FIFA/EPTS metadata (Metrica flavour: Metadata + DataFormatSpecifications)."""
    n_slots = sum(slots)
    o = [_xml_header(), "<main>\n  <Metadata>\n    <GlobalConfig>\n"]
    o.append("      <ProviderName>Metrica Sports</ProviderName>\n")
    o.append("      <AnonymizedProvider>true</AnonymizedProvider>\n")
    o.append(f"      <FrameRate>{FPS}</FrameRate>\n      <ProviderGlobalParameters>\n")
    if periods:
        for name, value in periods:
            o.append(f"        <ProviderParameter><Name>{name}</Name><Value>{value}</Value>"
                     "</ProviderParameter>\n")
    o.append("      </ProviderGlobalParameters>\n    </GlobalConfig>\n    <Sessions>\n")
    o.append('      <Session id="1">\n')
    o.append(f"        <FieldSize><Width>{PITCH_L}</Width><Height>{PITCH_W}</Height></FieldSize>\n")
    o.append("        <MatchParameters>\n")
    o.append(f'          <Score idLocalTeam="{TEAM_IDS[0]}" idVisitingTeam="{TEAM_IDS[1]}">'
             f"<LocalTeamScore>{scores[0]}</LocalTeamScore><VisitingTeamScore>{scores[1]}</VisitingTeamScore></Score>\n")
    o.append("        </MatchParameters>\n      </Session>\n    </Sessions>\n    <Teams>\n")
    for k in (0, 1):
        o.append(f'      <Team id="{TEAM_IDS[k]}"><Name>{TEAM_NAMES[k]}</Name>')
        if k == 0 and periods:
            o.append("<ProviderTeamsParameters><ProviderParameter><Name>attack_direction_first_half</Name>"
                     "<Value>left_to_right</Value></ProviderParameter></ProviderTeamsParameters>")
        o.append("</Team>\n")
    o.append("    </Teams>\n    <Players>\n")
    slot_team = [TEAM_IDS[0]] * slots[0] + [TEAM_IDS[1]] * slots[1]
    for i in range(n_slots):
        o.append(f'      <Player id="Track{i + 1}" teamId="{slot_team[i]}"><Name>Track {i + 1}</Name>'
                 f"<ShirtNumber>{i + 1}</ShirtNumber></Player>\n")
    o.append('    </Players>\n    <Devices>\n      <Device id="device1"><Name>Optical tracking</Name>'
             '<Sensors>\n')
    o.append('        <Sensor id="position"><Name>Position</Name><Channels>'
             '<Channel id="x"><Name>x</Name><Unit>normalized</Unit></Channel>'
             '<Channel id="y"><Name>y</Name><Unit>normalized</Unit></Channel></Channels></Sensor>\n')
    o.append('        <Sensor id="speed"><Name>Speed</Name><Channels>'
             '<Channel id="speed"><Name>speed</Name><Unit>m/s</Unit></Channel></Channels></Sensor>\n')
    o.append("      </Sensors></Device>\n    </Devices>\n    <PlayerChannels>\n")
    for i in range(n_slots):
        for ch in ("x", "y", "speed"):
            o.append(f'      <PlayerChannel id="Track{i + 1}_{ch}" playerId="Track{i + 1}" '
                     f'channelId="{ch}"/>\n')
    o.append("    </PlayerChannels>\n  </Metadata>\n  <DataFormatSpecifications>\n")
    o.append(f'    <DataFormatSpecification startFrame="1" endFrame="{spec_frames}" separator=":">\n')
    o.append('      <StringRegister name="frameCount"/>\n')
    o.append('      <SplitRegister separator=";">\n')
    for i in range(n_slots):
        refs = "".join(f'<PlayerChannelRef playerChannelId="Track{i + 1}_{ch}"/>'
                       for ch in ("x", "y", "speed"))
        o.append(f'        <SplitRegister separator=",">{refs}</SplitRegister>\n')
    o.append("      </SplitRegister>\n")
    o.append('      <SplitRegister separator=",">'
             '<BallChannelRef channelId="x"/><BallChannelRef channelId="y"/>'
             '<BallChannelRef channelId="speed"/>'
             "</SplitRegister>\n")
    o.append("    </DataFormatSpecification>\n  </DataFormatSpecifications>\n</main>\n")
    return "".join(o)


def write_homography(path: Path, H: np.ndarray, err: np.ndarray, valid: np.ndarray):
    cols = [f"h{i}{j}" for i in range(3) for j in range(3)]
    df = pd.DataFrame(H.reshape(-1, 9), columns=cols)
    df.insert(0, "frame", np.arange(len(H)))
    df["reprojection_error"] = err
    df["valid"] = valid.astype(int)
    df.to_csv(path, index=False, float_format="%.9g", na_rep="NaN", lineterminator="\n")


def smart_tagging_instances(rng, spec, segs):
    """SportsCode instances: set pieces, goals and phases (times in seconds from video start)."""
    inst = []

    def half_name(h):
        return "1st Half" if h == 1 else "2nd Half"

    def add(t0, t1, code, team, half, typ):
        inst.append({"start": max(0.0, t0 / FPS), "end": t1 / FPS, "code": code,
                     "labels": [("Team", team), ("Half", half_name(half)), ("Type", typ)]})

    for i, s in enumerate(segs):
        if s.kind == "poss":
            if s.origin == "kickoff":
                add(s.start - 3.5 * FPS, s.start + 1.5 * FPS, "SET PIECES", TEAM_NAMES[s.team], s.half,
                    "Kick Off (Start)")
            phases = rng.integers(0, len(PHASES), size=max(1, (s.end - s.start) // (8 * FPS)))
            pts = np.linspace(s.start, s.end + 1, len(phases) + 1)
            for j, ph in enumerate(phases):
                j0 = pts[j] + rng.uniform(-1.0, 0.5) * FPS
                j1 = pts[j + 1] + rng.uniform(-0.5, 1.0) * FPS
                add(j0, max(j1, j0 + FPS), PHASES[ph], TEAM_NAMES[s.team], s.half, PHASES[ph].title())
        else:
            if s.typ == "half_start":
                continue
            if s.typ == "Goal":
                g = s.goal_frame if s.goal_frame >= 0 else s.start
                add(g - 2 * FPS, g + 4 * FPS, "GOALS", TEAM_NAMES[s.exit_team], s.half, "Goal")
            if not s.has_restart:
                continue
            t_star = s.end + 1
            typ = "Kick Off (After Goal)" if s.typ == "Goal" else s.typ
            team = "N/A" if s.typ == "Throw-in" else TEAM_NAMES[s.team]
            add(t_star - rng.uniform(2.5, 6.0) * FPS, t_star + rng.uniform(1.0, 2.5) * FPS, "SET PIECES",
                team, s.half, typ)
    inst.sort(key=lambda d: (d["start"], d["code"]))
    return inst


def write_smart_tagging(path: Path, instances):
    o = [_xml_header(), "<file>\n  <SESSION_INFO><start_time>2026-01-01 00:00:00</start_time>"
         "</SESSION_INFO>\n", "  <ALL_INSTANCES>\n"]
    for n, d in enumerate(instances, start=1):
        o.append(f"    <instance>\n      <ID>{n}</ID>\n      <start>{d['start']:.2f}</start>\n"
                 f"      <end>{d['end']:.2f}</end>\n      <code>{d['code']}</code>\n")
        for group, text in d["labels"]:
            o.append(f"      <label><group>{group}</group><text>{text}</text></label>\n")
        o.append("    </instance>\n")
    o.append("  </ALL_INSTANCES>\n</file>\n")
    Path(path).write_text("".join(o), encoding="utf-8", newline="\n")


# --------------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------------
def generate_synthetic_match(out_dir, seed: int = 0, n_frames: int = 12000,
                             full_length: bool = False) -> dict:
    """Write a synthetic BAR-ATH bundle under ``out_dir`` and return a summary dict."""
    spec = make_spec(n_frames, full_length)
    out = Path(out_dir)
    tdir, bdir = out / "tactical", out / "broadcast"
    tdir.mkdir(parents=True, exist_ok=True)
    bdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    segs, ctrl = simulate_script(rng, spec)
    P, F, takers, dsign, poss_team, _graded = build_positions(rng, spec, segs, ctrl)
    B_true = build_ball(rng, spec, segs, P, takers, dsign)
    labels = labels_from_segments(spec, segs)
    alive = (labels == "H") | (labels == "A")
    tracked = tracked_mask(spec)
    hidden = hidden_ball_mask(rng, alive)
    B_obs = np.where(hidden[:, None], np.nan, B_true)
    Bspeed = _speed(B_true)
    Bspeed_obs = np.where(hidden, np.nan, Bspeed)

    X, ball, stats = build_tactical_tracks(rng, spec, P, B_obs, Bspeed_obs, tracked, alive)
    H, herr, hvalid = build_tactical_homography(rng, spec, tracked)
    Xb, ballb, Hb, errb, livemask = build_broadcast(rng, spec, P, B_true, Bspeed, F, tracked)

    goals = [s for s in segs if s.kind == "dead" and s.typ == "Goal"]
    scores = [sum(1 for s in goals if s.exit_team == k) for k in (0, 1)]
    periods = [("first_half_start", spec.half1[0] + 1), ("first_half_end", spec.half1[1] + 1),
               ("second_half_start", spec.half2[0] + 1), ("second_half_end", spec.half2[1] + 1)]
    mid_b = spec.n_broadcast // 2
    b_periods = [("first_half_start", 1), ("first_half_end", mid_b),
                 ("second_half_start", mid_b + 1), ("second_half_end", spec.n_broadcast)]
    paths = {
        "tactical_atd_xml": tdir / f"{GAME}_tactical_atd.xml",
        "tactical_atd_txt": tdir / f"{GAME}_tactical_atd.txt",
        "tactical_homography": tdir / f"{GAME}_tactical_homography.csv",
        "tactical_smart_tagging": tdir / f"{GAME}_tactical_smart_tagging.xml",
        "broadcast_atd_xml": bdir / f"{GAME}_broadcast_atd.xml",
        "broadcast_atd_txt": bdir / f"{GAME}_broadcast_atd.txt",
        "broadcast_homography": bdir / f"{GAME}_broadcast_homography.csv",
        "ground_truth": out / "synthetic_gt.csv",
    }
    Path(paths["tactical_atd_xml"]).write_text(
        atd_xml(spec.n_frames, TACTICAL_SLOTS, scores, periods), encoding="utf-8", newline="\n")
    write_atd_txt(paths["tactical_atd_txt"], X, ball)
    write_homography(paths["tactical_homography"], H, herr, hvalid)
    instances = smart_tagging_instances(rng, spec, segs)
    write_smart_tagging(paths["tactical_smart_tagging"], instances)
    Path(paths["broadcast_atd_xml"]).write_text(
        atd_xml(spec.n_broadcast, BROADCAST_SLOTS, scores, b_periods), encoding="utf-8", newline="\n")
    write_atd_txt(paths["broadcast_atd_txt"], Xb, ballb)
    write_homography(paths["broadcast_homography"], Hb, errb, livemask)
    pd.DataFrame({"frame": np.arange(spec.n_frames), "label": labels}).to_csv(
        paths["ground_truth"], index=False, lineterminator="\n")

    return {
        "spec": spec,
        "paths": {k: str(v) for k, v in paths.items()},
        "labels": labels,
        "n_goals": len(goals),
        "n_set_pieces": sum(1 for d in instances if d["code"] == "SET PIECES"),
        "n_throw_ins": sum(1 for d in instances if d["labels"][2][1] == "Throw-in"),
        "ball_hidden_fraction_alive": float(hidden[alive].mean()),
        "track_artifacts": stats,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="output directory (bundle root)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-frames", type=int, default=12000, help="tactical frames of the short match")
    ap.add_argument("--full-length", action="store_true", help="exactly 199,319 frames, official boundaries")
    args = ap.parse_args(argv)
    info = generate_synthetic_match(args.out, args.seed, args.n_frames, args.full_length)
    summary = {k: v for k, v in info.items() if k not in ("labels", "spec")}
    summary["n_frames"] = info["spec"].n_frames
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
