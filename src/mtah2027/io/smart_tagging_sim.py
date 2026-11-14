# ruff: noqa: E501
"""Deterministic Smart Tagging simulator for the external matches (Issue #43).

The external benchmark matches carry no Smart Tagging, so the value of Smart Tagging priors cannot be measured on
them directly.  This module derives a *simulated* tagging from the true labels of a match with the structure that
the BAR-ATH file has (aggregates only, ``scripts/smart_tagging_stats.py``) and then corrupts it:

* every alive possession run of at least ``min_run_s`` becomes one or more *attacking* phases of the possessing team
  (``Build Up``, ``Progression``, ``Creating Chances`` ... chunks of ``min_phase_s``-``max_phase_s``), each paired
  with the matching ``Defending in ... Third`` phase tagged with the *opponent* (``Team`` is the team performing
  the event, as in the real XML); unpaired phases (``Ball in Final Third`` ...) are added with probability
  ``extra_phase_prob``; a possession change gets an ``Attacking Transition`` / ``Defensive Transition`` pair with
  probability ``transition_prob``;
* every dead-ball run of at least ``setpiece_min_dead_s`` followed by play is a restart; a random subset (rate
  ``setpiece_per_10min``) becomes a ``SET PIECES`` tag with a ``set_piece_window_s`` window that contains the
  restart (throw-ins get ``Team = N/A``) and the kick-off of every period a 5 s ``Kick Off (Start)`` window;
* noise: Gaussian boundary jitter (``jitter_s``), missed tags (``fn_rate``), spurious tags with random team and
  position (``fp_rate``, relative to the true counts), and a *dead-time bleed*: the phase ends are extended by a
  random tail until the union coverage of the graded frames reaches ``target_cover`` (the BAR-ATH value), so a
  realistic share of the tagged frames are in fact dead.

CRUCIAL CAVEAT: the clean tagging is derived from the true labels, so any gain measured with it is optimistic; a
result is only credible if it survives the medium and high noise levels (``NOISE_LEVELS``).  The actual accuracy of
the real Smart Tagging is unknown (it is "inferred automatically", ``OVERVIEW.md``).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from mtah2027.io.smart_tagging_loader import SET_PIECE_CODE, TagEvent, build_smart_tagging

H, A, D = 0, 1, 2

# (attacking code, defending counterpart, weight): the counterparts pair 1:1 in the real file
PAIRED = (("build up", "defending in attacking third", 0.2),
          ("progression", "defending in middle third", 0.5),
          ("creating chances", "defending in defensive third", 0.3))
UNPAIRED = ("ball in final third", "ball in the box", "players in the box", "long ball")
SET_PIECE_TYPES = (("throw-in", 26), ("corner", 12), ("goal kick", 10), ("free kick", 5))


@dataclass(frozen=True)
class SimParams:
    jitter_s: float = 3.0            # sd of the Gaussian shift of every boundary
    fn_rate: float = 0.25            # probability that a true tag is missing
    fp_rate: float = 0.15            # spurious tags, as a fraction of the true count
    target_cover: float = 0.69       # BAR-ATH: union of the phases over the graded frames
    min_run_s: float = 4.0
    min_phase_s: float = 9.0
    max_phase_s: float = 30.0
    extra_phase_prob: float = 0.8
    transition_prob: float = 0.10
    transition_s: tuple = (10.0, 20.0)
    setpiece_per_10min: float = 5.7  # BAR-ATH: 59 set pieces in 104 graded minutes
    setpiece_min_dead_s: float = 5.0
    set_piece_window_s: float = 35.0
    max_tail_s: float = 60.0
    na_rate: float = 0.02            # phases with Team = N/A (BAR-ATH: 10 of 575 phases; every throw-in is N/A too)


NOISE_LEVELS: dict[str, SimParams] = {
    "low": SimParams(jitter_s=1.0, fn_rate=0.10, fp_rate=0.05),
    "medium": SimParams(jitter_s=3.0, fn_rate=0.25, fp_rate=0.15),
    "high": SimParams(jitter_s=6.0, fn_rate=0.40, fp_rate=0.30),
}
CLEAN = SimParams(jitter_s=0.0, fn_rate=0.0, fp_rate=0.0, target_cover=0.0)


def _runs(codes: np.ndarray, a: int, b: int):
    """Runs ``(label, start, end_exclusive)`` of the labels in the inclusive period ``[a, b]``."""
    seg = codes[a:b + 1]
    if len(seg) == 0:
        return []
    cut = np.flatnonzero(np.diff(seg) != 0) + 1
    starts = np.concatenate([[0], cut])
    ends = np.concatenate([cut, [len(seg)]])
    return [(int(seg[s]), a + int(s), a + int(e)) for s, e in zip(starts, ends)]


def _event(code, s, e, team, fps, half, typ=""):
    return TagEvent(code=code, start_frame=int(s), end_frame=int(e), start_s=s / fps, end_s=e / fps, team=team,
                    half=half, type=typ, tags={})


def clean_events(codes: np.ndarray, periods, fps: float, p: SimParams, rng: np.random.Generator):
    """Noise-free tags derived from the labels: ``(phases, set_pieces)`` as lists of TagEvent."""
    phases: list[TagEvent] = []
    sets: list[TagEvent] = []
    pair_w = np.array([w for *_, w in PAIRED])
    min_run, min_ph, max_ph = int(p.min_run_s * fps), int(p.min_phase_s * fps), int(p.max_phase_s * fps)
    cand: list[tuple[int, int]] = []              # restarts: (t*, team)
    for half, (a, b) in enumerate(periods, start=1):
        runs = _runs(codes, int(a), int(b))
        prev_alive = None
        for k, (lab, s, e) in enumerate(runs):
            if lab not in (H, A):
                continue
            team = lab
            if e - s >= min_run:
                n_chunks = max(1, int(np.ceil((e - s) / max_ph)))
                edges = np.linspace(s, e, n_chunks + 1).astype(int)
                for c in range(n_chunks):
                    cs, ce = int(edges[c]), max(int(edges[c + 1]), int(edges[c]) + min_ph)
                    i = int(rng.choice(len(PAIRED), p=pair_w / pair_w.sum()))
                    att, dfn, _ = PAIRED[i]
                    phases.append(_event(att, cs, ce, team, fps, half))
                    phases.append(_event(dfn, cs, ce, 1 - team, fps, half))
                    if rng.random() < p.extra_phase_prob:
                        phases.append(_event(UNPAIRED[int(rng.integers(len(UNPAIRED)))], cs, ce, team,
                                             fps, half))
                if prev_alive is not None and prev_alive != team and rng.random() < p.transition_prob:
                    dur = int(rng.uniform(*p.transition_s) * fps)
                    phases.append(_event("attacking transition", s, s + dur, team, fps, half))
                    phases.append(_event("defensive transition", s, s + dur, 1 - team, fps, half))
            prev_alive = team
            if k > 0 and runs[k - 1][0] == D and runs[k - 1][2] - runs[k - 1][1] >= p.setpiece_min_dead_s * fps:
                cand.append((s, team))
        # kick-off of the period: window of 5 s that opens 3.5 s before the first alive frame
        first = next((r for r in runs if r[0] in (H, A)), None)
        if first is not None:
            t0 = first[1]
            sets.append(_event(SET_PIECE_CODE, t0 - int(3.5 * fps), t0 + int(1.5 * fps), -1, fps, half,
                               "kick off (start)"))
    n_graded = sum(int(b) - int(a) + 1 for a, b in periods)
    want = p.setpiece_per_10min * n_graded / fps / 600.0
    prob = min(1.0, want / max(len(cand), 1))
    w = np.array([x for _, x in SET_PIECE_TYPES], float)
    win = int(p.set_piece_window_s * fps)
    for t_star, team in cand:
        if rng.random() >= prob:
            continue
        typ = SET_PIECE_TYPES[int(rng.choice(len(SET_PIECE_TYPES), p=w / w.sum()))][0]
        start = t_star - int(rng.uniform(3.0, 28.0) * fps)
        half = 1 if any(a <= t_star <= b for a, b in periods[:1]) else 2
        sets.append(_event(SET_PIECE_CODE, start, start + win, -1 if typ == "throw-in" else team, fps, half, typ))
    return phases, sets


def _coverage(events, n, graded):
    diff = np.zeros(n + 1, np.int32)
    for e in events:
        lo, hi = max(0, min(e.start_frame, n)), max(0, min(e.end_frame, n))
        if hi > lo:
            diff[lo] += 1
            diff[hi] -= 1
    return float(((np.cumsum(diff[:n]) > 0) & graded).sum()) / max(int(graded.sum()), 1)


def simulate_smart_tagging(codes: np.ndarray, periods, fps: float, *, noise: str | SimParams = "medium",
                           seed: int = 0, n_frames: int | None = None):
    """Simulated :class:`~mtah2027.io.smart_tagging_loader.SmartTagging` of a match with labels ``codes``.

    ``codes`` are ints (H=0, A=1, D=2, X=3), ``periods`` the inclusive graded blocks.  ``noise`` is a key of
    ``NOISE_LEVELS`` (``low`` / ``medium`` / ``high``), ``"clean"`` or a :class:`SimParams`.  Deterministic for a
    given ``seed``."""
    p = CLEAN if isinstance(noise, str) and noise == "clean" else NOISE_LEVELS[noise] if isinstance(noise, str) else noise
    codes = np.asarray(codes)
    n = int(n_frames if n_frames is not None else len(codes))
    periods = [(int(a), int(b)) for a, b in periods]
    rng = np.random.default_rng(int(seed) * 7919 + 17)
    graded = np.zeros(n, bool)
    for a, b in periods:
        graded[a:b + 1] = True
    phases, sets = clean_events(codes, periods, fps, p, rng)
    n_true_ph, n_true_sp = len(phases), len(sets)
    durs = np.array([e.end_frame - e.start_frame for e in phases]) if phases else np.array([int(20 * fps)])

    def jit() -> int:
        return int(round(rng.normal(0.0, p.jitter_s) * fps)) if p.jitter_s > 0 else 0

    def noisy(events, whole_window: bool):
        out = []
        for e in events:
            if rng.random() < p.fn_rate:
                continue
            if whole_window:
                d = jit()
                s, t = e.start_frame + d, e.end_frame + d
            else:
                s, t = e.start_frame + jit(), e.end_frame + jit()
                t = max(t, s + int(3 * fps))
            team = -1 if (not whole_window and rng.random() < p.na_rate) else e.team
            out.append(_event(e.code, s, t, team, fps, e.half, e.type))
        return out

    ph = noisy(phases, False)
    sp = noisy(sets, True)
    # spurious tags: random place and team, true duration distribution
    codes_all = [c for c, *_ in PAIRED] + [d for _, d, _ in PAIRED] + list(UNPAIRED)
    inside = np.flatnonzero(graded)
    for _ in range(int(round(p.fp_rate * n_true_ph))):
        if len(inside) == 0:
            break
        s = int(rng.choice(inside))
        d = int(rng.choice(durs))
        ph.append(_event(codes_all[int(rng.integers(len(codes_all)))], s, s + d, int(rng.integers(2)), fps, 1))
    for _ in range(int(round(p.fp_rate * n_true_sp))):
        if len(inside) == 0:
            break
        s = int(rng.choice(inside)) - int(rng.uniform(3.0, 28.0) * fps)
        typ = SET_PIECE_TYPES[int(rng.integers(len(SET_PIECE_TYPES)))][0]
        sp.append(_event(SET_PIECE_CODE, s, s + int(p.set_piece_window_s * fps), -1 if typ == "throw-in"
                         else int(rng.integers(2)), fps, 1, typ))
    # dead-time bleed: lengthen the phase ends until the union coverage reaches the BAR-ATH value
    if p.target_cover > 0 and ph:
        u = np.minimum(rng.exponential(1.0, len(ph)), 4.0)

        def with_tail(scale: float):
            return [replace(e, end_frame=e.end_frame + int(scale * ui * fps)) for e, ui in zip(ph, u)]

        if _coverage(ph, n, graded) < p.target_cover:
            lo, hi = 0.0, p.max_tail_s
            for _ in range(24):
                mid = 0.5 * (lo + hi)
                if _coverage(with_tail(mid), n, graded) < p.target_cover:
                    lo = mid
                else:
                    hi = mid
            ph = with_tail(hi)
    return build_smart_tagging(ph + sp, n, int(round(fps)))
