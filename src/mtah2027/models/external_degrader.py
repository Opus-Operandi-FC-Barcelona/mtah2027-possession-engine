"""Realistic degradation of clean external matches (Issue #19).

A clean open-data match (complete ball, stable player slots) is turned into something that looks like the
competition automatic tracking, so that the engine can be validated -- and, later, trained -- on matches
other than BAR-ATH under the conditions it will face:

1. **Ball occlusion blocks.**  The ball is hidden in contiguous blocks until ``hide_frac`` of the *alive*
   frames (``gt`` in H/A) is covered.  Block durations are drawn from the empirical ATD gap mix (5-20 s 33 %,
   20-60 s 48 %, > 60 s 19 %; log-uniform inside each class).  The start of every block is sampled with a
   probability proportional to ``(floor + mean crowding) ** gamma`` where the crowding is the number of
   players within ``density_radius_m`` of the (clean) ball -- crowded situations lose the ball more often.
   Blocks never cross a break between graded periods (``gt == X``) and never overlap.
2. **Tracking noise**: temporally correlated (AR(1), correlation time ``noise_corr_s``) Gaussian position
   jitter on players and ball.  White jitter would be unrealistic -- provider tracking is already smoothed --
   and destroys every acceleration-based cue.
3. **Identity cuts**: Poisson-distributed cut times; at each one a few players of the same team have their
   slots permuted from then on (identity switch) and are lost for a short gap (fragmentation).
4. **Duplicate tracks**: an unused slot of a team temporarily mirrors another player (+ small offset).
5. **Dropped players**: a track disappears for a few seconds.

Everything is deterministic given ``seed``.  The ground truth is *not* modified (it describes the real play).
Only aggregate numbers are reported in ``DegradeReport``.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import lfilter

from mtah2027.config import FPS, load_section, params_from_mapping
from mtah2027.features._utils import mask_runs
from mtah2027.validation.eval_bridge import to_codes

H, A, D, X = 0, 1, 2, 3


@dataclass(frozen=True)
class DegraderParams:
    hide_frac: float = 0.41
    dur_edges_s: tuple = (5.0, 20.0, 60.0, 120.0)
    dur_probs: tuple = (0.33, 0.48, 0.19)
    density_radius_m: float = 5.0
    density_floor: float = 0.25
    density_gamma: float = 1.0
    noise_player_m: float = 0.12
    noise_ball_m: float = 0.08
    noise_corr_s: float = 0.5
    cut_rate_per_min: float = 0.6
    cut_players: tuple = (2, 4)
    cut_gap_s: tuple = (0.2, 2.0)
    dup_rate_per_min: float = 0.3
    dup_duration_s: tuple = (5.0, 40.0)
    dup_offset_m: float = 0.4
    drop_rate_per_min: float = 0.5
    drop_duration_s: tuple = (2.0, 20.0)

    def __post_init__(self):
        if not 0.0 <= self.hide_frac < 1.0:
            raise ValueError("hide_frac must be in [0, 1)")
        if len(self.dur_edges_s) != len(self.dur_probs) + 1:
            raise ValueError("dur_edges_s needs one more entry than dur_probs")
        if min(self.dur_probs) < 0 or sum(self.dur_probs) <= 0:
            raise ValueError("dur_probs must be non-negative with a positive sum")
        if list(self.dur_edges_s) != sorted(self.dur_edges_s) or self.dur_edges_s[0] <= 0:
            raise ValueError("dur_edges_s must be positive and increasing")

    @classmethod
    def from_config(cls, path=None) -> "DegraderParams":
        return params_from_mapping(cls, load_section("external_degrader", path), "external_degrader")


@dataclass
class DegradeReport:
    """Aggregate description of one degradation (no per-frame data)."""

    hidden_alive_share: float
    hidden_graded_share: float
    n_blocks: int
    block_seconds: list = field(default_factory=list)
    mean_crowding_hidden: float = float("nan")
    mean_crowding_all: float = float("nan")
    n_cuts: int = 0
    n_duplicates: int = 0
    n_drops: int = 0


@dataclass
class DegradedMatch:
    players_xy: np.ndarray        # [N, S, 2] float32
    team_of_slot: np.ndarray      # [S] unchanged
    ball_xy: np.ndarray           # [N, 2] float32, NaN while hidden
    hidden: np.ndarray            # [N] bool, frames whose ball was hidden by the occlusion blocks
    report: DegradeReport


# --------------------------------------------------------------------------------------------
# Ball occlusion blocks
# --------------------------------------------------------------------------------------------
def crowding(players_xy: np.ndarray, ball_xy: np.ndarray, radius_m: float) -> np.ndarray:
    """Number of players within ``radius_m`` of the ball per frame (0 where the ball is unknown)."""
    diff = players_xy.astype(np.float32) - ball_xy[:, None, :].astype(np.float32)
    with np.errstate(invalid="ignore"):
        near = np.hypot(diff[..., 0], diff[..., 1]) <= radius_m           # NaN compares False
    return near.sum(axis=1).astype(np.float32)


def sample_block_frames(rng: np.random.Generator, p: DegraderParams, fps: float) -> int:
    """Duration (frames) of one block: class by ``dur_probs``, log-uniform inside the class."""
    probs = np.asarray(p.dur_probs, dtype=np.float64)
    k = int(rng.choice(len(probs), p=probs / probs.sum()))
    lo, hi = p.dur_edges_s[k], p.dur_edges_s[k + 1]
    return max(1, int(round(float(np.exp(rng.uniform(np.log(lo), np.log(hi)))) * fps)))


def _hide_blocks(codes: np.ndarray, crowd: np.ndarray, fps: float, p: DegraderParams,
                 rng: np.random.Generator) -> tuple[np.ndarray, list]:
    """Boolean hidden mask and the list of ``(start, end_exclusive)`` blocks."""
    n = len(codes)
    graded = codes != X
    alive = (codes == H) | (codes == A)
    hidden = np.zeros(n, dtype=bool)
    blocks: list[tuple[int, int]] = []
    target = int(round(p.hide_frac * int(alive.sum())))
    if target == 0 or not graded.any():
        return hidden, blocks
    run_end = np.zeros(n, dtype=np.int64)            # exclusive end of the graded run holding each frame
    longest = 0
    for a, b in zip(*mask_runs(graded)):
        run_end[a:b] = b
        longest = max(longest, int(b - a))
    cs_crowd = np.concatenate([[0.0], np.cumsum(crowd.astype(np.float64))])
    cs_alive = np.concatenate([[0], np.cumsum(alive.astype(np.int64))])
    covered = 0
    for _ in range(100000):
        if covered >= target:
            break
        dur = min(sample_block_frames(rng, p, fps), longest)
        starts = np.arange(n - dur + 1)
        free = np.concatenate([[0], np.cumsum(hidden.astype(np.int64))])
        ok = graded[starts] & (starts + dur <= run_end[starts]) & ((free[starts + dur] - free[starts]) == 0)
        if not ok.any():
            if dur <= int(fps):                      # nothing fits any more
                break
            continue
        mean_crowd = (cs_crowd[starts + dur] - cs_crowd[starts]) / dur
        weight = np.where(ok, (p.density_floor + mean_crowd) ** p.density_gamma, 0.0)
        s = int(rng.choice(len(starts), p=weight / weight.sum()))
        end = s + dur
        gained = np.cumsum(alive[s:end].astype(np.int64))
        if covered + int(gained[-1]) > target:       # trim the last block so the target is met exactly
            end = s + int(np.searchsorted(gained, target - covered, side="left")) + 1
        hidden[s:end] = True
        covered += int(cs_alive[end] - cs_alive[s])
        blocks.append((s, end))
    return hidden, blocks


# --------------------------------------------------------------------------------------------
# Tracking artefacts
# --------------------------------------------------------------------------------------------
def correlated_noise(shape: tuple, sigma: float, corr_s: float, fps: float,
                     rng: np.random.Generator) -> np.ndarray:
    """Stationary AR(1) Gaussian noise along axis 0 with std ``sigma`` and correlation time ``corr_s``."""
    white = rng.normal(0.0, 1.0, shape)
    phi = float(np.exp(-1.0 / max(corr_s * fps, 1e-6))) if corr_s > 0 else 0.0
    out = lfilter([np.sqrt(1.0 - phi * phi)], [1.0, -phi], white, axis=0)
    return (sigma * out).astype(np.float32)


def _n_events(rate_per_min: float, graded_minutes: float, rng: np.random.Generator) -> int:
    return int(rng.poisson(max(rate_per_min, 0.0) * graded_minutes))


def _apply_cuts(pxy, team, graded_idx, fps, p, rng) -> int:
    n_cuts = _n_events(p.cut_rate_per_min, len(graded_idx) / fps / 60.0, rng)
    if not n_cuts:
        return 0
    times = np.sort(rng.choice(graded_idx, size=n_cuts, replace=n_cuts > len(graded_idx)))
    done = 0
    for t in times:
        side = int(rng.integers(0, 2))
        slots = np.flatnonzero(team == side)
        live = slots[np.isfinite(pxy[t, slots, 0])]
        k = int(rng.integers(p.cut_players[0], p.cut_players[1] + 1))
        if len(live) < max(k, 2):
            continue
        chosen = rng.choice(live, size=k, replace=False)
        perm = np.roll(chosen, 1)                                         # cyclic identity switch
        pxy[t:, chosen] = pxy[t:, perm]
        gap = int(round(rng.uniform(*p.cut_gap_s) * fps))
        pxy[t:t + gap, chosen] = np.nan
        done += 1
    return done


def _apply_duplicates(pxy, team, graded_idx, fps, p, rng) -> int:
    n_ev = _n_events(p.dup_rate_per_min, len(graded_idx) / fps / 60.0, rng)
    done = 0
    for _ in range(n_ev):
        t = int(rng.choice(graded_idx))
        end = min(t + int(round(rng.uniform(*p.dup_duration_s) * fps)), len(pxy))
        slots = np.flatnonzero(team == int(rng.integers(0, 2)))
        empty = [s for s in slots if not np.isfinite(pxy[t:end, s, 0]).any()]
        live = [s for s in slots if np.isfinite(pxy[t:end, s, 0]).all()]
        if not empty or not live:
            continue
        dst, src = int(rng.choice(empty)), int(rng.choice(live))
        angle = rng.uniform(0, 2 * np.pi)
        pxy[t:end, dst] = pxy[t:end, src] + p.dup_offset_m * np.array([np.cos(angle), np.sin(angle)])
        done += 1
    return done


def _apply_drops(pxy, graded_idx, fps, p, rng) -> int:
    n_ev = _n_events(p.drop_rate_per_min, len(graded_idx) / fps / 60.0, rng)
    done = 0
    for _ in range(n_ev):
        t = int(rng.choice(graded_idx))
        dur = int(round(rng.uniform(*p.drop_duration_s) * fps))
        live = np.flatnonzero(np.isfinite(pxy[t, :, 0]))
        if not len(live):
            continue
        pxy[t:t + dur, int(rng.choice(live))] = np.nan
        done += 1
    return done


# --------------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------------
def degrade_match(players_xy: np.ndarray, team_of_slot: np.ndarray, ball_xy: np.ndarray, gt, *,
                  fps: float = FPS, seed: int = 0, params: DegraderParams | None = None,
                  hide_frac: float | None = None, artefacts: bool = True) -> DegradedMatch:
    """Degraded copy of a clean match (inputs are not modified).

    ``players_xy[N, S, 2]`` and ``ball_xy[N, 2]`` in metres (NaN = missing), ``team_of_slot[S]``,
    ``gt[N]`` labels (strings or codes; ``X`` frames are never touched by occlusion blocks and are ignored
    when counting).  ``hide_frac`` overrides the configured share; ``artefacts=False`` keeps only the ball
    occlusion (realistic duration / placement model, no noise nor identity damage).
    """
    p = params or DegraderParams.from_config()
    if hide_frac is not None:
        p = dataclasses.replace(p, hide_frac=float(hide_frac))
    codes = to_codes(gt)
    n = len(codes)
    pxy = np.array(players_xy, dtype=np.float32, copy=True)
    bxy = np.array(ball_xy, dtype=np.float32, copy=True)
    team = np.asarray(team_of_slot)
    if pxy.shape[0] != n or bxy.shape != (n, 2) or team.shape != (pxy.shape[1],):
        raise ValueError("players_xy, ball_xy, team_of_slot and gt have inconsistent shapes")
    rng = np.random.default_rng(seed)
    graded_idx = np.flatnonzero(codes != X)

    crowd = crowding(pxy, bxy, p.density_radius_m)
    hidden, blocks = _hide_blocks(codes, crowd, fps, p, rng)

    n_cuts = n_dups = n_drops = 0
    if artefacts and len(graded_idx):
        if p.noise_player_m > 0:
            pxy += correlated_noise(pxy.shape, p.noise_player_m, p.noise_corr_s, fps, rng)
        if p.noise_ball_m > 0:
            bxy += correlated_noise(bxy.shape, p.noise_ball_m, p.noise_corr_s, fps, rng)
        n_cuts = _apply_cuts(pxy, team, graded_idx, fps, p, rng)
        n_dups = _apply_duplicates(pxy, team, graded_idx, fps, p, rng)
        n_drops = _apply_drops(pxy, graded_idx, fps, p, rng)
    bxy[hidden] = np.nan

    alive = (codes == H) | (codes == A)
    graded = codes != X
    report = DegradeReport(
        hidden_alive_share=float(hidden[alive].mean()) if alive.any() else 0.0,
        hidden_graded_share=float(hidden[graded].mean()) if graded.any() else 0.0,
        n_blocks=len(blocks), block_seconds=[round((b - a) / fps, 2) for a, b in blocks],
        mean_crowding_hidden=float(crowd[hidden].mean()) if hidden.any() else float("nan"),
        mean_crowding_all=float(crowd[graded].mean()) if graded.any() else float("nan"),
        n_cuts=n_cuts, n_duplicates=n_dups, n_drops=n_drops)
    return DegradedMatch(players_xy=pxy, team_of_slot=team, ball_xy=bxy, hidden=hidden, report=report)
