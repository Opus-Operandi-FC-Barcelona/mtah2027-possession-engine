"""Per-state duration distributions of H / A possessions and D stoppages from external labels (Issue #34).

The table is estimated **only from the ground-truth labels of external open matches** (never BAR-ATH): the
lengths of the maximal runs of ``H``, ``A`` and ``D`` inside every graded block, leaving out the first and
the last run of each block (they are cut by the block boundary).  Each state gets a histogram on a fixed grid
of bins in seconds (``edges_s``, the last edge is the cap) plus a geometric tail beyond the cap, so the
log-pmf of a duration of ``d`` frames is ``log(mass_bin / frames_in_bin)`` inside the grid and
``log(tail) + log(1 - q) + (d - cap - 1) log q`` beyond it.  Add-``alpha`` smoothing keeps empty bins finite.

``tables(fps)`` returns the per-frame arrays used by :mod:`mtah2027.decoding.semi_markov`.  The table is
small enough to be stored next to the specialists (``models/specialists.json``) as plain JSON.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from mtah2027.config import load_section, params_from_mapping

H, A, D, X = 0, 1, 2, 3
N_STATES = 3


@dataclass(frozen=True)
class DurationParams:
    enabled: bool = False             # use the semi-Markov decoder
    weight: float = 1.0               # multiplier of the duration log-probabilities
    center: bool = True               # subtract the mean log-pmf so an average segment costs nothing extra
    apply_to_heuristic: bool = False  # the heuristic route as well (needs a table, e.g. the committed one)
    edges_s: tuple = (0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0,
                      18.0, 22.0, 27.0, 33.0, 40.0, 50.0, 60.0)
    alpha: float = 0.5                # additive smoothing of the bin counts
    min_tail_runs: int = 5            # fewer runs beyond the cap -> fixed tail rate
    default_tail_excess_s: float = 30.0

    @classmethod
    def from_config(cls, path=None) -> "DurationParams":
        return params_from_mapping(cls, load_section("duration", path), "duration")


@dataclass
class DurationModel:
    """Binned duration distribution of the three alive / dead states."""

    edges_s: tuple
    mass: list                        # [3][n_bins] probabilities inside the grid (sum with tail = 1)
    tail: list                        # [3] probability of a run longer than the cap
    tail_excess_s: list               # [3] mean excess over the cap of those runs (seconds)
    counts: list = field(default_factory=lambda: [0, 0, 0])   # runs used per state

    def to_dict(self) -> dict:
        return {"edges_s": [float(e) for e in self.edges_s],
                "mass": [[round(float(v), 8) for v in row] for row in self.mass],
                "tail": [round(float(v), 8) for v in self.tail],
                "tail_excess_s": [round(float(v), 4) for v in self.tail_excess_s],
                "counts": [int(c) for c in self.counts]}

    @classmethod
    def from_dict(cls, data: dict) -> "DurationModel":
        return cls(edges_s=tuple(data["edges_s"]), mass=[list(r) for r in data["mass"]],
                   tail=list(data["tail"]), tail_excess_s=list(data["tail_excess_s"]),
                   counts=list(data.get("counts", [0, 0, 0])))

    def tables(self, fps: float, params: DurationParams | None = None):
        """``(ld[3, cap + 1], ls[3, cap + 1], lq[3], a_tail[3], ls_tail[3], cap)`` for the decoder.

        ``ld[s, d]`` = log P(duration = d frames), ``ls[s, d]`` = log P(duration >= d) (index 0 unused),
        ``lq`` = log of the geometric tail ratio per frame, ``a_tail`` such that the tail log-pmf is
        ``a_tail + d * lq`` for ``d > cap``.  With ``center`` the mean log-pmf of every state is removed
        (a segment of average plausibility costs nothing beyond the transition matrix)."""
        p = params or DurationParams()
        edges = np.asarray(self.edges_s, dtype=np.float64)
        cap = int(round(edges[-1] * fps))
        d = np.arange(1, cap + 1)
        bin_of = np.clip(np.searchsorted(edges, d / fps, side="left") - 1, 0, len(edges) - 2)
        frames_in_bin = np.bincount(bin_of, minlength=len(edges) - 1).astype(np.float64)
        ld = np.full((N_STATES, cap + 1), -1.0e9)
        ls = np.zeros((N_STATES, cap + 1))
        lq = np.zeros(N_STATES)
        a_tail = np.zeros(N_STATES)
        ls_tail = np.zeros(N_STATES)
        for s in range(N_STATES):
            mass = np.asarray(self.mass[s], dtype=np.float64)
            pmf = mass[bin_of] / np.maximum(frames_in_bin[bin_of], 1.0)
            tail = max(float(self.tail[s]), 1e-9)
            q = float(np.clip(1.0 - 1.0 / max(self.tail_excess_s[s] * fps, 2.0), 0.5, 0.999999))
            ld[s, 1:] = np.log(np.maximum(pmf, 1e-300))
            surv = np.cumsum(pmf[::-1])[::-1] + tail            # P(duration >= d), d = 1..cap
            ls[s, 1:] = np.log(surv)
            lq[s] = np.log(q)
            ls_tail[s] = np.log(tail)          # log P(duration >= cap + 1)
            a_tail[s] = np.log(tail) + np.log(1.0 - q) - (cap + 1) * lq[s]
            if p.center:
                mean_ld = float(np.sum(pmf * ld[s, 1:]) + tail * (np.log(tail) + np.log(1.0 - q)))
                ld[s, 1:] -= mean_ld
                a_tail[s] -= mean_ld
        return (np.round(ld, 6), np.round(ls, 6), np.round(lq, 8), np.round(a_tail, 6),
                np.round(ls_tail, 6), cap)


def run_lengths(codes: np.ndarray, blocks: Sequence[tuple[int, int]]) -> list[list[int]]:
    """Interior run lengths (frames) of H, A, D inside ``blocks`` (inclusive); border runs are dropped."""
    out: list[list[int]] = [[], [], []]
    codes = np.asarray(codes)
    for a, b in blocks:
        seg = codes[int(a):int(b) + 1]
        if len(seg) < 3:
            continue
        change = np.flatnonzero(seg[1:] != seg[:-1]) + 1
        starts = np.concatenate([[0], change])
        ends = np.concatenate([change, [len(seg)]])
        for s0, e0 in list(zip(starts, ends))[1:-1]:
            state = int(seg[s0])
            if state < N_STATES:
                out[state].append(int(e0 - s0))
    return out


def fit_duration_model(sequences: Iterable[tuple[np.ndarray, Sequence[tuple[int, int]], float]],
                       params: DurationParams | None = None) -> DurationModel:
    """Duration table from ``(codes, blocks, fps)`` triples (ground-truth labels of other matches)."""
    p = params or DurationParams()
    edges = np.asarray(p.edges_s, dtype=np.float64)
    runs_s: list[list[float]] = [[], [], []]
    for codes, blocks, fps in sequences:
        for s, lens in enumerate(run_lengths(codes, blocks)):
            runs_s[s].extend(n / float(fps) for n in lens)
    mass, tail, excess, counts = [], [], [], []
    for s in range(N_STATES):
        r = np.asarray(runs_s[s], dtype=np.float64)
        counts.append(int(len(r)))
        inside = r[r <= edges[-1]]
        bins = np.clip(np.searchsorted(edges, inside, side="left") - 1, 0, len(edges) - 2)   # (lo, hi] bins
        hist = np.bincount(bins, minlength=len(edges) - 1).astype(np.float64) + p.alpha
        long_runs = r[r > edges[-1]]
        t_cnt = float(len(long_runs)) + p.alpha
        total = hist.sum() + t_cnt
        mass.append((hist / total).tolist())
        tail.append(float(t_cnt / total))
        excess.append(float(np.mean(long_runs - edges[-1])) if len(long_runs) >= p.min_tail_runs
                      else float(p.default_tail_excess_s))
    return DurationModel(edges_s=tuple(float(e) for e in edges), mass=mass, tail=tail, tail_excess_s=excess,
                         counts=counts)


def fit_from_bench(bench_dir, matches: Iterable[str] | None = None, exclude: str | None = None,
                   params: DurationParams | None = None, min_period_frames: int = 250) -> DurationModel:
    """Duration table from the ground-truth labels of the benchmark matches ``bench_dir/<match>.npz``.

    ``matches`` defaults to every file with a complete ball channel (the training matches of the specialists);
    ``exclude`` leaves one match out (leave-one-match-out evaluation)."""
    from pathlib import Path

    from mtah2027.validation import eval_bridge as eb

    files = sorted(Path(bench_dir).glob("*.npz"))
    wanted = None if matches is None else set(matches)
    seqs = []
    for f in files:
        if (wanted is not None and f.stem not in wanted) or f.stem == exclude:
            continue
        z = np.load(f, allow_pickle=True)
        if wanted is None and not np.isfinite(z["ball_xy"]).all():
            continue
        blocks = [(int(a), int(b)) for a, b in z["period_bounds"] if int(b) - int(a) >= min_period_frames]
        seqs.append((eb.to_codes(z["gt"]), blocks, float(z["fps"])))
    if not seqs:
        raise FileNotFoundError(f"no benchmark matches for the duration table in {bench_dir}")
    return fit_duration_model(seqs, params)
