"""Training rows of the specialists from the degraded EXTERNAL open-data matches (Issues #20 and #22).

The primary training data of both specialists are the open Bundesliga matches of the benchmark repository
(``mtah2027-external-benchmarks``; DFL / IDSSE open data, Bassek et al. 2025), whose per-frame labels are
produced *algorithmically* from the provider's own ball-status and possession channels (see that
repository's ``docs/LABELLING_RULES.md``) -- never by hand, and never from BAR-ATH.  Every match is degraded
with the realistic degrader of Issue #19 (several seeds), and the full front end (cleaning, orientation so
that the home team attacks +x, #13, #16, #17, #18) is computed on the degraded tracking, exactly as for the
competition match.

Team symmetry: BAR-ATH teams differ from Bundesliga teams, so every degraded match is also processed with the
home / away roles swapped (``team_of_slot`` flipped, labels ``H <-> A``).  The orientation stage then mirrors
the pitch again so that the new "home" attacks +x, which makes the model equivariant to who is called home.

Rows are sub-sampled every ``stride`` frames of the graded periods (a 0.4 s step at the default stride of
10) to decorrelate neighbours and keep training fast; the offset depends on the seed.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from mtah2027.config import load_section, params_from_mapping
from mtah2027.models.external_degrader import degrade_match
from mtah2027.models.specialist_features import build_feature_matrix
from mtah2027.validation import eval_bridge as eb

H, A, D, X = 0, 1, 2, 3


@dataclass(frozen=True)
class TrainingDataParams:
    seeds: tuple = (0, 1, 2)         # degradation seeds of every external match
    stride: int = 10                 # frames between two rows
    swap_teams: bool = True          # also add every match with home / away swapped
    only_complete_ball: bool = True  # Metrica samples lose the ball while it is dead (leaks D): excluded
    min_period_frames: int = 250

    @classmethod
    def from_config(cls, path=None) -> "TrainingDataParams":
        return params_from_mapping(cls, load_section("specialist_data", path), "specialist_data")


@dataclass
class MatchRows:
    """Rows of one (match, seed, swap) variant; ``vis`` rows have a visible ball, ``hid`` rows do not."""

    match: str
    seed: int
    swap: bool
    names: list[str]
    x_vis: np.ndarray
    y_vis: np.ndarray
    x_hid: np.ndarray
    y_hid: np.ndarray
    frames_vis: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    frames_hid: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))


@dataclass
class TrainingSet:
    names: list[str]
    parts: list[MatchRows]

    def stack(self, kind: str, exclude_match: str | None = None, only_match: str | None = None):
        """``(x, y, groups)`` of the ``vis`` or ``hid`` rows (groups = match name)."""
        xs, ys, gs = [], [], []
        for part in self.parts:
            if exclude_match is not None and part.match == exclude_match:
                continue
            if only_match is not None and part.match != only_match:
                continue
            x, y = (part.x_vis, part.y_vis) if kind == "vis" else (part.x_hid, part.y_hid)
            xs.append(x)
            ys.append(y)
            gs.append(np.full(len(y), part.match, dtype=object))
        if not xs:
            raise ValueError("no training rows selected")
        return np.concatenate(xs), np.concatenate(ys), np.concatenate(gs)

    @property
    def matches(self) -> list[str]:
        return sorted({p.match for p in self.parts})

    def summary(self) -> dict:
        return {"matches": self.matches, "n_variants": len(self.parts),
                "rows_visible": int(sum(len(p.y_vis) for p in self.parts)),
                "rows_hidden": int(sum(len(p.y_hid) for p in self.parts)),
                "seeds": sorted({p.seed for p in self.parts}),
                "swapped_variants": int(sum(p.swap for p in self.parts))}


def list_external_matches(bench_dir: str | Path, params: TrainingDataParams | None = None) -> list[Path]:
    """Benchmark ``.npz`` files usable for training (sorted; complete-ball matches only by default)."""
    p = params or TrainingDataParams.from_config()
    files = sorted(Path(bench_dir).glob("*.npz"))
    if p.only_complete_ball:
        files = [f for f in files if np.isfinite(np.load(f, allow_pickle=True)["ball_xy"]).all()]
    return files


def swap_codes(codes: np.ndarray) -> np.ndarray:
    out = np.asarray(codes).copy()
    out[codes == H], out[codes == A] = A, H
    return out


def rows_for_match(path: str | Path, seed: int, *, stride: int, swaps: Sequence[bool] = (False, True),
                   config_path=None, min_period_frames: int = 250) -> list[MatchRows]:
    """Rows of one external match degraded with ``seed`` (one :class:`MatchRows` per swap variant)."""
    from mtah2027.pipeline import compute_front_end, heuristic_emissions  # local import: avoid a cycle

    z = np.load(path, allow_pickle=True)
    players, team = z["players_xy"].astype(np.float32), z["team_of_slot"].astype(np.int8)
    ball = z["ball_xy"].astype(np.float32)
    fps, pitch = float(z["fps"]), tuple(float(v) for v in z["pitch"])
    periods = [(int(a), int(b)) for a, b in z["period_bounds"] if int(b) - int(a) >= min_period_frames]
    gt = eb.to_codes(z["gt"])
    dm = degrade_match(players, team, ball, gt, fps=fps, seed=seed)
    graded = np.flatnonzero(gt != X)
    keep = graded[(graded % stride) == (seed % stride)]
    out = []
    for swap in swaps:
        tm = (1 - team).astype(np.int8) if swap else team
        labels = swap_codes(gt) if swap else gt
        fe = compute_front_end(dm.players_xy, tm, dm.ball_xy, play_blocks=periods, fps=fps, pitch=pitch,
                               config_path=config_path, collective=True)
        em = heuristic_emissions(fe, fps, config_path)
        fm = build_feature_matrix(fe, em)
        vis = fe.visible[keep]
        x, y = fm.x[keep], labels[keep].astype(np.int64)
        out.append(MatchRows(match=Path(path).stem, seed=seed, swap=bool(swap), names=fm.names,
                             x_vis=x[vis], y_vis=y[vis], x_hid=x[~vis], y_hid=y[~vis],
                             frames_vis=keep[vis], frames_hid=keep[~vis]))
    return out


def build_training_set(bench_dir: str | Path, *, params: TrainingDataParams | None = None, config_path=None,
                       matches: Iterable[str] | None = None,
                       log: Callable[[str], None] | None = None) -> TrainingSet:
    """Rows of all external matches in ``bench_dir`` for all configured seeds (and the swapped variants)."""
    p = params or TrainingDataParams.from_config(config_path)
    files = list_external_matches(bench_dir, p)
    if matches is not None:
        wanted = set(matches)
        files = [f for f in files if f.stem in wanted]
    if not files:
        raise FileNotFoundError(f"no usable external benchmark matches in {bench_dir}")
    parts: list[MatchRows] = []
    for seed in p.seeds:
        for i, f in enumerate(files):
            t0 = time.perf_counter()
            parts += rows_for_match(f, seed + i, stride=p.stride,
                                    swaps=(False, True) if p.swap_teams else (False,),
                                    config_path=config_path, min_period_frames=p.min_period_frames)
            if log:
                log(f"training rows {f.stem} seed {seed + i}: {len(parts[-1].y_vis)} visible / "
                    f"{len(parts[-1].y_hid)} hidden ({time.perf_counter() - t0:.1f} s)")
    return TrainingSet(names=parts[0].names, parts=parts)

