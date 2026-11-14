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

Extra training sources (Issue #35, ``extra_sources``): the SkillCorner A-League broadcast tracking
(``skillcorner_*.npz``, already degraded: 10 -> 25 fps, players off screen, frames with ``observed == 0`` are
``X``) and the PFF FC World Cup 2022 tracking (``pff_*.npz``, whose ``ball_xy`` includes positions *estimated*
from the players: the ball is set to NaN where ``ball_visible`` is False, so the occlusions are the real
ones).  Neither goes through the synthetic degrader.  Their rows can be down-weighted
(``skillcorner_weight`` / ``pff_weight``) and the SkillCorner dead rows dropped
(``skillcorner_drop_dead``: its
``D`` is the gap between two ``phases_of_play``, not the provider's ball status).  They are only ever used
as training rows; the leave-one-match-out evaluation stays on the degraded IDSSE matches.

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
from mtah2027.models.external_degrader import DegraderParams, degrade_match
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
    extra_sources: tuple = ()        # Issue #35: file-name prefixes (skillcorner_, pff_)
    skillcorner_weight: float = 1.0  # sample weight of the SkillCorner rows
    skillcorner_drop_dead: bool = False   # keep only the SkillCorner H / A rows
    skillcorner_max_matches: int = 0      # 0 = all (sorted by name, deterministic)
    pff_weight: float = 1.0
    pff_max_matches: int = 0
    # Issue #38: config section of the degrader of the IDSSE training rows.  ``external_degrader_legacy`` (the
    # Issue #19 profile) keeps the adopted models; ``external_degrader`` = the calibrated degrader.
    degrader_section: str = "external_degrader_legacy"

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
    source: str = "idsse"            # "idsse" (degraded here) | "skillcorner" | "pff" (native, Issue #35)


NATIVE_SOURCES = ("skillcorner", "pff")      # tracking that is already occluded: no synthetic degradation


def source_of(stem: str) -> str:
    for name in NATIVE_SOURCES:
        if stem.startswith(name + "_"):
            return name
    return "idsse"


@dataclass
class TrainingSet:
    names: list[str]
    parts: list[MatchRows]
    base_seeds: tuple = ()           # degradation seeds requested for every match (before the match offset)
    weights: dict = field(default_factory=dict)       # source -> sample weight (absent = 1)
    drop_dead: tuple = ()                             # sources whose D rows are left out

    def stack(self, kind: str, exclude_match: str | None = None, only_match: str | None = None):
        """``(x, y, groups)`` of the ``vis`` or ``hid`` rows (groups = match name)."""
        x, y, g, _ = self.stack_weighted(kind, exclude_match, only_match)
        return x, y, g

    def stack_weighted(self, kind: str, exclude_match: str | None = None, only_match: str | None = None):
        """``(x, y, groups, weight)``; ``weight`` is ``None`` when every row has weight 1."""
        xs, ys, gs, ws = [], [], [], []
        for part in self.parts:
            if exclude_match is not None and part.match == exclude_match:
                continue
            if only_match is not None and part.match != only_match:
                continue
            x, y = (part.x_vis, part.y_vis) if kind == "vis" else (part.x_hid, part.y_hid)
            if part.source in self.drop_dead:
                keep = y != D
                x, y = x[keep], y[keep]
            xs.append(x)
            ys.append(y)
            gs.append(np.full(len(y), part.match, dtype=object))
            ws.append(np.full(len(y), float(self.weights.get(part.source, 1.0))))
        if not xs:
            raise ValueError("no training rows selected")
        w = np.concatenate(ws)
        return np.concatenate(xs), np.concatenate(ys), np.concatenate(gs), (None if (w == 1.0).all() else w)

    def select(self, sources: Iterable[str], weights: dict | None = None,
               drop_dead: Iterable[str] = (), exclude_matches: Iterable[str] = ()) -> "TrainingSet":
        """View with the rows of ``sources`` (minus ``exclude_matches``) and the given weights / policy."""
        keep, out = set(sources), set(exclude_matches)
        return TrainingSet(names=self.names,
                           parts=[q for q in self.parts if q.source in keep and q.match not in out],
                           base_seeds=self.base_seeds, weights=dict(weights or {}),
                           drop_dead=tuple(drop_dead))

    @property
    def matches(self) -> list[str]:
        return sorted({p.match for p in self.parts})

    def summary(self) -> dict:
        return {"matches": self.matches, "n_variants": len(self.parts),
                "rows_visible": int(sum(len(p.y_vis) for p in self.parts)),
                "rows_hidden": int(sum(len(p.y_hid) for p in self.parts)),
                "degradation_seeds": list(self.base_seeds) or sorted({p.seed for p in self.parts}),
                "swapped_variants": int(sum(p.swap for p in self.parts)),
                "sources": {src: sorted({p.match for p in self.parts if p.source == src})
                            for src in sorted({p.source for p in self.parts}) if src != "idsse"},
                "source_weights": {k: float(v) for k, v in sorted(self.weights.items())},
                "source_drop_dead": list(self.drop_dead)}


def list_external_matches(bench_dir: str | Path, params: TrainingDataParams | None = None) -> list[Path]:
    """Benchmark ``.npz`` files usable for training (sorted; complete-ball matches only by default).

    The extra sources of Issue #35 (``extra_sources`` prefixes) are appended after them, each limited to its
    first ``*_max_matches`` files by name, so the rows of the main matches never depend on them."""
    p = params or TrainingDataParams.from_config()
    files = sorted(Path(bench_dir).glob("*.npz"))
    prefixes = tuple(f"{n}_" for n in NATIVE_SOURCES)
    out = [f for f in files if not f.stem.startswith(prefixes)]
    if p.only_complete_ball:
        out = [f for f in out if np.isfinite(np.load(f, allow_pickle=True)["ball_xy"]).all()]
    for name in NATIVE_SOURCES:
        if f"{name}_" not in tuple(p.extra_sources):
            continue
        sel = [f for f in files if f.stem.startswith(f"{name}_")]
        cap = int(getattr(p, f"{name}_max_matches"))
        out += sel[:cap] if cap > 0 else sel
    return out


def swap_codes(codes: np.ndarray) -> np.ndarray:
    out = np.asarray(codes).copy()
    out[codes == H], out[codes == A] = A, H
    return out


def rows_for_match(path: str | Path, seed: int, *, stride: int, swaps: Sequence[bool] = (False, True),
                   config_path=None, min_period_frames: int = 250,
                   degrader_section: str = "external_degrader_legacy") -> list[MatchRows]:
    """Rows of one external match degraded with ``seed`` (one :class:`MatchRows` per swap variant)."""
    from mtah2027.pipeline import compute_front_end, heuristic_emissions  # local import: avoid a cycle

    z = np.load(path, allow_pickle=True)
    players, team = z["players_xy"].astype(np.float32), z["team_of_slot"].astype(np.int8)
    ball = z["ball_xy"].astype(np.float32)
    fps, pitch = float(z["fps"]), tuple(float(v) for v in z["pitch"])
    periods = [(int(a), int(b)) for a, b in z["period_bounds"] if int(b) - int(a) >= min_period_frames]
    gt = eb.to_codes(z["gt"])
    source = source_of(Path(path).stem)
    if source == "idsse":
        dm = degrade_match(players, team, ball, gt, fps=fps, seed=seed,
                           params=DegraderParams.from_config(config_path, degrader_section))
        dm_players, dm_ball = dm.players_xy, dm.ball_xy
    else:                      # already occluded tracking: use it as it is (Issue #35)
        if "ball_visible" in z.files:         # PFF: estimated ball positions are not observations
            ball = np.where(z["ball_visible"][:, None], ball, np.float32(np.nan)).astype(np.float32)
        if "observed" in z.files:             # SkillCorner: nothing is known where the pitch is off screen
            gt = np.where(z["observed"], gt, X).astype(gt.dtype)
        dm_players, dm_ball = players, ball
    in_block = np.zeros(len(gt), bool)      # Issue #51: the front end is blank outside the play blocks
    for a0, b0 in periods:
        in_block[a0:b0 + 1] = True
    graded = np.flatnonzero((gt != X) & in_block)
    keep = graded[(graded % stride) == (seed % stride)]
    out = []
    for swap in swaps:
        tm = (1 - team).astype(np.int8) if swap else team
        labels = swap_codes(gt) if swap else gt
        fe = compute_front_end(dm_players, tm, dm_ball, play_blocks=periods, fps=fps, pitch=pitch,
                               config_path=config_path, collective=True)
        em = heuristic_emissions(fe, fps, config_path)
        fm = build_feature_matrix(fe, em, play_blocks=periods, config_path=config_path)
        vis = fe.visible[keep]
        x, y = fm.x[keep], labels[keep].astype(np.int64)
        out.append(MatchRows(match=Path(path).stem, seed=seed, swap=bool(swap), names=fm.names, source=source,
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
            if source_of(f.stem) != "idsse" and seed != p.seeds[0]:
                continue                       # native sources have a single (undegraded) variant
            t0 = time.perf_counter()
            parts += rows_for_match(f, seed + i, stride=p.stride,
                                    swaps=(False, True) if p.swap_teams else (False,),
                                    config_path=config_path, min_period_frames=p.min_period_frames,
                                    degrader_section=p.degrader_section)
            if log:
                log(f"training rows {f.stem} seed {seed + i}: {len(parts[-1].y_vis)} visible / "
                    f"{len(parts[-1].y_hid)} hidden ({time.perf_counter() - t0:.1f} s)")
    weights = {"skillcorner": p.skillcorner_weight, "pff": p.pff_weight}
    return TrainingSet(names=parts[0].names, parts=parts, base_seeds=tuple(p.seeds), weights=weights,
                       drop_dead=("skillcorner",) if p.skillcorner_drop_dead else ())

