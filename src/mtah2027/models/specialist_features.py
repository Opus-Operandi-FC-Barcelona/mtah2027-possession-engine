"""Per-frame feature matrix of the learned specialists (Issues #20-#22).

One matrix serves both specialists: the columns that need the *current* ball position (distances, control,
touches, evidence of Issue #16 and the ball-based part of #13) are NaN while the ball is hidden, so that the
ball-absent specialist (B) can only rely on the collective physics (#17), the coherence / memory features
(#18) and the player-speed part of the dead-ball signals (#13), exactly as the specification asks.

Column groups (prefix of the feature name):

``sig:``   Issue #13 dead-ball signals (``dead_score``, ball-out, stillness, player speeds, geometry, ...).
``bk:``    Issue #16 ball kinematics (distances, closing speeds, control, touch score, evidence).
``cp:``    Issue #17 collective physics (goalkeepers, hull / spectral areas and their derivatives, ...).
``bm:``    Issue #18 phase coherence, line advance and ball memory.
``diff:``  home minus away of the paired physics / memory features (team-symmetric view).
``heur:``  the transparent wave-1 emission (its H-vs-A logit and dead probability) as a stacked prior.
``ctx:``   past / future context of the collective signals (#17, #18) and ``dead_score`` (#13): centred
           means over ``+-W`` frames and the change ``mean(t..t+W) - mean(t-W..t)`` for every configured
           window ``W`` (``specialist_features.context_windows``; block-local, NaN-safe).  They let
           the ball-absent
           specialist see the reversal of the collective movement and short stops without the ball.

Slot indices and the raw ``ball_visible`` copies are dropped (meaningless / redundant); one ``visible``
column remains.  Team symmetry is obtained at the data level: the training sets also contain every match
with the home / away roles swapped (see ``specialist_data``).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from mtah2027.config import FPS, load_section, params_from_mapping
from mtah2027.features._utils import window_mean
from mtah2027.features.collective_physics import CollectivePhysicsParams, central_derivative

PHYSICS_DIFFS = ["hull_area", "spectral_area", "d_hull", "d_spectral", "depth", "width", "centroid_vx",
                 "n_field"]
MEMORY_DIFFS = ["phase_R", "line_vx", "press_last", "press_next", "dist_last", "dist_next", "adv_press",
                "adv_x"]
_DROPPED = {"sig:ball_visible", "bk:ball_visible", "bm:ball_visible", "bk:nearest_slot_home",
            "bk:nearest_slot_away", "cp:gk_slot_home", "cp:gk_slot_away"}
BALL_BASED_PREFIXES = ("bk:",)
BALL_BASED_SIG = ("sig:ball_out", "sig:ball_out_dist", "sig:ball_still_s", "sig:t_since_ball",
                  "sig:t_to_ball",
                  "sig:geom_corner", "sig:geom_goal_area", "sig:geom_kickoff", "sig:geom_touchline",
                  "sig:geom_any")


# signals whose past / future context is added (feature names before the ``ctx:`` prefix)
CONTEXT_SIGNALS = (
    "cp:centroid_vx_home", "cp:centroid_vx_away", "cp:depth_home", "cp:depth_away", "cp:width_home",
    "cp:width_away", "cp:spectral_area_home", "cp:spectral_area_away", "bm:phase_R_home", "bm:phase_R_away",
    "bm:line_vx_home", "bm:line_vx_away", "diff:centroid_vx", "diff:line_vx", "diff:phase_R",
    "diff:spectral_area", "diff:depth", "sig:speed_mean", "sig:speed_p80", "sig:frac_walking",
    "sig:dead_score")


# Issue #56: goalkeeper depth of each team (orientation-normalised x, Issue #17); its +-gk_span change is
# added as ``ctx:`` columns when ``gk_context`` is on (off = the adopted feature set).
GK_SIGNALS = ("cp:gk_xy_home_x", "cp:gk_xy_away_x")

# Issue #64: carrier / swarm micro-geometry (features/carrier_swarm.py); signals that get ``ctx:`` windows
SWARM_SIGNALS = ("sw:swarm_home", "sw:swarm_away", "sw:carrier_vx_home", "sw:carrier_vx_away",
                 "sw:carrier_x_home", "sw:carrier_x_away", "sw:min_interteam_dist", "sw:duel_density_3m",
                 "diff:swarm", "diff:carrier_vx", "diff:carrier_x")

COUNT_COLUMNS = ("cp:n_field_home", "cp:n_field_away", "diff:n_field", "sig:n_active")


@dataclass(frozen=True)
class FeatureParams:
    context_windows: tuple = (50, 100)    # frames; empty = no context features
    # Issue #56 -- goalkeeper depth and its +-gk_span (6 s) change as context columns (off = adopted).
    gk_context: bool = False
    gk_span: int = 150
    # Issue #64 -- implicit carrier and 22v22 pressure micro-geometry (off = adopted feature set).
    carrier_swarm: bool = False
    # Issue #42 -- invariance to player dropout (all off = the adopted feature set).
    neutralise_counts: bool = False       # count-like columns (n_field_*, diff:n_field, n_active) set to 0
    shape_min_field: int = 0              # team shape is NaN below this many outfield players
    normalise_shape: bool = False         # divide hull / spectral area, width, depth by expected ratio
    shape_ratio_n: tuple = (4, 5, 6, 7, 8, 9, 10)   # for n outfield players tracked (10 = complete)
    shape_ratio_hull: tuple = (0.279, 0.401, 0.538, 0.668, 0.789, 0.899, 1.0)
    shape_ratio_spectral: tuple = (0.542, 0.659, 0.767, 0.850, 0.914, 0.961, 1.0)
    shape_ratio_width: tuple = (0.686, 0.798, 0.848, 0.892, 0.943, 0.969, 1.0)
    shape_ratio_depth: tuple = (0.638, 0.667, 0.786, 0.844, 0.893, 0.938, 1.0)

    @classmethod
    def from_config(cls, path=None) -> "FeatureParams":
        return params_from_mapping(cls, load_section("specialist_features", path), "specialist_features")


def _shape_ratio(n: np.ndarray, grid: Sequence[float], ratio: Sequence[float]) -> np.ndarray:
    """Expected shape / complete-shape ratio for ``n`` tracked outfield players (1 at and above the last grid
    point and where ``n`` is unknown)."""
    nn = np.asarray(n, dtype=np.float64)
    r = np.interp(nn, np.asarray(grid, dtype=np.float64), np.asarray(ratio, dtype=np.float64))
    return np.where(np.isfinite(nn) & (nn < grid[-1]), r, 1.0)


def robust_physics(phys, fp: FeatureParams, fps: float = FPS, config_path=None):
    """Sample-size-robust copy of the collective physics (Issue #42; ``phys`` itself is not modified).

    ``normalise_shape`` divides the area / extent of a team by the average fraction that a random subset of
    its ``n`` tracked outfield players has of the complete shape (measured on clean external matches) and
    re-derives the area derivatives; ``shape_min_field`` blanks (NaN) the shape of a team below that many
    players.  Both are no-ops with the default parameters."""
    if phys is None or not (fp.normalise_shape or fp.shape_min_field):
        return phys
    span = max(int(round(CollectivePhysicsParams.from_config(config_path).deriv_span_s * fps)), 2)
    upd: dict[str, np.ndarray] = {}
    for team in ("home", "away"):
        n = np.asarray(getattr(phys, f"n_field_{team}"), dtype=np.float64)
        low = n < fp.shape_min_field if fp.shape_min_field else np.zeros(len(n), bool)
        for stat, table in (("hull_area", fp.shape_ratio_hull), ("spectral_area", fp.shape_ratio_spectral),
                            ("width", fp.shape_ratio_width), ("depth", fp.shape_ratio_depth)):
            arr = np.array(getattr(phys, f"{stat}_{team}"), dtype=np.float32, copy=True)
            if fp.normalise_shape:
                arr = (arr / _shape_ratio(n, fp.shape_ratio_n, table)).astype(np.float32)
            arr[low] = np.nan
            upd[f"{stat}_{team}"] = arr
        for deriv, src in (("d_hull", "hull_area"), ("d_spectral", "spectral_area")):
            d = np.array(getattr(phys, f"{deriv}_{team}"), dtype=np.float32, copy=True)
            if fp.normalise_shape:
                d = central_derivative(upd[f"{src}_{team}"], span, fps).astype(np.float32)
            d[low] = np.nan
            upd[f"{deriv}_{team}"] = d
    return dataclasses.replace(phys, **upd)


def context_features(x: np.ndarray, names: list[str], blocks: Sequence[tuple[int, int]] | None,
                     windows: Sequence[int], signals: Sequence[str] | None = None
                     ) -> tuple[np.ndarray, list[str]]:
    """``ctx:`` columns of the feature matrix ``x`` (see the module docstring).

    Every graded block is processed on its own, so no window crosses a block / half boundary; frames
    outside the blocks are NaN.  ``blocks=None`` treats the whole array as one block."""
    n = len(x)
    spans = [(0, n - 1)] if blocks is None else [(int(a), int(b)) for a, b in blocks]
    sel = [(k, names.index(k)) for k in (CONTEXT_SIGNALS if signals is None else signals) if k in names]
    cols, out_names = [], []
    for key, j in sel:
        for w in windows:
            w = int(w)
            cm = np.full(n, np.nan, np.float32)
            dd = np.full(n, np.nan, np.float32)
            for a, b in spans:
                seg = x[a:b + 1, j]
                if not len(seg):
                    continue
                cm[a:b + 1] = window_mean(seg, -w, w)
                dd[a:b + 1] = window_mean(seg, 0, w) - window_mean(seg, -w, 0)
            cols += [cm, dd]
            out_names += [f"ctx:{key}:cm{w}", f"ctx:{key}:dd{w}"]
    if not cols:
        return np.zeros((n, 0), np.float32), []
    return np.stack(cols, axis=1).astype(np.float32), out_names


@dataclass
class FeatureMatrix:
    x: np.ndarray                  # [N, F] float32
    names: list[str]

    def column(self, name: str) -> np.ndarray:
        return self.x[:, self.names.index(name)]


def _add(cols: list, names: list, matrix: np.ndarray, keys: list[str], prefix: str) -> None:
    for j, key in enumerate(keys):
        full = f"{prefix}{key}"
        if full in _DROPPED:
            continue
        cols.append(matrix[:, j].astype(np.float32))
        names.append(full)


def build_feature_matrix(fe, heuristic, *, require_collective: bool = True,
                         play_blocks: Sequence[tuple[int, int]] | None = None,
                         params: FeatureParams | None = None, config_path=None) -> FeatureMatrix:
    """Stack the feature groups of a :class:`~mtah2027.pipeline.FrontEnd` (with ``collective=True``) and its
    heuristic :class:`~mtah2027.models.heuristic_emissions.EmissionResult`; ``play_blocks`` bound the
    past / future context windows (see ``ctx:``)."""
    if require_collective and (fe.phys is None or fe.mem is None or fe.sig is None or fe.bk is None):
        raise ValueError("the specialists need the dead-ball, ball-kinematics, collective-physics and "
                         "ball-memory features")
    n = len(fe.visible)
    cols: list[np.ndarray] = []
    names: list[str] = []
    cols.append(fe.visible.astype(np.float32))
    names.append("visible")
    fp = params or FeatureParams.from_config(config_path)
    phys = robust_physics(fe.phys, fp, config_path=config_path)
    for obj, prefix in ((fe.sig, "sig:"), (fe.bk, "bk:"), (phys, "cp:"), (fe.mem, "bm:")):
        if obj is None:
            continue
        matrix, keys = obj.matrix()
        _add(cols, names, matrix, keys, prefix)
    for base in PHYSICS_DIFFS + MEMORY_DIFFS:
        src = phys if base in PHYSICS_DIFFS else fe.mem
        if src is None:
            continue
        home, away = getattr(src, f"{base}_home", None), getattr(src, f"{base}_away", None)
        if home is None or away is None:
            continue
        cols.append((np.asarray(home, np.float32) - np.asarray(away, np.float32)).astype(np.float32))
        names.append(f"diff:{base}")
    if heuristic is not None:
        cols.append(np.asarray(heuristic.home_logit, dtype=np.float32))
        names.append("heur:home_logit")
        cols.append(np.asarray(heuristic.p_dead, dtype=np.float32))
        names.append("heur:p_dead")
    x = np.stack(cols, axis=1) if cols else np.zeros((n, 0), np.float32)
    if fp.neutralise_counts:          # Issue #42: the layout is kept, the columns carry no information
        for key in COUNT_COLUMNS:
            if key in names:
                x[:, names.index(key)] = 0.0
    if fp.context_windows:
        cx, cnames = context_features(x, names, play_blocks, fp.context_windows)
        x = np.concatenate([x, cx], axis=1)
        names = names + cnames
    if fp.gk_context:
        cx, cnames = context_features(x, names, play_blocks, (fp.gk_span,), signals=GK_SIGNALS)
        x = np.concatenate([x, cx], axis=1)
        names = names + cnames
    if fp.carrier_swarm:
        from mtah2027.features.carrier_swarm import NAMES as SW_NAMES
        from mtah2027.features.carrier_swarm import carrier_swarm

        sw = carrier_swarm(fe.pxy, fe.team, fe.pvel)
        scols = [sw[k] for k in SW_NAMES]
        snames = [f"sw:{k}" for k in SW_NAMES]
        for base in ("swarm", "carrier_vx", "carrier_x"):
            scols.append((sw[f"{base}_home"] - sw[f"{base}_away"]).astype(np.float32))
            snames.append(f"diff:{base}")
        sx = np.stack(scols, axis=1).astype(np.float32)
        x = np.concatenate([x, sx], axis=1)
        names = names + snames
        if fp.context_windows:
            cx, cnames = context_features(x, names, play_blocks, fp.context_windows, signals=SWARM_SIGNALS)
            x = np.concatenate([x, cx], axis=1)
            names = names + cnames
    return FeatureMatrix(x=x, names=names)


def ball_based_columns(names: list[str]) -> list[int]:
    """Indices of the columns that are undefined while the ball is hidden."""
    return [i for i, k in enumerate(names) if k.startswith(BALL_BASED_PREFIXES) or k in BALL_BASED_SIG]
