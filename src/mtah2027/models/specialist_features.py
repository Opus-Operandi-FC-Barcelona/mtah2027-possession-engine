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

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from mtah2027.config import load_section, params_from_mapping
from mtah2027.features._utils import window_mean

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


@dataclass(frozen=True)
class FeatureParams:
    context_windows: tuple = (50, 100)    # frames; empty = no context features

    @classmethod
    def from_config(cls, path=None) -> "FeatureParams":
        return params_from_mapping(cls, load_section("specialist_features", path), "specialist_features")


def context_features(x: np.ndarray, names: list[str], blocks: Sequence[tuple[int, int]] | None,
                     windows: Sequence[int]) -> tuple[np.ndarray, list[str]]:
    """``ctx:`` columns of the feature matrix ``x`` (see the module docstring).

    Every graded block is processed on its own, so no window crosses a block / half boundary; frames
    outside the blocks are NaN.  ``blocks=None`` treats the whole array as one block."""
    n = len(x)
    spans = [(0, n - 1)] if blocks is None else [(int(a), int(b)) for a, b in blocks]
    sel = [(k, names.index(k)) for k in CONTEXT_SIGNALS if k in names]
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
    for obj, prefix in ((fe.sig, "sig:"), (fe.bk, "bk:"), (fe.phys, "cp:"), (fe.mem, "bm:")):
        if obj is None:
            continue
        matrix, keys = obj.matrix()
        _add(cols, names, matrix, keys, prefix)
    for base in PHYSICS_DIFFS + MEMORY_DIFFS:
        src = fe.phys if base in PHYSICS_DIFFS else fe.mem
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
    fp = params or FeatureParams.from_config(config_path)
    if fp.context_windows:
        cx, cnames = context_features(x, names, play_blocks, fp.context_windows)
        x = np.concatenate([x, cx], axis=1)
        names = names + cnames
    return FeatureMatrix(x=x, names=names)


def ball_based_columns(names: list[str]) -> list[int]:
    """Indices of the columns that are undefined while the ball is hidden."""
    return [i for i, k in enumerate(names) if k.startswith(BALL_BASED_PREFIXES) or k in BALL_BASED_SIG]
