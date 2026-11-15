"""Specialist C: a learned turnover (H <-> A) gate (Issue #57).

The H <-> A entries of the transition matrix open only near rule-based touch peaks.  Specialist C predicts,
from the same per-frame feature matrix as the specialists A and B (so it works with a visible and with a
hidden ball), whether a *true* possession change happens within ``+- half_window`` frames of the frame:

* class ``0`` -- no change; class ``1`` -- H -> A (the away team wins the ball); class ``2`` -- A -> H.

The targets come from the ground truth of external matches only (a direct H <-> A change between two
consecutive live frames); the model is a calibrated :class:`~mtah2027.models.frame_classifier.FrameClassifier`
(LightGBM, ``deterministic=True``, ``num_threads=1``, ``seed=0``; scikit-learn fallback).  The transition
matrix maps the probabilities to the gate with ``log A_t = closed + (open - closed) * g`` (see
``transition_matrix.learned_gate_mode``).
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from mtah2027.config import REPO_ROOT, load_section, params_from_mapping
from mtah2027.models.frame_classifier import ClassifierParams, FrameClassifier

H, A, D, X = 0, 1, 2, 3
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "specialist_c.json"
MODEL_FORMAT = 1


@dataclass(frozen=True)
class SpecialistCParams:
    half_window: int = 5              # a frame is positive within +- this many frames of a true change
    n_estimators: int = 200
    learning_rate: float = 0.05
    num_leaves: int = 15
    min_child_samples: int = 40       # turnovers are rare (~1 % of the rows)
    subsample: float = 0.8
    colsample: float = 0.7
    reg_lambda: float = 5.0
    calib_folds: int = 0
    pff_weight: float = 0.5           # weight of the PFF rows (real occlusions) when they are used

    @classmethod
    def from_config(cls, path=None) -> "SpecialistCParams":
        return params_from_mapping(cls, load_section("specialist_c", path), "specialist_c")

    def classifier_params(self, base: ClassifierParams | None = None) -> ClassifierParams:
        base = base or ClassifierParams.from_config()
        return dataclasses.replace(
            base, n_estimators=self.n_estimators, learning_rate=self.learning_rate,
            num_leaves=self.num_leaves, min_child_samples=self.min_child_samples, subsample=self.subsample,
            colsample=self.colsample, reg_lambda=self.reg_lambda, calib_folds=self.calib_folds)


def change_events(codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Frames ``t`` of the direct changes ``H -> A`` and ``A -> H`` between frames ``t - 1`` and ``t``."""
    c = np.asarray(codes)
    prev, cur = c[:-1], c[1:]
    ha = np.flatnonzero((prev == H) & (cur == A)) + 1
    ah = np.flatnonzero((prev == A) & (cur == H)) + 1
    return ha.astype(np.int64), ah.astype(np.int64)


def turnover_targets(codes: np.ndarray, half_window: int = 5) -> np.ndarray:
    """Per-frame class ``0`` none / ``1`` H -> A / ``2`` A -> H within ``+- half_window`` of a true change.

    Where two windows overlap the later event wins (deterministic)."""
    n = len(codes)
    y = np.zeros(n, dtype=np.int64)
    ha, ah = change_events(codes)
    events = sorted([(int(t), 1) for t in ha] + [(int(t), 2) for t in ah])
    for t, cls in events:
        y[max(0, t - half_window):min(n, t + half_window + 1)] = cls
    return y


def effective_changes(codes: np.ndarray) -> list[tuple[int, int]]:
    """``(frame, direction)`` of every team change with ``D`` / ``X`` frames filled forward (1 = H -> A)."""
    c = np.asarray(codes)
    out = []
    prev_team = None
    for t in np.flatnonzero((c == H) | (c == A)):
        team = int(c[t])
        if prev_team is not None and team != prev_team:
            out.append((int(t), 1 if prev_team == H else 2))
        prev_team = team
    return out


def turnover_recall(gt: np.ndarray, pred: np.ndarray, tol: int = 25) -> dict:
    """Share of the true direct H <-> A changes matched by a predicted change of the same direction.

    Predicted changes are the team changes of ``pred`` with dead frames filled forward (so ``H D A`` counts as
    H -> A at the start of the ``A`` spell); ``tol`` frames either side, each predicted change used once."""
    ha, ah = change_events(gt)
    truth = sorted([(int(t), 1) for t in ha] + [(int(t), 2) for t in ah])
    used: set[int] = set()
    preds = effective_changes(pred)
    matched = 0
    for t, d in truth:
        best, best_gap = None, tol + 1
        for k, (tp, dp) in enumerate(preds):
            if dp != d or k in used:
                continue
            gap = abs(tp - t)
            if gap < best_gap:
                best, best_gap = k, gap
        if best is not None:
            used.add(best)
            matched += 1
    return {"n_true": len(truth), "matched": matched, "n_pred": len(preds)}


class SpecialistC:
    """Fitted gate model: calibrated three-class classifier over the specialists' feature layout."""

    def __init__(self, classifier: FrameClassifier, params: SpecialistCParams | None = None,
                 meta: dict | None = None):
        self.classifier = classifier
        self.params = params or SpecialistCParams.from_config()
        self.meta = dict(meta or {})

    @property
    def feature_names(self) -> list[str]:
        return self.classifier.feature_names

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(p_ha, p_ah)``: probability of a H -> A / A -> H change within the window of every row."""
        prob = self.classifier.predict_proba(x)
        return prob[:, 1], prob[:, 2]

    def to_dict(self) -> dict:
        return {"format": MODEL_FORMAT, "classifier": self.classifier.to_dict(), "meta": self.meta,
                "params": dataclasses.asdict(self.params)}

    @classmethod
    def from_dict(cls, data: dict, config_path=None) -> "SpecialistC":
        if data.get("format") != MODEL_FORMAT:
            raise ValueError(f"unsupported specialist C file format {data.get('format')!r}")
        params = params_from_mapping(SpecialistCParams, data.get("params", {}), "specialist_c (stored)")
        clf = FrameClassifier.from_dict(data["classifier"], params.classifier_params(
            ClassifierParams.from_config(config_path)))
        return cls(clf, params, data.get("meta", {}))


def train_specialist_c(x: np.ndarray, y: np.ndarray, groups: np.ndarray, feature_names: Sequence[str], *,
                       weights: np.ndarray | None = None, params: SpecialistCParams | None = None,
                       classifier_params: ClassifierParams | None = None) -> SpecialistC:
    """Fit C on rows ``x`` with turnover classes ``y`` (see :func:`turnover_targets`); ``groups`` = match."""
    p = params or SpecialistCParams.from_config()
    clf = FrameClassifier(p.classifier_params(classifier_params), feature_names)
    clf.fit(x, y, sample_weight=weights, groups=groups)
    return SpecialistC(clf, p, {"n_samples": int(len(y)),
                                "class_counts": [int((y == k).sum()) for k in range(3)]})


def save_specialist_c(model: SpecialistC, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(model.to_dict(), sort_keys=True, separators=(",", ":")) + "\n",
                   encoding="utf-8", newline="\n")
    return out


def load_specialist_c(path: str | Path = DEFAULT_MODEL_PATH, config_path=None) -> SpecialistC:
    return SpecialistC.from_dict(json.loads(Path(path).read_text(encoding="utf-8")), config_path)


def load_default_specialist_c(config_path=None) -> SpecialistC | None:
    """The committed model, or ``None`` when it is missing / unreadable (progressive fallback)."""
    try:
        return load_specialist_c(DEFAULT_MODEL_PATH, config_path)
    except Exception:  # noqa: BLE001 - optional component
        return None
