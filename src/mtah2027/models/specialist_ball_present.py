"""Specialist A -- possession model for the frames whose ball is VISIBLE (Issue #20).

A gradient-boosted classifier (:class:`~mtah2027.models.frame_classifier.FrameClassifier`: LightGBM with
``deterministic=True, num_threads=1, seed=0`` and a scikit-learn ``HistGradientBoostingClassifier``
fallback) over the feature matrix of ``specialist_features`` -- the dead-ball signals (#13), the ball
kinematics (#16), the collective physics (#17) and the coherence / memory features (#18), plus the
transparent wave-1 emission as a stacked prior.  It returns *calibrated* probabilities
``P_vis(H, A, D)`` (temperature fitted on out-of-fold predictions of whole held-out matches) and an
**ultra-confident anchor mask**: the frames where the winning class has ``P > anchor_prob`` (0.98) and keeps
winning without interruption for at least ``anchor_min_s`` (2 s).  The anchors are the pseudo-labels from
which Issue #21 builds the self-supervised dataset of the competition match itself.

Training data: the degraded external matches (``specialist_data``), visible-ball rows only; the labels are the
algorithmic ground truth of those matches.  Nothing from the competition match is ever used as a label.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mtah2027.config import FPS, load_section, params_from_mapping
from mtah2027.features._utils import mask_runs
from mtah2027.models.frame_classifier import ClassifierParams, FrameClassifier
from mtah2027.models.specialist_data import TrainingSet


@dataclass(frozen=True)
class SpecialistAParams:
    anchor_prob: float = 0.98         # ultra-confident: P(winning class) above this ...
    anchor_min_s: float = 2.0         # ... for at least this long without a change of winner
    min_train_rows: int = 2000

    @classmethod
    def from_config(cls, path=None) -> "SpecialistAParams":
        return params_from_mapping(cls, load_section("specialist_a", path), "specialist_a")


@dataclass
class VisibleResult:
    """Output of :meth:`SpecialistA.predict`; frames with a hidden ball are NaN / ``False`` / ``-1``."""

    prob: np.ndarray                  # [N, 3] float64 calibrated P(H, A, D), NaN where the ball is hidden
    anchor_mask: np.ndarray           # [N] bool, ultra-confident sustained frames
    anchor_label: np.ndarray          # [N] int8 winning class on anchor frames, -1 elsewhere
    meta: dict = field(default_factory=dict)


def anchor_mask(prob: np.ndarray, visible: np.ndarray, fps: float = FPS, *, threshold: float = 0.98,
                min_seconds: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """``(mask[N], label[N])`` of the sustained ultra-confident frames.

    A frame is *confident* when its ball is visible and ``max(prob) > threshold``; a run of consecutive
    confident frames with the same winning class is an anchor span when it lasts at least ``min_seconds``.
    """
    p = np.asarray(prob, dtype=np.float64)
    n = len(p)
    vis = np.asarray(visible, dtype=bool)
    ok = vis & np.isfinite(p).all(axis=1)
    win = np.where(ok, np.argmax(np.where(np.isfinite(p), p, -1.0), axis=1), -1)
    confident = ok & (np.where(ok, p.max(axis=1), 0.0) > threshold)
    mask = np.zeros(n, bool)
    label = np.full(n, -1, dtype=np.int8)
    need = max(1, int(round(min_seconds * fps)))
    for cls in range(p.shape[1]):
        starts, ends = mask_runs(confident & (win == cls))
        for a, b in zip(starts, ends):
            if b - a >= need:
                mask[a:b] = True
                label[a:b] = cls
    return mask, label


class SpecialistA:
    """Visible-ball specialist (see the module docstring)."""

    def __init__(self, classifier: FrameClassifier, params: SpecialistAParams | None = None):
        self.classifier = classifier
        self.params = params or SpecialistAParams.from_config()

    @property
    def feature_names(self) -> list[str]:
        return self.classifier.feature_names

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.classifier.predict_proba(x)

    def predict(self, x: np.ndarray, visible: np.ndarray, fps: float = FPS) -> VisibleResult:
        """Calibrated probabilities on the visible frames and the anchor mask (see :func:`anchor_mask`)."""
        vis = np.asarray(visible, dtype=bool)
        prob = np.full((len(vis), 3), np.nan)
        if vis.any():
            prob[vis] = self.classifier.predict_proba(np.asarray(x)[vis])
        mask, label = anchor_mask(prob, vis, fps, threshold=self.params.anchor_prob,
                                  min_seconds=self.params.anchor_min_s)
        return VisibleResult(prob=prob, anchor_mask=mask, anchor_label=label,
                             meta={"temperature": self.classifier.temperature,
                                   "anchor_share_of_visible": float(mask.sum() / max(vis.sum(), 1))})

    def to_dict(self) -> dict:
        return {"classifier": self.classifier.to_dict()}

    @classmethod
    def from_dict(cls, data: dict, params: SpecialistAParams | None = None,
                  classifier_params: ClassifierParams | None = None) -> "SpecialistA":
        return cls(FrameClassifier.from_dict(data["classifier"], classifier_params), params)


def train_specialist_a(training: TrainingSet, *, params: SpecialistAParams | None = None,
                       classifier_params: ClassifierParams | None = None,
                       exclude_match: str | None = None) -> SpecialistA:
    """Fit the specialist on the visible-ball rows of ``training`` (optionally leaving one match out)."""
    p = params or SpecialistAParams.from_config()
    x, y, groups, w = training.stack_weighted("vis", exclude_match=exclude_match)
    if len(y) < p.min_train_rows:
        raise ValueError(f"only {len(y)} visible-ball training rows (< {p.min_train_rows})")
    clf = FrameClassifier(classifier_params or ClassifierParams.from_config(), training.names)
    clf.fit(x, y, sample_weight=w, groups=groups)
    clf.meta["role"] = "A: ball visible"
    return SpecialistA(clf, p)
