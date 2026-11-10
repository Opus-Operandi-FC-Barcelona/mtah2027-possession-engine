"""Specialist B -- possession model for the frames whose ball is ABSENT (Issue #22).

When the ball is hidden nothing but the other twenty-two players, the memory of the last / next visible
ball and the dead-ball geometry of the players remain.  Specialist B is the same calibrated
gradient-boosting classifier as Specialist A but it is trained exclusively on hidden-ball rows, so that it
uses only the ball-free information: the collective physics (#17: goalkeepers, hull / spectral areas and
their derivatives), the phase coherence and ball memory (#18: last / next toucher, pressure around the last /
next seen ball, line advance), the player-speed part of the dead-ball signals (#13) and the transparent
wave-1 emission as a stacked prior.  The soft Smart Tagging priors of Issue #6 (phase of play) are added by
the fusion stage as log-odds, because the external training matches have no Smart Tagging.

Training data, in order of importance:

1. the degraded external matches (hidden-ball rows, algorithmic ground truth), with home / away swapped as
   augmentation;
2. optionally, the self-supervised block-masked rows of the target match itself (Issue #21), added by
   :meth:`SpecialistB.adapt` as a few extra boosting rounds with their own sample weight.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mtah2027.config import load_section, params_from_mapping
from mtah2027.models.frame_classifier import ClassifierParams, FrameClassifier
from mtah2027.models.pseudo_labeling import PseudoDataset
from mtah2027.models.specialist_data import TrainingSet


@dataclass(frozen=True)
class SpecialistBParams:
    min_train_rows: int = 2000
    adapt: bool = True                # adapt to the target match with the pseudo-labelled blocks (Issue #21)

    @classmethod
    def from_config(cls, path=None) -> "SpecialistBParams":
        return params_from_mapping(cls, load_section("specialist_b", path), "specialist_b")


class SpecialistB:
    """Ball-absent specialist (see the module docstring)."""

    def __init__(self, classifier: FrameClassifier, params: SpecialistBParams | None = None):
        self.classifier = classifier
        self.params = params or SpecialistBParams.from_config()

    @property
    def feature_names(self) -> list[str]:
        return self.classifier.feature_names

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Calibrated ``P(H, A, D)`` ``[N, 3]`` for the rows ``x`` (meant for hidden-ball frames)."""
        return self.classifier.predict_proba(x)

    def adapt(self, pseudo: PseudoDataset, min_rows: int = 1) -> "SpecialistB":
        """Specialist adapted to the target match with its pseudo-labelled masked blocks (Issue #21).

        Returns ``self`` when there are fewer than ``min_rows`` pseudo rows or the backend cannot continue
        boosting (scikit-learn fallback)."""
        if len(pseudo) < max(min_rows, 1) or self.classifier.backend != "lightgbm":
            return self
        if pseudo.names != self.feature_names:
            raise ValueError("pseudo-labelled features do not match the specialist's feature layout")
        return SpecialistB(self.classifier.adapt(pseudo.x, pseudo.y, pseudo.weight), self.params)

    def to_dict(self) -> dict:
        return {"classifier": self.classifier.to_dict()}

    @classmethod
    def from_dict(cls, data: dict, params: SpecialistBParams | None = None,
                  classifier_params: ClassifierParams | None = None) -> "SpecialistB":
        return cls(FrameClassifier.from_dict(data["classifier"], classifier_params), params)


def train_specialist_b(training: TrainingSet, *, params: SpecialistBParams | None = None,
                       classifier_params: ClassifierParams | None = None,
                       exclude_match: str | None = None) -> SpecialistB:
    """Fit the specialist on the hidden-ball rows of ``training`` (optionally leaving one match out)."""
    p = params or SpecialistBParams.from_config()
    x, y, groups = training.stack("hid", exclude_match=exclude_match)
    if len(y) < p.min_train_rows:
        raise ValueError(f"only {len(y)} hidden-ball training rows (< {p.min_train_rows})")
    clf = FrameClassifier(classifier_params or ClassifierParams.from_config(), training.names)
    clf.fit(x, y, groups=groups)
    clf.meta["role"] = "B: ball absent"
    return SpecialistB(clf, p)
