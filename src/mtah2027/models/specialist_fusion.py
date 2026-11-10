"""Fusion of Specialists A and B into the emissions ``log_probs[N, 3]`` of the decoder (Issue #22).

For every frame the emission is built from the specialist that fits its ball status:

* ball **visible** -> Specialist A (``P_vis``), ball **absent** -> Specialist B, the latter first adapted to
  the target match with the block-masked pseudo-labels of Issue #21 (A's ultra-confident anchors);
* the soft priors that the external training matches cannot teach -- the Smart Tagging phase of play (#6), an
  ATD owning-team channel and an ATD alive / dead status channel (#5, #13) -- are added to the H-vs-A / dead
  log-odds with the weights of the wave-1 emissions, only where those channels exist;
* the result is optionally tempered (``scale_*``, to account for the temporal correlation of the frames that
  the Viterbi pass would otherwise count as independent evidence) and blended with the heuristic emission
  (``blend_*``); probabilities are floored at ``prob_floor`` and the log-probabilities rounded to ``1e-6``.

``resolve_specialists`` implements the progressive fallback: train at run time from the external open
matches when they are on disk, else load the committed model file (``models/specialists.json``), else return
``None`` and the pipeline keeps the heuristic emissions.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from mtah2027.config import FPS, REPO_ROOT, load_section, params_from_mapping
from mtah2027.features._utils import sigmoid
from mtah2027.models.frame_classifier import ClassifierParams, lightgbm_available
from mtah2027.models.heuristic_emissions import ROUND_DECIMALS, EmissionParams, EmissionResult
from mtah2027.models.pseudo_labeling import PseudoLabelParams, build_pseudo_dataset
from mtah2027.models.specialist_ball_absent import (
    SpecialistB,
    SpecialistBParams,
    train_specialist_b,
)
from mtah2027.models.specialist_ball_present import (
    SpecialistA,
    SpecialistAParams,
    train_specialist_a,
)
from mtah2027.models.specialist_data import TrainingSet, build_training_set
from mtah2027.models.specialist_features import build_feature_matrix

DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "specialists.json"
BENCH_ENV = "MTAH_BENCH_DIR"
MODEL_FORMAT = 1


@dataclass(frozen=True)
class FusionParams:
    prob_floor: float = 0.001         # emission probabilities are floored here
    scale_visible: float = 1.0        # multiplier of the log-probabilities where the ball is visible
    scale_hidden: float = 1.0         # ... and where it is absent
    blend_visible: float = 0.0        # weight of the heuristic log-probabilities (0 = learned only)
    blend_hidden: float = 0.0
    use_pseudo_labels: bool = True    # adapt B to the target match (Issue #21)

    def __post_init__(self):
        if not 0 < self.prob_floor < 0.5:
            raise ValueError("prob_floor must be in (0, 0.5)")
        for name in ("blend_visible", "blend_hidden"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.scale_visible <= 0 or self.scale_hidden <= 0:
            raise ValueError("scales must be positive")

    @classmethod
    def from_config(cls, path=None) -> "FusionParams":
        return params_from_mapping(cls, load_section("specialist_fusion", path), "specialist_fusion")


@dataclass
class SpecialistSet:
    """Specialists A and B plus the fusion parameters and a description of their training data."""

    a: SpecialistA
    b: SpecialistB
    fusion: FusionParams = field(default_factory=FusionParams.from_config)
    training: dict = field(default_factory=dict)
    source: str = "memory"

    def to_dict(self) -> dict:
        return {"format": MODEL_FORMAT, "a": self.a.to_dict(), "b": self.b.to_dict(),
                "training": self.training}

    @classmethod
    def from_dict(cls, data: dict, config_path=None) -> "SpecialistSet":
        if data.get("format") != MODEL_FORMAT:
            raise ValueError(f"unsupported specialists file format {data.get('format')!r}")
        cp = ClassifierParams.from_config(config_path)
        return cls(a=SpecialistA.from_dict(data["a"], SpecialistAParams.from_config(config_path), cp),
                   b=SpecialistB.from_dict(data["b"], SpecialistBParams.from_config(config_path), cp),
                   fusion=FusionParams.from_config(config_path), training=dict(data.get("training", {})),
                   source="file")


@dataclass
class FusionResult:
    emissions: EmissionResult
    report: dict


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def apply_priors(prob: np.ndarray, *, phase: np.ndarray | None, owner: np.ndarray | None,
                 status_dead: np.ndarray | None, ep: EmissionParams) -> tuple[np.ndarray, np.ndarray]:
    """Add the soft priors to the model probabilities ``prob[N, 3]``; returns ``(prob, home_logit)``.

    With no prior channel the input is returned untouched (``home_logit = log(pH / pA)``)."""
    ph, pa, pd = prob[:, 0], prob[:, 1], prob[:, 2]
    z = np.log(np.clip(ph, 1e-6, 1)) - np.log(np.clip(pa, 1e-6, 1))
    if phase is None and owner is None and status_dead is None:
        return prob, z
    zd = _logit(pd)
    if status_dead is not None:
        st = np.asarray(status_dead, dtype=np.float64)
        zd = zd + np.where(np.isfinite(st), ep.w_status * (2.0 * np.nan_to_num(st) - 1.0), 0.0)
    if phase is not None:
        z = z + ep.w_phase * np.nan_to_num(np.asarray(phase, dtype=np.float64))
    if owner is not None:
        ow = np.asarray(owner, dtype=np.float64)
        z = z + np.where(np.isfinite(ow), ep.w_owner * (1.0 - 2.0 * np.nan_to_num(ow)), 0.0)
    p_dead = sigmoid(zd)
    alive = 1.0 - p_dead
    return np.stack([alive * sigmoid(z), alive * sigmoid(-z), p_dead], axis=1), z


def fuse_emissions(specialists: SpecialistSet, fe, heuristic: EmissionResult, *, play_blocks,
                   fps: float = FPS, rebuild: Callable[[np.ndarray], object], raw_ball: np.ndarray,
                   config_path=None) -> FusionResult:
    """Learned emissions of the target match (see the module docstring)."""
    from mtah2027.pipeline import heuristic_emissions  # local import: pipeline imports this module lazily

    fp = specialists.fusion
    n = len(fe.visible)
    t0 = time.perf_counter()
    fm = build_feature_matrix(fe, heuristic, play_blocks=play_blocks, config_path=config_path)
    if fm.names != specialists.a.feature_names or fm.names != specialists.b.feature_names:
        raise ValueError("feature layout of the match does not match the stored specialists")
    vis = np.asarray(fe.visible, dtype=bool)
    res_a = specialists.a.predict(fm.x, vis, fps)

    spec_b = specialists.b
    pseudo_report: dict = {}
    if fp.use_pseudo_labels and spec_b.params.adapt:
        pp = PseudoLabelParams.from_config(config_path)
        pseudo = build_pseudo_dataset(
            res_a.anchor_mask, res_a.anchor_label, vis, raw_ball, rebuild, play_blocks=play_blocks,
            heuristic=lambda f: heuristic_emissions(f, fps, config_path), fps=fps, params=pp,
            names=fm.names)
        pseudo_report = dict(pseudo.report)
        spec_b = spec_b.adapt(pseudo, min_rows=pp.min_rows)
        pseudo_report["adapted"] = spec_b is not specialists.b

    prob = np.full((n, 3), np.nan)
    prob[vis] = res_a.prob[vis]
    if (~vis).any():
        prob[~vis] = spec_b.predict_proba(fm.x[~vis])
    ep = EmissionParams.from_config(config_path)
    sig = fe.sig
    status = sig.status_dead if (sig is not None and sig.has_status) else None
    prob, home_logit = apply_priors(prob, phase=fe.phase, owner=fe.owner, status_dead=status, ep=ep)

    logp = np.log(np.maximum(prob, fp.prob_floor))
    scale = np.where(vis, fp.scale_visible, fp.scale_hidden)[:, None]
    blend = np.where(vis, fp.blend_visible, fp.blend_hidden)[:, None]
    logp = scale * ((1.0 - blend) * logp + blend * heuristic.log_probs)
    log_probs = np.round(logp, ROUND_DECIMALS)
    emissions = EmissionResult(
        log_probs=log_probs, p_dead=prob[:, 2], home_logit=home_logit, carried=heuristic.carried,
        meta={"with_owner": fe.owner is not None, "with_status": status is not None,
              "with_phase": fe.phase is not None, "learned": True})
    report = {"seconds": round(time.perf_counter() - t0, 2), "source": specialists.source,
              "visible_share": float(vis.mean()) if n else 0.0,
              "anchor_share_of_visible": res_a.meta["anchor_share_of_visible"],
              "temperature_a": specialists.a.classifier.temperature,
              "temperature_b": specialists.b.classifier.temperature,
              "pseudo_labels": pseudo_report, "training": specialists.training}
    return FusionResult(emissions=emissions, report=report)


# --------------------------------------------------------------------------------------------
# Training, persistence and the progressive fallback
# --------------------------------------------------------------------------------------------
def describe_training(ts: TrainingSet, bench_dir: str | Path, a: SpecialistA, b: SpecialistB) -> dict:
    """Aggregate description of the training data (stored next to the models; no per-frame data)."""
    return {"dataset": "external open tracking matches (IDSSE / DFL Bundesliga, Bassek et al. 2025)",
            "labels": "algorithmic (provider ball-status and possession channels; no hand labels, "
                      "no BAR-ATH data)", "degrader": "Issue #19 realistic degrader, several seeds, "
                                                      "home/away swapped copies",
            "bench_dir_name": Path(bench_dir).parent.name, **ts.summary(),
            "temperature_a": a.classifier.temperature, "temperature_b": b.classifier.temperature,
            "class_counts_a": a.classifier.meta.get("class_counts"),
            "class_counts_b": b.classifier.meta.get("class_counts"),
            "oof_log_loss_a": [a.classifier.meta.get("oof_log_loss_raw"),
                               a.classifier.meta.get("oof_log_loss_calibrated")],
            "oof_log_loss_b": [b.classifier.meta.get("oof_log_loss_raw"),
                               b.classifier.meta.get("oof_log_loss_calibrated")]}


def train_specialists(bench_dir: str | Path, *, config_path=None, training: TrainingSet | None = None,
                      exclude_match: str | None = None,
                      log: Callable[[str], None] | None = None) -> SpecialistSet:
    """Train A and B on the degraded external matches of ``bench_dir`` (optionally leaving one match out)."""
    ts = training or build_training_set(bench_dir, config_path=config_path, log=log)
    cp = ClassifierParams.from_config(config_path)
    a = train_specialist_a(ts, params=SpecialistAParams.from_config(config_path), classifier_params=cp,
                           exclude_match=exclude_match)
    if log:
        log(f"specialist A trained ({a.classifier.meta['n_samples']} rows, T={a.classifier.temperature:.3f})")
    b = train_specialist_b(ts, params=SpecialistBParams.from_config(config_path), classifier_params=cp,
                           exclude_match=exclude_match)
    if log:
        log(f"specialist B trained ({b.classifier.meta['n_samples']} rows, T={b.classifier.temperature:.3f})")
    return SpecialistSet(a=a, b=b, fusion=FusionParams.from_config(config_path),
                         training=describe_training(ts, bench_dir, a, b), source="trained")


def save_specialists(specialists: SpecialistSet, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(specialists.to_dict(), sort_keys=True, separators=(",", ":")) + "\n",
                   encoding="utf-8", newline="\n")
    return out


def load_specialists(path: str | Path = DEFAULT_MODEL_PATH, config_path=None) -> SpecialistSet:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return SpecialistSet.from_dict(data, config_path)


def find_bench_dir(explicit: str | Path | None = None) -> Path | None:
    """External benchmark cache directory: explicit argument, ``$MTAH_BENCH_DIR`` or the sibling repo."""
    candidates = [explicit, os.environ.get(BENCH_ENV),
                  REPO_ROOT.parent / "mtah2027-external-benchmarks" / "cache"]
    for cand in candidates:
        if cand and Path(cand).is_dir() and any(Path(cand).glob("*.npz")):
            return Path(cand)
    return None


def resolve_specialists(mode: str = "auto", *, bench_dir: str | Path | None = None,
                        model_path: str | Path | None = None, config_path=None, prefer_file: bool = False,
                        log: Callable[[str], None] | None = None) -> SpecialistSet | None:
    """Specialists for the pipeline, or ``None`` (heuristic emissions).

    ``mode``: ``off`` -> ``None``; ``train`` -> train from the external matches (error if absent);
    ``load`` -> the committed model file (error if absent); ``auto`` -> train when the external matches are
    on disk, else load the committed file, else ``None``.  Any failure in ``auto`` mode falls through to
    the next option and is logged.  ``prefer_file`` makes ``auto`` skip the run-time training (used for the
    synthetic CI run: deterministic and independent of the data on the local disk).
    """
    say = log or (lambda m: None)
    if mode == "off":
        return None
    if mode not in ("auto", "train", "load"):
        raise ValueError(f"unknown specialists mode '{mode}'")
    path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    strict = mode != "auto"
    if mode == "train" or (mode == "auto" and not prefer_file):
        found = find_bench_dir(bench_dir)
        if found is None:
            if mode == "train":
                raise FileNotFoundError("no external benchmark matches found (--bench-dir / $MTAH_BENCH_DIR)")
            say("specialists: external matches not on disk")
        else:
            try:
                return train_specialists(found, config_path=config_path, log=say)
            except Exception as exc:  # noqa: BLE001 - progressive fallback
                if strict:
                    raise
                say(f"specialists: training failed ({type(exc).__name__}: {exc})")
    if mode in ("auto", "load"):
        if path.is_file():
            try:
                return load_specialists(path, config_path)
            except Exception as exc:  # noqa: BLE001 - progressive fallback
                if strict:
                    raise
                say(f"specialists: could not load {path.name} ({type(exc).__name__}: {exc})")
        elif strict:
            raise FileNotFoundError(f"{path} not found")
        else:
            say(f"specialists: no committed model file ({path.name})")
    say("specialists: unavailable, using the heuristic emissions"
        + ("" if lightgbm_available() else " (lightgbm is not installed)"))
    return None
