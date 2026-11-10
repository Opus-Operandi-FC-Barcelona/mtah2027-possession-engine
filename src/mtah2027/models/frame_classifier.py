"""Calibrated three-class frame classifier shared by the specialists (Issues #20 and #22).

A thin, deterministic wrapper around LightGBM (``deterministic=True``, ``num_threads=1``, ``seed=0``) with a
scikit-learn ``HistGradientBoostingClassifier`` fallback when LightGBM is not installed.  Classes are the
emission states ``H=0, A=1, D=2``.

* **Calibration.**  The raw scores are turned into probabilities by a temperature ``T`` (``softmax(raw / T)``)
  fitted by minimising the log-loss of *out-of-fold* predictions, the folds being whole groups (matches), so
  that the temperature reflects the generalisation to an unseen match, not the in-sample over-confidence.
* **Adaptation.**  :meth:`FrameClassifier.adapt` continues boosting of a LightGBM model on a few new samples
  (the pseudo-labelled blocks of the target match, Issue #21).
* **Persistence.**  LightGBM models are stored as their text dump inside a JSON-serialisable dict; the
  scikit-learn fallback cannot be serialised portably and is only used for models trained in-process.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from mtah2027.config import load_section, params_from_mapping

try:                                                    # optional accelerator, identical interface
    import lightgbm as _lgb
except ImportError:                                     # pragma: no cover - exercised on lean installs
    _lgb = None

N_CLASSES = 3
_NEG = -30.0                                            # raw score of a class the model never saw


def lightgbm_available() -> bool:
    return _lgb is not None


@dataclass(frozen=True)
class ClassifierParams:
    backend: str = "auto"            # "auto" (LightGBM if installed), "lightgbm" or "sklearn"
    n_estimators: int = 150
    learning_rate: float = 0.08
    num_leaves: int = 15
    min_child_samples: int = 200
    subsample: float = 0.8
    colsample: float = 0.7
    reg_lambda: float = 5.0
    seed: int = 0
    calib_folds: int = 3             # group folds for the temperature (0 = no calibration, T = 1)
    temperature_bounds: tuple = (0.5, 4.0)
    adapt_rounds: int = 40           # boosting rounds added by ``adapt``
    adapt_learning_rate: float = 0.05

    def __post_init__(self):
        if self.backend not in ("auto", "lightgbm", "sklearn"):
            raise ValueError(f"unknown backend '{self.backend}'")
        lo_hi = self.temperature_bounds
        if len(lo_hi) != 2 or not 0 < lo_hi[0] <= lo_hi[1]:
            raise ValueError("temperature_bounds must be an increasing positive pair")

    @classmethod
    def from_config(cls, path=None) -> "ClassifierParams":
        return params_from_mapping(cls, load_section("specialist_model", path), "specialist_model")

    def resolved_backend(self) -> str:
        if self.backend == "auto":
            return "lightgbm" if _lgb is not None else "sklearn"
        if self.backend == "lightgbm" and _lgb is None:
            raise RuntimeError("backend 'lightgbm' requested but lightgbm is not installed")
        return self.backend


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def log_loss(prob: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> float:
    p = np.clip(prob[np.arange(len(y)), y], 1e-9, 1.0)
    nll = -np.log(p)
    return float(np.average(nll, weights=w))


def _fit_booster(params: ClassifierParams, backend: str, x: np.ndarray, y: np.ndarray,
                 w: np.ndarray | None, init_model: Any = None, rounds: int | None = None,
                 learning_rate: float | None = None):
    if backend == "lightgbm":
        cfg = {"objective": "multiclass", "num_class": N_CLASSES, "learning_rate":
               params.learning_rate if learning_rate is None else learning_rate,
               "num_leaves": params.num_leaves, "min_data_in_leaf": params.min_child_samples,
               "bagging_fraction": params.subsample, "bagging_freq": 1, "feature_fraction": params.colsample,
               "lambda_l2": params.reg_lambda, "seed": params.seed, "bagging_seed": params.seed,
               "feature_fraction_seed": params.seed, "data_random_seed": params.seed,
               "deterministic": True, "force_row_wise": True, "num_threads": 1, "verbosity": -1}
        data = _lgb.Dataset(np.ascontiguousarray(x, dtype=np.float32), label=y, weight=w,
                            free_raw_data=False)
        return _lgb.train(cfg, data, num_boost_round=rounds or params.n_estimators, init_model=init_model)
    from sklearn.ensemble import HistGradientBoostingClassifier

    model = HistGradientBoostingClassifier(
        max_iter=params.n_estimators, learning_rate=params.learning_rate, max_leaf_nodes=params.num_leaves,
        min_samples_leaf=params.min_child_samples, l2_regularization=params.reg_lambda,
        random_state=params.seed, early_stopping=False)
    return model.fit(x, y, sample_weight=w)


def _raw_scores(model: Any, backend: str, x: np.ndarray) -> np.ndarray:
    if backend == "lightgbm":
        raw = np.asarray(model.predict(np.ascontiguousarray(x, dtype=np.float32), raw_score=True))
        return raw.reshape(len(x), N_CLASSES).astype(np.float64)
    scores = np.asarray(model.decision_function(x), dtype=np.float64)
    if scores.ndim == 1:                                # binary problem: two classes only
        scores = np.stack([-scores / 2, scores / 2], axis=1)
    out = np.full((len(x), N_CLASSES), _NEG)
    out[:, np.asarray(model.classes_, dtype=int)] = scores
    return out


def fit_temperature(raw: np.ndarray, y: np.ndarray, bounds: tuple, w: np.ndarray | None = None) -> float:
    """Temperature minimising the log-loss of ``softmax(raw / T)`` (bounded 1-D search, deterministic)."""
    from scipy.optimize import minimize_scalar

    lo, hi = float(bounds[0]), float(bounds[1])
    res = minimize_scalar(lambda t: log_loss(softmax(raw / t), y, w), bounds=(lo, hi), method="bounded",
                          options={"xatol": 1e-4})
    return float(np.clip(res.x, lo, hi))


def group_folds(groups: np.ndarray, n_folds: int) -> list[np.ndarray]:
    """Boolean validation masks that split the distinct groups (sorted) into ``n_folds`` round-robin sets."""
    labels = np.unique(groups)
    n_folds = max(1, min(int(n_folds), len(labels)))
    return [np.isin(groups, labels[k::n_folds]) for k in range(n_folds)]


class FrameClassifier:
    """Calibrated H / A / D classifier (see the module docstring)."""

    def __init__(self, params: ClassifierParams | None = None, feature_names: Sequence[str] = ()):
        self.params = params or ClassifierParams.from_config()
        self.feature_names = list(feature_names)
        self.backend = self.params.resolved_backend()
        self.model: Any = None
        self.temperature = 1.0
        self.meta: dict = {}

    # ------------------------------------------------------------------------------------------
    def fit(self, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None,
            groups: np.ndarray | None = None) -> "FrameClassifier":
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        w = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
        if len(x) != len(y) or (y < 0).any() or (y >= N_CLASSES).any():
            raise ValueError("x / y inconsistent or labels outside 0..2")
        if self.feature_names and len(self.feature_names) != x.shape[1]:
            raise ValueError("feature_names does not match the number of columns")
        self.temperature = 1.0
        if self.params.calib_folds > 1 and groups is not None and len(np.unique(groups)) > 1:
            groups = np.asarray(groups)
            oof = np.zeros((len(y), N_CLASSES))
            for held in group_folds(groups, self.params.calib_folds):
                tr = ~held
                model = _fit_booster(self.params, self.backend, x[tr], y[tr], None if w is None else w[tr])
                oof[held] = _raw_scores(model, self.backend, x[held])
            self.temperature = fit_temperature(oof, y, self.params.temperature_bounds, w)
            self.meta["oof_log_loss_raw"] = round(log_loss(softmax(oof), y, w), 5)
            self.meta["oof_log_loss_calibrated"] = round(log_loss(softmax(oof / self.temperature), y, w), 5)
        self.model = _fit_booster(self.params, self.backend, x, y, w)
        self.meta["n_samples"] = int(len(y))
        self.meta["class_counts"] = [int((y == k).sum()) for k in range(N_CLASSES)]
        return self

    def raw_scores(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("classifier is not fitted")
        return _raw_scores(self.model, self.backend, np.asarray(x, dtype=np.float32))

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Calibrated ``P(H, A, D)`` ``[N, 3]`` (float64)."""
        return softmax(self.raw_scores(x) / self.temperature)

    # ------------------------------------------------------------------------------------------
    def adapt(self, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None,
              rounds: int | None = None) -> "FrameClassifier":
        """Copy of this classifier with ``rounds`` extra boosting rounds on ``(x, y)`` (LightGBM only)."""
        if self.backend != "lightgbm" or self.model is None:
            raise RuntimeError("adaptation needs a fitted LightGBM classifier")
        out = FrameClassifier(self.params, self.feature_names)
        out.temperature = self.temperature
        out.meta = dict(self.meta)
        w = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
        out.model = _fit_booster(self.params, "lightgbm", np.asarray(x, dtype=np.float32),
                                 np.asarray(y, dtype=np.int64), w, init_model=self.model,
                                 rounds=rounds or self.params.adapt_rounds,
                                 learning_rate=self.params.adapt_learning_rate)
        out.meta["n_adapt_samples"] = int(len(y))
        return out

    # ------------------------------------------------------------------------------------------
    def to_dict(self) -> dict:
        if self.backend != "lightgbm" or self.model is None:
            raise ValueError("only fitted LightGBM classifiers can be serialised")
        return {"backend": "lightgbm", "feature_names": self.feature_names, "temperature": self.temperature,
                "params": {k: (list(v) if isinstance(v, tuple) else v)
                           for k, v in dataclasses.asdict(self.params).items()},
                "meta": self.meta, "model": self.model.model_to_string()}

    @classmethod
    def from_dict(cls, data: dict, params: ClassifierParams | None = None) -> "FrameClassifier":
        if data.get("backend") != "lightgbm":
            raise ValueError("unsupported classifier backend in the stored model")
        if _lgb is None:
            raise RuntimeError("lightgbm is required to load the stored specialists")
        stored = params_from_mapping(ClassifierParams, data.get("params", {}), "stored classifier")
        out = cls(params or stored, data["feature_names"])
        out.backend = "lightgbm"
        out.temperature = float(data["temperature"])
        out.meta = dict(data.get("meta", {}))
        out.model = _lgb.Booster(model_str=data["model"])
        return out
