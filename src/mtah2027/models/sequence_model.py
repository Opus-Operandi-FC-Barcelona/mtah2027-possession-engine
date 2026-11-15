"""Sequence model over the per-frame feature matrix of the specialists (Issue #72; OFF by default).

A non-causal dilated temporal convolutional network (TCN) reads the whole feature sequence of a play block and
returns, for every frame,

* three emission logits (``H``, ``A``, ``D``) -- used, when ``sequence_model.mode`` asks for it, on the
  hidden-ball frames only (mode ``emissions``);
* three turnover logits (none / ``H -> A`` / ``A -> H`` within ``+- half_window`` frames, the targets of
  Specialist C) -- used as an additional opening of the ``H <-> A`` transition (mode ``turnover``), through the
  same gate mechanism as the learned gate of Issue #57 (``off`` / ``hard`` / ``max`` / ``blend``).

Training happens on the GPU (``scripts/train_sequence.py``, PyTorch) on degraded *external* matches only; the
weights are exported to a ``.npz`` file and the inference here is plain NumPy (no torch at run time, fixed
arithmetic order, so a run is reproducible).  The architecture is therefore deliberately simple: a 1x1
input projection, ``len(dilations)`` residual blocks ``h += W2 relu(conv3_d(h))`` (kernel 3, dilation ``d``,
zero padding) and two linear heads.  Features are standardised with the training mean / std and missing
values are set to the mean (zero after standardisation).
"""
# ruff: noqa: E501
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from mtah2027.config import load_section, params_from_mapping

MODEL_FORMAT = 1
MODES = ("off", "emissions", "turnover", "both")
GATE_MODES = ("hard", "max", "blend")
PROB_DECIMALS = 6                      # emission / turnover probabilities are quantised before they are mixed
JSON_FORMAT = 1
FEATURE_DROP_PREFIX = "ctx:"          # hand-made context windows: the network sees the context itself


@dataclass(frozen=True)
class SequenceParams:
    enabled: bool = False
    mode: str = "off"                  # off | emissions | turnover | both
    emission_weight: float = 0.5       # hidden frames: log p = (1 - w) log p_prod + w * emission_scale * log p_seq
    emission_scale: float = 0.5        # tempering of the sequence log-probabilities (as specialist_fusion.scale_hidden)
    visible_weight: float = 0.0        # Issue #73: same mix on ball-visible frames (0: visible frames untouched)
    prob_floor: float = 0.001
    turnover_gate_mode: str = "max"    # hard | max | blend (see transition_matrix.learned_gate)
    turnover_threshold: float = 0.5
    turnover_blend: float = 0.5
    half_window: int = 5               # frames: targets of the turnover head

    @classmethod
    def from_config(cls, path=None) -> "SequenceParams":
        p = params_from_mapping(cls, load_section("sequence_model", path), "sequence_model")
        if p.mode not in MODES:
            raise ValueError(f"sequence_model.mode must be one of {MODES}, got {p.mode!r}")
        if p.turnover_gate_mode not in GATE_MODES:
            raise ValueError(f"sequence_model.turnover_gate_mode must be one of {GATE_MODES}")
        return p

    @property
    def use_emissions(self) -> bool:
        return self.enabled and self.mode in ("emissions", "both")

    @property
    def use_turnover(self) -> bool:
        return self.enabled and self.mode in ("turnover", "both")


def turnover_targets(codes: np.ndarray, half_window: int = 5) -> np.ndarray:
    """Per-frame class 0 none / 1 H -> A / 2 A -> H within ``+- half_window`` of a direct change (as C)."""
    c = np.asarray(codes)
    n = len(c)
    y = np.zeros(n, dtype=np.int64)
    prev, cur = c[:-1], c[1:]
    events = sorted([(int(t) + 1, 1) for t in np.flatnonzero((prev == 0) & (cur == 1))]
                    + [(int(t) + 1, 2) for t in np.flatnonzero((prev == 1) & (cur == 0))])
    for t, cls in events:
        y[max(0, t - half_window):min(n, t + half_window + 1)] = cls
    return y


EXTRA_NAMES = tuple(f"sw:{k}" for k in ("swarm_home", "swarm_away", "carrier_vx_home", "carrier_vx_away",
                                         "carrier_x_home", "carrier_x_away", "min_interteam_dist",
                                         "duel_density_3m")) + ("diff:swarm", "diff:carrier_vx", "diff:carrier_x")


def extra_names(feature_names: Sequence[str]) -> list[str]:
    """Issue #74: the carrier / swarm columns (Issue #64) among ``feature_names`` (not part of the specialists'
    feature matrix: the pipeline computes them from the front end when a model asks for them)."""
    return [k for k in feature_names if k in EXTRA_NAMES]


def select_columns(names: Sequence[str]) -> list[str]:
    """Feature names fed to the network (all but the ``ctx:`` columns)."""
    return [k for k in names if not k.startswith(FEATURE_DROP_PREFIX)]


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class SequenceModel:
    """Fitted TCN (weights as NumPy arrays)."""

    def __init__(self, weights: dict, meta: dict | None = None):
        self.w = {k: np.asarray(v, dtype=np.float32) for k, v in weights.items()}     # as trained / stored
        self._d = {k: v.astype(np.float64) for k, v in self.w.items()}                # exact float32 values, float64 maths
        self.meta = dict(meta or {})
        self.feature_names: list[str] = list(self.meta["feature_names"])
        self.dilations: list[int] = [int(d) for d in self.meta["dilations"]]

    def extra_names(self) -> list[str]:
        return extra_names(self.feature_names)

    # -------------------------------------------------------------------------------------------
    def _conv3(self, h: np.ndarray, w: np.ndarray, b: np.ndarray, d: int) -> np.ndarray:
        """``y[t] = sum_k h[t + (k - 1) d] @ w[k] + b`` with zero padding (``w[3, C, C]``)."""
        n = len(h)
        pad = np.zeros((n + 2 * d, h.shape[1]), np.float64)
        pad[d:d + n] = h
        return pad[:n] @ w[0] + pad[d:d + n] @ w[1] + pad[2 * d:2 * d + n] @ w[2] + b

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(emission_logits[N, 3], turnover_logits[N, 3])`` of one standardised block ``x[N, F]``."""
        w = self._d
        h = x.astype(np.float64) @ w["w_in"] + w["b_in"]
        for i, d in enumerate(self.dilations):
            r = np.maximum(self._conv3(h, w[f"w1_{i}"], w[f"b1_{i}"], d), 0.0)
            h = h + r @ w[f"w2_{i}"] + w[f"b2_{i}"]
        return h @ w["w_e"] + w["b_e"], h @ w["w_t"] + w["b_t"]

    def standardise(self, x: np.ndarray, names: Sequence[str]) -> np.ndarray:
        idx = [list(names).index(k) for k in self.feature_names]
        z = (x[:, idx].astype(np.float32).astype(np.float64) - self._d["mean"]) / self._d["std"]
        return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

    def predict(self, x: np.ndarray, names: Sequence[str], blocks: Sequence[tuple[int, int]]
                ) -> tuple[np.ndarray, np.ndarray]:
        """Emission probabilities ``[N, 3]`` and turnover probabilities ``[N, 3]`` (NaN outside ``blocks``)."""
        if not set(self.feature_names) <= set(names):
            raise ValueError("feature layout does not match the sequence model")
        n = len(x)
        pe = np.full((n, 3), np.nan, dtype=np.float64)
        pt = np.full((n, 3), np.nan, dtype=np.float64)
        z = self.standardise(x, names)
        for a, b in blocks:
            a, b = int(a), min(int(b), n - 1)
            if b < a:
                continue
            le, lt = self.forward(z[a:b + 1])
            # float64 maths, then quantised: BLAS differences between machines (1e-12) cannot reach the decoder
            pe[a:b + 1] = np.round(_softmax(le), PROB_DECIMALS)
            pt[a:b + 1] = np.round(_softmax(lt), PROB_DECIMALS)
        return pe, pt

    # -------------------------------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, __meta__=np.array(json.dumps({"format": MODEL_FORMAT, **self.meta}, sort_keys=True)),
                 **self.w)
        return out

    def to_json_dict(self) -> dict:
        """Weights as shortest-repr float32 decimals: ``np.float32(float(v)) == v`` for every stored value."""
        weights = {k: {"shape": list(v.shape), "values": [float(str(x)) for x in v.ravel()]}
                   for k, v in sorted(self.w.items())}
        return {"meta": {"format": MODEL_FORMAT, **self.meta}, "weights": weights}

    @classmethod
    def from_json_dict(cls, d: dict) -> "SequenceModel":
        meta = dict(d["meta"])
        if meta.get("format") != MODEL_FORMAT:
            raise ValueError(f"unsupported sequence model format {meta.get('format')!r}")
        w = {k: np.asarray(v["values"], dtype=np.float32).reshape(v["shape"]) for k, v in d["weights"].items()}
        return cls(w, meta)

    @classmethod
    def load(cls, path: str | Path) -> "SequenceModel":
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["__meta__"]))
        if meta.get("format") != MODEL_FORMAT:
            raise ValueError(f"unsupported sequence model format {meta.get('format')!r}")
        return cls({k: z[k] for k in z.files if k != "__meta__"}, meta)


class SequenceEnsemble:
    """Mean of the (quantised) softmax probabilities of several TCNs, e.g. different training seeds."""

    def __init__(self, members: Sequence[SequenceModel], meta: dict | None = None):
        if not members:
            raise ValueError("an ensemble needs at least one member")
        self.members = list(members)
        self.meta = dict(meta or {})

    def extra_names(self) -> list[str]:
        return sorted({k for m in self.members for k in m.extra_names()})

    def predict(self, x: np.ndarray, names: Sequence[str], blocks: Sequence[tuple[int, int]]
                ) -> tuple[np.ndarray, np.ndarray]:
        outs = [m.predict(x, names, blocks) for m in self.members]
        pe = np.round(np.mean([o[0] for o in outs], axis=0), PROB_DECIMALS)
        pt = np.round(np.mean([o[1] for o in outs], axis=0), PROB_DECIMALS)
        return pe, pt

    def to_json_dict(self) -> dict:
        return {"format": JSON_FORMAT, "kind": "tcn_mean_softmax", "meta": self.meta,
                "members": [m.to_json_dict() for m in self.members]}


def save_sequence_json(model, path: str | Path, meta: dict | None = None) -> Path:
    """Write a model or ensemble as JSON (``models/sequence_tcn.json``)."""
    ens = model if isinstance(model, SequenceEnsemble) else SequenceEnsemble([model], {**model.meta, **(meta or {})})
    if meta and ens is model:
        ens.meta.update(meta)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ens.to_json_dict(), sort_keys=True, separators=(",", ":")) + chr(10),
                   encoding="utf-8", newline=chr(10))
    return out


def load_sequence_json(path: str | Path) -> "SequenceModel | SequenceEnsemble":
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if d.get("format") != JSON_FORMAT or d.get("kind") != "tcn_mean_softmax":
        raise ValueError("unsupported sequence model JSON")
    members = [SequenceModel.from_json_dict(m) for m in d["members"]]
    return members[0] if len(members) == 1 else SequenceEnsemble(members, d.get("meta", {}))


def random_model(feature_names: Sequence[str], *, channels: int = 8, dilations: Sequence[int] = (1, 2, 4),
                 seed: int = 0) -> SequenceModel:
    """Small random-weight model (tests, smoke runs of the integration)."""
    rng = np.random.default_rng(seed)
    f = len(feature_names)
    w: dict = {"mean": np.zeros(f, np.float32), "std": np.ones(f, np.float32),
               "w_in": rng.normal(0, 0.3, (f, channels)), "b_in": np.zeros(channels),
               "w_e": rng.normal(0, 0.3, (channels, 3)), "b_e": np.zeros(3),
               "w_t": rng.normal(0, 0.3, (channels, 3)), "b_t": np.zeros(3)}
    for i, _ in enumerate(dilations):
        w[f"w1_{i}"] = rng.normal(0, 0.2, (3, channels, channels))
        w[f"b1_{i}"] = np.zeros(channels)
        w[f"w2_{i}"] = rng.normal(0, 0.2, (channels, channels))
        w[f"b2_{i}"] = np.zeros(channels)
    return SequenceModel(w, {"feature_names": list(feature_names), "dilations": list(dilations),
                             "channels": channels})


def mix_hidden_emissions(log_probs: np.ndarray, seq_prob: np.ndarray, visible: np.ndarray,
                         p: SequenceParams) -> np.ndarray:
    """Mixed frames: ``(1 - w) log_probs + w * scale * log p_seq`` with ``w = emission_weight`` on hidden-ball
    frames and ``w = visible_weight`` (Issue #73, default 0: untouched) on ball-visible frames; NaN rows untouched."""
    lp = np.asarray(log_probs, dtype=np.float64)
    vis = np.asarray(visible, dtype=bool)
    finite = np.isfinite(seq_prob).all(axis=1)
    w = np.where(vis, p.visible_weight, p.emission_weight)
    mix = finite & (w != 0.0)
    if not mix.any():
        return log_probs
    out = lp.copy()
    ls = p.emission_scale * np.log(np.maximum(seq_prob[mix], p.prob_floor))
    wm = w[mix][:, None]
    out[mix] = (1.0 - wm) * lp[mix] + wm * ls
    return np.round(out, 6)


def gate_probabilities(turnover_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(p_ha, p_ah)`` as consumed by ``transition_matrix.learned_gate`` (NaN -> 0 there)."""
    return turnover_prob[:, 1], turnover_prob[:, 2]


def with_gate(tp, p: SequenceParams):
    """TransitionParams with the learned gate set from the sequence parameters."""
    return dataclasses.replace(tp, learned_gate_mode=p.turnover_gate_mode,
                               learned_gate_threshold=p.turnover_threshold, learned_gate_blend=p.turnover_blend)
