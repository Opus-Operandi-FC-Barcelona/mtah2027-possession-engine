"""Bridge to the vendored official scorer (``evaluation/possession_metrics.py``, Issue #4).

Nothing is re-implemented here: this module imports ``possession_metrics`` and exposes it for
in-memory arrays so that validation, tests and the KPI dashboard can score a prediction in
milliseconds.  ``evaluate_arrays`` returns exactly the dict of ``possession_metrics.evaluate``.

Labels are accepted either as strings ('H', 'A', 'D', 'X', case/whitespace tolerant, like the
official reader) or as integer codes with the official numbering H=0, A=1, D=2, X=3.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
EVALUATION_DIR = REPO_ROOT / "evaluation"
for _path in (str(REPO_ROOT), str(EVALUATION_DIR)):
    if _path not in sys.path:
        sys.path.append(_path)

import possession_metrics as pm  # noqa: E402  (vendored official scorer, needs the paths above)

LABELS = pm.LABELS                      # ('H', 'A', 'D', 'X')
H, A, D, X = pm.H, pm.A, pm.D, pm.X
N_VIDEO_FRAMES = 199319
PUBLIC_GT_PATH = EVALUATION_DIR / "gt" / "BAR-ATH_possession_gt_public.csv"

_public_gt_codes: np.ndarray | None = None


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of the official format check; truthy when the submission is valid."""

    ok: bool
    reason: str | None = None

    def __bool__(self) -> bool:
        return self.ok


def to_codes(labels) -> np.ndarray:
    """Labels (strings or codes) -> int8 codes 0..3 in the official order (H, A, D, X)."""
    arr = np.asarray(labels)
    if arr.ndim != 1:
        raise ValueError("labels must be a 1-D sequence")
    if arr.dtype.kind in "iu":
        if len(arr) and (arr.min() < 0 or arr.max() >= len(LABELS)):
            raise ValueError(f"label codes outside 0..{len(LABELS) - 1}")
        return arr.astype(np.int8)
    if arr.dtype.kind == "b":
        raise ValueError("boolean labels are not valid")
    text = np.char.upper(np.char.strip(arr.astype(str)))
    codes = np.full(len(text), -1, dtype=np.int8)
    for index, name in enumerate(LABELS):
        codes[text == name] = index
    if (codes < 0).any():
        raise ValueError(f"labels outside {list(LABELS)}: {sorted(set(text[codes < 0].tolist()))[:5]}")
    return codes


def to_letters(labels) -> np.ndarray:
    """Labels (strings or codes) -> object array of 'H'/'A'/'D'/'X'."""
    return np.array(LABELS, dtype=object)[to_codes(labels)]


def evaluate_arrays(gt_labels, pred_labels, taus: Sequence[int] = pm.TAUS, bootstrap_n: int = 0,
                    typed: str = "to") -> dict:
    """Every official metric for two label arrays of equal length (``possession_metrics.evaluate``)."""
    gt, pred = to_codes(gt_labels), to_codes(pred_labels)
    if len(gt) != len(pred):
        raise ValueError(f"length mismatch: gt {len(gt)} vs pred {len(pred)}")
    return pm.evaluate(gt, pred, taus=tuple(taus), bootstrap_n=bootstrap_n, typed=typed)


def spell_f1(gt_labels, pred_labels) -> float:
    """Ranking metric: mean possession-spell F1 over IoU 0.3-0.7 (``spell_f1_mean``)."""
    return float(evaluate_arrays(gt_labels, pred_labels)["spell_f1_mean"])


def macro_f1(gt_labels, pred_labels) -> float:
    """Tie-break metric: unweighted mean of the per-frame F1 of H, A and D."""
    return float(evaluate_arrays(gt_labels, pred_labels)["macro_f1"])


def validate_labels_array(labels, n_expected: int = N_VIDEO_FRAMES) -> ValidationResult:
    """Run the official ``validate_submission`` on an in-memory label array (frame = 0..n-1)."""
    try:
        arr = np.asarray(labels)
        if arr.ndim != 1:
            return ValidationResult(False, "labels must be a 1-D sequence")
        raw = to_letters(arr) if arr.dtype.kind in "iu" else arr.astype(object)
        table = pd.DataFrame({"frame": np.arange(len(raw)), "label": raw})
    except (TypeError, ValueError) as error:
        return ValidationResult(False, f"could not interpret labels: {error}")
    reason = pm.validate_submission(table, n_expected)
    return ValidationResult(reason is None, reason)


def validate_submission_file(path, n_expected: int = N_VIDEO_FRAMES) -> ValidationResult:
    """Run the official format check on a ``frame,label`` CSV (same rules as validate_submission.py)."""
    if not os.path.isfile(path):
        return ValidationResult(False, f"no such file: {path}")
    try:
        table = pm.read_label_csv(path)
    except Exception as error:  # unparseable CSV or missing columns, reported rather than raised
        return ValidationResult(False, f"could not read: {error}")
    reason = pm.validate_submission(table, n_expected)
    return ValidationResult(reason is None, reason)


def read_submission_codes(path, n_expected: int = N_VIDEO_FRAMES) -> np.ndarray:
    """Codes of a valid submission CSV; raises ``ValueError`` with the official reason otherwise."""
    result = validate_submission_file(path, n_expected)
    if not result:
        raise ValueError(f"{path}: {result.reason}")
    return pm.label_codes(pm.read_label_csv(path))


def public_gt_codes() -> np.ndarray:
    """Codes of the public 5-minute ground truth (199,319 rows, X outside 11185..18684)."""
    global _public_gt_codes
    if _public_gt_codes is None:
        table = pm.read_label_csv(PUBLIC_GT_PATH)
        reason = pm.validate_submission(table, N_VIDEO_FRAMES)
        if reason:
            raise ValueError(f"{PUBLIC_GT_PATH}: {reason}")
        _public_gt_codes = pm.label_codes(table)
        _public_gt_codes.setflags(write=False)
    return _public_gt_codes


def score_against_public_gt(labels, taus: Sequence[int] = pm.TAUS, bootstrap_n: int = 0) -> dict:
    """Official metrics of a full-length prediction against the public ground truth.

    Smoke test of alignment and format only: never use it to choose between variants.
    """
    pred = to_codes(labels)
    if len(pred) != N_VIDEO_FRAMES:
        raise ValueError(f"expected {N_VIDEO_FRAMES} labels, got {len(pred)}")
    return evaluate_arrays(public_gt_codes(), pred, taus=taus, bootstrap_n=bootstrap_n)
