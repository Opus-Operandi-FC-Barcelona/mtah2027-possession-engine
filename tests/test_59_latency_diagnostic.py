"""Issue #59: signed-latency diagnostic of eval_external (pred - gt, split by transition type)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import eval_external as ee  # noqa: E402


def test_latency_by_type_sign_and_split():
    pairs = [((100, 0, 1), (97, 0, 1)),      # H->A predicted 3 frames early
             ((200, 0, 2), (202, 0, 2)),     # alive -> D predicted 2 frames late
             ((300, 2, 0), (300, 2, 0))]     # restart exact
    out = ee._latency_by_type(pairs)
    assert out == {"ha": [-3], "to_d": [2], "from_d": [0]}


def test_score_reports_signed_latency():
    n = 200
    gt = np.array(["H"] * 100 + ["A"] * 100)
    pred = np.array(["H"] * 97 + ["A"] * 103)
    row = ee.score(gt, pred)
    assert row["lat_by_type"]["ha"] == [-3]
    assert row["lat_median"] == -3.0 and len(gt) == n


def test_latency_summary_signs():
    rows = [{"engine": {"lat_median": m}} for m in (-1.0, -2.0, 0.0, -1.0)]
    text = ee.latency_summary(rows, "engine")
    assert "early 75%" in text and "late 0%" in text
    assert np.isclose(np.median([-1, -2, 0, -1]), -1.0)
