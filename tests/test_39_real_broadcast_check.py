"""Issue #39: aggregate coverage report of the broadcast synchronisation (synthetic bundle only)."""
from __future__ import annotations

import json

import numpy as np
import pytest
import real_broadcast_check as rbc

from mtah2027.config import MatchConfig


def _blocks(spec) -> MatchConfig:
    return MatchConfig(n_video_frames=spec.n_frames, half1=spec.half1, half2_dead=spec.half2_dead,
                       half2_play=spec.half2_play)


def test_report_on_the_synthetic_bundle(synthetic_dir, synthetic_spec):
    rep = rbc.build_report(synthetic_dir, _blocks(synthetic_spec))
    st = rep["blocks"]["stats"]
    assert st["blocks"] > 0 and 0 <= st["blocks_accepted"] <= st["blocks"]
    assert 0.0 <= rep["fill"]["hidden_graded_frames_filled_pct"] <= 100.0
    assert rep["physical"]["injected_frames"] == st["frames_filled"]
    assert rep["fill"]["ball_missing_after_pct_of_graded"] <= rep["hidden_graded_pct_of_graded"] + 1e-6
    text = json.dumps(rep)
    assert len(text) < 8000                              # aggregates only


def test_physical_consistency_flags_jumps_and_off_pitch_points():
    n, fps = 400, 25.0
    before = np.full((n, 2), np.nan)
    before[:100] = [50.0, 30.0]
    before[200:] = [52.0, 31.0]
    after = before.copy()
    src = np.where(np.isfinite(before[:, 0]), 1, 0).astype(np.uint8)
    after[100:200] = np.linspace([50.1, 30.0], [52.0, 31.0], 100)       # smooth bridge
    src[100:200] = 4
    out = rbc.physical_consistency(before, after, src, fps, (105.3, 68.0))
    assert out["injected_frames"] == 100 and out["injected_runs"] == 1 and out["off_pitch_pct"] == 0.0
    assert out["in_run_speed_gt35_pct"] == 0.0
    assert out["border_jump_into_run"]["gt35_pct"] == 0.0 and out["border_jump_out_of_run"]["gt35_pct"] == 0.0
    bad = after.copy()
    bad[150:] = [500.0, 30.0]                                           # teleport + off pitch
    out = rbc.physical_consistency(before, bad, src, fps, (105.3, 68.0))
    assert out["off_pitch_pct"] > 0 and out["in_run_speed_gt35_pct"] > 0
    assert out["border_jump_out_of_run"]["gt35_pct"] == 100.0


def test_runs_helper():
    s, e = rbc.runs(np.array([0, 1, 1, 0, 1, 0, 0, 1, 1], bool))
    assert s.tolist() == [1, 4, 7] and e.tolist() == [3, 5, 9]
    with pytest.raises(AttributeError):
        rbc.nonexistent  # noqa: B018
