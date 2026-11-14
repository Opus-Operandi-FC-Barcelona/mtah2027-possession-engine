"""Smoke test of scripts/explore_occlusion_team.py (synthetic match saved as two fake benchmarks)."""
from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def explore():
    return importlib.import_module("explore_occlusion_team")


@pytest.fixture(scope="module")
def bench_dir(tmp_path_factory, synthetic_kin):
    td, spec = synthetic_kin["td"], synthetic_kin["spec"]
    labels = np.asarray(synthetic_kin["labels"])
    out = tmp_path_factory.mktemp("bench")
    ball = pd.DataFrame(td.ball_xy).ffill().bfill().to_numpy().astype(np.float32)      # complete ball
    for name in ("synth_a", "synth_b"):
        np.savez(out / f"{name}.npz", players_xy=td.players_xy, team_of_slot=td.team_of_slot.astype(np.int8),
                 ball_xy=ball, fps=np.int64(25), pitch=np.array([105.3, 68.0]),
                 period_bounds=np.array([spec.half1, spec.half2_play]), gt=labels,
                 frame_ids=np.arange(len(labels)))
    return out


def test_dataset_and_leave_one_match_out(explore, bench_dir):
    datasets = [explore.build_occlusion_dataset(bench_dir / f"{name}.npz", seed, with_engine=False)
                for seed, name in enumerate(("synth_a", "synth_b"))]
    d = datasets[0]
    assert d["X"].shape == (len(d["y"]), len(d["names"])) and d["X"].shape[1] > 60
    assert set(np.unique(d["y"])) <= {0, 1} and len(d["rule_home"]) == len(d["y"])
    assert d["hidden_share"] == pytest.approx(0.424, abs=0.01)
    assert len(d["groups"]["physics"]) > 20 and len(d["groups"]["memory"]) > 20
    res = explore.leave_one_match_out(datasets, stride=5, importance=True)
    assert 0.0 <= res["mean"] <= 1.0 and len(res["acc"]) == 2
    assert next(iter(res["importance"])) in d["names"]


def test_main_runs_and_reports(explore, bench_dir, capsys, tmp_path):
    out = tmp_path / "occ.json"
    assert explore.main(["--bench-dir", str(bench_dir), "--seeds", "0", "--no-engine", "--stride", "5",
                         "--json", str(out)]) == 0
    text = capsys.readouterr().out
    assert "model (#17+#18)" in text and "rule: middle change point" in text
    assert out.is_file()
