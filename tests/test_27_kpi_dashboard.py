"""Issue #27: the seven-KPI dashboard (aggregate numbers only)."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from mtah2027.config import ConfigError, KpiAlarms, MatchConfig, params_from_mapping
from mtah2027.pipeline import run_pipeline
from mtah2027.postprocessing.submission_contract import write_submission_csv
from mtah2027.validation import eval_bridge as eb
from mtah2027.validation import kpi_dashboard as kd

FPS = 25.0
H, A, D, X = 0, 1, 2, 3
ROOT = Path(__file__).resolve().parents[1]


def _blocks(spec) -> MatchConfig:
    return MatchConfig(
        n_video_frames=spec.n_frames, half1=spec.half1, half2_dead=spec.half2_dead, half2_play=spec.half2_play
    )


# ---------------------------------------------------------------------------------- config
def test_yaml_section_matches_defaults_and_rejects_typos():
    assert kd.KpiParams.from_config() == kd.KpiParams()
    with pytest.raises(ConfigError):
        params_from_mapping(kd.KpiParams, {"audit": 20}, "kpi_dashboard")


# ---------------------------------------------------------------------------------- KPI 1
def test_statistics_and_alarms():
    codes = np.array([H] * 100 + [D] * 50 + [A] * 150 + [H] * 100 + [X] * 20, dtype=np.int8)
    stats = kd.label_statistics(codes, [(0, 399)], FPS)
    assert stats["graded_frames"] == 400 and stats["n_possessions"] == 3
    assert stats["dead_pct_of_graded"] == pytest.approx(12.5)
    assert stats["home_pct_of_alive"] == pytest.approx(100 * 200 / 350)
    assert stats["mean_possession_s"] == pytest.approx(350 / 3 / FPS)
    res = kd.kpi_alarms(stats, KpiAlarms())
    assert res["status"] == kd.WARN
    assert {i["name"]: i["status"] for i in res["alarms"]} == {
        "possessions": "WARN",
        "mean_duration_s": "WARN",
        "dead_pct_of_graded": "WARN",
        "home_pct_of_alive": "OK",
    }
    wide = KpiAlarms(
        possessions=(1, 10), mean_duration_s=(1, 10), dead_pct_of_graded=(0, 50), home_pct_of_alive=(0, 100)
    )
    assert kd.kpi_alarms(stats, wide)["status"] == kd.OK


# ---------------------------------------------------------------------------------- KPI 3 helpers
def test_transition_latency_matches_same_label_within_window():
    gt = np.array([H] * 100 + [A] * 100 + [D] * 100, dtype=np.int8)
    pred = np.array([H] * 103 + [A] * 95 + [D] * 102, dtype=np.int8)
    lat, n_true = kd.transition_latency(gt, pred, FPS, window_s=2.0)
    assert n_true == 2 and sorted(lat.tolist()) == [2, 3]
    far = np.array([H] * 300, dtype=np.int8)  # no A, no D at all: nothing to match
    lat2, n2 = kd.transition_latency(gt, far, FPS, window_s=2.0)
    assert n2 == 2 and len(lat2) == 0


def test_external_kpi_on_two_fake_matches(external_like_dir):
    res = kd.kpi_external(external_like_dir, params=kd.KpiParams(), lomo=False)
    assert res["n_matches"] == 2 and res["primary_route"] == "heuristic"
    row = res["routes"]["heuristic"]
    assert 0.0 <= row["spell_f1"] <= 1.0 and row["n_matches"] == 2
    assert res["status"] in (kd.OK, kd.WARN)
    assert kd.kpi_external(external_like_dir / "nope", params=kd.KpiParams())["status"] == kd.SKIPPED


# ---------------------------------------------------------------------------------- KPI 4
def test_public_smoke_is_labelled_and_scores_the_public_gt_as_perfect():
    gt = eb.public_gt_codes().copy()
    res = kd.kpi_public_smoke(gt, bootstrap_n=10)
    assert res["note"] == "pipeline check only - not for model selection"
    assert res["spell_f1"] == pytest.approx(1.0) and res["status"] == kd.INFO


# ---------------------------------------------------------------------------------- KPI 5
def test_audit_export_writes_a_bounded_csv_and_reports_only_the_count(tmp_path):
    codes = np.full(2000, H, dtype=np.int8)
    for k in range(60):
        codes[k * 30 : (k + 1) * 30] = (H, A, D)[k % 3]
    out = tmp_path / "sub" / "audit.csv"
    res = kd.kpi_audit(codes, [(0, 1799)], out, 20, FPS)
    lines = out.read_text(encoding="utf-8").splitlines()
    assert res["n_exported"] == 20 and len(lines) == 21
    assert lines[0] == "frame,time_s,clock,from,to"
    frame, t, _clock, a, b = lines[1].split(",")
    assert codes[int(frame)] != codes[int(frame) - 1] and a != b
    assert float(t) == pytest.approx(int(frame) / FPS, abs=0.01)
    assert set(res) <= {"status", "n_exported", "requested", "note"}  # no frame list in the report
    short = kd.kpi_audit(np.array([H] * 10 + [A] * 10, dtype=np.int8), [(0, 19)], tmp_path / "s.csv", 20, FPS)
    assert short["n_exported"] == 1


def test_outputs_folder_is_git_ignored():
    if shutil.which("git") is None:
        pytest.skip("git not available")
    probe = subprocess.run(["git", "check-ignore", "-q", "outputs/audit_transitions.csv"], cwd=ROOT)
    if probe.returncode == 128:
        pytest.skip("not a git checkout")
    assert probe.returncode == 0


# ---------------------------------------------------------------------------------- KPI 6
def test_format_check_and_sha256(tmp_path):
    labels = np.full(500, X, dtype=np.int8)
    labels[100:201] = H
    labels[300:401] = A
    path = tmp_path / "ok.csv"
    write_submission_csv(path, labels, n_expected=500)
    res = kd.kpi_format(path, 500)
    assert res["status"] == kd.OK and res["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    bad = tmp_path / "bad.csv"
    bad.write_text("frame,label\n0,H\n", encoding="utf-8")
    assert kd.kpi_format(bad, 500)["status"] == kd.FAIL
    assert kd.kpi_format(tmp_path / "missing.csv", 500)["status"] == kd.FAIL


# ---------------------------------------------------------------------------------- KPI 7
def test_decision_rule_uses_only_the_external_evidence():
    def ext(h, s):
        return {"routes": {"heuristic": {"spell_f1": h}, "specialists_lomo": {"spell_f1": s}}}

    assert kd.decide_route(ext(0.40, 0.50), None, 0.04)["recommended"] == "specialists"
    assert kd.decide_route(ext(0.50, 0.40), None, 0.04)["recommended"] == "heuristic"
    inside = kd.decide_route(ext(0.50, 0.52), None, 0.04)
    assert inside["recommended"] == "heuristic" and "noise" in inside["basis"]
    assert kd.decide_route(None, None, 0.04)["recommended"] == "undetermined"
    only_h = {"routes": {"heuristic": {"spell_f1": 0.5}}}
    assert kd.decide_route(only_h, None, 0.04)["recommended"] == "undetermined"
    warn = kd.decide_route(ext(0.4, 0.5), {"spell_f1": 0.3}, 0.04)
    assert warn["agreement_note"] and warn["recommended"] == "specialists"


def test_route_agreement_diagnostics():
    base = np.array([H] * 100 + [A] * 100 + [D] * 100 + [H] * 100, dtype=np.int8)
    other = base.copy()
    other[150:160] = H
    res = kd.kpi_routes(other, base, [(0, 399)], FPS, None, 0.04)
    assert res["agreement"]["frame_agreement"] == pytest.approx(1 - 10 / 400)
    assert 0.0 < res["agreement"]["spell_f1"] <= 1.0
    assert res["heuristic"]["possessions"] == 3 and res["specialists"]["possessions"] == 5
    alone = kd.kpi_routes(None, base, [(0, 399)], FPS, None, 0.04)
    assert alone["agreement"] is None and alone["decision"]["recommended"] == "undetermined"


# ---------------------------------------------------------------------------------- full dashboard
def test_dashboard_on_the_synthetic_match(synthetic_dir, synthetic_spec, synthetic_gt, tmp_path):
    blocks = _blocks(synthetic_spec)
    sub = tmp_path / "sub.csv"
    res = run_pipeline(synthetic_dir, blocks=blocks, log=lambda m: None)
    write_submission_csv(sub, res.labels, n_expected=blocks.n_video_frames)
    out_json, audit = tmp_path / "kpi" / "kpi_report.json", tmp_path / "kpi" / "audit.csv"
    truth = eb.to_codes(synthetic_gt)
    report = kd.run_dashboard(
        data_dir=synthetic_dir,
        blocks=blocks,
        specialists=None,
        submission_path=sub,
        skip_external=True,
        real_data=False,
        out_json=out_json,
        audit_path=audit,
        truth=truth,
    )
    k = report["kpis"]
    assert set(k) == {
        "1_alarms",
        "2_blindness",
        "3_external",
        "4_public_smoke",
        "5_audit",
        "6_format",
        "7_routes",
    }
    assert k["1_alarms"]["status"] in (kd.OK, kd.WARN)
    assert 0.0 <= k["2_blindness"]["spell_f1"] <= 1.0 and k["2_blindness"]["status"] in (kd.OK, kd.WARN)
    assert k["3_external"]["status"] == kd.SKIPPED
    assert k["4_public_smoke"]["status"] == kd.SKIPPED
    assert "not for model selection" in k["4_public_smoke"]["note"]
    assert k["5_audit"]["n_exported"] == 20 and audit.is_file()
    digest = hashlib.sha256(sub.read_bytes()).hexdigest()
    assert k["6_format"]["status"] == kd.OK and k["6_format"]["sha256"] == digest
    assert k["7_routes"]["specialists"] is None and report["synthetic_truth"]["heuristic"]["spell_f1"] > 0.0
    saved = json.loads(out_json.read_text(encoding="utf-8"))
    assert saved["kpis"].keys() == k.keys()
    assert out_json.stat().st_size < 20000  # aggregates only, nothing per frame
    table = kd.format_report(report)
    assert "6 format + SHA256" in table and "not for model selection" in table
    assert len(table.splitlines()) < 20


def test_cli_runs_on_a_short_synthetic_match(tmp_path):
    import run_kpi_dashboard as cli

    ap = argparse.ArgumentParser()
    cli.add_arguments(ap)
    out_json = tmp_path / "r.json"
    args = ap.parse_args(
        [
            "--synthetic",
            "--synthetic-frames",
            "12000",
            "--skip-external",
            "--specialists",
            "off",
            "--out-json",
            str(out_json),
            "--audit-csv",
            str(tmp_path / "a.csv"),
        ]
    )
    assert cli.run(args) == 0
    assert json.loads(out_json.read_text(encoding="utf-8"))["kpis"]["6_format"]["status"] == "OK"
    assert (tmp_path / "a.csv").is_file()
