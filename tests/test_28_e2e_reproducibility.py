"""Issue #28 (wave 1): heuristic emissions, array-level core, end-to-end pipeline and CLI reproducibility."""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from mtah2027 import pipeline
from mtah2027.config import ConfigError, MatchConfig, params_from_mapping
from mtah2027.models import baseline_labels as bl
from mtah2027.models import heuristic_emissions as he
from mtah2027.postprocessing.submission_contract import enforce_submission_contract, write_submission_csv
from mtah2027.validation import eval_bridge as eb

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "generate_submission.py"
H, A, D, X = 0, 1, 2, 3
EP = he.EmissionParams()


def _blocks(spec) -> MatchConfig:
    return MatchConfig(n_video_frames=spec.n_frames, half1=spec.half1, half2_dead=spec.half2_dead,
                       half2_play=spec.half2_play)


def _em(n=300, **kw):
    base = dict(dead_score=np.full(n, 0.05), possession_evidence=np.zeros(n), evidence_valid=np.ones(n, bool),
                ball_visible=np.ones(n, bool))
    base.update(kw)
    return he.compute_emissions(base.pop("dead_score"), base.pop("possession_evidence"),
                                base.pop("evidence_valid"), base.pop("ball_visible"), params=EP, **base)


# ---------------------------------------------------------------------------- emissions
def test_emission_params_from_yaml_and_typos():
    assert he.EmissionParams.from_config() == EP
    with pytest.raises(ConfigError):
        params_from_mapping(he.EmissionParams, {"w_evidence_": 1}, "heuristic_emissions")


def test_emission_shape_rounding_and_dead_follows_score():
    ds = np.concatenate([np.full(100, 0.02), np.full(100, 0.98), np.full(100, 0.5)])
    r = _em(dead_score=ds)
    assert r.log_probs.shape == (300, 3) and r.log_probs.dtype == np.float64
    assert np.array_equal(r.log_probs, np.round(r.log_probs, 6))
    assert (r.log_probs[:100].argmax(1) != D).all() and (r.log_probs[100:200].argmax(1) == D).all()
    assert (r.log_probs <= 0).all() and r.log_probs.min() >= np.log(EP.prob_floor) - 1e-6
    assert np.allclose(np.exp(r.log_probs).sum(1), 1.0, atol=0.01)


def test_evidence_sign_decides_team_and_status_channel_pushes_dead():
    ev = np.concatenate([np.full(100, 0.8), np.full(100, -0.8)])
    r = _em(200, possession_evidence=ev, dead_score=np.full(200, 0.05))
    assert (r.log_probs[:100].argmax(1) == H).all() and (r.log_probs[100:].argmax(1) == A).all()
    st = np.concatenate([np.full(100, 1.0), np.full(100, 0.0)])
    r2 = _em(200, dead_score=np.full(200, 0.5), status_dead=st)
    assert (r2.log_probs[:100].argmax(1) == D).all() and (r2.log_probs[100:].argmax(1) != D).all()


def test_last_toucher_is_carried_forward_and_across_a_gap_in_both_directions():
    n = 400
    touches = np.full(n, np.nan)
    touches[50], touches[350] = 1.0, -1.0              # home touch, then an away touch after the gap
    vis = np.ones(n, bool)
    vis[100:300] = False
    valid = vis.copy()
    ev = np.zeros(n)
    r = _em(n, possession_evidence=ev, evidence_valid=valid, ball_visible=vis, touch_signed=touches)
    assert r.meta["with_touches"] is True
    team = np.where(r.home_logit > 0, H, A)
    assert (team[50:100] == H).all() and (team[300:350] == H).all()        # visible: forward carry only
    assert (team[100:150] == H).all() and (team[250:300] == A).all()       # gap: the change sits mid-gap
    assert 150 <= int(np.argmax(team[100:300] == A)) + 100 <= 250
    no_back = he.compute_emissions(np.full(n, 0.05), ev, valid, vis, touch_signed=touches,
                                   params=he.EmissionParams(carry_backward=False))
    assert (no_back.home_logit[100:300] > 0).all()                         # without the backward carry


def test_without_touches_confident_evidence_is_carried():
    n = 300
    ev = np.zeros(n)
    ev[:50] = 0.9
    vis = np.ones(n, bool)
    vis[50:] = False
    r = _em(n, possession_evidence=ev, evidence_valid=vis, ball_visible=vis)
    assert r.meta["with_touches"] is False and (r.home_logit[50:] > 0).all()
    assert r.home_logit[299] < r.home_logit[60]                            # confidence decays


def test_touch_signal_and_owner_channel_and_phase_prior():
    mask = np.array([0, 1, 0, 1, 1], bool)
    gap = np.array([5.0, 3.0, 1.0, -0.4, 0.0])
    sig = he.touch_signal(mask, gap, 1.0)
    assert np.isnan(sig[[0, 2, 4]]).all() and sig[1] == 1.0 and sig[3] == pytest.approx(-0.4)
    owner = np.full(100, np.nan)
    owner[:50], owner[50:] = 0.0, 1.0
    r = _em(100, owner_team=owner)
    assert (r.home_logit[:50] > 0).all() and (r.home_logit[50:] < 0).all()
    r2 = _em(100, phase_prior_home=np.full(100, 1.0))
    assert (r2.home_logit == pytest.approx(EP.w_phase)) and r2.meta["with_phase"]
    r3 = _em(100, phase_prior_home=np.full(100, -1.0))
    assert abs(r3.home_logit[0]) < EP.w_touch                              # the prior is weak


def test_resolve_owner_team_variants():
    n = 6
    arr = np.array([1, 1, 2, 2, 0, 1], dtype=np.float32)
    table = {"FIFATMA": 1, "FIFATMB": 2}
    out, key = he.resolve_owner_team({"owner": arr}, {"owner": table}, EP.owner_channels,
                                     ("FIFATMA", "FIFATMB"), ("Barcelona", "Athletic"), n)
    assert key == "owner" and np.array_equal(out[:4], [0, 0, 1, 1]) and np.isnan(out[4])
    out2, _ = he.resolve_owner_team({"possession": arr}, {"possession": {"away": 2, "home": 1}},
                                    EP.owner_channels, (), (), n)
    assert np.array_equal(out2[[0, 2]], [0, 1])
    assert he.resolve_owner_team({"x": arr}, None, EP.owner_channels)[0] is None
    assert he.resolve_owner_team({"speed": arr, "z": arr}, None, EP.owner_channels)[0] is None
    assert he.resolve_owner_team(None, None, EP.owner_channels)[0] is None
    # a numeric channel without a code table is only trusted when its values are the team ids
    ids = np.array([10, 10, 20, 20, 30, 10], dtype=np.float32)
    out3, _ = he.resolve_owner_team({"team": ids}, None, EP.owner_channels, ("10", "20"), ("a", "b"), n)
    assert np.array_equal(out3[:4], [0, 0, 1, 1]) and np.isnan(out3[4])
    unknown_ids = ("FIFATMA", "FIFATMB")
    assert he.resolve_owner_team({"team": ids}, None, EP.owner_channels, unknown_ids, (), n)[0] is None


def test_emission_input_validation():
    with pytest.raises(ValueError):
        he.compute_emissions(np.zeros(5), np.zeros(4), np.ones(5, bool), np.ones(5, bool), params=EP)


# ---------------------------------------------------------------------------- baseline
def test_baseline_nearest_player_and_majority():
    team = np.array([0, 0, 1, 1])
    players = np.zeros((6, 4, 2), dtype=np.float32)
    players[:, 0] = [0, 0]
    players[:, 1] = [50, 0]
    players[:, 2] = [10, 0]
    players[:, 3] = [90, 0]
    ball = np.array([[1, 0], [1, 0], [11, 0], [11, 0], [np.nan, np.nan], [np.nan, np.nan]], dtype=np.float32)
    nt = bl.nearest_player_team(players, team, ball)
    assert nt.tolist() == [0, 0, 1, 1, -1, -1]
    assert bl.smooth_majority(nt, 0).tolist() == [0, 0, 1, 1, 2, 2]
    sm = bl.smooth_majority(nt, 1)
    assert sm.tolist()[:4] == [0, 0, 0, 1] or sm.tolist()[:4] == [0, 0, 1, 1]
    assert (bl.smooth_majority(np.full(10, -1), 3) == D).all()
    assert bl.baseline_labels(players, team, ball, 0).tolist() == [0, 0, 1, 1, 2, 2]


# ---------------------------------------------------------------------------- pipeline on the synthetic match
@pytest.fixture(scope="module")
def short_run(synthetic_dir, synthetic_spec):
    blocks = _blocks(synthetic_spec)
    return pipeline.run_pipeline(synthetic_dir, None, blocks, log=lambda m: None), blocks


def test_pipeline_output_obeys_the_contract(short_run, synthetic_spec):
    result, blocks = short_run
    lab = result.labels
    assert lab.shape == (synthetic_spec.n_frames,) and lab.dtype == np.int8
    assert result.diagnostics["route"] == "heuristic"
    assert (lab[:blocks.half1[0]] == X).all() and (lab[blocks.half1[1] + 1:blocks.half2_dead[0]] == X).all()
    assert (lab[blocks.half2_dead[0]:blocks.half2_dead[1] + 1] == D).all()
    assert (lab[blocks.half2_play[1] + 1:] == X).all()
    graded = np.zeros(len(lab), bool)
    graded[blocks.half1[0]:blocks.half1[1] + 1] = True
    graded[blocks.half2_dead[0]:blocks.half2_play[1] + 1] = True
    assert not (lab[graded] == X).any()
    assert eb.validate_labels_array(lab, synthetic_spec.n_frames)
    stages = result.diagnostics["core"]["stages"]
    assert all(v["status"] in ("ok",) or v["status"].startswith("skipped") for v in stages.values())
    assert result.diagnostics["core"]["n_anchors"] > 0


def test_pipeline_twice_gives_identical_sha256(synthetic_dir, synthetic_spec, short_run, tmp_path):
    blocks = _blocks(synthetic_spec)
    again = pipeline.run_pipeline(synthetic_dir, None, blocks, log=lambda m: None).labels
    hashes = []
    for i, labels in enumerate((short_run[0].labels, again)):
        path = write_submission_csv(tmp_path / f"run{i}.csv", labels)
        hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
        assert eb.validate_submission_file(path, synthetic_spec.n_frames)
    assert hashes[0] == hashes[1]


def test_numba_and_fallback_decoders_give_the_same_file(synthetic_dir, synthetic_spec, short_run,
                                                        monkeypatch):
    monkeypatch.setenv("MTAH_DISABLE_NUMBA", "1")
    fallback = pipeline.run_pipeline(synthetic_dir, None, _blocks(synthetic_spec), log=lambda m: None).labels
    assert np.array_equal(fallback, short_run[0].labels)


def test_synthetic_score_beats_the_nearest_player_baseline(short_run, synthetic_gt, synthetic_dir,
                                                           synthetic_spec):
    from mtah2027.io.atd_loader import load_atd

    result, blocks = short_run
    gt = eb.to_codes(synthetic_gt)
    engine = eb.evaluate_arrays(gt, result.labels)
    td = load_atd(synthetic_dir, cache_dir=None, use_cache=False, n_frames=synthetic_spec.n_frames)
    base = enforce_submission_contract(bl.baseline_labels(td.players_xy, td.team_of_slot, td.ball_xy), blocks)
    baseline = eb.evaluate_arrays(gt, base)
    assert engine["spell_f1_mean"] > baseline["spell_f1_mean"]
    assert engine["macro_f1"] > baseline["macro_f1"]


def test_missing_smart_tagging_is_skipped_not_fatal(synthetic_dir, synthetic_spec, tmp_path):
    import shutil

    clone = tmp_path / "bundle"
    shutil.copytree(synthetic_dir, clone)
    for f in clone.rglob("*smart_tagging.xml"):
        f.unlink()
    res = pipeline.run_pipeline(clone, None, _blocks(synthetic_spec), log=lambda m: None)
    assert res.diagnostics["route"] == "heuristic"
    assert res.diagnostics["loader_stages"]["load_smart_tagging"]["status"].startswith("failed")
    assert res.diagnostics["core"]["stages"]["set_piece_anchors"]["status"].startswith("skipped")
    assert eb.validate_labels_array(res.labels, synthetic_spec.n_frames)


def test_failing_optional_stage_degrades_gracefully(synthetic_dir, synthetic_spec, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("module missing")

    monkeypatch.setattr(pipeline, "compute_ball_kinematics", boom)
    monkeypatch.setattr(pipeline, "anchor_set_pieces", boom)
    res = pipeline.run_pipeline(synthetic_dir, None, _blocks(synthetic_spec), log=lambda m: None)
    stages = res.diagnostics["core"]["stages"]
    assert stages["ball_kinematics"]["status"].startswith("failed")
    assert stages["set_piece_anchors"]["status"].startswith("failed")
    assert res.diagnostics["route"] == "heuristic"
    assert eb.validate_labels_array(res.labels, synthetic_spec.n_frames)


def test_failing_required_stage_falls_back_to_baseline(synthetic_dir, synthetic_spec, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("emissions unavailable")

    monkeypatch.setattr(pipeline, "compute_emissions", boom)
    res = pipeline.run_pipeline(synthetic_dir, None, _blocks(synthetic_spec), log=lambda m: None)
    assert res.diagnostics["route"] == "baseline-fallback"
    assert eb.validate_labels_array(res.labels, synthetic_spec.n_frames)
    assert (res.labels[synthetic_spec.half2_dead[0]:synthetic_spec.half2_dead[1] + 1] == D).all()


def test_run_core_on_arrays_without_files(synthetic_kin, synthetic_spec):
    td, spec = synthetic_kin["td"], synthetic_spec
    core = pipeline.run_core(td.players_xy, td.team_of_slot, td.ball_xy,
                             play_blocks=[spec.half1, spec.half2_play], dead_blocks=[spec.half2_dead],
                             log=lambda m: None, keep_features=True)
    assert core.codes.shape == (td.n_frames,)
    assert (core.codes[:spec.half1[0]] == X).all()
    assert set(np.unique(core.codes[spec.half1[0]:spec.half1[1] + 1])) <= {H, A, D}
    assert core.diagnostics["labels"]["n_possessions"] > 0
    assert core.features["log_probs"].shape == (td.n_frames, 3)
    assert "set_piece_anchors" in core.diagnostics["stages"]


def test_disabled_stages_are_reported(synthetic_kin, synthetic_spec, tmp_path):
    import yaml

    from mtah2027.config import DEFAULT_CONFIG_PATH

    cfg = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    cfg["pipeline"].update({"use_clean_tracks": False, "use_spell_optimizer": False})
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    td, spec = synthetic_kin["td"], synthetic_spec
    core = pipeline.run_core(td.players_xy, td.team_of_slot, td.ball_xy,
                             play_blocks=[spec.half1, spec.half2_play], config_path=path, log=lambda m: None)
    assert core.diagnostics["stages"]["clean_tracks"]["status"].startswith("skipped")
    assert core.diagnostics["stages"]["spell_optimizer"]["status"].startswith("skipped")


# ---------------------------------------------------------------------------- CLI
def _run_cli(*args, env_extra=None):
    env = dict(os.environ, PYTHONHASHSEED="0", **(env_extra or {}))
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, env=env,
                          cwd=ROOT, timeout=900)


def test_cli_short_synthetic_is_byte_reproducible_and_prints_only_aggregates(tmp_path):
    outs = []
    for i in range(2):
        out = tmp_path / f"sub{i}.csv"
        proc = _run_cli("--synthetic", "--synthetic-frames", "12000", "--out", str(out), "--quiet")
        assert proc.returncode == 0, proc.stdout + proc.stderr
        outs.append(out)
        assert "validate_submission: OK" in proc.stdout and "SHA256:" in proc.stdout
        assert "possessions:" in proc.stdout and "spellF1=" in proc.stdout
        assert "\n0,X" not in proc.stdout and "frame,label" not in proc.stdout       # no per-frame rows
    assert outs[0].read_bytes() == outs[1].read_bytes()
    assert hashlib.sha256(outs[0].read_bytes()).hexdigest() in proc.stdout
    assert eb.validate_submission_file(outs[0], 12000)


@pytest.mark.slow
def test_cli_full_length_synthetic_matches_the_contract(tmp_path):
    out = tmp_path / "full.csv"
    t0 = time.perf_counter()
    proc = _run_cli("--synthetic", "--out", str(out), "--quiet")
    elapsed = time.perf_counter() - t0
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert eb.validate_submission_file(out, 199319)
    codes = eb.read_submission_codes(out)
    assert len(codes) == 199319 and (codes[:11185] == X).all() and (codes[110808:113742] == D).all()
    assert not (codes[11185:90652] == X).any() and not (codes[113742:190198] == X).any()
    assert elapsed < 600


# ---------------------------------------------------------------------------- wave 2: learned emissions
MODEL_FILE = ROOT / "models" / "specialists.json"


@pytest.mark.skipif(not MODEL_FILE.is_file(), reason="models/specialists.json not present")
def test_cli_uses_the_committed_specialists_on_synthetic_and_is_reproducible(tmp_path):
    pytest.importorskip("lightgbm")
    outs = []
    for i in range(2):
        out = tmp_path / f"learned{i}.csv"
        proc = _run_cli("--synthetic", "--synthetic-frames", "12000", "--out", str(out))
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "specialists: file" in proc.stdout and "emissions: specialists" in proc.stdout
        assert "validate_submission: OK" in proc.stdout
        outs.append(out)
    assert outs[0].read_bytes() == outs[1].read_bytes()


def test_cli_specialists_off_and_missing_model_fall_back_or_fail_clearly(tmp_path):
    out = tmp_path / "off.csv"
    proc = _run_cli("--synthetic", "--synthetic-frames", "12000", "--out", str(out), "--specialists", "off",
                    "--quiet")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "emissions: heuristic" in proc.stdout and "specialists: none" in proc.stdout
    out2 = tmp_path / "auto.csv"
    proc = _run_cli("--synthetic", "--synthetic-frames", "12000", "--out", str(out2), "--model-path",
                    str(tmp_path / "missing.json"), "--quiet")
    assert proc.returncode == 0, proc.stdout + proc.stderr           # auto mode: progressive fallback
    assert "emissions: heuristic" in proc.stdout
    assert out.read_bytes() == out2.read_bytes()
    proc = _run_cli("--synthetic", "--synthetic-frames", "12000", "--out", str(tmp_path / "x.csv"),
                    "--specialists", "load", "--model-path", str(tmp_path / "missing.json"), "--quiet")
    assert proc.returncode != 0                                       # explicit mode: no silent fallback
