"""Issue #44: ensemble of specialists (per-frame mixing), committed-manifest loading and the decoder sweep harness."""
# ruff: noqa: E501
from __future__ import annotations

import dataclasses
import json
import sys

import numpy as np
import pytest

from mtah2027 import pipeline
from mtah2027.config import REPO_ROOT, load_section, params_from_mapping
from mtah2027.models import frame_classifier as fc
from mtah2027.models import specialist_ball_absent as sb
from mtah2027.models import specialist_data as sd
from mtah2027.models import specialist_fusion as sf

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import decoder_sweep as dsw  # noqa: E402

FAST = fc.ClassifierParams(n_estimators=20, min_child_samples=20, calib_folds=2, adapt_rounds=10)
FAST2 = dataclasses.replace(FAST, n_estimators=35)
TRAIN = dataclasses.replace(sd.TrainingDataParams(), seeds=(0,), stride=15, only_complete_ball=False,
                            min_period_frames=100)
needs_lgbm = pytest.mark.skipif(not fc.lightgbm_available(), reason="lightgbm not installed")


def _train(external_like_dir, cp):
    ts = sd.build_training_set(external_like_dir, params=TRAIN)
    a = sf.train_specialist_a(ts, params=sf.SpecialistAParams(min_train_rows=50), classifier_params=cp)
    b = sf.train_specialist_b(ts, params=sb.SpecialistBParams(min_train_rows=50), classifier_params=cp)
    return sf.SpecialistSet(a=a, b=b, fusion=sf.FusionParams(use_pseudo_labels=False),
                            training=sf.describe_training(ts, external_like_dir, a, b))


@pytest.fixture(scope="module")
def pair(external_like_dir):
    return _train(external_like_dir, FAST), _train(external_like_dir, FAST2)


@pytest.fixture(scope="module")
def match(synthetic_kin):
    td, spec = synthetic_kin["td"], synthetic_kin["spec"]
    ball = np.array(td.ball_xy, dtype=np.float32, copy=True)
    ball[spec.half1[0] + 1500:spec.half1[0] + 2600] = np.nan
    ball[spec.half2_play[0] + 800:spec.half2_play[0] + 1800] = np.nan
    return {"td": td, "spec": spec, "ball": ball, "blocks": [spec.half1, spec.half2_play]}


def _core(m, **kw):
    td = m["td"]
    return pipeline.run_core(td.players_xy, td.team_of_slot, m["ball"], play_blocks=m["blocks"], fps=td.fps,
                             log=lambda s: None, **kw)


def _fuse(m, spec):
    td = m["td"]
    fe = pipeline.compute_front_end(td.players_xy, td.team_of_slot, m["ball"], play_blocks=m["blocks"],
                                    fps=td.fps, collective=True)
    heur = pipeline.heuristic_emissions(fe, td.fps)
    return sf.fuse_emissions(spec, fe, heur, play_blocks=m["blocks"], fps=td.fps,
                             rebuild=lambda b: None, raw_ball=np.asarray(m["ball"], np.float32)).emissions


def test_mix_probabilities_modes_and_validation():
    p = np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
    q = np.array([[0.1, 0.6, 0.3], [0.2, 0.2, 0.6]])
    assert np.array_equal(sf.mix_probabilities([p], [3.0]), p)
    lin = sf.mix_probabilities([p, q], [1, 1], "linear")
    assert np.allclose(lin, (p + q) / 2) and np.allclose(lin.sum(axis=1), 1)
    geo = sf.mix_probabilities([p, q], [1, 1], "loglinear")
    ref = np.sqrt(p * q)
    assert np.allclose(geo, ref / ref.sum(axis=1, keepdims=True)) and np.allclose(geo.sum(axis=1), 1)
    assert np.allclose(sf.mix_probabilities([p, q], [1, 0], "loglinear"), p)
    assert np.allclose(sf.mix_probabilities([p, q], [3, 1], "linear"), 0.75 * p + 0.25 * q)
    # the log-linear mix is more peaked where the members agree, the linear one never leaves the hull
    assert geo[1, 2] > min(p[1, 2], q[1, 2]) and lin[1, 2] <= max(p[1, 2], q[1, 2])
    for bad in (dict(weights=[1], mode="linear"), dict(weights=[-1, 2], mode="linear"),
                dict(weights=[0, 0], mode="linear"), dict(weights=[1, 1], mode="median")):
        with pytest.raises(ValueError):
            sf.mix_probabilities([p, q], bad["weights"], bad["mode"])


@needs_lgbm
def test_identical_members_equal_the_single_set(pair, match):
    single, _ = pair
    ens = dataclasses.replace(single, ensemble=((single, 0.3), (single, 0.7)), mix_mode="loglinear")
    e1, e2 = _fuse(match, single), _fuse(match, ens)
    assert np.allclose(e1.log_probs, e2.log_probs, atol=2e-6)
    assert np.array_equal(_core(match, specialists=single).codes, _core(match, specialists=ens).codes)
    assert single.members() == ((single, 1.0),)
    assert [w for _, w in ens.members()] == pytest.approx([0.3, 0.7])


@needs_lgbm
def test_two_members_mix_per_frame_and_are_deterministic(pair, match):
    a, b = pair
    only_a = dataclasses.replace(a, ensemble=((a, 1.0), (b, 0.0)))
    assert np.allclose(_fuse(match, only_a).log_probs, _fuse(match, a).log_probs, atol=2e-6)
    mixes = {}
    for mode in sf.MIX_MODES:
        ens = dataclasses.replace(a, ensemble=((a, 0.5), (b, 0.5)), mix_mode=mode)
        first = _fuse(match, ens).log_probs
        assert np.array_equal(first, _fuse(match, ens).log_probs)            # deterministic
        assert np.array_equal(first, np.round(first, 6))                      # quantised at 1e-6
        mixes[mode] = first
    la, lb = _fuse(match, a).log_probs, _fuse(match, b).log_probs
    assert not np.array_equal(la, lb)
    lo, hi = np.minimum(la, lb), np.maximum(la, lb)
    inside = (mixes["linear"] >= lo - 1e-3 - 0.7) & (mixes["linear"] <= hi + 1e-3 + 0.7)
    assert inside.mean() > 0.99
    assert not np.array_equal(mixes["linear"], mixes["loglinear"])


@needs_lgbm
def test_manifest_loads_committed_files_only(pair, match, tmp_path):
    a, b = pair
    sf.save_specialists(a, tmp_path / "specialists_a.json")
    sf.save_specialists(b, tmp_path / "specialists_b.json")
    manifest = {"format": "ensemble", "mode": "linear",
                "members": [{"file": "specialists_a.json", "weight": 0.25},
                            {"file": "specialists_b.json", "weight": 0.75}]}
    path = tmp_path / "specialists_ens.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    ens = sf.load_specialists(path)
    assert ens.mix_mode == "linear" and len(ens.ensemble) == 2 and ens.source == "file"
    assert [w for _, w in ens.members()] == pytest.approx([0.25, 0.75])
    assert ens.training["ensemble"][0]["file"] == "specialists_a.json"
    via = sf.resolve_specialists("load", model_path=path)
    r1, r2 = _core(match, specialists=via).codes, _core(match, specialists=sf.load_specialists(path)).codes
    assert np.array_equal(r1, r2)                                              # byte-reproducible from files
    for broken in ({"format": "ensemble", "mode": "median", "members": manifest["members"]},
                   {"format": "ensemble", "mode": "linear", "members": []}):
        path.write_text(json.dumps(broken), encoding="utf-8")
        with pytest.raises(ValueError):
            sf.load_specialists(path)
    with pytest.raises(FileNotFoundError):
        path.write_text(json.dumps({**manifest, "members": [{"file": "missing.json", "weight": 1}]}), encoding="utf-8")
        sf.load_specialists(path)


def test_candidate_config_overrides_only_the_specialists_route():
    raw = {"route_overrides": {"specialists": {"transition_matrix": {"ha_occluded": -7.0, "ha_open": -2.5}}},
           "transition_matrix": {"ha_occluded": -6.0}}
    cfg = dsw.candidate_config(raw, {"transition_matrix.ha_occluded": -9.0, "specialist_fusion.scale_hidden": 0.7,
                                     "transition_matrix.dead_ramp": (0.4, 0.8)})
    spec = cfg["route_overrides"]["specialists"]
    assert spec["transition_matrix"] == {"ha_occluded": -9.0, "ha_open": -2.5, "dead_ramp": [0.4, 0.8]}
    assert spec["specialist_fusion"] == {"scale_hidden": 0.7}
    assert raw["route_overrides"]["specialists"]["transition_matrix"]["ha_occluded"] == -7.0     # input untouched
    assert cfg["transition_matrix"] == {"ha_occluded": -6.0}


@needs_lgbm
def test_cell_cache_reuses_the_front_end_and_is_exact(pair, match):
    spec, _ = pair
    calls = []
    orig = pipeline.compute_front_end

    def counting(*a, **k):
        calls.append(1)
        return orig(*a, **k)

    td = match["td"]
    assert dsw.candidate_config({}, {}) == {"route_overrides": {"specialists": {}}}
    pipeline.compute_front_end = counting
    try:
        cache = dsw.CellCache(match["ball"])
        cache.install()
        try:
            a1 = _core(match, specialists=spec)
            n_first = len(calls)
            b1 = pipeline.run_core(td.players_xy, td.team_of_slot, match["ball"], play_blocks=match["blocks"],
                                   fps=td.fps, log=lambda s: None, specialists=spec, config_path=None)
            assert len(calls) == n_first == 1                                  # the second run reuses it
            assert np.array_equal(a1.codes, b1.codes)
        finally:
            cache.uninstall()
        assert pipeline.compute_front_end is counting                        # restored
    finally:
        pipeline.compute_front_end = orig
    assert np.array_equal(_core(match, specialists=spec).codes, a1.codes)    # cached == uncached


def test_aggregation_table_and_deltas(tmp_path):
    def cell(world, seed, match, vals):
        row = {k: {"spell_f1": v, "macro_f1": v + 0.3} for k, v in vals.items()}
        (tmp_path / f"{world}__{match}__s{seed}.json").write_text(
            json.dumps({"cell": {"world": world, "match": match, "seed": seed}, "results": row}), encoding="utf-8")

    cell("b33", 0, "m1", {"base": 0.60, "cand": 0.63})
    cell("b33", 0, "m2", {"base": 0.50, "cand": 0.52})
    cell("b33", 1, "m1", {"base": 0.70, "cand": 0.69})
    cell("b33", 1, "m2", {"base": 0.50, "cand": 0.53})
    table = dsw.load_results([tmp_path])
    rows, worlds = dsw.build_rows(table, dsw.candidates_in(table), "base")
    assert worlds == ["b33"]
    cand = next(r for r in rows if r["name"] == "cand")["b33"]
    assert cand["mean"] == pytest.approx(0.5925) and cand["delta"] == pytest.approx(0.0175)
    assert cand["seed_delta"][0] == pytest.approx(0.025) and cand["seed_delta"][1] == pytest.approx(0.01)
    assert cand["macro"] == pytest.approx(0.8925)


def test_cells_for_worlds(tmp_path):
    for name in ("idsse_A", "idsse_B", "pff_1", "pff_2", "pff_3", "pff_4", "pff_5"):
        (tmp_path / f"{name}.npz").write_bytes(b"")
    cells = dsw.cells_for("b33", tmp_path, [0, 1])
    assert [(c["match"], c["seed"]) for c in cells] == [("idsse_A", 0), ("idsse_B", 0), ("idsse_A", 1), ("idsse_B", 1)]
    assert dsw.cells_for("b1", tmp_path, [2])[0]["world"] == "b1"
    folds = dsw.cells_for("pff", tmp_path, [0], n_pff=4)
    assert [c["fold"] for c in folds] == [0, 1, 2, 3]
    assert [c["match"] for c in dsw.cells_for("unseen", tmp_path, [0], n_unseen=2)] == ["pff_4", "pff_5"]
    assert dsw.model_path(tmp_path, folds[1], "mixed") == tmp_path / "pff_mixed" / "fold1.json"
    assert dsw.model_path(tmp_path, cells[0], "calib") == tmp_path / "lomo_calib" / "idsse_A.json"
    assert dsw.model_path(tmp_path, dsw.cells_for("unseen", tmp_path, [0])[0], "gap").name == "specialists_gap.json"


def test_issue_44_candidate_block_is_valid_and_not_applied():
    from mtah2027.decoding.transition_matrix import TransitionParams
    from mtah2027.models.specialist_fusion import FusionParams
    from mtah2027.postprocessing.spell_optimizer import SpellParams

    block = load_section("decoder_candidates")["issue_44"]
    cls = {"specialist_fusion": FusionParams, "transition_matrix": TransitionParams, "spell_optimizer": SpellParams}
    assert set(block) == set(cls)
    for section, values in block.items():
        built = params_from_mapping(cls[section], values, section)          # unknown key -> ConfigError
        assert all(getattr(built, k) == v for k, v in values.items())
    live = load_section("route_overrides")["specialists"]
    assert live["specialist_fusion"]["scale_visible"] != block["specialist_fusion"]["scale_visible"]   # OFF
    manifest = REPO_ROOT / "models" / "specialists_ensemble.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["format"] == "ensemble" and data["mode"] in sf.MIX_MODES
    assert sum(m["weight"] for m in data["members"]) == pytest.approx(1.0)
    assert all((manifest.parent / m["file"]).is_file() for m in data["members"])
    default = json.loads((REPO_ROOT / "models" / "specialists.json").read_text(encoding="utf-8"))
    assert default.get("format") != "ensemble"                                                         # default unchanged
    ens = sf.load_specialists(manifest)
    assert len(ens.ensemble) == 2 and ens.durations is not None
