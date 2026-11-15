"""Issue #72: sequence model (TCN over the specialists' feature matrix), NumPy inference, off by default."""
# ruff: noqa: E501
from __future__ import annotations

import dataclasses
import sys

import numpy as np
import pytest

from mtah2027 import pipeline
from mtah2027.config import REPO_ROOT, load_section
from mtah2027.models import frame_classifier as fc
from mtah2027.models import sequence_model as sq
from mtah2027.models import specialist_c as sc

sys.path.insert(0, str(REPO_ROOT / "scripts"))
needs_lgbm = pytest.mark.skipif(not fc.lightgbm_available(), reason="lightgbm not installed")


def test_default_config_is_e100_and_valid():
    p = sq.SequenceParams.from_config()
    assert p.enabled is True and p.mode == "emissions" and p.emission_weight == 1.0
    assert p.use_emissions and not p.use_turnover
    assert load_section("sequence_model")["enabled"] is True


def test_committed_model_is_a_two_seed_ensemble_and_attaches_only_when_enabled():
    from mtah2027.models import specialist_fusion as sf

    model = sq.load_sequence_json(REPO_ROOT / "models" / "sequence_tcn.json")
    assert isinstance(model, sq.SequenceEnsemble) and len(model.members) == 2
    assert [m.meta["seed"] for m in model.members] == [0, 1] and model.meta["training_seeds"] == [0, 1]
    assert all(len(m.feature_names) == 122 and m.meta["exclude"] is None for m in model.members)
    base = sf.SpecialistSet(a=None, b=None, fusion=None)
    on = sf.attach_sequence_model(base, REPO_ROOT / "models")
    assert isinstance(on.seq, sq.SequenceEnsemble)
    off_cfg = {"sequence_model": {"enabled": False, "mode": "emissions"}}
    assert sf.attach_sequence_model(base, REPO_ROOT / "models", off_cfg).seq is None
    msgs = []
    assert sf.attach_sequence_model(base, REPO_ROOT / "nowhere", None, msgs.append).seq is None and msgs


def test_params_validation(tmp_path):
    raw = {"sequence_model": {"enabled": True, "mode": "nope"}}
    with pytest.raises(ValueError):
        sq.SequenceParams.from_config(raw)
    p = sq.SequenceParams.from_config({"sequence_model": {"enabled": True, "mode": "both"}})
    assert p.use_emissions and p.use_turnover
    assert sq.SequenceParams.from_config({"sequence_model": {"enabled": True, "mode": "turnover"}}).use_emissions is False


def test_turnover_targets_match_specialist_c():
    rng = np.random.default_rng(0)
    codes = rng.choice(3, 400, p=[0.45, 0.45, 0.10]).astype(np.int8)
    codes = np.repeat(codes[::5], 5)
    for hw in (1, 5):
        assert np.array_equal(sq.turnover_targets(codes, hw), sc.turnover_targets(codes, hw))


def test_select_columns_drops_context_windows():
    assert sq.select_columns(["visible", "ctx:a:cm50", "sig:x", "ctx:b:dd100"]) == ["visible", "sig:x"]


def test_forward_is_deterministic_and_roundtrips(tmp_path):
    names = [f"f{i}" for i in range(5)]
    m = sq.random_model(names, channels=6, dilations=(1, 2, 4), seed=3)
    x = np.random.default_rng(1).normal(size=(120, 5)).astype(np.float32)
    le, lt = m.forward(x)
    assert le.shape == (120, 3) and lt.shape == (120, 3) and np.isfinite(le).all()
    path = m.save(tmp_path / "m.npz")
    back = sq.SequenceModel.load(path)
    le2, lt2 = back.forward(x)
    assert np.array_equal(le, le2) and np.array_equal(lt, lt2)
    assert np.array_equal(m.forward(x)[0], le)


def test_receptive_field_is_local_and_blocks_are_independent():
    names = ["a", "b"]
    m = sq.random_model(names, channels=6, dilations=(1, 2, 4), seed=0)      # receptive field +-7 frames
    x = np.random.default_rng(2).normal(size=(200, 2)).astype(np.float32)
    base, _ = m.forward(x)
    x2 = x.copy()
    x2[100] += 5.0
    out, _ = m.forward(x2)
    assert np.abs(out[:92] - base[:92]).max() == 0.0 and np.abs(out[109:] - base[109:]).max() == 0.0
    assert np.abs(out[100] - base[100]).max() > 0
    pe, pt = m.predict(x, names, [(0, 59), (100, 159)])
    assert np.isnan(pe[60:100]).all() and np.isfinite(pe[:60]).all() and np.allclose(pe[:60].sum(axis=1), 1.0)
    ref, _ = m.forward(m.standardise(x, names)[100:160])
    assert np.allclose(pe[100:160], sq._softmax(ref.astype(np.float64)))      # a block is decoded on its own


def test_standardise_uses_named_columns_and_fills_missing():
    m = sq.random_model(["b", "a"], channels=4, dilations=(1,))
    x = np.array([[1.0, 2.0, 9.0], [np.nan, 4.0, 9.0]], np.float32)           # columns a, b, c
    z = m.standardise(x, ["a", "b", "c"])
    assert z.tolist() == [[2.0, 1.0], [4.0, 0.0]]
    with pytest.raises(ValueError):
        m.predict(x, ["a", "z", "c"], [(0, 1)])


def test_mix_hidden_emissions_touches_only_hidden_frames():
    n = 10
    lp = np.round(np.log(np.full((n, 3), 1 / 3)), 6)             # production emissions are rounded
    seq = np.tile([0.8, 0.1, 0.1], (n, 1))
    seq[3] = np.nan
    vis = np.array([True] * 4 + [False] * 6)
    p = sq.SequenceParams(enabled=True, mode="emissions", emission_weight=0.5, emission_scale=0.5)
    out = sq.mix_hidden_emissions(lp, seq, vis, p)
    assert np.array_equal(out[:4], lp[:4])                                     # visible and NaN rows untouched
    expect = 0.5 * lp[5] + 0.5 * 0.5 * np.log(seq[5])
    assert np.allclose(out[5], np.round(expect, 6))
    assert out[5, 0] > out[5, 1]
    same = sq.mix_hidden_emissions(lp, seq, vis, dataclasses.replace(p, emission_weight=0.0))
    assert same is lp


def test_gate_helpers():
    from mtah2027.decoding.transition_matrix import TransitionParams

    prob = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
    ha, ah = sq.gate_probabilities(prob)
    assert list(ha) == [0.2, 0.8] and list(ah) == [0.1, 0.1]
    tp = sq.with_gate(TransitionParams(), sq.SequenceParams(turnover_gate_mode="hard", turnover_threshold=0.3))
    assert tp.learned_gate_mode == "hard" and tp.learned_gate_threshold == 0.3


@needs_lgbm
def test_pipeline_off_is_inert_and_modes_run(synthetic_kin, external_like_dir):
    import yaml

    from mtah2027.models import specialist_ball_absent as sb
    from mtah2027.models import specialist_data as sd
    from mtah2027.models import specialist_fusion as sf
    from mtah2027.models.specialist_features import build_feature_matrix

    td, spec = synthetic_kin["td"], synthetic_kin["spec"]
    ball = np.array(td.ball_xy, dtype=np.float32, copy=True)
    ball[spec.half1[0] + 1500:spec.half1[0] + 2600] = np.nan
    blocks = [spec.half1, spec.half2_play]
    fast = fc.ClassifierParams(n_estimators=20, min_child_samples=20, calib_folds=2, adapt_rounds=10)
    train = dataclasses.replace(sd.TrainingDataParams(), seeds=(0,), stride=15, only_complete_ball=False,
                                min_period_frames=100)
    ts = sd.build_training_set(external_like_dir, params=train)
    a = sf.train_specialist_a(ts, params=sf.SpecialistAParams(min_train_rows=50), classifier_params=fast)
    b = sf.train_specialist_b(ts, params=sb.SpecialistBParams(min_train_rows=50), classifier_params=fast)
    base = sf.SpecialistSet(a=a, b=b, fusion=sf.FusionParams(use_pseudo_labels=False))
    fe = pipeline.compute_front_end(td.players_xy, td.team_of_slot, ball, play_blocks=blocks, fps=td.fps,
                                    collective=True)
    fm = build_feature_matrix(fe, pipeline.heuristic_emissions(fe, td.fps), play_blocks=blocks)
    seq = sq.random_model(sq.select_columns(fm.names), channels=8, dilations=(1, 4, 16), seed=0)
    with_seq = dataclasses.replace(base, seq=seq)
    raw = yaml.safe_load((REPO_ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))

    def cfg(**kw):
        out = yaml.safe_load(yaml.safe_dump(raw))
        out["sequence_model"].update(kw)
        return out

    def run(config, specs):
        return pipeline.run_core(td.players_xy, td.team_of_slot, ball, play_blocks=blocks, fps=td.fps,
                                 log=lambda s: None, specialists=specs, config_path=config)

    off_cfg = cfg(enabled=False, mode="off")
    plain = run(off_cfg, base)
    off = run(off_cfg, with_seq)
    assert "sequence_model" not in off.diagnostics["stages"]                  # off: the stage does not even run
    assert np.array_equal(off.codes, plain.codes)
    em = run(cfg(enabled=True, mode="emissions", emission_weight=1.0), with_seq)
    assert em.diagnostics["stages"]["sequence_model"]["status"] == "ok"
    assert np.array_equal(em.codes, run(cfg(enabled=True, mode="emissions", emission_weight=1.0), with_seq).codes)
    tu = run(cfg(enabled=True, mode="turnover", turnover_gate_mode="max"), with_seq)
    assert tu.diagnostics["stages"]["sequence_model"]["status"] == "ok"
    both = run(cfg(enabled=True, mode="both"), with_seq)
    assert both.codes.shape == plain.codes.shape
    assert np.array_equal(run(cfg(enabled=True, mode="turnover"), base).codes, plain.codes)   # no model: inert


def _toy(names, seed=0):
    return sq.random_model(names, channels=8, dilations=(1, 2, 4), seed=seed)


def test_json_round_trip_is_exact_and_loader_picks_ensemble(tmp_path):
    names = [f"f{i}" for i in range(6)]
    m0, m1 = _toy(names, 0), _toy(names, 1)
    p = sq.save_sequence_json(sq.SequenceEnsemble([m0, m1], {"note": "toy"}), tmp_path / "m.json")
    ens = sq.load_sequence_json(p)
    assert isinstance(ens, sq.SequenceEnsemble) and len(ens.members) == 2
    for a, b in zip((m0, m1), ens.members):
        assert a.w.keys() == b.w.keys()
        assert all(np.array_equal(a.w[k], b.w[k]) and b.w[k].dtype == np.float32 for k in a.w)
    single = sq.load_sequence_json(sq.save_sequence_json(m0, tmp_path / "s.json"))
    assert isinstance(single, sq.SequenceModel) and np.array_equal(single.w["w_in"], m0.w["w_in"])
    x = np.random.default_rng(0).normal(size=(200, 6))
    pe, pt = ens.predict(x, names, [(0, 199)])
    want = np.round((m0.predict(x, names, [(0, 199)])[0] + m1.predict(x, names, [(0, 199)])[0]) / 2, 6)
    assert np.allclose(pe, want, atol=1e-6) and np.allclose(pe.sum(axis=1), 1.0, atol=2e-6)
    assert np.isfinite(pt).all()


def test_predictions_are_float64_and_quantised_against_blas_noise():
    names = [f"f{i}" for i in range(6)]
    m = _toy(names)
    x = np.random.default_rng(1).normal(size=(300, 6))
    pe, pt = m.predict(x, names, [(0, 299)])
    assert pe.dtype == np.float64 and np.array_equal(pe, np.round(pe, 6)) and np.array_equal(pt, np.round(pt, 6))
    z = m.standardise(x, names)
    assert z.dtype == np.float64
    # a 1e-11 perturbation (what a different BLAS does) must not change the quantised output
    noisy = _toy(names)
    noisy._d = {k: v * (1.0 + 1e-13) for k, v in noisy._d.items()}
    pe2, _ = noisy.predict(x, names, [(0, 299)])
    assert np.array_equal(pe, pe2)
