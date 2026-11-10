"""Issue #22: Specialist B, the A/B fusion into log_probs[N, 3], persistence and the progressive fallback."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from mtah2027 import pipeline
from mtah2027.config import ConfigError, params_from_mapping
from mtah2027.models import frame_classifier as fc
from mtah2027.models import pseudo_labeling as pl
from mtah2027.models import specialist_ball_absent as sb
from mtah2027.models import specialist_data as sd
from mtah2027.models import specialist_fusion as sf
from mtah2027.models.heuristic_emissions import EmissionParams

FAST = fc.ClassifierParams(n_estimators=20, min_child_samples=20, calib_folds=2, adapt_rounds=10)
TRAIN = dataclasses.replace(sd.TrainingDataParams(), seeds=(0,), stride=15, only_complete_ball=False,
                            min_period_frames=100)
needs_lgbm = pytest.mark.skipif(not fc.lightgbm_available(), reason="lightgbm not installed")


@pytest.fixture(scope="module")
def trained(external_like_dir):
    ts = sd.build_training_set(external_like_dir, params=TRAIN)
    a = sf.train_specialist_a(ts, params=sf.SpecialistAParams(min_train_rows=50), classifier_params=FAST)
    b = sf.train_specialist_b(ts, params=sb.SpecialistBParams(min_train_rows=50), classifier_params=FAST)
    return sf.SpecialistSet(a=a, b=b, fusion=sf.FusionParams(use_pseudo_labels=False),
                            training=sf.describe_training(ts, external_like_dir, a, b))


@pytest.fixture(scope="module")
def match(synthetic_kin):
    td, spec = synthetic_kin["td"], synthetic_kin["spec"]
    # hide the ball in long blocks so that both specialists are used
    ball = np.array(td.ball_xy, dtype=np.float32, copy=True)
    ball[spec.half1[0] + 1500:spec.half1[0] + 2600] = np.nan
    ball[spec.half2_play[0] + 800:spec.half2_play[0] + 1800] = np.nan
    return {"td": td, "spec": spec, "ball": ball, "blocks": [spec.half1, spec.half2_play]}


def _front_end(m, ball):
    td = m["td"]
    return pipeline.compute_front_end(td.players_xy, td.team_of_slot, ball, play_blocks=m["blocks"],
                                      fps=td.fps, collective=True)


def _core(m, **kw):
    td = m["td"]
    return pipeline.run_core(td.players_xy, td.team_of_slot, m["ball"], play_blocks=m["blocks"], fps=td.fps,
                             log=lambda s: None, **kw)


def test_params_and_validation():
    assert sb.SpecialistBParams.from_config().adapt is True
    fp = sf.FusionParams.from_config()
    assert fp.prob_floor == 0.001 and not fp.use_pseudo_labels
    with pytest.raises(ConfigError):
        params_from_mapping(sf.FusionParams, {"prob_flor": 1}, "specialist_fusion")
    with pytest.raises(ValueError):
        sf.FusionParams(blend_hidden=2.0)
    with pytest.raises(ValueError):
        sf.FusionParams(scale_visible=0.0)


def test_apply_priors_is_identity_without_channels_and_pushes_with_them():
    ep = EmissionParams()
    prob = np.tile([0.5, 0.3, 0.2], (6, 1))
    same, z = sf.apply_priors(prob, phase=None, owner=None, status_dead=None, ep=ep)
    assert same is prob and z == pytest.approx(np.log(0.5 / 0.3))
    owner = np.array([0, 0, 1, 1, np.nan, np.nan])
    status = np.array([1, 1, 0, 0, 0, np.nan])
    out, z2 = sf.apply_priors(prob, phase=np.ones(6), owner=owner, status_dead=status, ep=ep)
    assert np.allclose(out.sum(1), 1.0)
    assert out[0, 2] > out[2, 2]                                    # dead status raises P(D)
    assert z2[0] > z2[2] and z2[2] < 0                              # away owner flips the team
    assert z2[4] == pytest.approx(np.log(0.5 / 0.3) + ep.w_phase)   # unknown owner: only the phase prior


def test_fusion_uses_a_where_visible_and_b_where_hidden(trained, match):
    td = match["td"]
    fe = _front_end(match, match["ball"])
    heur = pipeline.heuristic_emissions(fe, td.fps)
    res = sf.fuse_emissions(trained, fe, heur, play_blocks=match["blocks"], fps=td.fps,
                            rebuild=lambda b: None, raw_ball=match["ball"])
    lp = res.emissions.log_probs
    assert lp.shape == (len(fe.visible), 3) and np.array_equal(lp, np.round(lp, 6))
    assert lp.min() >= np.log(trained.fusion.prob_floor) - 1e-6 and (lp <= 0).all()
    from mtah2027.models.specialist_features import build_feature_matrix

    x = build_feature_matrix(fe, heur, play_blocks=match["blocks"]).x
    vis = fe.visible
    exp_a = np.log(np.maximum(trained.a.predict_proba(x[vis]), 0.001))
    exp_b = np.log(np.maximum(trained.b.predict_proba(x[~vis]), 0.001))
    assert np.allclose(lp[vis], exp_a, atol=1e-5) and np.allclose(lp[~vis], exp_b, atol=1e-5)
    assert (~vis).sum() > 1000 and res.report["pseudo_labels"] == {}
    blended = dataclasses.replace(trained, fusion=dataclasses.replace(trained.fusion, blend_hidden=1.0,
                                                                       scale_visible=2.0))
    res2 = sf.fuse_emissions(blended, fe, heur, play_blocks=match["blocks"], fps=td.fps,
                             rebuild=lambda b: None, raw_ball=match["ball"])
    assert np.allclose(res2.emissions.log_probs[~vis], heur.log_probs[~vis], atol=1e-5)
    assert np.allclose(res2.emissions.log_probs[vis], np.round(2 * lp[vis], 6), atol=1e-5)


def test_pipeline_with_specialists_is_deterministic_and_reports_source(trained, match):
    r1, r2 = _core(match, specialists=trained), _core(match, specialists=trained)
    assert np.array_equal(r1.codes, r2.codes)
    assert r1.diagnostics["emission_source"] == "specialists"
    assert r1.diagnostics["specialists"]["source"] == "memory"
    base = _core(match)
    assert base.diagnostics["emission_source"] == "heuristic" and base.diagnostics["specialists"] == {}
    graded = np.zeros(len(r1.codes), bool)
    for a, b in match["blocks"]:
        graded[a:b + 1] = True
    assert (r1.codes[graded] < 3).all() and (r1.codes[~graded] == 3).all()


def test_failing_specialists_fall_back_to_the_heuristic_route(trained, match):
    broken = sf.SpecialistSet(a=trained.a, b=trained.b, fusion=trained.fusion)
    broken.b = None                                                  # any failure inside the fusion
    out = _core(match, specialists=broken)
    assert out.diagnostics["emission_source"] == "heuristic"
    assert out.diagnostics["stages"]["specialists"]["status"].startswith("failed")
    assert np.array_equal(out.codes, _core(match).codes)


@needs_lgbm
def test_adaptation_with_pseudo_labels_changes_b_and_is_reported(trained, match):
    td = match["td"]
    on = sf.SpecialistSet(a=trained.a, b=trained.b, fusion=sf.FusionParams(use_pseudo_labels=True),
                          training=trained.training)
    fe = _front_end(match, match["ball"])
    heur = pipeline.heuristic_emissions(fe, td.fps)

    def rebuild(ball):
        return _front_end(match, ball)

    easy = dataclasses.replace(sf.SpecialistAParams(), anchor_prob=0.5, anchor_min_s=1.0)
    on.a.params = easy                                               # synthetic data: relax the anchor test
    res = sf.fuse_emissions(on, fe, heur, play_blocks=match["blocks"], fps=td.fps, rebuild=rebuild,
                            raw_ball=match["ball"])
    rep = res.report["pseudo_labels"]
    assert rep["blocks"] > 0 and rep["rows"] > 0
    ds = pl.PseudoDataset(x=np.zeros((0, 3), np.float32), y=np.zeros(0, np.int64), weight=np.zeros(0),
                          names=["a", "b", "c"])
    assert trained.b.adapt(ds) is trained.b                          # nothing to learn from


def test_specialist_b_validation(trained):
    ds = pl.PseudoDataset(x=np.zeros((5, 2), np.float32), y=np.zeros(5, np.int64), weight=np.ones(5),
                          names=["x", "y"])
    if fc.lightgbm_available():
        with pytest.raises(ValueError):
            trained.b.adapt(ds)
    with pytest.raises(ValueError):
        sf.SpecialistSet.from_dict({"format": 99})


@needs_lgbm
def test_save_load_roundtrip_and_resolve_modes(trained, tmp_path, external_like_dir, match, monkeypatch):
    path = sf.save_specialists(trained, tmp_path / "m" / "specialists.json")
    assert path.read_bytes() == sf.save_specialists(trained, tmp_path / "again.json").read_bytes()
    back = sf.load_specialists(path)
    x = np.random.default_rng(0).normal(size=(40, len(trained.a.feature_names))).astype(np.float32)
    assert np.allclose(back.a.predict_proba(x), trained.a.predict_proba(x))
    assert np.allclose(back.b.predict_proba(x), trained.b.predict_proba(x))
    assert back.source == "file" and back.training["matches"] == trained.training["matches"]
    assert sf.resolve_specialists("off") is None
    assert sf.resolve_specialists("load", model_path=path).source == "file"
    with pytest.raises(FileNotFoundError):
        sf.resolve_specialists("load", model_path=tmp_path / "nope.json")
    with pytest.raises(ValueError):
        sf.resolve_specialists("sometimes")
    monkeypatch.setattr(sf, "find_bench_dir", lambda explicit=None: None)
    logs = []
    assert sf.resolve_specialists("auto", model_path=tmp_path / "nope.json", log=logs.append) is None
    assert any("unavailable" in m for m in logs)
    assert sf.resolve_specialists("auto", model_path=path).source == "file"
    with pytest.raises(FileNotFoundError):
        sf.resolve_specialists("train")
    (tmp_path / "bad.json").write_text("{}", encoding="utf-8")
    assert sf.resolve_specialists("auto", model_path=tmp_path / "bad.json") is None
    monkeypatch.setattr(sf, "find_bench_dir", lambda explicit=None: external_like_dir)
    monkeypatch.setattr(sf, "build_training_set", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert sf.resolve_specialists("auto", model_path=path).source == "file"      # training failed -> file
    with pytest.raises(RuntimeError):
        sf.resolve_specialists("train")


def test_find_bench_dir(tmp_path, monkeypatch, external_like_dir):
    monkeypatch.delenv(sf.BENCH_ENV, raising=False)
    assert sf.find_bench_dir(external_like_dir) == external_like_dir
    assert sf.find_bench_dir(tmp_path) != tmp_path
    monkeypatch.setenv(sf.BENCH_ENV, str(external_like_dir))
    assert sf.find_bench_dir(tmp_path) == external_like_dir


def test_context_features_are_block_local_and_nan_safe():
    from mtah2027.models import specialist_features as sf

    n = 300
    names = ["cp:depth_home", "other"]
    x = np.zeros((n, 2), np.float32)
    x[:150, 0] = 1.0
    x[150:, 0] = 5.0
    x[40:45, 0] = np.nan
    cx, cn = sf.context_features(x, names, [(0, 149), (150, 299)], (20,))
    assert cn == ["ctx:cp:depth_home:cm20", "ctx:cp:depth_home:dd20"]
    assert np.allclose(cx[:150, 0], 1.0) and np.allclose(cx[150:, 0], 5.0)     # nothing crosses the boundary
    assert np.allclose(cx[:150, 1], 0.0)                                         # constant block: no change
    cx2, _ = sf.context_features(x, names, [(0, 299)], (20,))
    assert cx2[149, 1] > 0 and cx2[150, 1] > 0                                  # one block sees the step
    cx3, _ = sf.context_features(x, names, [(0, 149)], (20,))
    assert np.isnan(cx3[200]).all()                                              # outside the blocks: NaN
    assert sf.context_features(x, names, None, ())[0].shape == (n, 0)
