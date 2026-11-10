"""Issue #20: calibrated classifier, feature matrix, training rows and Specialist A with its anchor mask."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from mtah2027.config import ConfigError, params_from_mapping
from mtah2027.models import frame_classifier as fc
from mtah2027.models import specialist_ball_present as sa
from mtah2027.models import specialist_data as sd
from mtah2027.models import specialist_features as sfe

FAST = fc.ClassifierParams(n_estimators=25, min_child_samples=20, calib_folds=2)


def _blobs(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n)
    x = rng.normal(size=(n, 5)).astype(np.float32)
    x[:, 0] += 2.5 * (y == 0) - 2.5 * (y == 1)
    x[:, 1] += 3.0 * (y == 2)
    groups = np.where(np.arange(n) % 2 == 0, "m1", "m2")
    return x, y, groups


def test_params_from_yaml_and_typos():
    assert fc.ClassifierParams.from_config().calib_folds >= 0
    assert sa.SpecialistAParams.from_config().anchor_prob == 0.98
    with pytest.raises(ConfigError):
        params_from_mapping(fc.ClassifierParams, {"n_estimator": 3}, "specialist_model")
    with pytest.raises(ValueError):
        fc.ClassifierParams(backend="xgboost")


def test_classifier_learns_is_calibrated_and_deterministic():
    x, y, g = _blobs()
    a = fc.FrameClassifier(FAST, [f"f{i}" for i in range(5)]).fit(x, y, groups=g)
    b = fc.FrameClassifier(FAST, [f"f{i}" for i in range(5)]).fit(x, y, groups=g)
    p = a.predict_proba(x)
    assert p.shape == (len(y), 3) and np.allclose(p.sum(1), 1.0)
    assert (p.argmax(1) == y).mean() > 0.9
    assert np.array_equal(p, b.predict_proba(x))                      # deterministic
    assert a.params.temperature_bounds[0] <= a.temperature <= a.params.temperature_bounds[1]
    assert a.meta["oof_log_loss_calibrated"] <= a.meta["oof_log_loss_raw"] + 1e-6


def test_missing_class_and_bad_input():
    x, y, _ = _blobs()
    keep = y != 2
    clf = fc.FrameClassifier(dataclasses.replace(FAST, calib_folds=0)).fit(x[keep], y[keep])
    assert clf.predict_proba(x)[:, 2].max() < 0.05
    with pytest.raises(ValueError):
        fc.FrameClassifier(FAST).fit(x, y + 5)
    with pytest.raises(RuntimeError):
        fc.FrameClassifier(FAST).predict_proba(x)


def test_serialisation_roundtrip_and_adaptation():
    if not fc.lightgbm_available():
        pytest.skip("lightgbm not installed")
    x, y, g = _blobs()
    clf = fc.FrameClassifier(FAST, [f"f{i}" for i in range(5)]).fit(x, y, groups=g)
    back = fc.FrameClassifier.from_dict(clf.to_dict())
    assert np.allclose(back.predict_proba(x), clf.predict_proba(x))
    shifted = clf.adapt(x, (y + 1) % 3, rounds=30)
    assert not np.allclose(shifted.predict_proba(x), clf.predict_proba(x))
    with pytest.raises(ValueError):
        fc.FrameClassifier.from_dict({"backend": "sklearn"})


def test_sklearn_fallback_works_and_cannot_be_saved():
    x, y, g = _blobs()
    clf = fc.FrameClassifier(dataclasses.replace(FAST, backend="sklearn"), [f"f{i}" for i in range(5)])
    clf.fit(x, y, groups=g)
    assert (clf.predict_proba(x).argmax(1) == y).mean() > 0.9
    with pytest.raises(ValueError):
        clf.to_dict()
    with pytest.raises(RuntimeError):
        clf.adapt(x, y)


def test_anchor_mask_requires_sustained_confidence_and_visibility():
    n, fps = 400, 25.0
    prob = np.tile([0.005, 0.005, 0.99], (n, 1)).astype(float)
    vis = np.ones(n, bool)
    prob[100:140] = [0.99, 0.005, 0.005]               # 40 frames = 1.6 s of class H: too short
    prob[200:300] = [0.005, 0.99, 0.005]               # 4 s of class A ...
    prob[250:252] = [0.4, 0.3, 0.3]                    # ... interrupted once (50 and 48 frames: both long)
    vis[300:340] = False                               # hidden: never an anchor
    mask, label = sa.anchor_mask(prob, vis, fps, threshold=0.98, min_seconds=2.0)
    assert not mask[100:140].any() and not mask[250:252].any() and not mask[300:340].any()
    assert mask[205:245].all() and (label[205:245] == 1).all() and (label[~mask] == -1).all()
    assert mask[0:100].all() and (label[0:100] == 2).all()
    assert not sa.anchor_mask(np.full((50, 3), np.nan), np.ones(50, bool), fps)[0].any()


def test_feature_matrix_layout_and_ball_free_columns(synthetic_kin):
    from mtah2027.pipeline import compute_front_end, heuristic_emissions

    td, spec = synthetic_kin["td"], synthetic_kin["spec"]
    blocks = [spec.half1, spec.half2_play]
    fe = compute_front_end(td.players_xy, td.team_of_slot, td.ball_xy, play_blocks=blocks, collective=True)
    fm = sfe.build_feature_matrix(fe, heuristic_emissions(fe, td.fps))
    assert fm.x.shape == (len(fe.visible), len(fm.names)) and len(set(fm.names)) == len(fm.names)
    assert not any("slot" in k or k.endswith("ball_visible") for k in fm.names)
    assert {"visible", "heur:home_logit", "diff:hull_area", "bm:since_seen_s"} <= set(fm.names)
    nothing = np.full_like(td.ball_xy, np.nan)
    fe2 = compute_front_end(td.players_xy, td.team_of_slot, nothing, play_blocks=blocks, collective=True)
    fm2 = sfe.build_feature_matrix(fe2, None)
    bk_cols = [i for i in sfe.ball_based_columns(fm2.names) if fm2.names[i].startswith("bk:")]
    block = np.nan_to_num(fm2.x[spec.half1[0]:spec.half1[1]][:, bk_cols])
    assert bk_cols and (block.std(axis=0) == 0).all()          # no ball -> constant (NaN / 0) columns
    bare = dataclasses.replace(fe, phys=None)
    with pytest.raises(ValueError):
        sfe.build_feature_matrix(bare, None)


def test_rows_for_match_swap_symmetry_and_training(external_like_dir):
    path = sorted(external_like_dir.glob("*.npz"))[0]
    plain, swapped = sd.rows_for_match(path, 0, stride=20, min_period_frames=100)
    assert (plain.swap, swapped.swap) == (False, True)
    assert plain.x_vis.shape[1] == len(plain.names) == len(swapped.names)
    assert len(plain.y_vis) + len(plain.y_hid) > 100
    both = np.intersect1d(plain.frames_vis, swapped.frames_vis)
    ya = dict(zip(plain.frames_vis.tolist(), plain.y_vis.tolist()))
    yb = dict(zip(swapped.frames_vis.tolist(), swapped.y_vis.tolist()))
    assert len(both) and all(yb[f] == sd.swap_codes(np.array([ya[f]]))[0] for f in both.tolist())
    assert np.array_equal(sd.swap_codes(np.array([0, 1, 2, 3])), [1, 0, 2, 3])
    p = dataclasses.replace(sd.TrainingDataParams(), seeds=(0,), stride=20, only_complete_ball=False,
                            min_period_frames=100)
    ts = sd.build_training_set(external_like_dir, params=p)
    assert ts.matches == ["fake_match_0", "fake_match_1"] and ts.summary()["swapped_variants"] == 2
    spec_a = sa.train_specialist_a(ts, params=sa.SpecialistAParams(min_train_rows=100),
                                   classifier_params=FAST)
    x, y, _ = ts.stack("vis")
    res = spec_a.predict(x, np.ones(len(x), bool))
    assert res.prob.shape == (len(x), 3) and np.allclose(res.prob.sum(1), 1.0)
    assert (res.prob.argmax(1) == y).mean() > 0.5
    none = spec_a.predict(x, np.zeros(len(x), bool))
    assert np.isnan(none.prob).all() and not none.anchor_mask.any()
    with pytest.raises(ValueError):
        sa.train_specialist_a(ts, params=sa.SpecialistAParams(min_train_rows=10 ** 9), classifier_params=FAST)
    with pytest.raises(FileNotFoundError):
        sd.build_training_set(external_like_dir / "missing", params=p)
