"""Issue #60: seed bagging of the specialists (extra LightGBM seeds, same training data, linear mean)."""
# ruff: noqa: E501
from __future__ import annotations

import dataclasses
import json
import sys

import numpy as np
import pytest

from mtah2027.config import REPO_ROOT
from mtah2027.models import frame_classifier as fc
from mtah2027.models import specialist_ball_absent as sb
from mtah2027.models import specialist_data as sd
from mtah2027.models import specialist_fusion as sf

sys.path.insert(0, str(REPO_ROOT / "scripts"))

FAST = fc.ClassifierParams(n_estimators=20, min_child_samples=20, calib_folds=2, adapt_rounds=10)
TRAIN = dataclasses.replace(sd.TrainingDataParams(), seeds=(0,), stride=15, only_complete_ball=False,
                            min_period_frames=100)
needs_lgbm = pytest.mark.skipif(not fc.lightgbm_available(), reason="lightgbm not installed")


def _train(external_like_dir, seed):
    ts = sd.build_training_set(external_like_dir, params=TRAIN)
    cp = dataclasses.replace(FAST, seed=seed)
    a = sf.train_specialist_a(ts, params=sf.SpecialistAParams(min_train_rows=50), classifier_params=cp)
    b = sf.train_specialist_b(ts, params=sb.SpecialistBParams(min_train_rows=50), classifier_params=cp)
    return sf.SpecialistSet(a=a, b=b, fusion=sf.FusionParams(use_pseudo_labels=False),
                            training=sf.describe_training(ts, external_like_dir, a, b))


def test_seed_argument_is_part_of_the_training_api():
    import inspect

    assert inspect.signature(sf.train_specialists).parameters["seed"].default is None
    for script in ("train_cv_models.py", "train_specialists.py"):
        assert "--seed" in (REPO_ROOT / "scripts" / script).read_text(encoding="utf-8")


def test_bag_manifest_roundtrip_and_validation(tmp_path):
    out = sf.save_bag_manifest([tmp_path / "m0.json", tmp_path / "m1.json", tmp_path / "m2.json"],
                               tmp_path / "specialists_bag.json")
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["format"] == "ensemble" and data["mode"] == "linear"
    assert [m["file"] for m in data["members"]] == ["m0.json", "m1.json", "m2.json"]
    assert {m["weight"] for m in data["members"]} == {1.0}
    assert out.read_bytes() == sf.save_bag_manifest([tmp_path / "m0.json", tmp_path / "m1.json",
                                                     tmp_path / "m2.json"], out).read_bytes()
    for bad in ([], [tmp_path / "a.json", tmp_path / "a.json"]):
        with pytest.raises(ValueError):
            sf.save_bag_manifest(bad, tmp_path / "x.json")


def test_bag_mean_is_the_linear_mean_of_the_members():
    rng = np.random.default_rng(0)
    ps = [rng.dirichlet(np.ones(3), size=50) for _ in range(3)]
    bag = sf.mix_probabilities(ps, [1.0, 1.0, 1.0], "linear")
    assert np.allclose(bag, sum(ps) / 3) and np.allclose(bag.sum(axis=1), 1)


@needs_lgbm
def test_seeds_give_different_but_reproducible_models_and_a_loadable_bag(external_like_dir, tmp_path):
    m0, m0b, m1 = (_train(external_like_dir, 0), _train(external_like_dir, 0), _train(external_like_dir, 1))
    f0, f0b, f1 = (sf.save_specialists(m, tmp_path / n) for m, n in
                   ((m0, "specialists_s0.json"), (m0b, "again.json"), (m1, "specialists_s1.json")))
    assert f0.read_bytes() == f0b.read_bytes()                  # same seed: byte-identical
    assert f0.read_bytes() != f1.read_bytes()                   # another seed: another model
    bag = sf.load_specialists(sf.save_bag_manifest([f0, f1], tmp_path / "specialists_bag.json"))
    assert bag.mix_mode == "linear" and [w for _, w in bag.members()] == pytest.approx([0.5, 0.5])
    again = sf.load_specialists(tmp_path / "specialists_bag.json")
    x = np.random.default_rng(1).normal(size=(40, len(m0.a.feature_names)))
    p = [mem.b.predict_proba(x) for mem, _ in bag.members()]
    q = [mem.b.predict_proba(x) for mem, _ in again.members()]
    assert all(np.array_equal(u, v) for u, v in zip(p, q))
