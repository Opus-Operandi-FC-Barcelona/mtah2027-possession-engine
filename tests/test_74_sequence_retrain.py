"""Issue #74: carrier / swarm input of the sequence model and the boundary-weighted loss (both off by default)."""
# ruff: noqa: E501
from __future__ import annotations

import dataclasses
import json
import sys

import numpy as np
import pytest

from mtah2027 import pipeline
from mtah2027.config import REPO_ROOT
from mtah2027.models import frame_classifier as fc
from mtah2027.models import sequence_model as sq
from mtah2027.models.specialist_features import FeatureParams, build_feature_matrix, carrier_swarm_columns

sys.path.insert(0, str(REPO_ROOT / "scripts"))
needs_lgbm = pytest.mark.skipif(not fc.lightgbm_available(), reason="lightgbm not installed")


def _fe(synthetic_kin):
    td, spec = synthetic_kin["td"], synthetic_kin["spec"]
    blocks = [spec.half1, spec.half2_play]
    fe = pipeline.compute_front_end(td.players_xy, td.team_of_slot, td.ball_xy, play_blocks=blocks, fps=td.fps,
                                    collective=True)
    return td, blocks, fe


def test_extra_columns_are_the_feature_matrix_columns_of_64(synthetic_kin):
    td, blocks, fe = _fe(synthetic_kin)
    sx, names = carrier_swarm_columns(fe)
    assert tuple(names) == sq.EXTRA_NAMES and sx.shape == (len(td.players_xy), len(names))
    em = pipeline.heuristic_emissions(fe, td.fps)
    on = build_feature_matrix(fe, em, play_blocks=blocks, params=dataclasses.replace(FeatureParams(), carrier_swarm=True))
    off = build_feature_matrix(fe, em, play_blocks=blocks, params=FeatureParams())
    assert [k for k in on.names if k in sq.EXTRA_NAMES] == list(names)
    assert np.array_equal(on.x[:, [on.names.index(k) for k in names]], sx, equal_nan=True)
    assert not set(sq.EXTRA_NAMES) & set(off.names)                    # off: the adopted feature set is untouched
    assert sq.select_columns(on.names)[:len(sq.select_columns(off.names))] == sq.select_columns(off.names)


def test_extra_names_of_models_and_ensembles():
    base = [f"f{i}" for i in range(4)]
    plain = sq.random_model(base, channels=4, dilations=(1, 2), seed=0)
    wide = sq.random_model(base + list(sq.EXTRA_NAMES[:2]), channels=4, dilations=(1, 2), seed=1)
    assert plain.extra_names() == [] and wide.extra_names() == list(sq.EXTRA_NAMES[:2])
    assert sq.SequenceEnsemble([plain, wide]).extra_names() == sorted(sq.EXTRA_NAMES[:2])


@needs_lgbm
def test_pipeline_computes_the_extra_columns_for_a_model_that_needs_them(synthetic_kin, external_like_dir):
    import yaml

    from mtah2027.models import specialist_ball_absent as sb
    from mtah2027.models import specialist_data as sd
    from mtah2027.models import specialist_fusion as sf

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
    names = sq.select_columns(fm.names) + list(sq.EXTRA_NAMES)
    seq = sq.random_model(names, channels=8, dilations=(1, 4, 16), seed=0)
    raw = yaml.safe_load((REPO_ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    raw["sequence_model"].update(enabled=True, mode="emissions", emission_weight=1.0)

    def run():
        return pipeline.run_core(td.players_xy, td.team_of_slot, ball, play_blocks=blocks, fps=td.fps,
                                 log=lambda s: None, specialists=dataclasses.replace(base, seq=seq), config_path=raw)

    r1 = run()
    assert r1.diagnostics["stages"]["sequence_model"]["status"] == "ok"
    assert np.array_equal(r1.codes, run().codes)


def test_boundary_mask_is_the_window_around_direct_changes():
    y = np.array([0] * 10 + [1] * 10 + [3] * 5 + [1] * 5 + [0] * 10)      # H->A at 10, D gap, A->H at 30
    near = sq.turnover_targets(np.where(y == 3, 9, y), 3) > 0
    assert np.flatnonzero(near).tolist() == list(range(7, 14)) + list(range(27, 34))   # not around the D gap


def test_training_cli_defaults_leave_the_72_network_unchanged():
    torch = pytest.importorskip("torch")
    import train_sequence as ts

    sample = lambda: ({"x": np.zeros((6, 3), np.float16), "y": np.zeros(6, np.int8)}, slice(0, 6))  # noqa: E731
    x, y, t, nb = ts.batch(sample, 1, 6, np.zeros(3, np.float32), np.ones(3, np.float32), "cpu", 5)
    assert not bool(nb.any()) and x.shape == (1, 6, 3) and isinstance(torch.zeros(1), torch.Tensor)
    json.dumps({})
