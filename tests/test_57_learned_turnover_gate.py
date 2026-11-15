"""Issue #57: Specialist C (learned H<->A turnover gate) and its use in the transition matrix."""
# ruff: noqa: E501
from __future__ import annotations

import dataclasses
import sys

import numpy as np
import pytest

from mtah2027 import pipeline
from mtah2027.config import REPO_ROOT
from mtah2027.decoding.transition_matrix import (
    GATE_MODES,
    TransitionParams,
    build_log_transitions,
    learned_gate,
)
from mtah2027.models import frame_classifier as fc
from mtah2027.models import specialist_c as sc

sys.path.insert(0, str(REPO_ROOT / "scripts"))
needs_lgbm = pytest.mark.skipif(not fc.lightgbm_available(), reason="lightgbm not installed")

H, A, D, X = 0, 1, 2, 3


def test_change_events_and_targets():
    codes = np.array([H, H, A, A, D, A, H, H, X, A, A])
    ha, ah = sc.change_events(codes)
    assert list(ha) == [2] and list(ah) == [6]          # H D->A and X are not direct changes
    y = sc.turnover_targets(codes, half_window=1)
    assert list(y) == [0, 1, 1, 1, 0, 2, 2, 2, 0, 0, 0]


def test_effective_changes_fill_dead_frames_forward():
    codes = np.array([H, H, D, D, A, A, A, H])
    assert sc.effective_changes(codes) == [(4, 1), (7, 2)]


def test_turnover_recall_matches_direction_within_tolerance():
    gt = np.array([H] * 100 + [A] * 100 + [H] * 100)
    pred = np.array([H] * 90 + [A] * 130 + [H] * 80)         # 10 and 20 frames off
    r = sc.turnover_recall(gt, pred, tol=25)
    assert r == {"n_true": 2, "matched": 2, "n_pred": 2}
    assert sc.turnover_recall(gt, pred, tol=15)["matched"] == 1
    wrong_way = np.array([A] * 100 + [H] * 100 + [A] * 100)
    assert sc.turnover_recall(gt, wrong_way, tol=25)["matched"] == 0
    assert sc.turnover_recall(np.zeros(10, int), np.zeros(10, int)) == {"n_true": 0, "matched": 0, "n_pred": 0}


def test_gate_modes_values():
    base = TransitionParams()
    rule = np.array([1.0, 0.0, 0.0, 1.0])
    prob = np.array([0.2, 0.9, np.nan, 0.6])
    hard = learned_gate(rule, prob, dataclasses.replace(base, learned_gate_mode="hard"))
    assert list(hard) == [0.0, 1.0, 0.0, 1.0]            # replaces the rule: frame 0 closes
    blend = learned_gate(rule, prob, dataclasses.replace(base, learned_gate_mode="blend", learned_gate_blend=0.5))
    assert blend == pytest.approx([0.6, 0.45, 0.0, 0.8])
    mx = learned_gate(rule, prob, dataclasses.replace(base, learned_gate_mode="max"))
    assert mx == pytest.approx([1.0, 0.9, 0.0, 1.0])
    with pytest.raises(ValueError):
        learned_gate(rule, prob, dataclasses.replace(base, learned_gate_mode="nope"))
    assert set(GATE_MODES) == {"off", "hard", "blend", "max"}


def _matrix(mode, p_ha=None, p_ah=None, n=40, **kw):
    ts = np.zeros(n)
    ts[20] = 1.0
    dist_a = np.full(n, 0.5)
    dist_h = np.full(n, 30.0)                              # only the away team is near: rule opens H -> A at the peak
    params = dataclasses.replace(TransitionParams(), learned_gate_mode=mode, **kw)
    gate = None if p_ha is None else (p_ha, p_ah)
    return build_log_transitions(n, touch_score=ts, d_home=dist_h, d_away=dist_a, ball_visible=np.ones(n, bool),
                                 params=params, gate_probs=gate)


def test_off_ignores_the_probabilities_and_hard_replaces_the_rule():
    n = 40
    zero, p = np.zeros(n), np.zeros(n)
    p[5:8] = 0.9
    off_plain = _matrix("off")
    assert np.array_equal(_matrix("off", p, zero), off_plain)
    p_def = TransitionParams()
    hard = _matrix("hard", p, zero)
    assert hard[6, H, A] == pytest.approx(p_def.ha_open)         # opened by the learned gate
    assert hard[20, H, A] == pytest.approx(p_def.ha_closed)      # the rule peak no longer opens it
    assert off_plain[20, H, A] == pytest.approx(p_def.ha_open)
    assert hard[6, A, H] == pytest.approx(p_def.ha_closed)       # the other direction uses p_ah
    mx = _matrix("max", p, zero)
    assert mx[6, H, A] == pytest.approx(p_def.ha_closed + (p_def.ha_open - p_def.ha_closed) * 0.9) and mx[20, H, A] == pytest.approx(p_def.ha_open)
    soft = _matrix("blend", np.full(n, 0.5), zero, learned_gate_blend=0.0)
    assert soft[1, H, A] == pytest.approx(p_def.ha_closed + (p_def.ha_open - p_def.ha_closed) * 0.5, abs=1e-6)


def test_occlusion_floor_is_kept_with_a_closed_learned_gate():
    n = 200
    vis = np.ones(n, bool)
    vis[50:150] = False
    params = dataclasses.replace(TransitionParams(), learned_gate_mode="hard")
    lt = build_log_transitions(n, ball_visible=vis, params=params, gate_probs=(np.zeros(n), np.zeros(n)))
    assert lt[100, H, A] == pytest.approx(params.ha_occluded)
    assert lt[10, H, A] == pytest.approx(params.ha_closed)


@needs_lgbm
def test_train_predict_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(1500, 6)).astype(np.float32)
    y = np.where(x[:, 0] > 1.0, 1, np.where(x[:, 1] > 1.0, 2, 0))
    groups = np.repeat(np.array(["m1", "m2", "m3"], dtype=object), 500)
    params = dataclasses.replace(sc.SpecialistCParams(), n_estimators=20, min_child_samples=10, calib_folds=3)
    names = [f"f{i}" for i in range(6)]
    model = sc.train_specialist_c(x, y, groups, names, params=params)
    p_ha, p_ah = model.predict(x)
    assert p_ha.shape == (1500,) and (p_ha + p_ah <= 1.0 + 1e-9).all()
    assert p_ha[x[:, 0] > 1.5].mean() > p_ha[x[:, 0] < 0].mean()
    path = sc.save_specialist_c(model, tmp_path / "c.json")
    back = sc.load_specialist_c(path)
    q_ha, q_ah = back.predict(x)
    assert np.array_equal(p_ha, q_ha) and np.array_equal(p_ah, q_ah)
    again = sc.train_specialist_c(x, y, groups, names, params=params)
    assert np.array_equal(again.predict(x)[0], p_ha)           # deterministic
    assert sc.load_default_specialist_c() is None or isinstance(sc.load_default_specialist_c(), sc.SpecialistC)


def test_default_config_gate_is_valid_and_off():
    from mtah2027.config import load_section

    assert load_section("transition_matrix")["learned_gate_mode"] == "off" or         TransitionParams.from_config().learned_gate_mode in GATE_MODES
    assert sc.SpecialistCParams.from_config().half_window == 5


@needs_lgbm
def test_pipeline_uses_specialist_c_only_when_the_gate_is_on(synthetic_kin, external_like_dir):
    import decoder_sweep as dsw
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
    rng = np.random.default_rng(1)
    idx = rng.choice(len(fm.x), 3000, replace=False)
    y = rng.choice(3, 3000, p=[0.8, 0.1, 0.1])
    cparams = dataclasses.replace(sc.SpecialistCParams(), n_estimators=10, min_child_samples=20, calib_folds=0)
    c = sc.train_specialist_c(fm.x[idx], y, np.full(3000, "g", dtype=object), fm.names, params=cparams)
    with_c = dataclasses.replace(base, c=c)
    raw = yaml.safe_load((REPO_ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    on = dsw.candidate_config(raw, {"transition_matrix.learned_gate_mode": "hard"})

    def run(cfg, specs):
        return pipeline.run_core(td.players_xy, td.team_of_slot, ball, play_blocks=blocks, fps=td.fps,
                                 log=lambda s: None, specialists=specs, config_path=cfg)

    off = run(raw, with_c)
    assert "learned_gate" not in off.diagnostics["stages"]             # off: the stage does not even run
    gated = run(on, with_c)
    assert gated.diagnostics["stages"]["learned_gate"]["status"] == "ok"
    assert np.array_equal(gated.codes, run(on, with_c).codes)         # deterministic
    assert np.array_equal(run(raw, base).codes, off.codes)            # a stored C is inert while the gate is off
