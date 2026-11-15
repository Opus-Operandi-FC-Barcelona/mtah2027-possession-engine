"""Issue #78: linear-chain CRF on top of the sequence model (C1 global transitions, C2 feature-conditioned H<->A)."""
# ruff: noqa: E501
from __future__ import annotations

import dataclasses
import importlib.util
import itertools
import sys

import numpy as np
import pytest

from mtah2027 import pipeline
from mtah2027.config import REPO_ROOT, load_section
from mtah2027.models import frame_classifier as fc
from mtah2027.models import sequence_crf as sc
from mtah2027.models import sequence_model as sq

sys.path.insert(0, str(REPO_ROOT / "scripts"))
needs_lgbm = pytest.mark.skipif(not fc.lightgbm_available(), reason="lightgbm not installed")


def _toy_chain(t=6, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(t, 3)), rng.normal(size=(t, 3, 3)), rng.integers(0, 3, size=t)


def test_partition_function_matches_brute_force_and_viterbi_is_consistent():
    for seed in range(3):
        u, tr, y = _toy_chain(6, seed)
        paths = list(itertools.product(range(3), repeat=len(u)))
        scores = np.array([sc.chain_score(u, tr, np.array(p)) for p in paths])
        m = scores.max()
        assert sc.chain_log_partition(u, tr) == pytest.approx(m + np.log(np.exp(scores - m).sum()), abs=1e-9)
        assert sc.chain_nll(u, tr, y) >= 0.0
        assert np.array_equal(sc.chain_viterbi(u, tr), np.array(paths[int(scores.argmax())]))
        assert sc.chain_score(u, tr, sc.chain_viterbi(u, tr)) == pytest.approx(m, abs=1e-5)


def test_chain_transitions_zero_diagonal_and_ha_columns():
    raw = np.arange(9, dtype=float).reshape(3, 3) - 5
    ha = np.array([[-1.0, -2.0], [-3.0, -4.0]])
    tr = sc.chain_transitions(raw, 2, ha)
    assert np.all(tr[:, np.arange(3), np.arange(3)] == 0.0)
    assert tr[1, sc.H, sc.A] == -3.0 and tr[1, sc.A, sc.H] == -4.0
    assert tr[0, sc.H, sc.D] == raw[0, 2] and tr[0, sc.D, sc.A] == raw[2, 1]
    assert np.array_equal(sc.chain_transitions(raw, 2)[:, sc.H, sc.A], np.full(2, raw[0, 1]))


def test_override_ha_replaces_only_finite_rows_and_only_the_ha_entries():
    rng = np.random.default_rng(0)
    lt = rng.normal(size=(5, 3, 3))
    orig = lt.copy()
    ha = np.array([[-1.23456789, -2.5], [np.nan, np.nan], [-0.5, -0.25], [np.nan, np.nan], [-9.0, -8.0]])
    out = sc.override_ha_transitions(lt, ha)
    ok = np.isfinite(ha[:, 0])
    assert np.array_equal(out[~ok], lt[~ok])
    assert np.array_equal(out[ok, sc.H, sc.A], np.round(ha[ok, 0], 6))
    assert np.array_equal(out[ok, sc.A, sc.H], np.round(ha[ok, 1], 6))
    keep = np.ones((3, 3), bool)
    keep[sc.H, sc.A] = keep[sc.A, sc.H] = False
    assert np.array_equal(out[:, keep], lt[:, keep])
    assert np.array_equal(lt, orig)                                    # the input is not modified


def _crf_model(names, seed=0, head=True):
    m = sq.random_model(names, channels=8, dilations=(1, 2, 4), seed=seed)
    w = dict(m.w)
    if head:
        rng = np.random.default_rng(seed + 100)
        w["w_ha"] = rng.normal(0, 0.5, (8, 2)).astype(np.float32)
        w["b_ha"] = np.array([6.0, 6.0], np.float32)
        w["crf_trans"] = np.zeros((3, 3), np.float32)
    return sq.SequenceModel(w, m.meta)


def test_predict_all_head_is_negative_nan_outside_blocks_and_none_without_head():
    names = [f"f{i}" for i in range(5)]
    x = np.random.default_rng(1).normal(size=(120, 5))
    blocks = [(10, 59), (70, 110)]
    m = _crf_model(names)
    pe, pt, ph = m.predict_all(x, names, blocks)
    assert m.has_crf_transition and ph.shape == (120, 2)
    assert np.all(ph[10:60] <= 0) and np.isfinite(ph[10:60]).all()
    assert np.isnan(ph[:10]).all() and np.isnan(ph[60:70]).all()
    assert np.array_equal(pe, m.predict(x, names, blocks)[0], equal_nan=True)
    plain = _crf_model(names, head=False)
    assert plain.predict_all(x, names, blocks)[2] is None and not plain.has_crf_transition
    # the emissions of the production forward are untouched by the extra keys
    bare = sq.SequenceModel({k: v for k, v in m.w.items() if k not in ("w_ha", "b_ha", "crf_trans")}, m.meta)
    assert np.array_equal(bare.predict(x, names, blocks)[0], pe, equal_nan=True)
    ens = sq.SequenceEnsemble([m, _crf_model(names, 1)])
    _, _, eh = ens.predict_all(x, names, blocks)
    assert eh.shape == (120, 2) and np.isfinite(eh[10:60]).all()
    assert sq.SequenceEnsemble([plain, plain]).predict_all(x, names, blocks)[2] is None


def test_json_round_trip_keeps_the_head_bit_exact(tmp_path):
    names = [f"f{i}" for i in range(5)]
    m = _crf_model(names)
    back = sq.load_sequence_json(sq.save_sequence_json(m, tmp_path / "m.json"))
    assert back.has_crf_transition
    assert all(np.array_equal(m.w[k], back.w[k]) for k in m.w)
    x = np.random.default_rng(2).normal(size=(60, 5))
    assert np.array_equal(m.predict_all(x, names, [(0, 59)])[2], back.predict_all(x, names, [(0, 59)])[2], equal_nan=True)


def test_config_flag_is_off_by_default():
    p = sq.SequenceParams.from_config()
    assert p.crf_transition is False and p.use_crf_transition is False
    assert load_section("sequence_model")["crf_transition"] is False
    assert sq.SequenceParams(enabled=True, crf_transition=True).use_crf_transition


def _train_module():
    pytest.importorskip("torch")
    spec = importlib.util.spec_from_file_location("train_sequence_78", REPO_ROOT / "scripts" / "train_sequence.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_torch_crf_nll_matches_numpy_with_padding_and_has_finite_gradients():
    mod = _train_module()
    import torch

    rng = np.random.default_rng(3)
    b, t = 4, 11
    u, tr = rng.normal(size=(b, t, 3)), rng.normal(size=(b, t, 3, 3))
    y = rng.integers(0, 3, size=(b, t))
    lengths = [11, 7, 2, 1]
    valid = np.arange(t)[None, :] < np.array(lengths)[:, None]
    y = np.where(valid, y, 3)
    tu, tt = torch.tensor(u, requires_grad=True), torch.tensor(tr, requires_grad=True)
    nll = mod.crf_nll(tu, tt, torch.tensor(y), torch.tensor(valid))
    want = sum(sc.chain_nll(u[i, :n], tr[i, :n], y[i, :n]) for i, n in enumerate(lengths)) / sum(lengths)
    assert float(nll) == pytest.approx(want, abs=1e-9)
    nll.backward()
    assert torch.isfinite(tu.grad).all() and torch.isfinite(tt.grad).all()
    assert tu.grad[3, 1:].abs().sum() == 0 and tu.grad[2, 2:].abs().sum() == 0      # padding gets no gradient


@pytest.mark.parametrize("variant", ["c1", "c2"])
def test_exported_torch_network_matches_numpy_inference(variant):
    mod = _train_module()
    import torch

    torch.manual_seed(0)
    names = [f"f{i}" for i in range(5)]
    net = mod.TCN(5, 8, dilations=(1, 2, 4), crf=variant).eval()
    with torch.no_grad():
        net.he.bias.add_(torch.tensor([0.3, -0.2, 0.1]))
        if variant == "c2":
            net.ha.weight.normal_(0, 0.5)
        net.crf_raw.normal_(-3.0, 1.0)
    sm = net.export(np.zeros(5, np.float32), np.ones(5, np.float32), names, (1, 2, 4))
    x = np.random.default_rng(4).normal(size=(40, 5)).astype(np.float32)
    with torch.no_grad():
        le, lt, trans = net.forward_crf(torch.tensor(x)[None])
    np_le, np_lt, np_ha = sm.forward_all(x.astype(np.float64))
    assert np.allclose(np_le, le[0].numpy(), atol=1e-4) and np.allclose(np_lt, lt[0].numpy(), atol=1e-4)
    assert np.allclose(sm.w["crf_trans"], sc.global_transitions(sm.w["crf_trans"]))
    if variant == "c2":
        assert sm.has_crf_transition
        assert np.allclose(np_ha[:, 0], trans[0, :, 0, 1].numpy(), atol=1e-4)
        assert np.allclose(np_ha[:, 1], trans[0, :, 1, 0].numpy(), atol=1e-4)
        assert np.all(np_ha <= 0)
    else:
        assert not sm.has_crf_transition and np_ha is None
    plain = mod.TCN(5, 8, dilations=(1, 2, 4)).export(np.zeros(5, np.float32), np.ones(5, np.float32), names, (1, 2, 4))
    assert "crf_trans" not in plain.w and "w_ha" not in plain.w          # the #72 network is unchanged


def test_crf_training_reduces_the_loss_on_a_toy_task():
    mod = _train_module()
    import torch
    from torch.nn import functional as F

    torch.manual_seed(0)
    b, t, f = 6, 64, 4
    rng = np.random.default_rng(0)
    state = np.cumsum(rng.random((b, t)) < 0.05, axis=1) % 3                   # long runs of H / A / D
    x = np.eye(3)[state] @ rng.normal(size=(3, f)) + rng.normal(0, 0.3, (b, t, f))
    xt, yt = torch.tensor(x, dtype=torch.float32), torch.tensor(state)
    valid = torch.ones(b, t, dtype=torch.bool)
    net = mod.TCN(f, 8, dilations=(1, 2, 4), crf="c2")
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)

    def loss():
        le, _, tr = net.forward_crf(xt)
        return mod.crf_nll(le, tr, yt, valid) + 0.5 * F.cross_entropy(le.reshape(-1, 3), yt.reshape(-1))

    first = float(loss())
    for _ in range(60):
        opt.zero_grad()
        loss().backward()
        opt.step()
    assert float(loss()) < 0.5 * first


@needs_lgbm
def test_pipeline_crf_transition_flag(synthetic_kin, external_like_dir):
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
    names = sq.select_columns(fm.names)
    head = _crf_model(names)
    bare = sq.SequenceModel({k: v for k, v in head.w.items() if k not in ("w_ha", "b_ha", "crf_trans")}, head.meta)
    with_head = dataclasses.replace(base, seq=head)
    no_head = dataclasses.replace(base, seq=bare)
    raw = yaml.safe_load((REPO_ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))

    def cfg(**kw):
        out = yaml.safe_load(yaml.safe_dump(raw))
        out["sequence_model"].update(kw)
        return out

    def run(config, specs):
        return pipeline.run_core(td.players_xy, td.team_of_slot, ball, play_blocks=blocks, fps=td.fps,
                                 log=lambda s: None, specialists=specs, config_path=config)

    on = dict(enabled=True, mode="emissions", emission_weight=1.0)
    off = run(cfg(**on), with_head)
    crf = run(cfg(**on, crf_transition=True), with_head)
    assert crf.diagnostics["stages"]["sequence_model"]["status"] == "ok"
    assert np.array_equal(crf.codes, run(cfg(**on, crf_transition=True), with_head).codes)       # deterministic
    # flag off: a head in the model changes nothing
    assert np.array_equal(off.codes, run(cfg(**on), no_head).codes)
    # flag on without a head: skipped, identical to the flag off
    assert np.array_equal(run(cfg(**on, crf_transition=True), no_head).codes, run(cfg(**on), no_head).codes)
