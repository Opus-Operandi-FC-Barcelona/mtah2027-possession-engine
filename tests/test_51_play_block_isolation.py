"""Issue 51: the front end of a graded block must not depend on tracking outside the play blocks.

Warm-up, half-time and post-match frames are filled with garbage (random players and balls); every feature of
every graded play frame and the final labels must equal the clean run."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from mtah2027 import pipeline
from mtah2027.models import frame_classifier as fc
from mtah2027.models import specialist_ball_absent as sb
from mtah2027.models import specialist_data as sd
from mtah2027.models import specialist_fusion as sf
from mtah2027.models.specialist_features import build_feature_matrix

FAST = fc.ClassifierParams(n_estimators=20, min_child_samples=20, calib_folds=2, adapt_rounds=10)
TRAIN = dataclasses.replace(sd.TrainingDataParams(), seeds=(0,), stride=15, only_complete_ball=False,
                            min_period_frames=100)
needs_lgbm = pytest.mark.skipif(not fc.lightgbm_available(), reason="lightgbm not installed")


@pytest.fixture(scope="module")
def setup(synthetic_kin):
    td, spec, smart = synthetic_kin["td"], synthetic_kin["spec"], synthetic_kin["st"]
    blocks = [spec.half1, spec.half2_play]
    n = td.n_frames
    inside = np.zeros(n, bool)
    for a, b in blocks:
        inside[a:b + 1] = True
    assert (~inside).sum() > 500            # the synthetic match has warm-up and half-time frames
    rng = np.random.default_rng(51)
    outside = ~inside
    k = int(outside.sum())
    gp = np.array(td.players_xy, dtype=np.float32, copy=True)
    gp[outside] = rng.uniform([0, 0], [105.3, 68.0], size=(k,) + gp.shape[1:]).astype(np.float32)
    gb = np.array(td.ball_xy, dtype=np.float32, copy=True)
    gb[outside] = rng.uniform([0, 0], [105.3, 68.0], size=(k, 2)).astype(np.float32)
    chans = {key: np.array(v, copy=True) for key, v in td.ball_channels.items()}
    gchans = {key: v.copy() for key, v in chans.items()}
    for v in gchans.values():
        v[outside] = rng.uniform(0, 30, size=k).astype(v.dtype)
    clean = (np.array(td.players_xy, dtype=np.float32), np.array(td.ball_xy, dtype=np.float32), chans)
    return {"td": td, "blocks": blocks, "inside": inside, "smart": smart, "garbage": (gp, gb, gchans),
            "clean": clean}


def _fe(m, which, config=None):
    td = m["td"]
    p, b, c = m[which]
    fe = pipeline.compute_front_end(p, td.team_of_slot, b, play_blocks=m["blocks"], fps=td.fps,
                                    ball_channels=c, ball_channel_codes=td.ball_channel_codes,
                                    team_ids=td.team_ids, team_names=td.team_names, smart_tagging=m["smart"],
                                    collective=True, config_path=config)
    em = pipeline.heuristic_emissions(fe, td.fps, config)
    return fe, build_feature_matrix(fe, em, play_blocks=m["blocks"]), em


def _equal(a, b, mask):
    return np.array_equal(np.asarray(a)[mask], np.asarray(b)[mask], equal_nan=True)


def test_mask_outside_blocks_only_touches_outside_frames():
    n = 100
    p = np.ones((n, 3, 2), np.float32)
    b = np.ones((n, 2), np.float32)
    ch = {"x": np.ones(n, np.float32), "n": np.ones(7, np.float32)}
    mp, mb, mc, out = pipeline.mask_outside_blocks(p, b, ch, [(10, 40), (60, 80)])
    ins = ~out
    assert out.sum() == 100 - 31 - 21 and ins[10] and ins[40] and ins[60] and ins[80] and not ins[41]
    assert np.isnan(mp[out]).all() and np.isnan(mb[out]).all() and np.isnan(mc["x"][out]).all()
    assert (mp[ins] == 1).all() and (mb[ins] == 1).all() and (mc["x"][ins] == 1).all()
    assert (p == 1).all() and (b == 1).all() and (ch["x"] == 1).all()       # inputs untouched
    assert mc["n"] is not None and len(mc["n"]) == 7                       # non per-frame channel untouched


def test_front_end_features_of_graded_frames_ignore_everything_outside_the_blocks(setup):
    fe0, fm0, em0 = _fe(setup, "clean")
    fe1, fm1, em1 = _fe(setup, "garbage")
    g = setup["inside"]
    assert fm0.names == fm1.names
    assert _equal(fm0.x, fm1.x, g)
    for key in ("pxy", "bxy", "pvel", "bvel", "bacc", "pspeed", "bspeed", "visible", "dead_score"):
        assert _equal(getattr(fe0, key), getattr(fe1, key), g), key
    assert _equal(em0.log_probs, em1.log_probs, g)
    assert not np.isfinite(fe1.bxy[~g]).any()          # outside frames are blank, never garbage


def test_legacy_behaviour_leaks_across_the_blocks(setup, monkeypatch):
    """Sensitivity of the test: with the isolation switched off the garbage does reach graded frames."""
    orig = pipeline.PipelineParams.from_config.__func__
    monkeypatch.setattr(pipeline.PipelineParams, "from_config", classmethod(
        lambda cls, path=None: dataclasses.replace(orig(cls, path), isolate_play_blocks=False)))
    _, fm0, _ = _fe(setup, "clean")
    _, fm1, _ = _fe(setup, "garbage")
    assert not _equal(fm0.x, fm1.x, setup["inside"])


def test_final_labels_of_the_play_blocks_ignore_garbage_outside(setup):
    td, blocks, g = setup["td"], setup["blocks"], setup["inside"]
    out = []
    for which in ("clean", "garbage"):
        p, b, c = setup[which]
        out.append(pipeline.run_core(p, td.team_of_slot, b, play_blocks=blocks, fps=td.fps, ball_channels=c,
                                     ball_channel_codes=td.ball_channel_codes, team_ids=td.team_ids,
                                     team_names=td.team_names, smart_tagging=setup["smart"],
                                     log=lambda m: None).codes)
    assert np.array_equal(out[0][g], out[1][g])
    assert (out[0][g] != 3).all() and (out[1][~g] == 3).all()


@needs_lgbm
def test_specialist_route_ignores_garbage_outside_the_blocks(setup, external_like_dir):
    ts = sd.build_training_set(external_like_dir, params=TRAIN)
    a = sf.train_specialist_a(ts, params=sf.SpecialistAParams(min_train_rows=50), classifier_params=FAST)
    b = sf.train_specialist_b(ts, params=sb.SpecialistBParams(min_train_rows=50), classifier_params=FAST)
    spec = sf.SpecialistSet(a=a, b=b, fusion=sf.FusionParams(use_pseudo_labels=True),
                            training=sf.describe_training(ts, external_like_dir, a, b))
    td, blocks, g = setup["td"], setup["blocks"], setup["inside"]
    outs = []
    for which in ("clean", "garbage"):
        p, ball, c = setup[which]
        ball = np.array(ball, copy=True)
        ball[blocks[0][0] + 1500:blocks[0][0] + 2600] = np.nan     # a long hidden stretch exercises B
        outs.append(pipeline.run_core(p, td.team_of_slot, ball, play_blocks=blocks, fps=td.fps,
                                      ball_channels=c, ball_channel_codes=td.ball_channel_codes,
                                      smart_tagging=setup["smart"], specialists=spec, log=lambda m: None))
    assert outs[0].diagnostics["emission_source"] == "specialists"
    assert np.array_equal(outs[0].codes[g], outs[1].codes[g])
