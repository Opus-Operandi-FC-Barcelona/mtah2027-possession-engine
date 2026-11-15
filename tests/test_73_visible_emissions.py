"""Issue #73: sequence-model emissions on ball-visible frames (``sequence_model.visible_weight``)."""
from __future__ import annotations

import dataclasses

import numpy as np

from mtah2027.config import load_section
from mtah2027.models import sequence_model as sq


def _data():
    n = 10
    lp = np.round(np.log(np.full((n, 3), 1 / 3)), 6)
    seq = np.tile([0.8, 0.1, 0.1], (n, 1))
    seq[3] = np.nan
    vis = np.array([True] * 4 + [False] * 6)
    return lp, seq, vis


def test_code_default_is_zero_and_config_is_the_adopted_v50s():
    assert sq.SequenceParams().visible_weight == 0.0
    cfg = load_section("sequence_model")
    p = sq.SequenceParams.from_config()
    assert (p.visible_weight, p.emission_scale, p.emission_weight) == (0.5, 0.8, 1.0)
    assert cfg["visible_weight"] == 0.5


def test_default_leaves_visible_frames_untouched():
    lp, seq, vis = _data()
    p = sq.SequenceParams(enabled=True, mode="emissions", emission_weight=1.0, emission_scale=0.5)
    out = sq.mix_hidden_emissions(lp, seq, vis, p)
    assert np.array_equal(out[:4], lp[:4])
    assert np.allclose(out[5], np.round(0.5 * np.log(seq[5]), 6))


def test_visible_weight_mixes_visible_frames_and_skips_nan_rows():
    lp, seq, vis = _data()
    p = sq.SequenceParams(enabled=True, mode="emissions", emission_weight=1.0, emission_scale=0.8,
                          visible_weight=0.25)
    out = sq.mix_hidden_emissions(lp, seq, vis, p)
    assert np.array_equal(out[3], lp[3])                                   # NaN sequence row untouched
    expect_vis = 0.75 * lp[0] + 0.25 * 0.8 * np.log(seq[0])
    assert np.allclose(out[0], np.round(expect_vis, 6))
    assert np.allclose(out[5], np.round(0.8 * np.log(seq[5]), 6))          # hidden: weight 1.0, scale 0.8


def test_visible_only_and_no_op_cases():
    lp, seq, vis = _data()
    p = sq.SequenceParams(enabled=True, mode="emissions", emission_weight=0.0, emission_scale=0.5,
                          visible_weight=0.5)
    out = sq.mix_hidden_emissions(lp, seq, vis, p)
    assert np.array_equal(out[4:], lp[4:]) and not np.array_equal(out[0], lp[0])
    same = sq.mix_hidden_emissions(lp, seq, vis, dataclasses.replace(p, visible_weight=0.0))
    assert same is lp


def test_output_is_quantised_float64():
    lp, seq, vis = _data()
    p = sq.SequenceParams(enabled=True, mode="emissions", emission_weight=1.0, visible_weight=0.5)
    out = sq.mix_hidden_emissions(lp, seq, vis, p)
    assert out.dtype == np.float64 and np.array_equal(out, np.round(out, 6))
