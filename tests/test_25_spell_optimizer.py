"""Issue #25: spellF1-oriented post-processing."""
from __future__ import annotations

import numpy as np
import pytest

from mtah2027.config import ConfigError, params_from_mapping
from mtah2027.postprocessing import spell_optimizer as so

FPS = 25
H, A, D, X = 0, 1, 2, 3
P = so.SpellParams()


def _labels(spec):
    """``[(state, frames), ...]`` -> code array."""
    return np.concatenate([np.full(n, c, dtype=np.int8) for c, n in spec])


def _opt(codes, **kw):
    n = len(codes)
    kw.setdefault("play_blocks", ((0, n - 1),))
    kw.setdefault("params", P)
    return so.optimize_spells(codes, **kw)


def _n_runs(codes, state=None):
    change = np.flatnonzero(np.diff(codes.astype(int)) != 0)
    starts = np.concatenate([[0], change + 1])
    return int(len(starts) if state is None else (codes[starts] == state).sum())


def test_yaml_section_matches_defaults_and_rejects_typos():
    assert so.SpellParams.from_config() == P
    with pytest.raises(ConfigError):
        params_from_mapping(so.SpellParams, {"micro_poss": 1}, "spell_optimizer")


def test_micro_possession_absorbed_when_ball_not_visible():
    codes = _labels([(H, 100), (A, 20), (H, 100)])            # 0.8 s A blip
    out, st = _opt(codes)
    assert (out == H).all() and st["micro_poss"] == 1


def test_micro_possession_kept_with_visible_ball_and_double_touch():
    codes = _labels([(H, 100), (A, 20), (H, 100)])
    vis = np.ones(len(codes), bool)
    touch = np.zeros(len(codes))
    touch[102], touch[115] = 0.9, 0.8                          # gain + pass inside the A run
    out, _ = _opt(codes, ball_visible=vis, touch_score=touch)
    assert np.array_equal(out, codes)
    touch[115] = 0.0                                           # a single touch is not enough
    out, _ = _opt(codes, ball_visible=vis, touch_score=touch)
    assert (out == H).all()
    out, _ = _opt(codes, ball_visible=np.zeros(len(codes), bool), touch_score=np.where(touch > 0, 0.9, 0))
    assert (out == H).all()


def test_micro_possession_of_normal_length_untouched():
    codes = _labels([(H, 100), (A, 35), (H, 100)])             # 1.4 s > 1.2 s
    out, st = _opt(codes)
    assert np.array_equal(out, codes) and st["micro_poss"] == 0


def test_micro_possession_not_absorbed_between_different_teams_or_dead():
    codes = _labels([(H, 100), (A, 20), (D, 100)])
    out, _ = _opt(codes)
    assert np.array_equal(out, codes)


def test_micro_dead_absorbed_between_same_team_and_split_between_different():
    codes = _labels([(H, 100), (D, 10), (H, 100)])             # 0.4 s D
    out, st = _opt(codes)
    assert (out == H).all() and st["micro_dead"] == 1
    codes = _labels([(H, 100), (D, 10), (A, 100)])
    out, _ = _opt(codes)
    assert (out[:105] == H).all() and (out[105:] == A).all()
    long_d = _labels([(H, 100), (D, 20), (H, 100)])            # 0.8 s D stays (a real stoppage)
    out, _ = _opt(long_d)
    assert np.array_equal(out, long_d)


def test_isolated_possession_between_dead_runs_becomes_dead():
    codes = _labels([(D, 100), (H, 10), (D, 100)])
    out, st = _opt(codes)
    assert (out == D).all() and st["isolated"] == 1


def test_forced_dead_applied_and_protected():
    codes = _labels([(H, 400)])
    forced = np.zeros(400, bool)
    forced[200:210] = True                                    # 0.4 s forced D, shorter than min_dead
    out, st = _opt(codes, forced_dead_mask=forced)
    assert (out[200:210] == D).all() and (out[:200] == H).all() and (out[210:] == H).all()
    assert st["forced_dead_frames"] == 10 and _n_runs(out, H) == 2


def test_x_frames_are_never_changed_and_split_segments():
    codes = _labels([(H, 100), (X, 5), (A, 10), (X, 5), (H, 100)])
    out, _ = _opt(codes)
    assert (out[100:105] == X).all() and (out[115:120] == X).all()
    assert (out[105:115] == A).all()                           # not between H and H once X separates them


def test_never_crosses_block_boundaries():
    codes = _labels([(H, 100), (A, 20), (H, 100)])
    out, _ = _opt(codes, play_blocks=((0, 99), (100, 219)))     # the A blip is a block of its own
    assert (out[:100] == H).all()
    out2, _ = _opt(codes, play_blocks=((0, 99), (120, 219)))
    assert (out2[100:120] == A).all()                          # outside every block: untouched


def test_dead_bridge_inserted_in_long_possession_with_support():
    n = 1500                                                   # 60 s of one possession
    codes = _labels([(H, n)])
    ds = np.zeros(n)
    ds[600:750] = 0.95                                         # 6 s of dead evidence
    sup = np.zeros(n, bool)
    sup[650:700] = True
    out, st = _opt(codes, dead_score=ds, dead_support=sup)
    assert st["bridges"] == 1
    assert (out[605:745] == D).all() and (out[:590] == H).all() and (out[760:] == H).all()
    assert _n_runs(out, H) == 2


def test_dead_bridge_requires_length_support_and_long_possession():
    n = 1500
    ds = np.zeros(n)
    ds[600:750] = 0.95
    sup = np.zeros(n, bool)
    sup[650:700] = True
    _, st = _opt(_labels([(H, n)]), dead_score=ds)                         # no geometric support
    assert st["bridges"] == 0
    short = ds.copy()
    short[600:750] = 0.0
    short[600:660] = 0.95                                                  # 2.4 s only
    _, st = _opt(_labels([(H, n)]), dead_score=short, dead_support=sup)
    assert st["bridges"] == 0
    _, st = _opt(_labels([(H, 400)]), dead_score=ds[:400] + 0.95, dead_support=np.ones(400, bool))
    assert st["bridges"] == 0                                              # possession under 25 s


def test_idempotent_and_deterministic():
    rng = np.random.default_rng(0)
    codes = np.repeat(rng.integers(0, 3, 300).astype(np.int8), rng.integers(3, 60, 300))
    out1, _ = _opt(codes)
    out2, _ = _opt(out1)
    out3, _ = _opt(codes)
    assert np.array_equal(out1, out3) and np.array_equal(out1, out2)


def test_input_not_modified_and_length_checked():
    codes = _labels([(H, 100), (A, 20), (H, 100)])
    keep = codes.copy()
    _opt(codes)
    assert np.array_equal(codes, keep)
    with pytest.raises(ValueError):
        _opt(codes, ball_visible=np.ones(5, bool))


def test_blocks_not_starting_at_zero_give_the_same_result():
    inner = _labels([(H, 100), (A, 20), (H, 100), (D, 10), (H, 60), (X, 4), (A, 60)])
    base, _ = _opt(inner, play_blocks=((0, len(inner) - 1),))
    padded = np.concatenate([np.full(777, X, dtype=np.int8), inner, np.full(50, X, dtype=np.int8)])
    out, _ = so.optimize_spells(padded, play_blocks=((777, 777 + len(inner) - 1),), params=P)
    assert np.array_equal(out[777:777 + len(inner)], base)
    assert (out[:777] == X).all() and (out[777 + len(inner):] == X).all()
    n = len(padded)
    ds = np.zeros(n)
    ds[777 + 20:777 + 80] = 0.95
    out2, st = so.optimize_spells(padded, play_blocks=((777, 777 + len(inner) - 1),), dead_score=ds,
                                  dead_support=np.ones(n, bool), params=P)
    assert out2.shape == padded.shape and st["micro_poss"] >= 1
