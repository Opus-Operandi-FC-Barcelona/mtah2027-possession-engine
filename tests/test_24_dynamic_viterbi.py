"""Issue #24: dynamic Viterbi decoder (numba kernel + identical pure-Python fallback)."""
from __future__ import annotations

import itertools
import time

import numpy as np
import pytest

from mtah2027.config import HALF1, HALF2_DEAD, HALF2_PLAY, N_VIDEO_FRAMES
from mtah2027.decoding import dynamic_viterbi as dv
from mtah2027.decoding.transition_matrix import build_log_transitions

INIT = np.array([-0.6931, -0.6931, -3.0])


def _random_problem(n, seed, coarse=False):
    rng = np.random.default_rng(seed)
    lp = np.log(rng.dirichlet(np.ones(3), size=n))
    tr = -np.abs(rng.normal(0, 3, size=(n, 3, 3)))
    for s in range(3):
        tr[:, s, s] = 0.0
    if coarse:     # many exact ties
        lp, tr = np.round(lp * 2) / 2, np.round(tr)
    return lp, tr


def _score(path, lp, tr, init):
    s = init[path[0]] + lp[0, path[0]]
    for t in range(1, len(path)):
        s += tr[t, path[t - 1], path[t]] + lp[t, path[t]]
    return s


def test_matches_brute_force_on_small_problems():
    for seed in range(12):
        lp, tr = _random_problem(7, seed)
        lp, tr = np.round(lp, 6), np.round(tr, 6)
        best = max(itertools.product(range(3), repeat=7), key=lambda p: (_score(p, lp, tr, INIT), p[::-1]))
        got = dv.viterbi_dynamic(lp, tr, INIT, use_numba=False)
        assert _score(got, lp, tr, INIT) == pytest.approx(_score(best, lp, tr, INIT), abs=1e-9)


@pytest.mark.skipif(not dv.numba_available(), reason="numba not installed")
@pytest.mark.parametrize("coarse", [False, True])
def test_numba_and_fallback_are_identical(coarse):
    for seed in range(4):
        lp, tr = _random_problem(6000, seed, coarse)
        a = dv.viterbi_dynamic(lp, tr, INIT, use_numba=True)
        b = dv.viterbi_dynamic(lp, tr, INIT, use_numba=False)
        assert a.dtype == np.int8 and np.array_equal(a, b)


def test_environment_switch_disables_numba(monkeypatch):
    monkeypatch.setenv("MTAH_DISABLE_NUMBA", "1")
    assert not dv.numba_available()
    lp, tr = _random_problem(50, 1)
    assert len(dv.viterbi_dynamic(lp, tr, INIT)) == 50


def test_tie_break_prefers_lowest_state_and_is_stable():
    n = 6
    lp = np.zeros((n, 3))
    tr = np.zeros((n, 3, 3))
    for use in (False, True) if dv.numba_available() else (False,):
        assert np.array_equal(dv.viterbi_dynamic(lp, tr, np.zeros(3), use_numba=use), np.zeros(n, np.int8))


def test_dynamic_transitions_gate_the_change():
    n = 200
    lp = np.full((n, 3), -5.0)
    lp[:100, 0] = -0.1
    lp[100:, 1] = -0.1                       # evidence flips at 100
    closed = build_log_transitions(n)        # H<->A closed (-11.5): 100 frames x 4.9 > 11.5 -> flips anyway
    path = dv.viterbi_dynamic(lp, closed, INIT)
    assert path[0] == 0 and path[-1] == 1 and (np.diff(path.astype(int)) != 0).sum() == 1
    # a 2-frame blip of contrary evidence is overruled by the closed matrix but not by an open one
    blip = np.full((n, 3), -5.0)
    blip[:, 0] = -0.1
    blip[100:102, 0], blip[100:102, 1] = -5.0, -0.1
    assert (dv.viterbi_dynamic(blip, closed, INIT) == 0).all()
    open_tr = closed.copy()
    open_tr[:, 0, 1] = open_tr[:, 1, 0] = -0.5
    assert (dv.viterbi_dynamic(blip, open_tr, INIT)[100:102] == 1).all()


def test_input_validation():
    lp, tr = _random_problem(10, 0)
    with pytest.raises(ValueError):
        dv.viterbi_dynamic(lp[:, :2], tr, INIT)
    with pytest.raises(ValueError):
        dv.viterbi_dynamic(lp, tr[:5], INIT)
    with pytest.raises(ValueError):
        dv.viterbi_dynamic(lp, tr, INIT[:2])
    assert dv.viterbi_dynamic(lp[:0], tr[:0], INIT).shape == (0,)


def test_default_log_init_from_config():
    init = dv.default_log_init()
    assert init.shape == (3,) and init[2] < init[0]


def test_decode_halves_blocks_x_and_forced_dead():
    n = 1000
    rng = np.random.default_rng(3)
    lp = np.log(rng.dirichlet(np.ones(3), size=n))
    tr = np.zeros((n, 3, 3)) + np.where(np.eye(3) > 0, 0, -4.0)
    half1, dead2, play2 = (100, 399), (500, 549), (550, 899)
    out = dv.decode_match_halves(lp, tr, play_blocks=(half1, play2), dead_blocks=(dead2,), log_init=INIT)
    assert out.shape == (n,) and out.dtype == np.int8
    assert (out[:100] == dv.X).all() and (out[400:500] == dv.X).all() and (out[900:] == dv.X).all()
    assert (out[500:550] == dv.D).all()
    assert set(np.unique(out[100:400])) <= {0, 1, 2} and set(np.unique(out[550:900])) <= {0, 1, 2}


def test_blocks_are_decoded_independently():
    n = 600
    lp = np.full((n, 3), -4.0)
    lp[:, 0] = -0.2                                   # H everywhere ...
    lp[300:, 0], lp[300:, 1] = -4.0, -0.2             # ... A in the second block
    tr = np.where(np.eye(3) > 0, 0.0, -50.0) * np.ones((n, 1, 1))
    out = dv.decode_match_halves(lp, tr, play_blocks=((0, 299), (300, 599)), dead_blocks=(), log_init=INIT)
    assert (out[:300] == 0).all() and (out[300:] == 1).all()   # impossible in a single decode (-50 switch)


@pytest.mark.slow
def test_full_length_decoding_is_fast_with_numba():
    if not dv.numba_available():
        pytest.skip("numba not installed")
    lp, tr = _random_problem(N_VIDEO_FRAMES, 5)
    dv.viterbi_dynamic(lp[:100], tr[:100], INIT)           # compile outside the timing
    t0 = time.perf_counter()
    out = dv.decode_match_halves(lp, tr, play_blocks=(HALF1, HALF2_PLAY), dead_blocks=(HALF2_DEAD,),
                                 log_init=INIT, use_numba=True)
    assert time.perf_counter() - t0 < 1.0
    assert (out[HALF2_DEAD[0]:HALF2_DEAD[1] + 1] == dv.D).all()
