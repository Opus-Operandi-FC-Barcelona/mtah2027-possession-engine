"""Issue #62: decoding by expected spellF1 (posterior sampling + MBR selection)."""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "evaluation", ROOT):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import possession_metrics as pm  # noqa: E402

from mtah2027.decoding import dynamic_viterbi as dv  # noqa: E402
from mtah2027.decoding import expected_spellf1 as ef  # noqa: E402

INIT = np.array([-0.6931, -0.6931, -3.0])
NUMBA = [False, True] if dv.numba_available() else [False]


def _problem(n, seed, strength=1.0):
    rng = np.random.default_rng(seed)
    lp = np.log(rng.dirichlet(np.ones(3) * 0.5 / strength, size=n))
    tr = np.log(rng.dirichlet(np.ones(3) * 2.0, size=(n, 3)))
    return np.round(lp, 6), np.round(tr, 6)


def _path_logp(path, lp, tr, init):
    s = init[path[0]] + lp[0, path[0]]
    for t in range(1, len(path)):
        s += tr[t, path[t - 1], path[t]] + lp[t, path[t]]
    return s


@pytest.mark.parametrize("use_numba", NUMBA)
def test_sampler_matches_brute_force_posterior(use_numba):
    n = 5
    lp, tr = _problem(n, 3)
    paths = list(itertools.product(range(3), repeat=n))
    logp = np.array([_path_logp(p, lp, tr, INIT) for p in paths])
    post = np.exp(logp - logp.max())
    post /= post.sum()
    k = 40000
    smp = ef.sample_posterior_paths(lp, tr, INIT, k, seed=7, use_numba=use_numba)
    assert smp.shape == (k, n) and smp.dtype == np.int8
    index = {p: i for i, p in enumerate(paths)}
    emp = np.zeros(len(paths))
    for row in smp:
        emp[index[tuple(int(v) for v in row)]] += 1
    emp /= k
    assert np.abs(emp - post).max() < 0.012


def test_temperature_flattens_the_posterior():
    lp, tr = _problem(40, 5)
    cold = ef.sample_posterior_paths(lp, tr, INIT, 300, 1, temperature=0.5, use_numba=False)
    hot = ef.sample_posterior_paths(lp, tr, INIT, 300, 1, temperature=4.0, use_numba=False)
    distinct = lambda a: len({tuple(r) for r in a.tolist()})  # noqa: E731
    assert distinct(hot) > distinct(cold)


@pytest.mark.skipif(not dv.numba_available(), reason="numba not installed")
def test_numba_and_python_sampling_agree():
    lp, tr = _problem(3000, 11)
    a = ef.sample_posterior_paths(lp, tr, INIT, 8, 3, use_numba=True)
    b = ef.sample_posterior_paths(lp, tr, INIT, 8, 3, use_numba=False)
    assert np.array_equal(a, b)


def test_sampling_is_deterministic_and_seed_dependent():
    lp, tr = _problem(500, 2)
    a = ef.sample_posterior_paths(lp, tr, INIT, 5, 0)
    b = ef.sample_posterior_paths(lp, tr, INIT, 5, 0)
    c = ef.sample_posterior_paths(lp, tr, INIT, 5, 1)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def _random_paths(n, count, seed, min_run=3):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(count):
        lab, cur = [], int(rng.integers(0, 3))
        while len(lab) < n:
            lab.extend([cur] * int(rng.integers(min_run, 60)))
            cur = int(rng.choice(3, p=[0.4, 0.4, 0.2]))
        out.append(np.array(lab[:n], dtype=np.int8))
    return out


@pytest.mark.parametrize("use_numba", NUMBA)
def test_fast_scorer_equals_official(use_numba):
    paths = _random_paths(1500, 14, 4)
    paths += [np.full(1500, 2, np.int8), np.zeros(1500, np.int8)]
    runs = [(0, 1500)]
    u = ef.utility_matrix(paths, use_numba=use_numba)
    for a, gt in enumerate(paths):
        for b, pred in enumerate(paths):
            if a == b:
                continue
            ref = pm.spell_scores(gt, pred, runs)["mean_f1"]
            assert u[a, b] == pytest.approx(ref, abs=1e-12)
            fast = ef.pair_spell_f1(ef.spells_of(gt), ef.spells_of(pred), use_numba=use_numba)
            assert fast == pytest.approx(ref, abs=1e-12)


def test_select_mbr_prefers_the_central_candidate():
    base = np.array([0] * 200 + [1] * 200 + [0] * 200, dtype=np.int8)
    noisy = base.copy()
    noisy[100:110] = 1                       # splits the first possession
    shifted = np.roll(base, 3)
    odd = np.array([1] * 600, dtype=np.int8)
    assert ef.select_mbr([odd, noisy, base, shifted, base.copy()], use_numba=False) == 2
    assert ef.select_mbr([base], use_numba=False) == 0


def test_block_decoder_is_deterministic_and_keeps_layout():
    n = 400
    lp, tr = _problem(n, 9, strength=3.0)
    params = ef.ExpectedSpellF1Params(enabled=True, k=20, seed=0)
    a = ef.decode_match_halves_mbr(lp, tr, params=params, play_blocks=[(50, 149), (250, 349)],
                                   dead_blocks=[(200, 249)], log_init=INIT)
    b = ef.decode_match_halves_mbr(lp, tr, params=params, play_blocks=[(50, 149), (250, 349)],
                                   dead_blocks=[(200, 249)], log_init=INIT)
    assert np.array_equal(a, b)
    assert (a[:50] == 3).all() and (a[150:200] == 3).all() and (a[350:] == 3).all()
    assert (a[200:250] == 2).all()
    assert set(np.unique(a[50:150])) <= {0, 1, 2}


def test_k_zero_is_plain_viterbi():
    lp, tr = _problem(300, 1)
    got = ef.decode_block_mbr(lp, tr, INIT, ef.ExpectedSpellF1Params(enabled=True, k=0))
    assert np.array_equal(got, dv.viterbi_dynamic(lp, tr, INIT))


def test_default_config_is_off():
    p = ef.ExpectedSpellF1Params.from_config()
    assert not p.enabled and p.k == 100 and p.seed == 0
