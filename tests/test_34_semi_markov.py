"""Issue #34 -- explicit-duration (semi-Markov) decoder and the duration table."""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from mtah2027.decoding.duration_model import (
    DurationModel,
    DurationParams,
    fit_duration_model,
    run_lengths,
)
from mtah2027.decoding.dynamic_viterbi import decode_match_halves, numba_available
from mtah2027.decoding.semi_markov import viterbi_semi_markov


def _model(fps=10.0):
    rng = np.random.default_rng(3)
    seq = np.concatenate([np.full(int(rng.integers(2, 12)), rng.integers(0, 3)) for _ in range(300)])
    params = DurationParams(edges_s=(0.0, 0.2, 0.4, 0.8), alpha=0.5)
    return fit_duration_model([(seq, [(0, len(seq) - 1)], fps)], params), params


def _score(path, lp, tr, init, tables, w):
    ld, ls, lq, a_tail, ls_tail, cap = tables
    n = len(path)
    change = np.flatnonzero(path[1:] != path[:-1]) + 1
    starts = [0, *change.tolist()]
    ends = [*change.tolist(), n]
    total = init[path[0]]
    for k, (u, e) in enumerate(zip(starts, ends)):
        s, d = int(path[u]), e - u
        total += lp[u:e, s].sum()
        if u > 0:
            total += tr[u, path[u - 1], s]
        first, last = u == 0, e == n
        if first and last:
            continue
        if first or last:
            total += w * (ls[s, d] if d <= cap else ls_tail[s] + (d - cap - 1) * lq[s])
        else:
            total += w * (ld[s, d] if d <= cap else a_tail[s] + d * lq[s])
    return total


def test_run_lengths_and_fit():
    codes = np.array([0, 0, 1, 1, 1, 2, 2, 0, 0, 0, 0, 1])
    assert run_lengths(codes, [(0, 11)]) == [[4], [3], [2]]       # border runs dropped
    m, p = _model()
    assert sum(m.counts) > 100
    for s in range(3):
        assert abs(sum(m.mass[s]) + m.tail[s] - 1.0) < 1e-6
    assert np.allclose(DurationModel.from_dict(m.to_dict()).mass, m.mass, atol=1e-7)
    ld, ls, lq, a_tail, ls_tail, cap = m.tables(10.0, p)
    assert cap == 8 and ld.shape == (3, 9) and np.all(np.diff(ls[:, 1:], axis=1) <= 1e-9)


@pytest.mark.parametrize("w", [0.0, 0.7, 1.0])
def test_decoder_is_exact_against_brute_force(w):
    m, p = _model()
    tables = m.tables(10.0, p)
    rng = np.random.default_rng(int(10 * w) + 1)
    n = 9                                  # longer than cap (8) so the geometric tail is exercised
    lp = np.round(np.log(rng.dirichlet([1, 1, 1], n)), 6)
    tr = np.round(rng.normal(-2.0, 1.5, size=(n, 3, 3)), 6)
    init = np.log([0.4, 0.4, 0.2])
    best = -np.inf
    for states in itertools.product(range(3), repeat=n):
        best = max(best, _score(np.array(states), lp, tr, init, tables, w))
    got = viterbi_semi_markov(lp, tr, init, tables, w, use_numba=False)
    assert _score(got, lp, tr, init, tables, w) == pytest.approx(best, abs=1e-9)


def test_numba_and_numpy_kernels_agree_and_blocks():
    m, p = _model(25.0)
    tables = m.tables(25.0, p)
    rng = np.random.default_rng(0)
    n = 4000
    lp = np.log(rng.dirichlet([1, 1, 1], n))
    tr = np.zeros((n, 3, 3))
    tr[:, 0, 1] = tr[:, 1, 0] = -5.0
    tr[:, 2, :2] = -3.0
    tr[:, :2, 2] = -4.0
    init = np.log([0.4, 0.4, 0.2])
    a = viterbi_semi_markov(lp, tr, init, tables, 1.0, use_numba=False)
    if numba_available():
        assert (a == viterbi_semi_markov(lp, tr, init, tables, 1.0, use_numba=True)).all()
    out = decode_match_halves(lp, tr, play_blocks=[(100, 1999), (2500, 3999)], dead_blocks=[(2000, 2499)],
                              log_init=init, duration=(tables, 1.0))
    assert (out[:100] == 3).all() and (out[2000:2500] == 2).all()
    assert set(np.unique(out[100:2000])) <= {0, 1, 2}
    assert (out[2500:] == viterbi_semi_markov(lp[2500:], tr[2500:], init, tables, 1.0, use_numba=False)).all()


def test_long_segments_use_the_tail():
    m, p = _model()
    tables = m.tables(10.0, p)
    n = 60
    lp = np.zeros((n, 3))
    lp[:, 0] = 0.5                          # state H preferred everywhere: one run, longer than the cap
    tr = np.full((n, 3, 3), -3.0)
    out = viterbi_semi_markov(lp, tr, np.log([0.4, 0.4, 0.2]), tables, 1.0, use_numba=False)
    assert (out == 0).all()


def test_pipeline_switch_uses_the_duration_decoder(synthetic_kin, synthetic_spec, tmp_path):
    import yaml

    from mtah2027 import pipeline
    from mtah2027.config import DEFAULT_CONFIG_PATH

    m, _ = _model(25.0)
    cfg = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    td, spec = synthetic_kin["td"], synthetic_spec
    kw = dict(play_blocks=[spec.half1, spec.half2_play], dead_blocks=[spec.half2_dead], log=lambda m_: None)
    off = pipeline.run_core(td.players_xy, td.team_of_slot, td.ball_xy, durations=m, **kw)
    assert "duration" not in off.diagnostics["stages"]
    cfg["duration"].update({"enabled": True, "apply_to_heuristic": True})
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    on = pipeline.run_core(td.players_xy, td.team_of_slot, td.ball_xy, durations=m, config_path=path, **kw)
    assert on.codes.shape == off.codes.shape
    assert set(np.unique(on.codes[spec.half1[0]:spec.half1[1] + 1])) <= {0, 1, 2}
    cfg["duration"]["apply_to_heuristic"] = False             # the heuristic route is not selected
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    skipped = pipeline.run_core(td.players_xy, td.team_of_slot, td.ball_xy, durations=m, config_path=path,
                                **kw)
    assert (skipped.codes == off.codes).all()
    assert skipped.diagnostics["stages"]["duration"]["status"].startswith("skipped")


def test_committed_specialists_carry_the_duration_table():
    from mtah2027.models.specialist_fusion import DEFAULT_MODEL_PATH, load_specialists

    spec = load_specialists(DEFAULT_MODEL_PATH)
    assert spec.durations is not None and min(spec.durations.counts) > 100
    ld, ls, lq, a_tail, ls_tail, cap = spec.durations.tables(25.0)
    assert cap == 1500 and np.isfinite(ld[:, 1:]).all()
