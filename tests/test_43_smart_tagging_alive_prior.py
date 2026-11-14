# ruff: noqa: E501
"""Issue #43: alive-ball prior and transition cue from Smart Tagging, and the Smart Tagging simulator."""
from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np

from mtah2027.decoding.transition_matrix import TransitionParams, build_log_transitions
from mtah2027.io import smart_tagging_loader as stl
from mtah2027.io.smart_tagging_sim import NOISE_LEVELS, simulate_smart_tagging
from mtah2027.models.heuristic_emissions import EmissionParams, compute_emissions
from mtah2027.models.specialist_fusion import apply_priors
from mtah2027.pipeline import live_phase_mask

FPS = 25.0


def _event(code, a, b, team, typ=""):
    return stl.TagEvent(code=code, start_frame=a, end_frame=b, start_s=a / 25, end_s=b / 25, team=team, half=1,
                        type=typ, tags={})


def _labels(n=6000):
    """H run, D, A run, D, H run (alternating) inside one period."""
    codes = np.full(n, 3, dtype=np.int8)
    spans = [(100, 900, 0), (900, 1100, 2), (1100, 2300, 1), (2300, 2600, 2), (2600, 3800, 0), (3800, 4000, 2),
             (4000, 5200, 1)]
    for a, b, c in spans:
        codes[a:b] = c
    return codes, [(100, 5199)]


# ---- loader ------------------------------------------------------------------------------------------
def test_loader_exposes_phase_cover_and_transitions():
    events = [_event("progression", 100, 300, 0), _event("defending in middle third", 100, 300, 1),
              _event("defensive transition", 500, 700, 0), _event("attacking transition", 500, 700, 1),
              _event("ball in the box", 900, 950, -1), _event(stl.SET_PIECE_CODE, 1000, 1100, 0, "corner")]
    s = stl.build_smart_tagging(events, 2000)
    assert s.phase_cover[100] and s.phase_cover[299] and not s.phase_cover[300] and s.phase_cover[920]
    assert not s.phase_cover[1050]                     # set-piece windows are not phases
    assert s.transitions == [(500, 1), (500, 1)]       # the defender's tag is flipped to the possessing side


# ---- the prior on the emissions -----------------------------------------------------------------------
def test_emissions_alive_prior_off_by_default_and_hidden_only():
    n = 40
    ds = np.full(n, 0.6)
    ev, valid = np.zeros(n), np.zeros(n, bool)
    vis = np.arange(n) < 20
    alive = np.ones(n, bool)
    base = compute_emissions(ds, ev, valid, vis, phase_alive=alive, params=EmissionParams())
    plain = compute_emissions(ds, ev, valid, vis, params=EmissionParams())
    assert np.array_equal(base.log_probs, plain.log_probs)           # w_phase_alive = 0 -> identical
    on = compute_emissions(ds, ev, valid, vis, phase_alive=alive, params=EmissionParams(w_phase_alive=1.5))
    assert np.array_equal(on.p_dead[:20], plain.p_dead[:20])         # visible frames untouched
    assert (on.p_dead[20:] < plain.p_dead[20:] - 0.05).all()         # hidden + live phase -> less D
    off_cover = compute_emissions(ds, ev, valid, vis, phase_alive=np.zeros(n, bool),
                                  params=EmissionParams(w_phase_alive=1.5))
    assert np.array_equal(off_cover.log_probs, plain.log_probs)      # no live phase -> identical


def test_specialist_prior_lowers_d_logit_only_on_hidden_live_frames():
    n = 30
    prob = np.tile([0.2, 0.2, 0.6], (n, 1))
    vis = np.arange(n) < 10
    alive = np.arange(n) % 2 == 0
    ep = EmissionParams(w_phase_alive=2.0)
    out, _ = apply_priors(prob, phase=None, owner=None, status_dead=None, ep=ep, phase_alive=alive, visible=vis)
    hit = alive & ~vis
    assert (out[hit, 2] < 0.6 - 0.1).all() and np.allclose(out[~hit, 2], 0.6)
    same, _ = apply_priors(prob, phase=None, owner=None, status_dead=None, ep=EmissionParams(), phase_alive=alive,
                           visible=vis)
    assert np.array_equal(same, prob)


# ---- live mask ----------------------------------------------------------------------------------------
def test_live_phase_mask_excludes_pre_restart_and_post_goal():
    n = 1000
    cover = np.ones(n, bool)
    anchors = SimpleNamespace(
        forced_dead_mask=np.zeros(n, bool),
        forced_dead_intervals=[],
        anchors=[SimpleNamespace(tag_start=100, tag_end=300, t_star=250, resolved=True),
                 SimpleNamespace(tag_start=400, tag_end=500, t_star=-1, resolved=False)])
    anchors.forced_dead_mask[600:650] = True
    st = SimpleNamespace(phase_cover=cover, goals=[stl.Goal(0, 800, 810, 1)])
    live = live_phase_mask(st, anchors, n, goal_tail_s=2.0)
    assert not live[100:250].any() and live[250:400].all()              # pre-t* excluded, post-t* live
    assert not live[400:500].any()                                      # unresolved: whole window excluded
    assert not live[600:650].any() and live[650:800].all()
    assert not live[800:860].any() and live[860:].all()                 # goal + 2 s tail (no linked kick-off)
    none = live_phase_mask(st, None, n, goal_tail_s=2.0)
    assert none[100:250].all()


# ---- transition cue -----------------------------------------------------------------------------------
def test_transition_cue_opens_towards_the_gaining_team_inside_occlusions_only():
    n = 600
    visible = np.ones(n, bool)
    visible[100:400] = False                                             # one 12 s occlusion
    off = build_log_transitions(n, ball_visible=visible, transitions=[(250, 1), (20, 0)], params=TransitionParams())
    on_p = dataclasses.replace(TransitionParams(), phase_cue=True)
    on = build_log_transitions(n, ball_visible=visible, transitions=[(250, 1), (20, 0)], params=on_p)
    assert np.array_equal(off, build_log_transitions(n, ball_visible=visible, transitions=None, params=on_p))
    assert on[250, 0, 1] == on_p.phase_cue_open and on[250, 1, 0] == off[250, 1, 0]   # H->A (away gains) opens
    assert on[20, 1, 0] == off[20, 1, 0]                                 # visible ball: not hidden-only opened
    wide = dataclasses.replace(on_p, phase_cue_hidden_only=False)
    assert build_log_transitions(n, ball_visible=visible, transitions=[(20, 0)], params=wide)[20, 1, 0] \
        == on_p.phase_cue_open


# ---- simulator ----------------------------------------------------------------------------------------
def test_simulator_is_deterministic_and_structured():
    codes, periods = _labels()
    a = simulate_smart_tagging(codes, periods, FPS, noise="clean", seed=1, n_frames=len(codes))
    b = simulate_smart_tagging(codes, periods, FPS, noise="clean", seed=1, n_frames=len(codes))
    assert [dataclasses.astuple(e)[:6] for e in a.events] == [dataclasses.astuple(e)[:6] for e in b.events]
    assert a.codes.get("progression", 0) + a.codes.get("build up", 0) + a.codes.get("creating chances", 0) > 0
    # the paired defending phase is tagged with the opponent: the flipped prior points at the possessing side
    h_run = a.phase_prior_home[300:800]
    assert (h_run > 0).all()
    a_run = a.phase_prior_home[1200:2200]
    assert (a_run < 0).all()
    assert not a.phase_cover[:100].any() and not a.phase_cover[5400:].any()
    assert any(s.type == "kick off (start)" for s in a.set_pieces)
    c = simulate_smart_tagging(codes, periods, FPS, noise="clean", seed=2, n_frames=len(codes))
    assert len(c.events) > 0


def test_simulator_noise_levels_are_ordered_and_reach_the_target_coverage():
    rng = np.random.default_rng(0)
    n = 60000
    codes = np.full(n, 3, dtype=np.int8)
    pos = 500
    lab = 0
    while pos < n - 1000:
        dur = int(rng.uniform(100, 700))
        codes[pos:pos + dur] = lab
        pos += dur
        lab = 1 - lab
        dead = int(rng.uniform(100, 500))
        codes[pos:pos + dead] = 2
        pos += dead
    periods = [(500, pos - 1)]
    graded = codes != 3
    cover = {}
    for level in ("clean", "low", "medium", "high"):
        s = simulate_smart_tagging(codes, periods, FPS, noise=level, seed=0, n_frames=n)
        cover[level] = float(s.phase_cover[graded].mean())
    assert cover["clean"] < 0.69
    assert abs(cover["low"] - 0.69) < 0.02 and cover["medium"] >= 0.68   # target is a floor (spurious tags add)
    assert cover["high"] >= 0.68
    assert NOISE_LEVELS["low"].jitter_s < NOISE_LEVELS["medium"].jitter_s < NOISE_LEVELS["high"].jitter_s
    s = simulate_smart_tagging(codes, periods, FPS, noise="medium", seed=0, n_frames=n)
    dead_cov = float(s.phase_cover[graded & (codes == 2)].mean())
    assert 0.15 < dead_cov < 0.7                                         # realistic dead-time contamination
