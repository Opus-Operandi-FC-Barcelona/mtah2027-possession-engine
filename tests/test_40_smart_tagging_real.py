"""Issue #40: defending phases credit the opponent in the phase prior; aggregate Smart Tagging check."""
from __future__ import annotations

import json

import numpy as np
import pytest
import smart_tagging_check as stc

from mtah2027.config import MatchConfig
from mtah2027.io import smart_tagging_loader as st

XML = """<?xml version="1.0"?>
<file><ALL_INSTANCES>
<instance><ID>1</ID><start>10.00</start><end>20.00</end><code>PROGRESSION</code>
 <label><group>Team</group><text>Barcelona</text></label>
 <label><group>Half</group><text>1st Half</text></label></instance>
<instance><ID>2</ID><start>10.00</start><end>20.00</end><code>DEFENDING IN MIDDLE THIRD</code>
 <label><group>Team</group><text>Atletic Club</text></label>
 <label><group>Half</group><text>1st Half</text></label></instance>
<instance><ID>3</ID><start>30.00</start><end>34.00</end><code>Defensive_Transition</code>
 <label><group>Team</group><text>Barcelona</text></label></instance>
<instance><ID>4</ID><start>40.00</start><end>44.00</end><code>DEFENDING IN DEFENSIVE THIRD</code>
 <label><group>Team</group><text>N/A</text></label></instance>
</ALL_INSTANCES></file>
"""


@pytest.fixture()
def path(tmp_path):
    p = tmp_path / "st.xml"
    p.write_text(XML, encoding="utf-8")
    return p


def test_defending_phase_credits_the_opponent(path):
    s = st.load_smart_tagging(path=path, n_frames=1500)
    # paired phases (attack H + defence A) reinforce each other instead of cancelling
    assert s.phase_count_home[300] == 2 and s.phase_count_away[300] == 0
    assert s.phase_prior_home[300] == 1.0 and s.phase_team[300] == st.HOME
    # defensive transition of H -> A possesses; N/A stays unknown
    assert s.phase_count_away[800] == 1 and s.phase_prior_home[800] == -1.0
    assert s.phase_count_home[1050] == 0 and s.phase_count_away[1050] == 0 and s.phase_prior_home[1050] == 0.0
    assert s.n_flipped == 2


def test_flip_can_be_disabled_and_events_keep_the_tagged_team(path):
    s = st.load_smart_tagging(path=path, n_frames=1500, defending_codes=())
    assert s.n_flipped == 0 and s.phase_count_home[300] == 1 and s.phase_count_away[300] == 1
    ev = {e.code: e for e in st.parse_events(path)}
    assert ev["defending in middle third"].team == st.AWAY            # the raw tag is never rewritten
    assert st.possession_side(ev["defending in middle third"]) == st.HOME
    assert st.possession_side(ev["defensive transition"]) == st.AWAY
    assert st.possession_side(ev["defending in defensive third"]) == st.NO_TEAM


def _blocks(spec) -> MatchConfig:
    return MatchConfig(n_video_frames=spec.n_frames, half1=spec.half1, half2_dead=spec.half2_dead,
                       half2_play=spec.half2_play)


def test_aggregate_report_on_the_synthetic_match(synthetic_dir, synthetic_spec):
    rep = stc.build_report(synthetic_dir, _blocks(synthetic_spec))
    a = rep["anchors"]
    assert a["n_anchors"] == rep["set_pieces"] > 0 and 0 <= a["resolved_pct"] <= 100
    assert "throw_ins" in a and a["throw_ins"]["total"] >= 0
    assert rep["emissions"]["phase_prior_in_front_end"] is True
    assert rep["phase"]["agreement_by_code"]
    text = json.dumps(rep)
    assert text.count("t_star") < 40 and len(text) < 20000               # aggregates, no event list


def test_phase_report_agreement_unit():
    ev = [st.TagEvent("defending in middle third", 0, 10, 0.0, 0.4, st.AWAY, 1, "", {}),
          st.TagEvent("progression", 10, 20, 0.4, 0.8, st.HOME, 1, "", {})]
    smart = st.build_smart_tagging(ev, n_frames=20)
    labels = np.zeros(20, np.int8)                                         # home always in possession
    rep = stc.phase_report(smart, labels, MatchConfig(n_video_frames=20, half1=(0, 9), half2_dead=(10, 9),
                                                      half2_play=(10, 19)))
    by = rep["agreement_by_code"]
    assert by["defending in middle third"]["pred_equals_possession_side_pct"] == 100.0
    assert by["defending in middle third"]["pred_equals_raw_tag_team_pct"] == 0.0
    assert by["progression"]["pred_equals_possession_side_pct"] == 100.0


# ------------------------------------------------------------------ collective anchor gated on dead evidence
def _collective_case(lag: float):
    import dataclasses
    from types import SimpleNamespace

    from mtah2027.features.setpiece_anchor import AnchorParams, anchor_set_pieces

    n, s = 900, 12
    speed = np.full((n, s), 0.8, np.float32)
    speed[200:] = 3.5                 # large jump with no dead evidence before it (a counter-attack)
    speed[300:500] = 0.3              # dead stretch
    speed[500:] = 1.6                 # smaller jump right after it: the real restart
    xy = np.zeros((n, s, 2), np.float32)
    xy[..., 0] = np.linspace(10, 90, s)[None, :]
    dead = np.zeros(n)
    dead[300:500] = 0.9
    sp = [SimpleNamespace(type="free kick", team=-1, start_frame=100, end_frame=800, half=1)]
    p = dataclasses.replace(AnchorParams(), collective_dead_lag_s=lag)
    res = anchor_set_pieces(np.full((n, 2), np.nan, np.float32), xy, np.arange(s) % 2, sp, (),
                            players_speed=speed, dead_score=dead, params=p)
    return res.anchors[0]


def test_collective_anchor_is_gated_on_dead_evidence():
    ungated, gated = _collective_case(0.0), _collective_case(1.0)
    assert ungated.method == "collective" and ungated.t_star < 260          # the counter-attack jump
    assert gated.method == "collective" and abs(gated.t_star - 500) <= 12   # the restart after the dead run


def test_default_config_enables_the_gate():
    from mtah2027.features.setpiece_anchor import AnchorParams

    assert AnchorParams.from_config().collective_dead_lag_s == 1.0


def test_anchor_accuracy_summary_unit():
    import eval_anchor_accuracy as eaa

    gt = np.array([0, 0, 2, 2, 2, 2, 2, 2, 1, 1, 2, 0, 0], np.int8)
    assert eaa.true_restarts(gt, fps=1.0, min_dead_s=5.0) == [8]
    rows = [("ball", "ball", -2, 1.0), ("ball", "ball", 40, 1.0), ("collective", "unresolved", None, 0.0)]
    out = eaa.summarise(rows)
    assert out["ball->ball"]["within_3f_pct"] == 50.0 and out["collective->unresolved"]["resolved"] == 0
