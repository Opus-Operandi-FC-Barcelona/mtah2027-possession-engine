"""Issue #6: Smart Tagging loader."""
from __future__ import annotations

import numpy as np
import pytest

from mtah2027.io import smart_tagging_loader as st
from starter import load_smart_tagging as starter_st

XML = """<?xml version="1.0"?>
<file><ALL_INSTANCES>
<instance><ID>1</ID><start>10.00</start><end>14.00</end><code>Build Up</code>
 <label><group>TEAM</group><text>BARCELONA</text></label>
 <label><group>half</group><text>1st half</text></label>
 <label><group>Type</group><text>Build Up</text></label></instance>
<instance><ID>2</ID><start>12.00</start><end>20.00</end><code>progression</code>
 <label><group>Team</group><text>Atletic Club</text></label>
 <label><group>Half</group><text>1st Half</text></label></instance>
<instance><ID>3</ID><start>13.00</start><end>16.00</end><code>Final Third</code>
 <label><group>Team</group><text>atletic club</text></label></instance>
<instance><ID>4</ID><start>30.00</start><end>34.00</end><code>set pieces</code>
 <label><group>Team</group><text>N/A</text></label>
 <label><group>Half</group><text>2nd Half</text></label>
 <label><group>Type</group><text>THROW-IN</text></label></instance>
<instance><ID>5</ID><start>40.02</start><end>41.01</end><code>GOALS</code>
 <label><group>Team</group><text>Barcelona</text></label>
 <label><group>Half</group><text>2nd Half</text></label>
 <label><group>Type</group><text>Goal</text></label></instance>
</ALL_INSTANCES></file>
"""


@pytest.fixture()
def tiny(tmp_path):
    path = tmp_path / "st.xml"
    path.write_text(XML, encoding="utf-8")
    return st.load_smart_tagging(path=path, n_frames=1500)


def test_frames_halfopen_and_case_insensitive(tiny):
    ev = {e.code: e for e in tiny.events}
    assert set(ev) == {"build up", "progression", "final third", "set pieces", "goals"}
    assert (ev["build up"].start_frame, ev["build up"].end_frame) == (250, 350)
    assert ev["build up"].team == st.HOME and ev["build up"].half == 1 and ev["build up"].type == "build up"
    assert ev["progression"].team == st.AWAY and ev["final third"].team == st.AWAY
    assert ev["goals"].start_frame == round(40.02 * 25) == 1001 and ev["goals"].end_frame == 1025


def test_set_pieces_and_goals_lists(tiny):
    assert len(tiny.set_pieces) == 1 and len(tiny.goals) == 1
    sp = tiny.set_pieces[0]
    assert (sp.type, sp.team, sp.start_frame, sp.end_frame, sp.half) == ("throw-in", st.NO_TEAM, 750, 850, 2)
    assert tiny.set_pieces_of("Throw-In") == [sp]
    g = tiny.goals[0]
    assert (g.team, g.half) == (st.HOME, 2)
    assert tiny.in_set_piece[750] and tiny.in_set_piece[849] and not tiny.in_set_piece[850]
    assert tiny.in_set_piece.sum() == 100


def test_per_frame_phase_prior_excludes_set_pieces_and_goals(tiny):
    h, a = tiny.phase_count_home, tiny.phase_count_away
    assert h[249] == 0 and h[250] == 1 and h[349] == 1 and h[350] == 0       # half-open end
    assert a[300] == 1 and a[325] == 2 and a[399] == 2 and a[400] == 1 and a[499] == 1 and a[500] == 0
    assert h[760] == 0 and a[760] == 0 and h[1010] == 0                      # set piece / goal ignored
    assert tiny.phase_team[300] == st.NO_TEAM                                 # tie
    assert tiny.phase_team[330] == st.AWAY and tiny.phase_team[760] == st.NO_TEAM
    assert tiny.phase_prior_home[300] == 0.0 and tiny.phase_prior_home[100] == 0.0
    assert tiny.phase_prior_home[260] == 1.0 and tiny.phase_prior_home[450] == -1.0
    assert tiny.phase_prior_home[330] == pytest.approx(-1 / 3, abs=1e-6)
    assert tiny.phase_team.dtype == np.int8 and len(h) == 1500


def test_clipping_to_frame_range(tmp_path):
    path = tmp_path / "st.xml"
    path.write_text(XML, encoding="utf-8")
    short = st.load_smart_tagging(path=path, n_frames=300)
    assert short.n_frames == 300 and short.phase_count_home[299] == 1 and not short.in_set_piece.any()


def test_matches_starter_on_synthetic(synthetic_dir, synthetic_info):
    n = synthetic_info["spec"].n_frames
    mine = st.load_smart_tagging(synthetic_dir, n_frames=n)
    theirs = starter_st.load(data_dir=str(synthetic_dir))
    assert len(mine.events) == len(theirs)
    for a, b in zip(mine.events, theirs):
        assert (a.start_frame, a.end_frame) == (b["start_frame"], b["end_frame"])
        assert a.code == b["code"].strip().lower()
    assert len(mine.set_pieces) == synthetic_info["n_set_pieces"]
    assert sum(1 for s in mine.set_pieces if s.type == "throw-in") == synthetic_info["n_throw_ins"]
    assert all(s.team == st.NO_TEAM for s in mine.set_pieces if s.type == "throw-in")
    assert len(mine.goals) == synthetic_info["n_goals"]
    assert {e.team for e in mine.events} <= {st.HOME, st.AWAY, st.NO_TEAM}
    assert mine.phase_count_home.sum() > 0 and mine.phase_count_away.sum() > 0


def test_time_base_check_on_synthetic(synthetic_dir, synthetic_info):
    spec = synthetic_info["spec"]
    mine = st.load_smart_tagging(synthetic_dir, n_frames=spec.n_frames)
    report = st.check_time_base(mine, spec.half1[0], spec.half2_play[0])
    assert report["agrees"] is True
    assert len(report["halves"]) == 2


def test_missing_file_raises(tmp_path):
    with pytest.raises(IOError):
        st.load_smart_tagging(tmp_path / "nothing")
