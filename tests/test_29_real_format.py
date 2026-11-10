"""Real-format hardening: Metrica's FIFA/EPTS export (open Sample Game 3 structure), tolerant side loaders.

``tests/fixtures/metrica_sample3_structure_*`` is the XML structure of Metrica's public Sample Game 3
(three of its eleven ``DataFormatSpecification`` blocks, the players they use, ``FieldSize`` and the period
provider parameters) plus 45 raw lines taken from the open data, renumbered to match the trimmed ranges.
"""
from __future__ import annotations

import importlib.util
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from mtah2027.io import atd_loader as al
from mtah2027.io import homography_loader as hl
from mtah2027.io import smart_tagging_loader as st
from mtah2027.pipeline import match_from_periods, run_pipeline_files

FIX = Path(__file__).parent / "fixtures"
META, RAW = FIX / "metrica_sample3_structure_metadata.xml", FIX / "metrica_sample3_structure_tracking.txt"


@pytest.fixture(scope="module")
def td():
    return al.load_atd_files(META, RAW)


def _reference(frame_id: int):
    """Independent re-parse of one raw line ({player id: (x, y)}, ball) via its specification."""
    root = ET.parse(META).getroot()
    pc = {e.get("id"): e.get("playerId") for e in root.iter("PlayerChannel")}
    spec = next(s for s in root.iter("DataFormatSpecification")
                if int(s.get("startFrame")) <= frame_id <= int(s.get("endFrame")))
    ids = [pc[r.get("playerChannelId")] for r in spec.iter("PlayerChannelRef")][::2]
    line = next(ln for ln in RAW.read_text().splitlines() if ln.startswith(f"{frame_id}:"))
    _, players, ball = line.split(":")
    xy = {pid: tuple(float(v) for v in grp.split(",")) for pid, grp in zip(ids, players.split(";"))}
    return xy, tuple(float(v) for v in ball.split(","))


def test_layout_has_one_segment_per_specification():
    lay = al.parse_atd_meta(META)
    assert [(s.start_frame, s.end_frame) for s in lay.segments] == [(1, 15), (16, 30), (31, 45)]
    assert lay.player_channels == ("x", "y") and lay.ball_channels == ("x", "y")
    assert lay.field_size == (105.0, 68.0)
    assert lay.periods == ((0, 29), (30, 44))
    assert lay.local_team_id == "FIFATMA" and lay.team_ids[0] == "FIFATMA"
    assert len(lay.player_ids) > 22            # substitutions: the union of the specifications
    assert int((lay.team_of_slot == 0).sum()) == int(sum(p.startswith("P") for p in lay.player_ids)) - int(
        (lay.team_of_slot == 1).sum())


def test_positions_follow_the_specification_of_each_frame(td):
    assert td.n_frames == 45 and td.report["n_specifications"] == 3 and td.report["contiguous"]
    assert td.pitch == (105.0, 68.0) and td.fps == 25.0 and td.periods == ((0, 29), (30, 44))
    for frame_id in (1, 15, 16, 30, 31, 45):
        xy, ball = _reference(frame_id)
        row = td.players_xy[frame_id - 1]
        for slot, pid in enumerate(td.player_ids):
            if pid in xy:
                np.testing.assert_allclose(row[slot], np.array(xy[pid]) * [105.0, 68.0], rtol=1e-5)
            else:
                assert np.isnan(row[slot]).all(), f"{pid} should be off the pitch at frame {frame_id}"
        if np.isnan(ball[0]):
            assert np.isnan(td.ball_xy[frame_id - 1]).all()
        else:
            np.testing.assert_allclose(td.ball_xy[frame_id - 1], np.array(ball) * [105.0, 68.0], rtol=1e-5)
    assert (np.isfinite(td.players_xy[..., 0]).sum(axis=1) == 22).all()


def test_no_speed_channel_and_team_assignment(td):
    assert np.isnan(td.players_speed).all() and "speed" not in td.ball_channels
    root = ET.parse(META).getroot()
    team = {p.get("id"): p.get("teamId") for p in root.iter("Player")}
    first = next(t.get("id") for t in root.iter("Team"))
    for slot, pid in enumerate(td.player_ids):
        assert td.team_of_slot[slot] == (0 if team[pid] == first else 1)


def test_cache_round_trip_keeps_periods(td, tmp_path):
    path = al.save_tracking(td, tmp_path / "t.npz")
    again = al.load_tracking_cache(path)
    assert again.periods == td.periods and again.player_ids == td.player_ids
    np.testing.assert_array_equal(again.players_xy, td.players_xy)


def test_matches_kloppy():
    pytest.importorskip("kloppy")
    from kloppy import metrica

    ds = metrica.load_tracking_epts(meta_data=str(META), raw_data=str(RAW), coordinates="metrica")
    ours = al.load_atd_files(META, RAW)
    assert len(ds.frames) == ours.n_frames and ds.metadata.frame_rate == ours.fps
    checked = 0
    for frame in ds.frames[::2]:
        row = ours.players_xy[frame.frame_id - 1]
        for player, data in frame.players_data.items():
            slot = ours.player_ids.index(player.player_id)
            if data.coordinates is None or np.isnan(data.coordinates.x):
                assert np.isnan(row[slot]).all()
                continue
            np.testing.assert_allclose(row[slot], [data.coordinates.x * 105.0, data.coordinates.y * 68.0],
                                       atol=1e-2)
            checked += 1
        ball = frame.ball_coordinates
        if np.isnan(ball.x):
            assert np.isnan(ours.ball_xy[frame.frame_id - 1]).all()
        else:
            np.testing.assert_allclose(ours.ball_xy[frame.frame_id - 1],
                                       [ball.x * 105.0, ball.y * 68.0], atol=1e-2)
    assert checked > 400


_RAGGED = """<?xml version="1.0"?>
<main><Metadata><GlobalConfig><FrameRate>25</FrameRate></GlobalConfig>
<Teams><Team id="A"><Name>A</Name></Team><Team id="B"><Name>B</Name></Team></Teams>
<Players><Player id="p1" teamId="A"/><Player id="p2" teamId="B"/><Player id="p3" teamId="B"/></Players>
<Devices><Device id="d"><Sensors><Sensor id="s"><Channels><Channel id="x"><Unit>m</Unit></Channel>
<Channel id="y"><Unit>m</Unit></Channel></Channels></Sensor></Sensors></Device></Devices>
<PlayerChannels>{channels}</PlayerChannels></Metadata>
<DataFormatSpecifications>
<DataFormatSpecification startFrame="1" endFrame="2" separator=":"><StringRegister name="frameCount"/>
<SplitRegister separator=";">{two}</SplitRegister><SplitRegister separator=","><BallChannelRef channelId="x"/>
<BallChannelRef channelId="y"/></SplitRegister></DataFormatSpecification>
<DataFormatSpecification startFrame="3" endFrame="4" separator=":"><StringRegister name="frameCount"/>
<SplitRegister separator=";">{three}</SplitRegister>
<SplitRegister separator=","><BallChannelRef channelId="x"/>
<BallChannelRef channelId="y"/></SplitRegister></DataFormatSpecification>
</DataFormatSpecifications></main>
"""


def _grp(pid):
    return (f'<SplitRegister separator=","><PlayerChannelRef playerChannelId="{pid}_x"/>'
            f'<PlayerChannelRef playerChannelId="{pid}_y"/></SplitRegister>')


def test_specifications_of_different_width_and_crlf(tmp_path):
    channels = "".join(f'<PlayerChannel id="{p}_{c}" playerId="{p}" channelId="{c}"/>'
                       for p in ("p1", "p2", "p3") for c in "xy")
    meta, raw = tmp_path / "m.xml", tmp_path / "r.txt"
    meta.write_text(_RAGGED.format(channels=channels, two=_grp("p1") + _grp("p2"),
                                   three=_grp("p3") + _grp("p1") + _grp("p2")), encoding="utf-8")
    raw.write_bytes(b"1:1,2;3,4:5,6\r\n2:1.5,2;3,4:NaN,NaN\r\n3:7,8;9,10;11,12:1,1\r\n4:NaN,NaN;9,10;11,12:2,2\r\n")
    out = al.load_atd_files(meta, raw)       # metres, FieldSize absent -> default pitch
    assert out.player_ids == ("p1", "p2", "p3") and out.n_frames == 4
    assert out.report["malformed_lines"] == 0 and out.report["contiguous"]
    np.testing.assert_allclose(out.players_xy[0, :, 0], [1, 3, np.nan], equal_nan=True)
    np.testing.assert_allclose(out.players_xy[2, :, 0], [9, 11, 7])
    assert np.isnan(out.players_xy[3, 2]).all() and out.players_xy[3, 1, 1] == 12
    np.testing.assert_allclose(out.ball_xy[:, 0], [5, np.nan, 1, 2], equal_nan=True)


def test_pipeline_runs_on_the_real_structure_without_optional_inputs():
    messages: list[str] = []
    res = run_pipeline_files(META, RAW, log=messages.append)
    assert res.diagnostics["route"] == "heuristic" and len(res.labels) == 45
    assert int((np.asarray(res.labels) == 3).sum()) == 0
    stages = res.diagnostics["core"]["stages"]
    assert stages["set_piece_anchors"]["status"].startswith("skipped")
    assert res.diagnostics["loader_stages"]["load_smart_tagging"]["status"].startswith("skipped")
    assert not [k for k, v in stages.items() if v["status"].startswith("failed")]
    match = match_from_periods(((0, 29), (30, 44)), 45)
    assert match.half2_play == (30, 44) and match.half2_dead == (30, 29)


def test_check_real_format_script(capsys):
    script = Path(__file__).parents[1] / "scripts" / "check_real_format.py"
    spec = importlib.util.spec_from_file_location("check_real_format", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main(["--meta", str(META), "--raw", str(RAW)]) == 0
    out = capsys.readouterr().out
    assert '"n_specifications": 3' in out and '"x_in_graded": 0' in out


# --------------------------------------------------------------------------------------------
# Smart Tagging / homography variants
# --------------------------------------------------------------------------------------------
VARIANT_XML = """<?xml version="1.0" encoding="utf-8"?>
<File>
  <Instance>
    <ID> 1 </ID><Start> 10,00 </Start><End>
       14.00
    </End><Code>  Set_Piece  </Code>
    <Label><Text> Barcelona </Text></Label>
    <Label><Text>1st   Half</Text></Label>
    <Label><Text>Throw-In</Text></Label>
  </Instance>
  <instance><start>20</start><end>25</end><code>GOAL</code>
    <label><group> Team </group><text>  atletic   club </text></label>
    <label><group>HALF</group><text>second half</text></label></instance>
  <instance><start>oops</start><end>3</end><code>Build Up</code></instance>
  <instance><start>30</start><end>34</end><code>Build   Up</code>
    <label><group>Team</group><text>N/A</text></label></instance>
</File>
"""


def test_smart_tagging_variants(tmp_path):
    path = tmp_path / "st.xml"
    path.write_text(VARIANT_XML, encoding="utf-8")
    events = st.parse_events(path)
    assert len(events) == 3                              # the instance without usable times is skipped
    first, goal, phase = events
    assert first.code == st.SET_PIECE_CODE and first.team == st.HOME and first.half == 1
    assert first.type == "throw-in" and first.start_frame == 250 and first.end_frame == 350
    assert goal.code == st.GOAL_CODE and goal.team == st.AWAY and goal.half == 2
    assert phase.code == "build up" and phase.team == st.NO_TEAM
    built = st.build_smart_tagging(events, 1000)
    assert len(built.set_pieces) == 1 and len(built.goals) == 1 and built.in_set_piece[300]


def test_homography_tolerates_crlf_extra_columns_and_bom(tmp_path):
    path = tmp_path / "h.csv"
    header = " Frame ,H00,h01,h02,h10,h11,h12,h20,h21,h22,Reprojection_Error,Valid,source,note\r\n"
    path.write_bytes(("﻿" + header +
                      "0,1,0,0,0,1,0,0,0,1,0.5,True,a,x\r\n"
                      "1,1,0,0,0,1,0,0,0,1,0.5,False,a,x\r\n"
                      "2,2,0,0,0,2,0,0,0,1,0.1,1,b,y\r\n").encode("utf-8"))
    hom = hl.parse_homography_csv(path)
    assert hom.n_frames == 3
    assert hom.valid.tolist() == [True, False, True] and np.isnan(hom.H[1]).all()
    assert hom.usable(1.0).tolist() == [True, False, True]
