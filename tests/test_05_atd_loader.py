"""Issue #5: streaming ATD parser."""
from __future__ import annotations

import os
import time

import numpy as np
import pytest

from mtah2027.config import N_VIDEO_FRAMES, PITCH
from mtah2027.io import atd_loader as al

SPEC_XML = """<?xml version="1.0" encoding="UTF-8"?>
<main><Metadata><GlobalConfig><FrameRate>25</FrameRate></GlobalConfig><Sessions/>
<Teams><Team id="T1"><Name>Home</Name></Team><Team id="T2"><Name>Away</Name></Team></Teams>
<Players><Player id="P1" teamId="T1"/><Player id="P2" teamId="T2"/><Player id="P3" teamId="T2"/></Players>
<Devices><Device id="d"><Sensors>
<Sensor id="p"><Channels><Channel id="x"><Unit>normalized</Unit></Channel>
<Channel id="y"><Unit>normalized</Unit></Channel></Channels></Sensor>
<Sensor id="s"><Channels><Channel id="speed"><Unit>m/s</Unit></Channel></Channels></Sensor>
</Sensors></Device></Devices>
<PlayerChannels>{pchannels}</PlayerChannels></Metadata>
<DataFormatSpecifications><DataFormatSpecification startFrame="1" endFrame="{end}" separator=":">
<StringRegister name="frameCount"/>
<SplitRegister separator=";">{players}</SplitRegister>
<SplitRegister separator=",">{ball}</SplitRegister>
</DataFormatSpecification></DataFormatSpecifications></main>
"""
ROW = "0.5,0.5,1.0;0.1,0.2,3.0;0.3,0.3,2.0"


def _spec_xml(ball_channels, end=100):
    ids = ["P1", "P2", "P3"]
    chans = ("x", "y", "speed")
    pch = "".join(f'<PlayerChannel id="{p}_{c}" playerId="{p}" channelId="{c}"/>' for p in ids for c in chans)
    players = "".join(
        '<SplitRegister separator=",">'
        + "".join(f'<PlayerChannelRef playerChannelId="{p}_{c}"/>' for c in chans) + "</SplitRegister>"
        for p in ids)
    ball = "".join(f'<BallChannelRef channelId="{c}"/>' for c in ball_channels)
    return SPEC_XML.format(pchannels=pch, end=end, players=players, ball=ball)


def _write(tmp_path, xml, lines):
    meta, raw = tmp_path / "m_atd.xml", tmp_path / "m_atd.txt"
    meta.write_text(xml, encoding="utf-8")
    raw.write_text("\n".join(lines) + "\n", encoding="ascii")
    return meta, raw


def test_matches_reference_parser_tactical(synthetic_info, tactical_atd):
    meta, raw = synthetic_info["paths"]["tactical_atd_xml"], synthetic_info["paths"]["tactical_atd_txt"]
    td = al.load_atd_files(meta, raw)
    ids, players, ball = tactical_atd
    assert td.n_frames == len(ids) == synthetic_info["spec"].n_frames
    assert td.players_xy.shape == (len(ids), 46, 2) and td.players_xy.dtype == np.float32
    np.testing.assert_allclose(td.players_xy[..., 0], players[..., 0] * PITCH[0], rtol=1e-6, equal_nan=True)
    np.testing.assert_allclose(td.players_xy[..., 1], players[..., 1] * PITCH[1], rtol=1e-6, equal_nan=True)
    np.testing.assert_allclose(td.players_speed, players[..., 2], rtol=1e-6, equal_nan=True)
    np.testing.assert_allclose(td.ball_xy[:, 0], ball[:, 0] * PITCH[0], rtol=1e-6, equal_nan=True)
    np.testing.assert_allclose(td.ball_xy[:, 1], ball[:, 1] * PITCH[1], rtol=1e-6, equal_nan=True)
    np.testing.assert_allclose(td.ball_speed, ball[:, 2], rtol=1e-6, equal_nan=True)
    assert set(td.ball_channels) == {"x", "y", "speed"}
    assert td.report["from_spec"] and td.report["contiguous"] and td.present.all()
    assert (td.frame_index == np.arange(td.n_frames)).all()
    assert td.team_names == ("Barcelona", "Atletic Club")


def test_team_of_slot_and_broadcast(synthetic_info, broadcast_atd):
    paths = synthetic_info["paths"]
    tac = al.load_atd_files(paths["tactical_atd_xml"], paths["tactical_atd_txt"])
    assert tac.team_of_slot.tolist() == [0] * 23 + [1] * 23
    bro = al.load_atd_files(paths["broadcast_atd_xml"], paths["broadcast_atd_txt"], source="broadcast")
    assert bro.n_slots == 51 and bro.team_of_slot.tolist() == [0] * 25 + [1] * 26
    assert bro.n_frames == len(broadcast_atd[0]) == synthetic_info["spec"].n_broadcast
    np.testing.assert_allclose(bro.players_xy[..., 0], broadcast_atd[1][..., 0] * PITCH[0], rtol=1e-6,
                               equal_nan=True)
    assert bro.source == "broadcast"


def test_alignment_and_missing_frames(tmp_path):
    lines = [
        "1:0.5,0.5,1.0;NaN,NaN,NaN;0.1,0.2,3.0:0.25,0.5,2.0",
        "2:0.5,0.5,1.0;NaN,NaN,NaN;0.1,0.2,3.0:NaN,NaN,NaN",
        "5:0.9,0.1,1.0;0.2,0.3,0.0;NaN,NaN,NaN:0.5,0.5,0.0",
    ]
    meta, raw = _write(tmp_path, _spec_xml(("x", "y", "speed"), end=6), lines)
    td = al.load_atd_files(meta, raw)
    assert td.n_frames == 6                       # endFrame declared in the XML
    assert td.present.tolist() == [True, True, False, False, True, False]
    assert np.isnan(td.players_xy[2]).all() and np.isnan(td.ball_xy[3]).all()
    np.testing.assert_allclose(td.players_xy[0, 0], [0.5 * 105.3, 0.5 * 68.0], rtol=1e-6)
    np.testing.assert_allclose(td.players_xy[4, 0], [0.9 * 105.3, 0.1 * 68.0], rtol=1e-6)   # id 5 -> row 4
    np.testing.assert_allclose(td.ball_xy[0], [0.25 * 105.3, 0.5 * 68.0], rtol=1e-6)
    assert not td.report["contiguous"]
    assert td.team_of_slot.tolist() == [0, 1, 1]
    assert al.load_atd_files(meta, raw, n_frames=10).n_frames == 10
    with pytest.raises(al.AtdFormatError):
        al.load_atd_files(meta, raw, n_frames=3)


def test_extra_ball_channels_numeric_and_text(tmp_path):
    chans = ("x", "y", "speed", "z", "ball_owning_team_id", "ball_state")
    lines = [
        f"1:{ROW}:0.5,0.5,5.0,1.5,T1,alive",
        f"2:{ROW}:0.5,0.5,5.0,NaN,T2,dead",
        f"3:{ROW}:NaN,NaN,NaN,NaN,NaN,NaN",
    ]
    meta, raw = _write(tmp_path, _spec_xml(chans, end=3), lines)
    td = al.load_atd_files(meta, raw)
    assert set(td.ball_channels) == set(chans)
    assert td.ball_channels["z"][0] == pytest.approx(1.5)
    assert np.isnan(td.ball_channels["z"][1:]).all()
    assert td.ball_channels["ball_owning_team_id"][:2].tolist() == [0.0, 1.0]
    assert td.ball_channels["ball_state"][:2].tolist() == [1.0, 0.0]
    assert np.isnan(td.ball_channels["ball_state"][2])
    assert td.ball_channel_codes["ball_state"] == {"alive": 1, "dead": 0}
    assert td.ball_channel_codes["ball_owning_team_id"] == {"T1": 0, "T2": 1}
    assert td.report["ball_extra_channels"] == ["z", "ball_owning_team_id", "ball_state"]
    assert td.report["malformed_lines"] == 0 and td.report["n_records"] == 3


def test_fallback_without_specification(tmp_path):
    xml = _spec_xml(("x", "y", "speed")).split("<DataFormatSpecifications>")[0] + "</main>"
    lines = [f"1:{ROW}:0.25,0.5,2.0", "2:NaN,NaN,NaN;NaN,NaN,NaN;NaN,NaN,NaN:NaN,NaN,NaN"]
    meta, raw = _write(tmp_path, xml, lines)
    td = al.load_atd_files(meta, raw)
    assert not td.report["from_spec"] and td.n_frames == 2 and td.n_slots == 3
    np.testing.assert_allclose(td.ball_xy[0], [0.25 * 105.3, 0.5 * 68.0], rtol=1e-6)


def test_malformed_lines_are_counted_and_skipped(tmp_path):
    lines = [f"1:{ROW}:0.25,0.5,2.0", "2:0.5,0.5,1.0;0.1,0.2", "", f"3:{ROW}:0.25,0.5,2.0"]
    meta, raw = _write(tmp_path, _spec_xml(("x", "y", "speed"), end=3), lines)
    td = al.load_atd_files(meta, raw)
    assert td.report["malformed_lines"] == 1 and td.present.tolist() == [True, False, True]


def test_small_blocks_give_identical_result(synthetic_info):
    paths = synthetic_info["paths"]
    a = al.load_atd_files(paths["tactical_atd_xml"], paths["tactical_atd_txt"])
    b = al.load_atd_files(paths["tactical_atd_xml"], paths["tactical_atd_txt"], block_bytes=50_000)
    np.testing.assert_array_equal(a.players_xy, b.players_xy)
    np.testing.assert_array_equal(a.ball_xy, b.ball_xy)


def test_load_atd_with_cache(synthetic_dir, tmp_path):
    cache = tmp_path / "cache"
    first = al.load_atd(synthetic_dir, cache_dir=cache)
    assert len(list(cache.glob("atd_tactical_*.npz"))) == 1
    second = al.load_atd(synthetic_dir, cache_dir=cache)
    np.testing.assert_array_equal(first.players_xy, second.players_xy)
    np.testing.assert_array_equal(first.ball_xy, second.ball_xy)
    assert second.team_of_slot.tolist() == first.team_of_slot.tolist()
    assert second.report == first.report and second.team_names == first.team_names
    assert set(second.ball_channels) == set(first.ball_channels)
    # touching the source changes the key -> a new cache entry
    raw = synthetic_dir / "tactical" / "BAR-ATH_tactical_atd.txt"
    st = raw.stat()
    os.utime(raw, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    try:
        al.load_atd(synthetic_dir, cache_dir=cache)
        assert len(list(cache.glob("atd_tactical_*.npz"))) == 2
    finally:
        os.utime(raw, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert al.load_atd(synthetic_dir, cache_dir=cache, use_cache=False).n_frames == first.n_frames
    assert len(list(cache.glob("atd_tactical_*.npz"))) == 2


def test_load_atd_broadcast_and_missing(synthetic_dir, tmp_path):
    bro = al.load_atd(synthetic_dir, source="broadcast", cache_dir=None)
    assert bro.n_slots == 51 and bro.source == "broadcast"
    with pytest.raises(IOError):
        al.load_atd(tmp_path / "nothing", cache_dir=None)


@pytest.mark.slow
def test_full_length_performance(synthetic_full_dir, tmp_path):
    t0 = time.perf_counter()
    td = al.load_atd(synthetic_full_dir, cache_dir=tmp_path / "c")
    parse_s = time.perf_counter() - t0
    print(f"\nfull-length tactical ATD parse: {parse_s:.2f} s")
    assert td.n_frames == N_VIDEO_FRAMES and td.n_slots == 46
    assert td.report["matches_expected"] and td.report["contiguous"]
    assert parse_s < 60.0
    t0 = time.perf_counter()
    again = al.load_atd(synthetic_full_dir, cache_dir=tmp_path / "c")
    assert time.perf_counter() - t0 < parse_s
    np.testing.assert_array_equal(td.ball_xy, again.ball_xy)
    bro = al.load_atd(synthetic_full_dir, source="broadcast", cache_dir=None)
    assert bro.n_frames == 164488 and bro.n_slots == 51
