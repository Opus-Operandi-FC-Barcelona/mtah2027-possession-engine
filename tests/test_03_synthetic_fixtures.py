import hashlib
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import games
from starter import load_atd, load_homography, load_smart_tagging

FPS = 25
LINE_RE = re.compile(r"^\d+:(?:[^;:]+;)+[^;:]+:[^;:]+$")
GROUP_RE = re.compile(r"^(?:-?\d+\.\d{5}|NaN),(?:-?\d+\.\d{5}|NaN),(?:\d+\.\d{2}|NaN)$")


def pitch_mask(players):
    """Slots whose normalised position lies inside the pitch."""
    x, y = players[..., 0], players[..., 1]
    return (x >= 0) & (x <= 1) & (y >= 0) & (y <= 1)


def sha_tree(root: Path):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def spells(labels):
    """[(label, start, end)] of H/A runs (a D or X breaks a spell)."""
    out, start = [], None
    for i, lab in enumerate(list(labels) + ["X"]):
        if start is not None and (lab != labels[start]):
            out.append((labels[start], start, i - 1))
            start = None
        if start is None and lab in ("H", "A"):
            start = i
    return out


# ---------------------------------------------------------------- layout and formats
def test_layout_matches_games_asset_paths(synthetic_dir):
    tactical = games.asset_paths(data_dir=str(synthetic_dir))
    broadcast = games.asset_paths(data_dir=str(synthetic_dir), source="broadcast")
    for name in ("atd_meta", "atd_raw", "homography", "smart_tagging"):
        assert Path(tactical[name]).is_file(), name
    for name in ("atd_meta", "atd_raw", "homography"):
        assert Path(broadcast[name]).is_file(), name
    assert (synthetic_dir / "synthetic_gt.csv").is_file()
    assert games.require(["atd_raw", "atd_meta", "homography", "smart_tagging"], data_dir=str(synthetic_dir))


def test_atd_txt_line_format_and_rows(synthetic_dir, synthetic_spec):
    path = Path(games.asset_paths(data_dir=str(synthetic_dir))["atd_raw"])
    with open(path, encoding="ascii") as handle:
        lines = handle.read().split("\n")
    assert lines[-1] == ""
    lines = lines[:-1]
    assert len(lines) == synthetic_spec.n_frames
    for line in lines[:50] + lines[-50:] + lines[::97]:
        assert LINE_RE.match(line), line[:80]
        frame, slots, ball = line.split(":")
        groups = slots.split(";")
        assert len(groups) == 46
        for group in groups + [ball]:
            assert GROUP_RE.match(group), group
    assert lines[0].startswith("1:")
    assert "NaN,NaN,NaN" in lines[0]


def test_frame_ids_contiguous_and_arrays(tactical_atd, broadcast_atd, synthetic_spec):
    ids, players, ball = tactical_atd
    assert np.array_equal(ids, np.arange(1, synthetic_spec.n_frames + 1))
    assert players.shape == (synthetic_spec.n_frames, 46, 3)
    ids_b, players_b, _ = broadcast_atd
    assert np.array_equal(ids_b, np.arange(1, synthetic_spec.n_broadcast + 1))
    assert players_b.shape[1] == 51
    xy = players[..., :2]
    assert np.nanmin(xy) >= -0.1 and np.nanmax(xy) <= 1.1
    assert np.nanmin(players[..., 2]) >= 0.0


def test_ball_missing_on_about_41_percent_of_alive_frames(tactical_atd, synthetic_gt):
    _, _, ball = tactical_atd
    alive = np.isin(synthetic_gt, ["H", "A"])
    missing = np.isnan(ball[alive, 0]).mean()
    assert abs(missing - 0.41) < 0.03
    # contiguous blocks: long runs, not isolated frames
    hidden = np.isnan(ball[:, 0]) & alive
    edges = np.flatnonzero(np.diff(np.concatenate(([0], hidden.astype(int), [0]))))
    runs = edges[1::2] - edges[0::2]
    assert runs.max() >= 5 * FPS
    assert np.median(runs) >= 3 * FPS
    assert runs.max() <= 61 * FPS + 400   # blocks merge only across adjacent alive stretches


def test_atd_xml_structure(synthetic_dir, synthetic_spec):
    root = ET.parse(games.asset_paths(data_dir=str(synthetic_dir))["atd_meta"]).getroot()
    teams = root.findall("./Metadata/Teams/Team")
    assert [t.findtext("Name") for t in teams] == ["Barcelona", "Atletic Club"]
    players = root.findall("./Metadata/Players/Player")
    assert len(players) == 46
    assert {p.get("teamId") for p in players} == {t.get("id") for t in teams}
    assert [p.get("teamId") for p in players[:23]] == [teams[0].get("id")] * 23
    assert players[0].findtext("Name") == "Track 1" and players[0].findtext("ShirtNumber") == "1"
    assert int(root.findtext("./Metadata/GlobalConfig/FrameRate")) == FPS
    channels = {c.get("id"): c.findtext("Unit") for c in root.iter("Channel")}
    assert channels == {"x": "normalized", "y": "normalized", "speed": "m/s"}
    spec = root.find("./DataFormatSpecifications/DataFormatSpecification")
    assert int(spec.get("startFrame")) == 1 and int(spec.get("endFrame")) == synthetic_spec.n_frames
    assert len(spec.findall(".//PlayerChannelRef")) == 46 * 3
    assert len(spec.findall(".//BallChannelRef")) == 3
    assert len(root.findall("./Metadata/PlayerChannels/PlayerChannel")) == 46 * 3
    broadcast = ET.parse(games.asset_paths(data_dir=str(synthetic_dir), source="broadcast")["atd_meta"])
    assert len(broadcast.getroot().findall("./Metadata/Players/Player")) == 51


# ---------------------------------------------------------------- starter loaders
def test_starter_smart_tagging_parses(synthetic_dir, synthetic_spec):
    events = load_smart_tagging.load(data_dir=str(synthetic_dir))
    assert len(events) > 20
    codes = {e["code"] for e in events}
    assert {"SET PIECES", "GOALS"} <= codes
    assert len(codes) >= 4  # phases besides set pieces and goals
    set_pieces = [e for e in events if e["code"] == "SET PIECES"]
    types = {e["tags"]["Type"] for e in set_pieces}
    assert {"Throw-in", "Kick Off (Start)"} <= types
    assert types & {"Corner Kick", "Goal Kick", "Free Kick"}
    throw_ins = [e for e in set_pieces if e["tags"]["Type"] == "Throw-in"]
    assert throw_ins and all(e["tags"]["Team"] == "N/A" for e in throw_ins)
    teams = {e["tags"]["Team"] for e in events}
    assert teams <= {"Barcelona", "Atletic Club", "N/A"}
    for e in events:
        assert 0 <= e["start_frame"] <= e["end_frame"] <= synthetic_spec.n_frames + 400
        assert e["tags"]["Half"] in ("1st Half", "2nd Half")
    kick_offs = [e for e in set_pieces if e["tags"]["Type"] == "Kick Off (Start)"]
    assert sorted(e["tags"]["Half"] for e in kick_offs) == ["1st Half", "2nd Half"]
    starts = {"1st Half": synthetic_spec.half1[0], "2nd Half": synthetic_spec.half2_play[0]}
    for e in kick_offs:
        assert e["start_frame"] <= starts[e["tags"]["Half"]] <= e["end_frame"]


def test_starter_homography_parses(synthetic_dir, synthetic_spec):
    matrices, error, valid = load_homography.load(data_dir=str(synthetic_dir))
    assert matrices.shape == (synthetic_spec.n_frames, 3, 3)
    assert valid.all() and np.isfinite(matrices).all()
    corners = np.array(load_homography.pitch_corners(matrices[100]))
    assert corners[:, 0].min() > -50 and corners[:, 0].max() < 1970
    assert corners[:, 1].min() > 0 and corners[:, 1].max() < 1080
    centre = load_homography.project(matrices[100], 0.5, 0.5)
    assert 0 < centre[0] < 1920 and 0 < centre[1] < 1080
    assert np.median(error[synthetic_spec.half1[0]:synthetic_spec.half1[1]]) < 0.6
    report = load_homography.check_alignment(data_dir=str(synthetic_dir))
    assert report["contiguous"] and report["n_rows"] == synthetic_spec.n_frames

    mb, eb, vb = load_homography.load(data_dir=str(synthetic_dir), source="broadcast")
    assert mb.shape == (synthetic_spec.n_broadcast, 3, 3)
    assert 0.6 < vb.mean() < 0.95
    assert np.isnan(mb[~vb]).all() and np.isfinite(mb[vb]).all()


def test_homography_csv_header(synthetic_dir):
    path = games.asset_paths(data_dir=str(synthetic_dir))["homography"]
    header = open(path, encoding="utf-8").readline().strip()
    assert header == "frame,h00,h01,h02,h10,h11,h12,h20,h21,h22,reprojection_error,valid"


def test_alignment_checks_short_match(synthetic_dir, synthetic_spec):
    report = load_atd.check_alignment_raw(data_dir=str(synthetic_dir))
    assert report["n_records"] == synthetic_spec.n_frames
    assert report["contiguous"] and report["frame_id_range"] == (1, synthetic_spec.n_frames)
    assert report["first_video_frame"] == 0
    assert report["agrees"] is False  # the official length is 199,319; this is the short match


# ---------------------------------------------------------------- ground truth and play
def test_ground_truth_blocks_and_labels(synthetic_gt, synthetic_spec):
    s = synthetic_spec
    assert len(synthetic_gt) == s.n_frames
    assert set(np.unique(synthetic_gt)) == {"H", "A", "D", "X"}
    assert (synthetic_gt[:s.half1[0]] == "X").all()
    assert (synthetic_gt[s.half1[1] + 1:s.half2_dead[0]] == "X").all()
    assert (synthetic_gt[s.half2_dead[0]:s.half2_dead[1] + 1] == "D").all()
    assert (synthetic_gt[s.half2_play[1] + 1:] == "X").all()
    assert (synthetic_gt[s.half1[0]:s.half1[1] + 1] != "X").all()
    assert (synthetic_gt[s.half2_play[0]:s.half2_play[1] + 1] != "X").all()
    assert synthetic_gt[s.half1[0]] in ("H", "A")           # kick-off starts live play
    assert synthetic_gt[s.half2_play[0]] in ("H", "A")


def test_ground_truth_csv_contract(synthetic_dir, synthetic_spec):
    table = pd.read_csv(synthetic_dir / "synthetic_gt.csv")
    assert list(table.columns) == ["frame", "label"]
    assert np.array_equal(table["frame"].to_numpy(), np.arange(synthetic_spec.n_frames))


def test_possession_durations_and_kpis(synthetic_gt, synthetic_spec):
    s = synthetic_spec
    sp = spells(list(synthetic_gt))
    assert len(sp) >= 10
    edge = {s.half1[0], s.half2_play[0], s.half1[1], s.half2_play[1]}
    for lab, a, b in sp:
        dur = (b - a + 1) / FPS
        if a in edge or b in edge or b + 1 in edge:
            continue
        # interior spells are 5-30 s long; adjacent same-team runs may merge only across a dead ball
        assert 4.9 <= dur <= 30.1, (lab, a, b, dur)
    # alternation: consecutive spells without a dead ball between them are different teams
    for (l1, _, b1), (l2, a2, _) in zip(sp, sp[1:]):
        if a2 == b1 + 1:
            assert l1 != l2
    graded = synthetic_gt != "X"
    dead_pct = 100 * (synthetic_gt[graded] == "D").mean()
    home_pct = 100 * (synthetic_gt == "H").sum() / np.isin(synthetic_gt, ["H", "A"]).sum()
    assert 10 <= dead_pct <= 45
    assert 35 <= home_pct <= 75


def test_dead_ball_slow_players_and_stationary_ball(tactical_atd, synthetic_gt, synthetic_spec):
    _, players, ball = tactical_atd
    s = synthetic_spec
    dead = (synthetic_gt == "D") & (np.arange(len(synthetic_gt)) > s.half2_dead[1])  # restarts after play
    alive = np.isin(synthetic_gt, ["H", "A"])
    assert dead.sum() > 100
    speed_dead = np.nanmean(players[dead][..., 2])
    speed_alive = np.nanmean(players[alive][..., 2])
    assert speed_dead < 0.8 * speed_alive
    ball_dead = ball[dead & ~np.isnan(ball[:, 0]), 2]
    ball_alive = ball[alive & ~np.isnan(ball[:, 0]), 2]
    assert np.nanmedian(ball_dead) < 0.5 and np.nanmedian(ball_alive) > 1.0
    outside = ~pitch_mask(ball[:, None, :])[:, 0] & ~np.isnan(ball[:, 0])
    assert outside[synthetic_gt == "D"].any()      # the ball leaves the pitch in some dead stretches
    assert outside[alive].mean() < 0.02          # only the first frames of a throw-in or corner


def test_possessing_team_is_nearest_to_ball(tactical_atd, synthetic_gt):
    _, players, ball = tactical_atd
    x, y = players[..., 0] * 105.3, players[..., 1] * 68.0
    on_pitch = pitch_mask(players)
    d = np.hypot(x - (ball[:, 0] * 105.3)[:, None], y - (ball[:, 1] * 68.0)[:, None])
    d[~on_pitch] = np.inf
    near_home, near_away = d[:, :23].min(1), d[:, 23:].min(1)
    for lab, closer in (("H", near_home < near_away), ("A", near_away < near_home)):
        m = (synthetic_gt == lab) & ~np.isnan(ball[:, 0])
        assert closer[m].mean() > 0.8


def test_about_22_players_on_pitch_and_artifacts(tactical_atd, synthetic_gt, synthetic_info):
    _, players, _ = tactical_atd
    on_pitch = pitch_mask(players)
    alive = np.isin(synthetic_gt, ["H", "A"])
    assert 21.5 <= on_pitch[alive].sum(1).mean() <= 23.5
    spec = synthetic_info["spec"]
    assert on_pitch[spec.half1[1] + 400:spec.half2_dead[0] - 400].sum() == 0
    off = (players[..., 1] < -0.02) | (players[..., 1] > 1.02)
    assert off[alive].any()                       # sideline staff / substitutes
    art = synthetic_info["track_artifacts"]
    assert art["duplicates"] > 0 and art["id_breaks"] + art["id_swaps"] > 0
    # duplicate tracks: two same-team slots closer than 0.6 m in the same frame
    found = False
    for lo, hi in ((0, 23), (23, 46)):
        xs = players[:, lo:hi, 0] * 105.3
        ys = players[:, lo:hi, 1] * 68.0
        for t in np.flatnonzero(alive)[::5]:
            ok = np.flatnonzero(on_pitch[t, lo:hi])
            if len(ok) < 2:
                continue
            dd = np.hypot(xs[t, ok, None] - xs[t, None, ok], ys[t, ok, None] - ys[t, None, ok])
            np.fill_diagonal(dd, np.inf)
            if dd.min() < 0.6:
                found = True
                break
    assert found


def test_broadcast_offset_cropped_with_cuts(broadcast_atd, tactical_atd, synthetic_dir, synthetic_spec):
    _, players_b, _ = broadcast_atd
    _, players_t, _ = tactical_atd
    _, _, valid = load_homography.load(data_dir=str(synthetic_dir), source="broadcast")
    n_visible = (~np.isnan(players_b[..., 0])).sum(1)
    assert (n_visible[~valid] == 0).all()                 # cuts carry no tracking
    assert n_visible[valid].mean() < 0.8 * (~np.isnan(players_t[..., 0])).sum(1).max()  # cropped view
    assert synthetic_spec.n_broadcast != synthetic_spec.n_frames
    assert 0.05 < (~valid).mean() < 0.4
    flips = np.flatnonzero(np.diff(valid.astype(int)) != 0)
    assert len(flips) >= 4                                # several cuts


# ---------------------------------------------------------------- determinism and sizes
def test_generation_is_byte_identical(synth_module, tmp_path, synthetic_dir):
    synth_module.generate_synthetic_match(tmp_path / "again", seed=0, n_frames=12000)
    assert sha_tree(tmp_path / "again") == sha_tree(synthetic_dir)


def test_different_seed_differs(synth_module, tmp_path, synthetic_dir):
    synth_module.generate_synthetic_match(tmp_path / "other", seed=1, n_frames=12000)
    a = sha_tree(tmp_path / "other")["synthetic_gt.csv"]
    b = sha_tree(synthetic_dir)["synthetic_gt.csv"]
    assert a != b


def test_configurable_size(synth_module, tmp_path):
    info = synth_module.generate_synthetic_match(tmp_path / "small", seed=3, n_frames=6000)
    assert info["spec"].n_frames == 6000
    lines = sum(1 for _ in open(info["paths"]["tactical_atd_txt"], encoding="ascii"))
    assert lines == 6000
    with pytest.raises(ValueError):
        synth_module.make_spec(100)


def test_cli_runs(tmp_path):
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, str(root / "scripts" / "generate_synthetic_fixtures.py"),
         "--out", str(tmp_path / "cli"), "--n-frames", "3000", "--seed", "2"],
        capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert (tmp_path / "cli" / "tactical" / "BAR-ATH_tactical_atd.txt").is_file()


def test_full_length_mode_matches_official_layout(synth_module, tmp_path_factory):
    out = tmp_path_factory.mktemp("synthetic_full")
    info = synth_module.generate_synthetic_match(out, seed=0, full_length=True)
    d = str(out)
    spec = info["spec"]
    assert spec.n_frames == 199319 and spec.n_broadcast == 164488
    report = load_atd.check_alignment_raw(data_dir=d)
    assert report["agrees"] and report["frame_id_range"] == (1, 199319)
    assert load_atd.check_alignment_raw(data_dir=d, source="broadcast")["agrees"]
    assert load_homography.check_alignment(data_dir=d)["agrees"]
    assert load_homography.check_alignment(data_dir=d, source="broadcast")["agrees"]
    events = load_smart_tagging.load(data_dir=d)
    assert load_smart_tagging.check_time_base(events)["agrees"]
    gt = info["labels"]
    blocks = games.half_video_blocks("BAR-ATH")
    assert (blocks[0][1], blocks[0][2]) == spec.half1 and blocks[1][2] == spec.half2_play[1]
    assert (gt[:11185] == "X").all() and (gt[90652:110808] == "X").all() and (gt[190198:] == "X").all()
    assert (gt[110808:113742] == "D").all()
    assert (gt[11185:90652] != "X").all() and (gt[113742:190198] != "X").all()
    sp = spells(list(gt))
    assert 150 <= len(sp) <= 600
    alive = np.isin(gt, ["H", "A"])
    assert 35 <= 100 * (gt == "H").sum() / alive.sum() <= 75
    assert abs(info["ball_hidden_fraction_alive"] - 0.41) < 0.01
