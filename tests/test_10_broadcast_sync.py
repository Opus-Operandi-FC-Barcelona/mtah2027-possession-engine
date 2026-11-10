"""Issue #10: broadcast block synchronisation and ball injection (synthetic data only)."""
# ruff: noqa: E501  (long docstring / comment lines)
from __future__ import annotations

import numpy as np
import pytest
import yaml

from mtah2027.config import ConfigError, params_from_mapping
from mtah2027.preprocessing import broadcast_sync as bs

FPS = 25.0
PITCH = (105.3, 68.0)


def _make_match(n=5000, seed=0):
    """22 players + ball: Ornstein-Uhlenbeck velocities whose energy follows slowly changing tempo phases."""
    rng = np.random.default_rng(seed)
    s = 22
    team = np.repeat([0, 1], 11).astype(np.int8)
    energy = np.zeros(n)
    t = 0
    while t < n:
        length = int(rng.uniform(3, 12) * FPS)
        energy[t:t + length] = rng.uniform(0.3, 4.0)
        t += length
    drive = {"x": rng.normal(size=(n, s)), "y": rng.normal(size=(n, s))}
    pull = rng.uniform(0.5, 1.5, s)
    vel = np.zeros((s, 2))
    pos = np.stack([rng.uniform(5, 100, s), rng.uniform(5, 63, s)], axis=1)
    out = np.zeros((n, s, 2))
    low, high = np.array([1.0, 1.0]), np.array([PITCH[0] - 1, PITCH[1] - 1])
    for i in range(n):
        vel += (-vel / 40.0 + energy[i] * pull[:, None] * 0.35
                * np.stack([drive["x"][i], drive["y"][i]], axis=1))
        q = pos + vel / FPS
        hit = (q < low) | (q > high)
        vel = np.where(hit, -vel, vel)
        pos = np.clip(q, low, high)
        out[i] = pos
    ball = out[:, 0, :] + 0.5
    return out.astype(np.float32), team, ball.astype(np.float32)


@pytest.fixture(scope="module")
def match():
    return _make_match()


@pytest.fixture(scope="module")
def sim():
    import simulate_broadcast_value as sv

    return sv


@pytest.fixture(scope="module")
def feed(match, sim):
    players, team, ball = match
    return sim.fake_broadcast(players, team, ball, [(0, len(ball) - 1)], fps=FPS, shot_s=(10.0, 30.0), seed=3)


def _hide(ball, a, b):
    out = ball.copy()
    out[a:b] = np.nan
    return out


# --------------------------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------------------------
def test_default_off_and_config_section():
    assert bs.BroadcastSyncParams().enabled is False
    p = bs.BroadcastSyncParams.from_config()
    assert p.enabled is False
    assert p.min_corr == 0.65 and p.min_block_frames == 75 and p.max_reproj_error == 5.0
    with pytest.raises(ConfigError):
        params_from_mapping(bs.BroadcastSyncParams, {"min_cor": 0.5})
    assert bs.SRC_BROADCAST == 4


# --------------------------------------------------------------------------------------------
# blocks
# --------------------------------------------------------------------------------------------
def test_split_blocks_min_length_and_cap():
    usable = np.zeros(2000, bool)
    usable[100:160] = True          # 60 frames: too short
    usable[300:400] = True          # 100 frames: kept
    usable[500:1500] = True         # 1000 frames: split (cap 400)
    blocks = bs.split_blocks(usable, bs.BroadcastSyncParams(max_block_s=16.0), FPS)
    assert (300, 400) in blocks
    assert all(b - a >= 75 for a, b in blocks)
    pieces = [(a, b) for a, b in blocks if a >= 500]
    assert pieces[0][0] == 500 and pieces[-1][1] == 1500
    assert all(pieces[i][1] == pieces[i + 1][0] for i in range(len(pieces) - 1))
    assert max(b - a for a, b in pieces) <= 500
    assert bs.split_blocks(usable, bs.BroadcastSyncParams(max_block_s=0), FPS)[-1] == (500, 1500)


def test_fingerprint_is_crop_invariant(match):
    players, team, _ = match
    full = bs.fingerprint(players, team, FPS)
    cropped = players.copy()
    cropped[np.abs(cropped[..., 0] - 52.0) > 20.0] = np.nan      # a camera window of 40 m
    part = bs.fingerprint(cropped, team, FPS)
    ok = np.isfinite(full).all(axis=1) & np.isfinite(part).all(axis=1)
    assert ok.mean() > 0.8
    assert np.corrcoef(full[ok, 0], part[ok, 0])[0, 1] > 0.6
    few = players.copy()
    few[:, 3:] = np.nan                                           # only 3 players visible
    assert np.isnan(bs.fingerprint(few, team, FPS)).all()


# --------------------------------------------------------------------------------------------
# synchronisation
# --------------------------------------------------------------------------------------------
def test_recovers_offsets_and_injects_only_into_holes(match, feed, sim):
    players, team, ball = match
    hidden = _hide(ball, 800, 3200)
    res = bs.sync_broadcast(players, team, hidden, feed["players_xy"], feed["team_of_slot"], feed["ball_xy"],
                            feed["usable"], fps=FPS, pitch=PITCH, graded=[(0, len(ball) - 1)])
    acc = sim.offset_accuracy(res.blocks, feed["offset"])
    assert acc["accepted"] >= 5
    assert acc["within_tol"] == 1.0
    # tracking is never overwritten, only NaN frames are filled and flagged 4
    seen = np.isfinite(hidden).all(axis=1)
    assert np.array_equal(res.ball_xy[seen], hidden[seen])
    assert (res.ball_source[seen] == 1).all()
    filled = res.ball_source == bs.SRC_BROADCAST
    assert filled.any() and not (filled & seen).any()
    assert not (filled & ~np.isfinite(res.ball_xy).all(axis=1)).any()
    assert filled[:800].sum() == 0 and filled[3200:].sum() == 0 and filled[800:3200].sum() > 100
    assert np.linalg.norm(res.ball_xy[filled] - ball[filled], axis=1).mean() < 1.5   # noise 0.5 m, offsets exact
    st = res.stats
    assert st["blocks_accepted"] == acc["accepted"] and st["frames_filled"] == int(filled.sum())
    assert 0.0 < st["fill_rate"] <= 1.0


def test_local_pass_propagates_to_short_shots(match, sim):
    players, team, ball = match
    short = sim.fake_broadcast(players, team, ball, [(0, len(ball) - 1)], fps=FPS, shot_s=(3.0, 8.0),
                               replay_prob=0.0, seed=4)
    args = (players, team, _hide(ball, 800, 3200), short["players_xy"], short["team_of_slot"], short["ball_xy"],
            short["usable"])
    off = bs.sync_broadcast(*args, fps=FPS, pitch=PITCH, params=bs.BroadcastSyncParams(local_rounds=0))
    on = bs.sync_broadcast(*args, fps=FPS, pitch=PITCH, params=bs.BroadcastSyncParams(local_rounds=3))
    assert on.stats["blocks_accepted"] > off.stats["blocks_accepted"]
    assert sim.offset_accuracy(on.blocks, short["offset"])["within_tol"] >= 0.95


def test_unrelated_broadcast_is_rejected(match, sim):
    players, team, ball = match
    other = _make_match(5000, seed=11)
    feed = sim.fake_broadcast(other[0], other[1], other[2], [(0, 4999)], fps=FPS, seed=5)
    res = bs.sync_broadcast(players, team, _hide(ball, 800, 3200), feed["players_xy"], feed["team_of_slot"],
                            feed["ball_xy"], feed["usable"], fps=FPS, pitch=PITCH)
    # the position / ball tests must stop the accidental fingerprint matches of another match
    wrong = [b for b in res.blocks if b.accepted]
    assert len(wrong) <= 0.15 * max(len(res.blocks), 1)


def test_invalid_homography_frames_are_never_used(match, feed):
    players, team, ball = match
    none = np.zeros_like(feed["usable"])
    res = bs.sync_broadcast(players, team, _hide(ball, 800, 3200), feed["players_xy"], feed["team_of_slot"],
                            feed["ball_xy"], none, fps=FPS, pitch=PITCH)
    assert res.blocks == [] and res.stats["blocks_accepted"] == 0 and res.stats["frames_filled"] == 0
    assert np.isnan(res.ball_xy[800:3200]).all()


def test_ball_disagreement_and_injection_rules():
    t_ball = np.full((300, 2), np.nan, dtype=np.float32)
    b_ball = np.zeros((300, 2), dtype=np.float32)
    b_ball[:, 0] = 10.0
    t_ball[100:200] = b_ball[0:100] + 8.0                         # offset 100, 8 m apart
    assert bs.ball_disagreement(b_ball, t_ball, 0, 100, 100, 25) == pytest.approx(8.0 * np.sqrt(2.0))
    assert np.isnan(bs.ball_disagreement(b_ball, t_ball, 0, 20, 100, 25))     # too little overlap
    blk = bs.BlockSync(0, 100, offset=150, corr=0.9, accepted=True)
    out, src = bs.inject_broadcast_ball(t_ball, b_ball, [blk], PITCH, 1.0)
    assert (src[150:200] == 1).all()                              # tracking kept
    assert (src[200:250] == bs.SRC_BROADCAST).all() and np.allclose(out[200:250, 0], 10.0)
    assert (src[:100] == 0).all()
    rej = bs.BlockSync(0, 100, offset=0, corr=0.9, accepted=False)
    _, src2 = bs.inject_broadcast_ball(t_ball, b_ball, [rej], PITCH, 1.0)
    assert (src2 == bs.SRC_BROADCAST).sum() == 0
    off = bs.BlockSync(0, 100, offset=0, corr=0.9, accepted=True)
    far = b_ball.copy()
    far[:, 0] = 300.0                                             # outside the pitch
    _, src3 = bs.inject_broadcast_ball(t_ball, far, [off], PITCH, 1.0)
    assert (src3 == bs.SRC_BROADCAST).sum() == 0


def test_higher_correlation_block_wins_a_contested_frame():
    t_ball = np.full((100, 2), np.nan, dtype=np.float32)
    b_ball = np.zeros((100, 2), dtype=np.float32)
    b_ball[:50, 0], b_ball[50:, 0] = 20.0, 40.0
    lo = bs.BlockSync(0, 50, offset=0, corr=0.7, accepted=True)
    hi = bs.BlockSync(50, 100, offset=-50, corr=0.9, accepted=True)
    out, _ = bs.inject_broadcast_ball(t_ball, b_ball, [lo, hi], PITCH, 1.0)
    assert np.allclose(out[:50, 0], 40.0)


def test_deterministic(match, feed):
    players, team, ball = match
    args = (players, team, _hide(ball, 800, 3200), feed["players_xy"], feed["team_of_slot"], feed["ball_xy"],
            feed["usable"])
    a = bs.sync_broadcast(*args, fps=FPS, pitch=PITCH)
    b = bs.sync_broadcast(*args, fps=FPS, pitch=PITCH)
    assert np.array_equal(a.ball_source, b.ball_source)
    assert [(x.start, x.offset, x.accepted) for x in a.blocks] == [(x.start, x.offset, x.accepted) for x in b.blocks]


# --------------------------------------------------------------------------------------------
# synthetic bundle (files) and pipeline wiring
# --------------------------------------------------------------------------------------------
def test_files_present_and_sync_on_synthetic_bundle(synthetic_dir, synthetic_spec, tmp_path):
    from mtah2027.io.atd_loader import load_atd

    assert bs.broadcast_files_present(synthetic_dir)
    assert not bs.broadcast_files_present(tmp_path)
    td = load_atd(synthetic_dir, cache_dir=None, use_cache=False, n_frames=synthetic_spec.n_frames)
    graded = [tuple(synthetic_spec.half1), tuple(synthetic_spec.half2)]
    res = bs.sync_broadcast_files(td, synthetic_dir, graded)
    assert res.stats["blocks"] >= 5 and res.stats["blocks_accepted"] >= 3
    acc = [b for b in res.blocks if b.accepted]
    assert all(b.cost < 1.0 for b in acc)                         # refined to the 0.12 m position noise
    # the generator hides the tactical ball in gaps: the TV ball fills some, tracking is never overwritten
    seen = np.isfinite(td.ball_xy).all(axis=1)
    assert np.array_equal(res.ball_xy[seen], td.ball_xy[seen])
    n_inj = int((res.ball_source == bs.SRC_BROADCAST).sum())
    assert n_inj > 0 and res.stats["frames_filled"] <= n_inj


def _config_with(tmp_path, enabled: bool):
    from mtah2027.config import DEFAULT_CONFIG_PATH

    cfg = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    cfg["broadcast_sync"]["enabled"] = enabled
    path = tmp_path / f"cfg_{enabled}.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def test_pipeline_wiring_off_on_and_missing(synthetic_dir, synthetic_spec, tmp_path):
    from mtah2027.config import MatchConfig
    from mtah2027.pipeline import run_pipeline

    match_cfg = MatchConfig(n_video_frames=synthetic_spec.n_frames, half1=synthetic_spec.half1,
                            half2_dead=synthetic_spec.half2_dead, half2_play=synthetic_spec.half2)
    quiet = lambda m: None   # noqa: E731
    off = run_pipeline(synthetic_dir, _config_with(tmp_path, False), match_cfg, log=quiet)
    assert off.diagnostics["broadcast"] is None
    assert "broadcast_sync" in off.diagnostics["loader_stages"]
    on = run_pipeline(synthetic_dir, _config_with(tmp_path, True), match_cfg, log=quiet)
    st = on.diagnostics["broadcast"]
    assert st["status"] == "ok" and st["blocks_accepted"] >= 3
    assert on.diagnostics["route"] == "heuristic" and st["frames_filled"] > 0
    # broadcast files absent: the stage is skipped, the route still runs
    import shutil

    bare = tmp_path / "bare"
    shutil.copytree(synthetic_dir / "tactical", bare / "tactical")
    miss = run_pipeline(bare, _config_with(tmp_path, True), match_cfg, log=quiet)
    assert miss.diagnostics["broadcast"] is None
    assert "broadcast files not found" in miss.diagnostics["loader_stages"]["broadcast_sync"]["status"]
    assert np.array_equal(miss.labels, off.labels)
