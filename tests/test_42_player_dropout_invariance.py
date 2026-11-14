"""Issue #42: invariance to player dropout (strata, native speed channel, sample-size-robust shape)."""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

from mtah2027.features.collective_physics import compute_collective_physics
from mtah2027.io import atd_loader as al
from mtah2027.models import specialist_features as sfe
from mtah2027.preprocessing.kinematics_and_orientation import (
    compute_kinematics_and_orientation,
    fill_speed_from_native,
)
from mtah2027.validation import dropout_strata as ds

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_05_atd_loader import _write  # noqa: E402

H, A, D, X = 0, 1, 2, 3


# ----------------------------------------------- strata ------------------------------------------------
def test_tracked_counts_and_masks():
    p = np.full((4, 22, 2), np.nan, np.float32)
    p[0, :22] = 1.0
    p[1, :19] = 1.0
    p[2, :15] = 1.0
    n = ds.tracked_counts(p)
    assert n.tolist() == [22, 19, 15, 0]
    m = ds.stratum_masks(n)
    assert m["lt16"].tolist() == [False, False, True, True]
    assert m["lt20"].tolist() == [False, True, True, True]
    assert m["ge20"].tolist() == [True, False, False, False]


def _scenario():
    n = 600
    gt = np.full(n, X, np.int8)
    gt[50:550] = H
    gt[250:350] = A
    gt[350:400] = D
    pred = gt.copy()
    pred[250:300] = H             # misses the H->A turnover at 250 by 50 frames (> tol 25) -> missed
    pred[100:130] = D             # spurious dead
    tracked = np.full(n, 22, np.int16)
    tracked[200:420] = 14         # dropout episode: contains the turnover and the dead stretch
    visible = np.ones(n, bool)
    visible[200:420] = False
    return gt, pred, tracked, visible


def test_stratum_counts_known_numbers():
    gt, pred, tracked, visible = _scenario()
    c = ds.stratum_counts(gt, pred, tracked, visible, tol=25)
    lt16, ge20 = c["lt16"], c["ge20"]
    assert ge20["frames"] == 500 - 220
    assert lt16["frames"] == 220 and lt16["gt_d"] == 50 and lt16["pred_d"] == 50
    assert lt16["alive"] == 170 and lt16["alive_ok"] == 120        # 50 A frames predicted H
    assert lt16["turnovers"] == 1 and lt16["turnovers_missed"] == 1
    assert ge20["pred_d"] == 30 and ge20["gt_d"] == 0
    assert ge20["turnovers"] == 0                                   # the only H<->A turnover is inside
    assert c["lt20"]["frames"] == lt16["frames"]                    # nothing between 16 and 19 here
    s = ds.summarize(c)
    assert s["lt16"]["pred_d_pct_hidden"] == pytest.approx(100.0 * 50 / 220)
    assert 0.0 <= s["lt16"]["macro_f1"] <= 1.0
    assert s["ge20"]["ha_acc_pct"] == pytest.approx(100.0 * 250 / 280)     # the 30 spurious D frames
    rows = ds.format_table(s)
    assert rows[0].startswith("stratum") and len(rows) == 4


def test_pool_is_additive_and_macro_f1_is_exact():
    gt, pred, tracked, visible = _scenario()
    one = ds.stratum_counts(gt, pred, tracked, visible)
    two = ds.pool([one, one])
    assert two["lt20"]["frames"] == 2 * one["lt20"]["frames"]
    assert ds.summarize(two)["lt20"]["macro_f1"] == pytest.approx(ds.summarize(one)["lt20"]["macro_f1"])
    perfect = ds.stratum_counts(gt, gt, tracked, visible)
    assert ds.summarize(perfect)["lt20"]["macro_f1"] == pytest.approx(1.0)
    assert ds.summarize(perfect)["lt20"]["turnovers_missed"] == 0


def test_abstentions_in_graded_frames_are_filled():
    gt, _, tracked, visible = _scenario()
    pred = gt.copy()
    pred[60:80] = X
    c = ds.stratum_counts(gt, pred, tracked, visible)
    assert sum(c[k]["tp_h"] for k in ("lt20", "ge20")) > 0
    assert c["ge20"]["alive_ok"] == c["ge20"]["alive"]            # X filled forward with H, which is correct


# ---------------------------------------------------------- native speed channel ----------------------------
def test_speed_alias_rules():
    assert al.speed_alias(["x", "y", "s"], {"s": "km/h"}) == "s"
    assert al.speed_alias(["x", "y", "speed"], {"speed": "m/s"}) is None
    assert al.speed_alias(["x", "y", "v"], {"v": "km/h"}) == "v"              # unit-based
    assert al.speed_alias(["x", "y", "z"], {"z": "km/h"}) is None             # never a coordinate
    assert al.speed_alias(["x", "y", "s", "speed"], {}) is None               # explicit speed wins


def _s_xml(unit, end=3):
    xml = ("""<?xml version="1.0" encoding="UTF-8"?>
<main><Metadata><GlobalConfig><FrameRate>25</FrameRate></GlobalConfig><Sessions/>
<Teams><Team id="T1"><Name>Home</Name></Team><Team id="T2"><Name>Away</Name></Team></Teams>
<Players><Player id="P1" teamId="T1"/><Player id="P2" teamId="T2"/></Players>
<Devices><Device id="d"><Sensors>
<Sensor id="p"><Channels><Channel id="x"><Unit>normalized</Unit></Channel>
<Channel id="y"><Unit>normalized</Unit></Channel></Channels></Sensor>
<Sensor id="q"><Channels><Channel id="s"><Unit>UNIT</Unit></Channel></Channels></Sensor>
</Sensors></Device></Devices>
<PlayerChannels>""" + "".join(f'<PlayerChannel id="{p}_{c}" playerId="{p}" channelId="{c}"/>'
                              for p in ("P1", "P2") for c in ("x", "y", "s"))
            + """</PlayerChannels></Metadata>
<DataFormatSpecifications><DataFormatSpecification startFrame="1" endFrame="END" separator=":">
<StringRegister name="frameCount"/>
<SplitRegister separator=";">""" + "".join(
        '<SplitRegister separator=",">' + "".join(f'<PlayerChannelRef playerChannelId="{p}_{c}"/>'
                                                  for c in ("x", "y", "s")) + "</SplitRegister>"
        for p in ("P1", "P2")) + """</SplitRegister>
<SplitRegister separator=","><BallChannelRef channelId="x"/><BallChannelRef channelId="y"/>
<BallChannelRef channelId="s"/></SplitRegister>
</DataFormatSpecification></DataFormatSpecifications></main>""")
    return xml.replace("UNIT", unit).replace("END", str(end))


def test_loader_maps_s_kmh_to_speed_ms(tmp_path):
    lines = ["1:0.5,0.5,36.0;0.1,0.2,7.2:0.25,0.5,18.0",
             "2:0.5,0.5,NaN;0.1,0.2,3.6:0.25,0.5,NaN",
             "3:0.5,0.5,0.0;0.1,0.2,0.0:0.25,0.5,0.0"]
    meta, raw = _write(tmp_path, _s_xml("km/h"), lines)
    td = al.load_atd_files(meta, raw)
    assert "speed" in td.report["player_channels"] and "s" not in td.report["player_channels"]
    np.testing.assert_allclose(td.players_speed[0], [10.0, 2.0], rtol=1e-6)          # km/h -> m/s
    assert np.isnan(td.players_speed[1, 0]) and td.players_speed[1, 1] == pytest.approx(1.0)
    assert td.ball_speed[0] == pytest.approx(5.0)
    assert set(td.ball_channels) == {"x", "y", "speed"}


def test_loader_keeps_ms_unit_when_not_kmh(tmp_path):
    meta, raw = _write(tmp_path, _s_xml("m/s"), ["1:0.5,0.5,6.0;0.1,0.2,7.0:0.25,0.5,8.0"])
    td = al.load_atd_files(meta, raw)
    np.testing.assert_allclose(td.players_speed[0], [6.0, 7.0])


def test_fill_speed_from_native_uses_edges_and_missing_only():
    n = 40
    valid = np.zeros((n, 2), bool)
    valid[5:35, 0] = True
    valid[10:12, 1] = True                    # a 2-frame fragment
    sg = np.where(valid, 2.0, np.nan).astype(np.float32)
    sg[20, 0] = np.nan                        # a hole inside a valid position run
    native = np.full((n, 2), 5.0, np.float32)
    native[:, 1] = np.nan
    out, used = fill_speed_from_native(sg, valid, native, edge_frames=3)
    assert used[5:8, 0].all() and used[32:35, 0].all()        # run edges
    assert used[20, 0] and out[20, 0] == 5.0                   # SG missing, native available
    assert out[15, 0] == 2.0 and not used[15, 0]
    assert not used[:, 1].any() and np.allclose(out[10:12, 1], 2.0)   # no native value -> SG kept
    assert np.isnan(out[0, 0])                                 # no position -> stays NaN


def test_kinematics_uses_native_speed_only_when_given(synthetic_kin):
    td, spec = synthetic_kin["td"], synthetic_kin["spec"]
    xy = td.players_xy[:2000]
    base = compute_kinematics_and_orientation(xy, td.ball_xy[:2000], td.team_of_slot,
                                              halves=(spec.half1, spec.half2_play))
    nat = np.full(base.players_speed.shape, 7.0, np.float32)
    with_nat = compute_kinematics_and_orientation(xy, td.ball_xy[:2000], td.team_of_slot,
                                                  halves=(spec.half1, spec.half2_play),
                                                  players_speed_native=nat)
    differs = np.isfinite(base.players_speed) & (base.players_speed != with_nat.players_speed)
    assert differs.any() and (with_nat.players_speed[differs] == 7.0).all()
    same = np.isfinite(base.players_speed) & ~differs
    np.testing.assert_array_equal(base.players_speed[same], with_nat.players_speed[same])


# ------------------------------------- sample-size-robust collective features --------------------------
def _team_frames(n_frames=60, dropped=0, seed=0):
    rng = np.random.default_rng(seed)
    base = np.concatenate([rng.uniform([10, 5], [45, 63], size=(11, 2)),
                           rng.uniform([60, 5], [95, 63], size=(11, 2))]).astype(np.float32)
    team = np.array([0] * 11 + [1] * 11, np.int8)
    p = np.repeat(base[None], n_frames, axis=0)
    p += rng.normal(0, 0.05, p.shape).astype(np.float32)
    for k in range(dropped):
        p[:, 1 + k] = np.nan        # outfield home players
    return p, team


def test_robust_physics_is_identity_by_default_and_masks_or_normalises():
    p, team = _team_frames(dropped=4)
    phys = compute_collective_physics(p, team, use_numba=False)
    off = sfe.FeatureParams()
    assert sfe.robust_physics(phys, off) is phys
    mask = sfe.robust_physics(phys, dataclasses.replace(off, shape_min_field=8))
    assert np.isnan(mask.hull_area_home).all() and np.isfinite(mask.hull_area_away).all()
    assert np.isnan(mask.d_hull_home).all() and np.isfinite(mask.n_field_home).all()
    norm = sfe.robust_physics(phys, dataclasses.replace(off, normalise_shape=True))
    full, _ = _team_frames(dropped=0)
    full_phys = compute_collective_physics(full, team, use_numba=False)
    # six players are left of ten: normalised hull / width are closer to the complete team than the raw ones
    raw_err = np.nanmean(np.abs(phys.width_home - full_phys.width_home))
    norm_err = np.nanmean(np.abs(norm.width_home - full_phys.width_home))
    assert norm_err < raw_err
    assert np.nanmean(norm.hull_area_home) > np.nanmean(phys.hull_area_home)
    np.testing.assert_array_equal(norm.hull_area_away, phys.hull_area_away)    # complete team: unchanged
    assert np.isfinite(norm.d_hull_home).sum() > 0


def test_shape_ratio_interpolates_and_is_one_when_complete():
    r = sfe._shape_ratio(np.array([4.0, 7.0, 10.0, 11.0, np.nan]), (4, 7, 10), (0.3, 0.7, 1.0))
    np.testing.assert_allclose(r, [0.3, 0.7, 1.0, 1.0, 1.0])


def test_feature_params_flags_change_values_not_layout(synthetic_kin):
    from mtah2027.pipeline import compute_front_end, heuristic_emissions

    td, spec = synthetic_kin["td"], synthetic_kin["spec"]
    blocks = [spec.half1, spec.half2_play]
    fe = compute_front_end(td.players_xy, td.team_of_slot, td.ball_xy, play_blocks=blocks, collective=True)
    em = heuristic_emissions(fe, td.fps)
    base = sfe.build_feature_matrix(fe, em)
    neut = sfe.build_feature_matrix(fe, em, params=dataclasses.replace(sfe.FeatureParams(),
                                                                       neutralise_counts=True))
    assert neut.names == base.names and neut.x.shape == base.x.shape
    for key in sfe.COUNT_COLUMNS:
        if key in base.names:
            assert (neut.column(key) == 0).all() and base.column(key).std() >= 0
    other = [i for i, k in enumerate(base.names) if k not in sfe.COUNT_COLUMNS]
    np.testing.assert_array_equal(neut.x[:, other], base.x[:, other])
    robust = sfe.build_feature_matrix(fe, em, params=dataclasses.replace(
        sfe.FeatureParams(), normalise_shape=True, shape_min_field=3))
    assert robust.names == base.names


def test_default_config_keeps_the_adopted_feature_set():
    fp = sfe.FeatureParams.from_config()
    assert not fp.neutralise_counts and fp.shape_min_field == 0 and not fp.normalise_shape
    assert len(fp.shape_ratio_n) == len(fp.shape_ratio_hull) == len(fp.shape_ratio_width)


def _moving_lines(n, s_value):
    """P1 runs at 3 m/s along x (normalised steps), P2 is static; the ``s`` channel is ``s_value``."""
    step = 3.0 / 25.0 / 105.3
    return [f"{i + 1}:{0.2 + step * (i % 400):.7f},0.5,{s_value};0.8,0.5,0.0:0.5,0.5,0.0" for i in range(n)]


@pytest.mark.parametrize("s_value, declared, expected_factor, expected_ms", [
    (3.0, "km/h", 3.6, 3.0),          # declared km/h but the values are m/s (the competition ATD)
    (10.8, "km/h", 1.0, 3.0),         # really km/h
    (3.0, "m/s", 1.0, 3.0),
])
def test_speed_unit_follows_the_data_not_the_label(tmp_path, s_value, declared, expected_factor, expected_ms):
    n = 20000
    meta, raw = _write(tmp_path, _s_xml(declared, end=n), _moving_lines(n, s_value))
    td = al.load_atd_files(meta, raw)
    assert td.report["speed_unit_factor_applied"] == pytest.approx(expected_factor)
    assert np.nanmedian(td.players_speed[:, 0]) == pytest.approx(expected_ms, rel=0.01)
