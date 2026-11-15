"""Issue #64: implicit-carrier / swarm micro-geometry features (off by default)."""
from __future__ import annotations

import dataclasses
import math

import numpy as np

from mtah2027.features.carrier_swarm import NAMES, CarrierSwarmParams, carrier_swarm
from mtah2027.models import specialist_features as sfe


def _scene():
    """1 frame, 2 slots per team + a goalkeeper each (slot 0 home GK x=1, slot 3 away GK x=104).
    Home outfield slot 1 at (50, 30) runs +x at 3 m/s
    away slot 4 at (52, 30) runs -x at 2 m/s (closing 5 m/s)."""
    pxy = np.full((1, 6, 2), np.nan, np.float32)
    vel = np.zeros((1, 6, 2), np.float32)
    team = np.array([0, 0, 0, 1, 1, 1])
    pxy[0, 0] = (1, 34)
    pxy[0, 1] = (50, 30)
    pxy[0, 2] = (30, 10)
    pxy[0, 3] = (104, 34)
    pxy[0, 4] = (52, 30)
    pxy[0, 5] = (80, 50)
    vel[0, 1] = (3, 0)
    vel[0, 4] = (-2, 0)
    return pxy, vel, team


def test_known_values():
    pxy, vel, team = _scene()
    o = carrier_swarm(pxy, team, vel)
    exp = math.exp(-1.0) * 5.0                      # d = 2 m, closing 5 m/s
    assert abs(o["swarm_home"][0] - exp) < 1e-5 and abs(o["swarm_away"][0] - exp) < 1e-5
    assert abs(o["carrier_vx_home"][0] - 3.0) < 1e-6 and abs(o["carrier_x_home"][0] - 50.0) < 1e-6
    assert abs(o["carrier_vx_away"][0] + 2.0) < 1e-6 and abs(o["carrier_x_away"][0] - 52.0) < 1e-6
    assert abs(o["min_interteam_dist"][0] - 2.0) < 1e-6 and o["duel_density_3m"][0] == 1


def test_goalkeepers_static_and_far_players_excluded():
    pxy, vel, team = _scene()
    vel[0, 1] = (0, 0)                               # home player no longer moves -> no home carrier
    o = carrier_swarm(pxy, team, vel)
    assert np.isnan(o["swarm_home"][0]) and np.isfinite(o["swarm_away"][0])
    pxy[0, 4] = (60, 30)                             # rival now 10 m away: beyond the radius
    vel[0, 1] = (3, 0)
    o = carrier_swarm(pxy, team, vel)
    assert o["swarm_home"][0] == 0.0 and o["duel_density_3m"][0] == 0


def test_diverging_players_have_no_pressure_and_nan_frame():
    pxy, vel, team = _scene()
    vel[0, 1] = (-3, 0)
    vel[0, 4] = (2, 0)          # moving apart
    assert carrier_swarm(pxy, team, vel)["swarm_home"][0] == 0.0
    pxy2 = np.concatenate([pxy, np.full_like(pxy, np.nan)])
    vel2 = np.concatenate([vel, vel])
    o = carrier_swarm(pxy2, team, vel2, params=CarrierSwarmParams(chunk=1))
    assert all(np.isnan(o[k][1]) for k in NAMES)


def test_chunking_and_determinism():
    rng = np.random.default_rng(0)
    pxy = rng.uniform(0, 100, (50, 8, 2)).astype(np.float32)
    vel = rng.normal(0, 3, (50, 8, 2)).astype(np.float32)
    team = np.array([0] * 4 + [1] * 4)
    a = carrier_swarm(pxy, team, vel, params=CarrierSwarmParams(chunk=7))
    b = carrier_swarm(pxy, team, vel, params=CarrierSwarmParams(chunk=50))
    c = carrier_swarm(pxy, team, vel)
    for k in NAMES:
        np.testing.assert_array_equal(a[k], b[k])
        np.testing.assert_array_equal(b[k], c[k])


def test_default_off_and_feature_matrix(synthetic_kin):
    from mtah2027.pipeline import compute_front_end, heuristic_emissions

    assert not sfe.FeatureParams.from_config().carrier_swarm
    td, spec = synthetic_kin["td"], synthetic_kin["spec"]
    blocks = [spec.half1, spec.half2_play]
    fe = compute_front_end(td.players_xy, td.team_of_slot, td.ball_xy, play_blocks=blocks, collective=True)
    em = heuristic_emissions(fe, td.fps)
    base = sfe.build_feature_matrix(fe, em, play_blocks=blocks)
    on = sfe.build_feature_matrix(fe, em, play_blocks=blocks,
                                  params=dataclasses.replace(sfe.FeatureParams(), carrier_swarm=True))
    assert on.names[:len(base.names)] == base.names and len(on.names) > len(base.names)
    np.testing.assert_array_equal(on.x[:, :base.x.shape[1]], base.x)
    assert "sw:swarm_home" in on.names and "diff:swarm" in on.names
    assert any(k.startswith("ctx:sw:swarm_home") for k in on.names)
    on2 = sfe.build_feature_matrix(fe, em, play_blocks=blocks,
                                   params=dataclasses.replace(sfe.FeatureParams(), carrier_swarm=True))
    np.testing.assert_array_equal(on.x, on2.x)
