"""Issue #17: dynamic goalkeepers, spectral / convex-hull areas and their derivatives."""
from __future__ import annotations

import time

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from mtah2027.config import ConfigError, params_from_mapping
from mtah2027.features import collective_physics as cp

FPS = 25.0
PARAMS = cp.CollectivePhysicsParams()


# ---------------------------------------------------------------------------------- config
def test_yaml_section_matches_defaults_and_rejects_typos():
    assert cp.CollectivePhysicsParams.from_config() == PARAMS
    with pytest.raises(ConfigError):
        params_from_mapping(cp.CollectivePhysicsParams, {"min_player": 3}, "collective_physics")


# ---------------------------------------------------------------------------------- goalkeepers
def test_dynamic_goalkeeper_is_the_extreme_x_player_of_each_team():
    # slots 0-2 home, 3-5 away; frame 1 changes which slot is deepest, frame 2 has NaNs, frame 3 is empty
    xy = np.array([
        [[5, 30], [20, 20], [30, 40], [100, 30], [80, 10], [70, 50]],
        [[25, 30], [3, 20], [30, 40], [70, 30], [80, 10], [101, 50]],
        [[np.nan, np.nan], [20, 20], [30, 40], [np.nan, 5], [80, 10], [70, 50]],
        [[np.nan] * 2] * 6,
    ], dtype=np.float32)
    team = np.array([0, 0, 0, 1, 1, 1])
    np.testing.assert_array_equal(cp.dynamic_goalkeeper(xy, team, 0), [0, 1, 1, -1])
    np.testing.assert_array_equal(cp.dynamic_goalkeeper(xy, team, 1), [3, 5, 4, -1])
    r = cp.compute_collective_physics(xy, team, fps=FPS, params=cp.CollectivePhysicsParams(min_players=2))
    np.testing.assert_array_equal(r.gk_slot_home, [0, 1, 1, -1])
    np.testing.assert_allclose(r.gk_xy_home[1], [3, 20])
    assert np.isnan(r.gk_xy_home[3]).all() and np.isnan(r.gk_xy_away[3]).all()
    np.testing.assert_array_equal(r.n_field_home, [2, 2, 1, 0])


def test_goalkeeper_velocity_and_exclusion_from_areas():
    n = 60
    t = np.arange(n)[:, None]
    gk = np.concatenate([3.0 + 0.2 * t, np.full((n, 1), 34.0)], axis=1)           # moves 5 m/s along x
    field = np.array([[20, 20], [20, 48], [40, 20], [40, 48]], dtype=np.float64)    # 20 x 28 rectangle
    xy = np.concatenate([gk[:, None, :], np.broadcast_to(field, (n, 4, 2))], axis=1).astype(np.float32)
    xy = np.concatenate([xy, np.full((n, 1, 2), np.nan, np.float32)], axis=1)
    team = np.array([0, 0, 0, 0, 0, 1])
    r = cp.compute_collective_physics(xy, team, fps=FPS, params=PARAMS)
    assert (r.gk_slot_home == 0).all()
    np.testing.assert_allclose(r.gk_vel_home[10], [5.0, 0.0], atol=1e-4)
    np.testing.assert_allclose(r.hull_area_home, 20 * 28, rtol=1e-6)              # the goalkeeper is excluded
    assert np.isnan(r.hull_area_away).all() and (r.n_field_away == 0).all()


# ---------------------------------------------------------------------------------- areas
def test_spectral_area_of_a_cross_is_pi_a_b_over_two():
    a, b = 8.0, 5.0
    x = np.array([[a, -a, 0, 0, 0.0]]) + 50.0
    y = np.array([[0, 0, b, -b, 0.0]]) + 30.0
    # with the centre point: var_x = 2 a^2 / 5, var_y = 2 b^2 / 5
    assert cp.spectral_areas(x, y)[0] == pytest.approx(np.pi * np.sqrt((2 * a * a / 5) * (2 * b * b / 5)))
    assert cp.spectral_areas(x[:, :4], y[:, :4])[0] == pytest.approx(np.pi * a * b / 2)


def test_spectral_area_is_rotation_invariant_and_handles_nan():
    rng = np.random.default_rng(0)
    pts = rng.normal(0, [6.0, 3.0], (1, 10, 2))
    th = 0.7
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    a = cp.spectral_areas(pts[..., 0], pts[..., 1])
    q = pts @ rot.T
    b = cp.spectral_areas(q[..., 0], q[..., 1])
    assert a[0] == pytest.approx(b[0], rel=1e-9)
    nanx = np.array([[1.0, np.nan, 3.0, 4.0]])
    nany = np.array([[1.0, 2.0, np.nan, 5.0]])
    assert np.isnan(cp.spectral_areas(nanx, nany)[0])                             # only two finite points
    collinear = cp.spectral_areas(np.array([[0.0, 1, 2, 3]]), np.array([[0.0, 1, 2, 3]]))
    assert collinear[0] == pytest.approx(0.0, abs=1e-6)


def test_hull_area_known_shapes_and_degenerate_rows():
    x = np.array([[0, 10, 10, 0, 5, np.nan],               # square with an interior point and a NaN
                  [0, 4, 0, np.nan, np.nan, np.nan],       # right triangle 4 x 3 -> 6
                  [0, 1, 2, 3, np.nan, np.nan],            # collinear
                  [0, 1, np.nan, np.nan, np.nan, np.nan],  # too few points
                  [3, 3, 3, 3, 3, 3]], dtype=float)
    y = np.array([[0, 0, 10, 10, 5, 7],
                  [0, 0, 3, np.nan, np.nan, np.nan],
                  [0, 1, 2, 3, np.nan, np.nan],
                  [0, 1, np.nan, np.nan, np.nan, np.nan],
                  [4, 4, 4, 4, 4, 4]], dtype=float)
    got = cp.convex_hull_areas(x, y)
    assert got[0] == pytest.approx(100.0) and got[1] == pytest.approx(6.0)
    assert got[2] == pytest.approx(0.0) and np.isnan(got[3]) and got[4] == pytest.approx(0.0)


def test_hull_matches_scipy_including_ties_and_numba_equals_fallback():
    rng = np.random.default_rng(1)
    x = rng.integers(0, 12, (400, 10)).astype(float)                              # many ties / collinearities
    y = rng.integers(0, 12, (400, 10)).astype(float)
    x[rng.random(x.shape) < 0.15] = np.nan
    pure = cp.convex_hull_areas(x, y, use_numba=False)
    for i in range(len(x)):
        ok = np.isfinite(x[i]) & np.isfinite(y[i])
        pts = np.stack([x[i, ok], y[i, ok]], axis=1)
        if len(pts) < 3:
            assert np.isnan(pure[i])
            continue
        try:
            want = ConvexHull(pts).volume
        except Exception:                                    # all collinear (QhullError)
            want = 0.0
        assert pure[i] == pytest.approx(want, abs=1e-9)
    xr, yr = rng.uniform(0, 100, (300, 10)), rng.uniform(0, 60, (300, 10))
    pure_r = cp.convex_hull_areas(xr, yr, use_numba=False)
    for i in range(0, 300, 7):
        assert pure_r[i] == pytest.approx(ConvexHull(np.stack([xr[i], yr[i]], axis=1)).volume)
    if cp.numba_available():
        np.testing.assert_array_equal(cp.convex_hull_areas(x, y, use_numba=True), pure)
        np.testing.assert_array_equal(cp.convex_hull_areas(xr, yr, use_numba=True), pure_r)


# ---------------------------------------------------------------------------------- derivatives
def test_area_derivatives_of_a_uniformly_expanding_block():
    n = 200
    t = np.arange(n, dtype=np.float64)
    half = 5.0 + 0.04 * t                                                         # half side, m
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float)
    field = 50.0 + half[:, None, None] * corners[None]                            # [n, 4, 2]
    gk = np.broadcast_to(np.array([[2.0, 34.0]]), (n, 1, 2))
    home = np.concatenate([gk, field + [0, 4]], axis=1)                           # expanding square
    away = np.concatenate([np.broadcast_to(field[:1] + [30, 0], (n, 4, 2)),       # static square
                           np.broadcast_to(np.array([[103.0, 34.0]]), (n, 1, 2))], axis=1)
    xy = np.concatenate([home, away], axis=1).astype(np.float32)
    team = np.array([0] * 5 + [1] * 5)
    r = cp.compute_collective_physics(xy, team, fps=FPS, params=PARAMS)
    side = 2 * half
    np.testing.assert_allclose(r.hull_area_home, side * side, rtol=1e-5)
    want = 2.0 * side * (0.08 * FPS)                  # d(side^2)/dt per second; exact for a quadratic
    np.testing.assert_allclose(r.d_hull_home[30:-30], want[30:-30], rtol=1e-4)
    # four corners of a square: equal eigenvalues var = side^2 / 4 -> spectral area pi * side^2 / 4
    np.testing.assert_allclose(r.spectral_area_home, np.pi * side * side / 4, rtol=1e-5)
    np.testing.assert_allclose(r.d_spectral_home[30:-30], np.pi / 4 * want[30:-30], rtol=1e-4)
    assert np.isnan(r.d_hull_home[:25]).all() and np.isnan(r.d_hull_home[-25:]).all()   # 2 s span
    np.testing.assert_allclose(r.d_hull_away[60:-60], 0.0, atol=1e-3)
    np.testing.assert_allclose(r.depth_home, side, rtol=1e-5)
    np.testing.assert_allclose(r.width_home, side, rtol=1e-5)


def test_centroid_velocity_along_x_and_matrix_export():
    n = 120
    t = np.arange(n)[:, None, None]
    base = np.array([[0.0, 0], [8, 0], [0, 8], [8, 8], [4, 4.0]])[None]
    home = base + [30.0, 20.0] + np.concatenate([0.12 * t, 0 * t], axis=2)        # 3 m/s towards +x
    gk = np.broadcast_to(np.array([[1.0, 34.0]]), (n, 1, 2))
    xy = np.concatenate([gk, home, np.full((n, 3, 2), np.nan)], axis=1).astype(np.float32)
    team = np.array([0] * 6 + [1] * 3)
    r = cp.compute_collective_physics(xy, team, fps=FPS, params=PARAMS)
    np.testing.assert_allclose(r.centroid_vx_home[50:70], 3.0, atol=1e-3)
    m, names = r.matrix()
    assert m.shape == (n, len(names)) and m.dtype == np.float32
    assert "hull_area_home" in names and "gk_xy_away_y" in names


def test_too_few_players_gives_nan_and_shape_errors():
    xy = np.zeros((5, 4, 2), dtype=np.float32)
    xy[..., 0] = np.arange(4) * 10.0
    r = cp.compute_collective_physics(xy, np.array([0, 0, 1, 1]), fps=FPS, params=PARAMS)
    assert np.isnan(r.hull_area_home).all() and np.isnan(r.spectral_area_away).all()
    with pytest.raises(ValueError):
        cp.compute_collective_physics(xy, np.array([0, 1]), fps=FPS, params=PARAMS)


# ---------------------------------------------------------------------------------- synthetic match
def test_synthetic_match_features_are_plausible(synthetic_kin):
    kin, spec = synthetic_kin["kin"], synthetic_kin["spec"]
    r = cp.compute_collective_physics(kin.players_xy, synthetic_kin["td"].team_of_slot,
                                      players_vel=kin.players_vel, fps=FPS, params=PARAMS)
    a, b = spec.half1
    sl = slice(a, b + 1)
    hull, spec_area = r.hull_area_home[sl], r.spectral_area_home[sl]
    assert np.isfinite(hull).mean() > 0.95
    assert 100.0 < np.nanmedian(hull) < 7140.0
    both = np.isfinite(hull) & np.isfinite(spec_area)
    assert (hull[both] > 0).all() and (spec_area[both] > 0).all()
    # oriented: the home goalkeeper defends the own (low-x) half, the away one the opposite half
    assert np.nanmedian(r.gk_xy_home[sl, 0]) < 52.65 < np.nanmedian(r.gk_xy_away[sl, 0])


# ---------------------------------------------------------------------------------- performance
@pytest.mark.slow
def test_full_size_computation_is_fast():
    n, s = 199319, 46
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 100, (n, s, 2)).astype(np.float32)
    team = np.array([0] * 23 + [1] * 23)
    cp.compute_collective_physics(xy[:2000], team, fps=FPS, params=PARAMS)      # warm up the njit kernel
    t0 = time.perf_counter()
    r = cp.compute_collective_physics(xy, team, fps=FPS, params=PARAMS)
    seconds = time.perf_counter() - t0
    assert r.hull_area_home.shape == (n,) and np.isfinite(r.hull_area_home).all()
    if cp.numba_available():
        assert seconds < 5.0
