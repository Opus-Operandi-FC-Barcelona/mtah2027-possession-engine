"""Issue #7: homography loader."""
from __future__ import annotations

import numpy as np
import pytest

from mtah2027.config import PITCH
from mtah2027.io import homography_loader as hl
from starter import load_homography as starter_h


def test_tactical_matches_starter(synthetic_dir, synthetic_info):
    hom = hl.load_homography(synthetic_dir)
    matrices, err, valid = starter_h.load(data_dir=str(synthetic_dir))
    assert hom.H.shape == (synthetic_info["spec"].n_frames, 3, 3) == hom.H_inv.shape
    np.testing.assert_array_equal(hom.H, matrices)
    np.testing.assert_array_equal(hom.reprojection_error, err)
    assert hom.valid.all() and valid.all()
    ident = np.einsum("nij,njk->nik", hom.H, hom.H_inv)
    np.testing.assert_allclose(ident, np.broadcast_to(np.eye(3), ident.shape), atol=1e-7)


def test_broadcast_has_invalid_frames(synthetic_dir):
    hom = hl.load_homography(synthetic_dir, source="broadcast")
    matrices, _, valid = starter_h.load(data_dir=str(synthetic_dir), source="broadcast")
    assert 0 < hom.valid.sum() < hom.n_frames
    np.testing.assert_array_equal(hom.valid, valid)
    assert np.isnan(hom.H[~hom.valid]).all() and np.isnan(hom.H_inv[~hom.valid]).all()
    assert np.isfinite(hom.H_inv[hom.valid]).all()
    np.testing.assert_array_equal(hom.H[hom.valid], matrices[valid])
    # usable() adds the reprojection filter
    u = hom.usable(5.0)
    assert (u <= hom.valid).all() and u.sum() <= hom.valid.sum()
    assert not hom.usable(0.0).any()


def test_projection_metres_vs_normalised_starter(synthetic_dir):
    hom = hl.load_homography(synthetic_dir)
    f = 123
    x_m, y_m = np.array([0.0, 10.0, 52.65, 105.3]), np.array([0.0, 20.0, 34.0, 68.0])
    u, v = hl.project_pitch_to_pixels(hom.H[f], x_m, y_m)
    su, sv = starter_h.project(hom.H[f], x_m / PITCH[0], y_m / PITCH[1])
    np.testing.assert_allclose(u, su)
    np.testing.assert_allclose(v, sv)
    # the pitch centre lands inside the 1920x1080 image
    cu, cv = hl.project_pitch_to_pixels(hom.H[f], PITCH[0] / 2, PITCH[1] / 2)
    assert 0 < cu < 1920 and 0 < cv < 1080


def test_round_trip_batched(synthetic_dir):
    hom = hl.load_homography(synthetic_dir)
    rng = np.random.default_rng(0)
    n = hom.n_frames
    x = rng.uniform(0, PITCH[0], (n, 23))
    y = rng.uniform(0, PITCH[1], (n, 23))
    u, v = hom.project_pitch_to_pixels(x, y)
    assert u.shape == (n, 23)
    bx, by = hom.project_pixels_to_pitch(u, v)
    np.testing.assert_allclose(bx, x, atol=1e-6)
    np.testing.assert_allclose(by, y, atol=1e-6)
    # 1-D per-frame points (e.g. the ball) too
    u1, v1 = hom.project_pitch_to_pixels(x[:, 0], y[:, 0])
    b1x, b1y = hom.project_pixels_to_pitch(u1, v1)
    np.testing.assert_allclose(b1x, x[:, 0], atol=1e-6)
    np.testing.assert_allclose(b1y, y[:, 0], atol=1e-6)


def test_invalid_frames_give_nan(synthetic_dir):
    hom = hl.load_homography(synthetic_dir, source="broadcast")
    bad = int(np.flatnonzero(~hom.valid)[0])
    u, v = hl.project_pitch_to_pixels(hom.H[bad], 50.0, 30.0)
    assert np.isnan(u) and np.isnan(v)
    x, y = hom.project_pitch_to_pixels(np.full(hom.n_frames, 50.0), np.full(hom.n_frames, 30.0))
    assert np.isnan(x[~hom.valid]).all() and np.isfinite(x[hom.valid]).all()


def test_singular_matrix_is_not_valid(tmp_path):
    cols = "frame,h00,h01,h02,h10,h11,h12,h20,h21,h22,reprojection_error,valid"
    rows = ["0,1,0,0,0,1,0,0,0,1,0.5,1", "1,1,2,3,2,4,6,3,6,9,0.5,1",
            "2" + ",NaN" * 9 + ",9,0"]
    path = tmp_path / "h.csv"
    path.write_text("\n".join([cols] + rows) + "\n")
    hom = hl.parse_homography_csv(path)
    assert hom.valid.tolist() == [True, False, False]
    assert np.isnan(hom.H_inv[1]).all()


def test_bad_columns_and_missing_file(tmp_path):
    path = tmp_path / "h.csv"
    path.write_text("frame,h00\n0,1\n")
    with pytest.raises(hl.HomographyFormatError):
        hl.parse_homography_csv(path)
    with pytest.raises(IOError):
        hl.load_homography(tmp_path / "nothing")
