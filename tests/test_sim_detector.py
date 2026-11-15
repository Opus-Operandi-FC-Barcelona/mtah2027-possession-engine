import numpy as np
import pytest

from mtah2027.vision.sim_detector import MOTION, SV_BALL640, DetectorProfile, simulate_detections


def test_probabilities_sum_to_one():
    for p in (SV_BALL640, MOTION):
        assert abs(p.probabilities().sum() - 1.0) < 1e-12


def test_bad_profile_rejected():
    with pytest.raises(ValueError):
        DetectorProfile("bad", 0.5, 0.4, 0.9).probabilities()


def test_measured_cdf_reproduced():
    truth = np.zeros((200000, 2))
    ret, xy = simulate_detections(truth, SV_BALL640, rate=1.0, seed=3)
    d = np.hypot(xy[:, 0], xy[:, 1])
    n = len(truth)
    for r, want in ((1.0, 0.410), (2.0, 0.597), (5.0, 0.757)):
        assert abs(np.sum(ret & (d <= r)) / n - want) < 0.005
    assert abs(ret.mean() - 0.95) < 0.005
    assert np.nanmax(d[ret]) <= 30.0


def test_rate_scales_and_nests():
    truth = np.zeros((50000, 2))
    r100, _ = simulate_detections(truth, SV_BALL640, rate=1.0, seed=1)
    r50, _ = simulate_detections(truth, SV_BALL640, rate=0.5, seed=1)
    r25, _ = simulate_detections(truth, SV_BALL640, rate=0.25, seed=1)
    assert abs(r50.mean() / r100.mean() - 0.5) < 0.02
    assert abs(r25.mean() / r100.mean() - 0.25) < 0.02
    assert np.all(r50 <= r100) and np.all(r25 <= r50)


def test_deterministic_and_nan_when_not_returned():
    truth = np.random.default_rng(0).uniform(0, 100, (500, 2))
    a = simulate_detections(truth, MOTION, rate=0.5, seed=7)
    b = simulate_detections(truth, MOTION, rate=0.5, seed=7)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1], equal_nan=True)
    assert np.isnan(a[1][~a[0]]).all() and np.isfinite(a[1][a[0]]).all()
