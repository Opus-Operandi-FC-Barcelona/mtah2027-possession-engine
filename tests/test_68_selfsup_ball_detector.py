# ruff: noqa: E501
"""Issue #68: self-supervised ball dataset, aggregate evaluation and the guarded GPU detector (synthetic data)."""
from __future__ import annotations

import numpy as np
import pytest

from mtah2027.io.homography_loader import Homographies
from mtah2027.vision import ball_dataset as bd
from mtah2027.vision import ball_eval as be
from mtah2027.vision.attention_crops import VisionParams

PITCH = (105.3, 68.0)


def _hom(n: int) -> Homographies:
    H = np.tile(np.diag([1920.0, 1080.0, 1.0]), (n, 1, 1))
    return Homographies(H=H, H_inv=np.linalg.inv(H), reprojection_error=np.zeros(n), valid=np.ones(n, bool))


def test_split_has_guard_band_and_no_leak():
    lab = bd.split_labels(20000, block_frames=3000, val_every=5, guard_frames=250)
    val = np.flatnonzero(lab == 1)
    tr = np.flatnonzero(lab == 0)
    assert len(val) and len(tr)
    assert (val // 3000 % 5 == 4).all()
    # every train frame is at least 250 frames away from every validation frame
    gaps = np.searchsorted(val, tr)
    left = np.where(gaps > 0, tr - val[np.clip(gaps - 1, 0, len(val) - 1)], 10 ** 9)
    right = np.where(gaps < len(val), val[np.clip(gaps, 0, len(val) - 1)] - tr, 10 ** 9)
    assert min(left.min(), right.min()) > 250


def test_near_gap_mask_uses_tracking_only():
    ball = np.zeros((200, 2))
    ball[100:110] = np.nan
    m = bd.near_gap_mask(ball, 20)
    assert m[85] and m[125] and not m[70] and not m[135]


def test_positive_window_always_contains_ball_with_margin():
    rng = np.random.default_rng(0)
    p = bd.DatasetParams()
    u = rng.uniform(40, 1880, 500)
    v = rng.uniform(40, 1040, 500)
    org, ball = bd.random_positive_windows(u, v, rng, p, 1920, 1080)
    assert ((ball >= 0) & (ball < p.crop)).all()
    assert (org >= 0).all() and (org[:, 0] <= 1920 - p.crop).all() and (org[:, 1] <= 1080 - p.crop).all()
    assert bd.yolo_line(320.0, 160.0, 640).startswith("0 0.500000 0.250000")


def test_negative_windows_exclude_ball():
    rng = np.random.default_rng(1)
    p = bd.DatasetParams()
    cen = rng.uniform(300, 1600, (300, 2))
    ball = cen + rng.uniform(-150, 150, (300, 2))
    keep, org = bd.negative_windows(cen, ball, rng, p, 1920, 1080)
    bu, bv = ball[:, 0] - org[:, 0], ball[:, 1] - org[:, 1]
    assert not ((bu > -24) & (bu < p.crop + 24) & (bv > -24) & (bv < p.crop + 24))[keep].any()


def test_simulated_prior_never_uses_the_ball_at_t():
    n = 600
    t = np.arange(n)
    ball = np.stack([20 + 0.05 * t, np.full(n, 30.0)], 1)
    players = np.full((n, 4, 2), np.nan)
    frames = np.array([300, 310, 320])
    v = VisionParams()
    a = bd.simulated_gap_prior(frames, ball, players, 25.0, v, np.random.default_rng(3), np.array([40, 80]))
    ball2 = ball.copy()
    ball2[frames] += 50.0                      # moving the true ball at t must not change the prior
    b = bd.simulated_gap_prior(frames, ball2, players, 25.0, v, np.random.default_rng(3), np.array([40, 80]))
    assert np.isfinite(a).all() and np.allclose(a, b)
    assert np.abs(a[:, 0] - ball[frames, 0]).max() < 5.0       # interpolation of a straight track is close


def test_profile_is_aggregate_and_guarded():
    err = np.array([0.1, 0.4, 3.0, np.inf, 12.0])
    conf = np.array([0.9, 0.6, 0.2, 0.0, 0.05])
    cnt = np.array([1, 2, 1, 0, 1])
    prof = be.profile(err, conf, cnt, n_false_positive=2)
    be.assert_aggregate(prof)
    assert prof["top1_within_pct"]["0.3m"] == 20.0 and prof["top1_within_pct"]["5m"] == 60.0
    assert abs(sum(prof["top1_error_hist_pct"]) - 100.0) < 0.5
    with pytest.raises(ValueError):
        be.assert_aggregate({"rows": list(range(50))})
    e, c, k = be.top1_errors(3, [0, 0, 2], np.array([[0, 0], [5, 5], [1, 1]], float), [0.2, 0.9, 0.5],
                             np.zeros((3, 2)))
    assert e[0] == pytest.approx(np.hypot(5, 5)) and c[0] == 0.9 and np.isinf(e[1]) and k.tolist() == [2, 0, 1]


def test_build_dataset_on_a_synthetic_video(tmp_path):
    cv2 = pytest.importorskip("cv2")
    n, W, H = 60, 1920, 1080
    video = tmp_path / "v.mp4"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (W, H))
    ball = np.stack([30 + 0.3 * np.arange(n), np.full(n, 34.0)], 1)
    for i in range(n):
        img = np.full((H, W, 3), 40, np.uint8)
        u, v = int(ball[i, 0] / PITCH[0] * W), int(ball[i, 1] / PITCH[1] * H)
        cv2.circle(img, (u, v), 3, (255, 255, 255), -1)
        w.write(img)
    w.release()
    players = np.full((n, 6, 2), np.nan)
    players[:, :3] = np.array([[50.0, 20.0], [60.0, 40.0], [70.0, 30.0]])
    p = bd.DatasetParams(block_frames=20, val_every=3, guard_frames=2, train_stride=2, val_stride=2, neg_prob=1.0)
    params = VisionParams(detector="dryrun")
    stats = bd.build_dataset(video, ball, players, _hom(n), [(0, n - 1)], params, tmp_path / "ds", fps=25.0,
                             pitch=PITCH, p=p)
    be.assert_aggregate(stats)
    assert stats["train_positive_crops"] > 0 and stats["val_crops_atd"] > 0 and stats["val_crops_prior"] > 0
    meta = np.load(tmp_path / "ds" / "val_meta.npz")
    tr_frames = {int(f.name[1:7]) for f in (tmp_path / "ds/images/train").iterdir()}
    assert not (tr_frames & set(meta["frame"].tolist()))
    assert (tmp_path / "ds" / "data.yaml").is_file()
    # a positive label sits on the synthetic white ball
    lab = next((tmp_path / "ds/labels/train").glob("f*_p.txt")).read_text().split()
    img = cv2.imread(str((tmp_path / "ds/images/train" / next((tmp_path / "ds/labels/train").glob("f*_p.txt")).name)
                         ).replace(".txt", ".jpg")).astype(int).sum(axis=2)
    cx, cy = int(float(lab[1]) * 640), int(float(lab[2]) * 640)
    assert img[max(cy - 3, 0):cy + 4, max(cx - 3, 0):cx + 4].max() > 600


def test_finetuned_detector_needs_torch_and_weights(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("ultralytics")
    from mtah2027.vision.ft_ball import FineTunedBallDetector

    with pytest.raises(Exception):
        FineTunedBallDetector(str(tmp_path / "missing.pt"))
