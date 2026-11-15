#!/usr/bin/env python
"""Evaluate ball detectors on the validation crops of the self-supervised dataset (Issue #68; aggregates only).

Reads ``<ds>/val_meta.npz`` + ``<ds>/images/val`` (made by build_ball_dataset.py), runs one or more YOLO weights
on the GPU, reprojects the candidates to pitch metres with the homography inverse of each frame and compares
them with the ATD ball.  Two crop strategies (``atd``: window around the ATD ball plus a random offset, the
ceiling; ``prior``: window around a search point that does not use the ball, as in a real gap) and the subset
of frames within 2 s of a tracking gap.  Prints / writes ONLY aggregates.

    python scripts/eval_ball_detector.py --data-dir <data> --ds <ds> --model ft=<best.pt>:960 \
        --model sv=<sv.pt>:640 --json reports/issue68_detector_profile.json
"""
# ruff: noqa: E501
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for _extra in (REPO_ROOT / "src", REPO_ROOT, REPO_ROOT / "scripts"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))


def run_model(det, files, batch: int):
    import cv2

    rows, xy_px, conf = [], [], []
    t0 = time.perf_counter()
    n_imgs = 0
    for i in range(0, len(files), batch):
        imgs = [cv2.imread(str(f)) for f in files[i:i + batch]]
        res = det.detect(imgs)
        n_imgs += len(imgs)
        for j, r in enumerate(res):
            for c in r:
                rows.append(i + j)
                xy_px.append(c[:2])
                conf.append(c[2])
    secs = time.perf_counter() - t0
    return (np.asarray(rows, np.int64), np.asarray(xy_px, float).reshape(-1, 2), np.asarray(conf, float),
            n_imgs / max(secs, 1e-9))


def evaluate_model(det, files, meta, hom, pitch, batch: int) -> dict:
    from mtah2027.vision.attention_crops import crop_points_to_pitch
    from mtah2027.vision.ball_eval import profile, top1_errors

    rows, xy_px, conf, thr = run_model(det, files, batch)
    frame = meta["frame"]
    org = np.stack([meta["x0"], meta["y0"]], 1)
    xy_m = (crop_points_to_pitch(xy_px, org[rows], hom.H_inv[frame[rows]], pitch)
            if len(rows) else np.zeros((0, 2)))
    truth = meta["truth_xy"]
    err, topc, cnt = top1_errors(len(files), rows, xy_m, conf, truth)
    cand_err = np.hypot(*(xy_m - truth[rows]).T) if len(rows) else np.zeros(0)
    inside = meta["ball_inside"]
    res: dict = {"throughput_crops_per_s": round(float(thr), 1)}
    for strat, k in (("atd", 0), ("prior", 1)):
        for sub, extra in (("all", np.ones(len(files), bool)), ("near_gap", meta["near_gap"])):
            m = (meta["kind"] == k) & extra & inside     # only crops that contain the ball are scored
            n_fp = int((cand_err[m[rows]] > 5.0).sum()) if len(rows) else 0
            res[f"{strat}_{sub}"] = profile(err[m], topc[m], cnt[m], n_fp)
    pm = (meta["kind"] == 1) & ~inside
    res["prior_ball_outside_crops"] = {
        "crops": int(pm.sum()),
        "frames_with_detection_pct": round(100.0 * float((cnt[pm] > 0).mean()), 2) if pm.any() else None}
    res["prior_ball_inside_pct"] = round(100.0 * float(inside[meta["kind"] == 1].mean()), 2)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--ds", required=True)
    ap.add_argument("--model", action="append", required=True, help="name=weights.pt:imgsz (repeatable)")
    ap.add_argument("--conf", type=float, default=0.02)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    from run_vision_recovery import load_inputs

    from mtah2027.pipeline import resolve_config
    from mtah2027.vision.ball_eval import assert_aggregate
    from mtah2027.vision.ft_ball import FineTunedBallDetector

    cfg_path, _ = resolve_config(args.config)
    data_dir = args.data_dir or os.environ.get("MTAH_DATA_DIR") or str(REPO_ROOT / "data")
    ball, players, hom, blocks, fps, pitch, _video = load_inputs(data_dir, cfg_path)
    from mtah2027.vision import attention_crops as ac
    plan = ac.plan_crops(ball, players, hom, blocks, fps=fps, pitch=pitch, params=ac.VisionParams.from_config(cfg_path))
    ds = Path(args.ds)
    meta = dict(np.load(ds / "val_meta.npz"))
    names = [f"f{int(f):06d}_{'atd' if k == 0 else 'prior'}.png" for f, k in zip(meta["frame"], meta["kind"])]
    files = [ds / "images" / "val" / n for n in names]
    rep: dict = {"val_crops": int(len(files)), "conf_floor": args.conf, "models": {}}
    for spec in args.model:
        name, rest = spec.split("=", 1)
        weights, _, sz = rest.rpartition(":")
        det = FineTunedBallDetector(weights, imgsz=int(sz), conf=args.conf, max_candidates=10, batch=args.batch)
        rep["models"][name] = evaluate_model(det, files, meta, hom, pitch, args.batch)
        thr = rep["models"][name]["throughput_crops_per_s"]
        rep["models"][name]["full_match_cost"] = {
            "hidden_graded_frames": int(plan.n_hidden), "planned_crops_every_k": int(len(plan)),
            "gpu_inference_minutes_crops_only": round(len(plan) / max(thr, 1e-9) / 60.0, 1),
            "all_hidden_frames_minutes_crops_only": round(plan.n_hidden / max(thr, 1e-9) / 60.0, 1)}
    assert_aggregate(rep)
    text = json.dumps(rep, indent=1)
    print(text)
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
