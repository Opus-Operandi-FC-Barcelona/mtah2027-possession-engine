#!/usr/bin/env python
"""Teacher-agreement relabelling of the self-supervised training crops (Issue #68; aggregates only).

The ATD ball projected with the homography is only roughly aligned with the ball seen in the video (a few
pixels to ~1 m).  Two independent algorithmic sources are combined instead: the tracking (where the ball
should be) and a public single-class football-ball YOLO run locally as TEACHER (where a ball-like blob
is).  A training crop keeps a positive label only when the teacher has a candidate of confidence >=
``--min-conf`` within ``--radius-px`` of the ATD pixel; the label is then the teacher's box centre.  Crops
without agreement are dropped from training (their ball position is unknown), negatives are kept.  The
validation labels / metrics stay ATD-based and are never touched.  Nothing leaves the machine.

Output: ``<out>`` with ``images`` (junctions to the source images), refined ``labels``, ``train_list.txt``,
``data.yaml``.

    python scripts/refine_ball_labels.py --ds <ds> --out <ds_ref> --teacher football-ball-detection.pt
"""
# ruff: noqa: E501
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--min-conf", type=float, default=0.4)
    ap.add_argument("--radius-px", type=float, default=40.0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    from ultralytics import YOLO

    ds, out = Path(args.ds), Path(args.out)
    (out / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (out / "labels" / "val").mkdir(parents=True, exist_ok=True)
    if not (out / "images").exists():
        subprocess.run(["cmd", "/c", "mklink", "/J", str(out / "images"), str(ds / "images")], check=True,
                       stdout=subprocess.DEVNULL)
    for f in (ds / "labels" / "val").glob("*.txt"):
        (out / "labels" / "val" / f.name).write_text(f.read_text())
    imgs = sorted((ds / "images" / "train").glob("*.jpg"))
    if args.limit:
        imgs = imgs[:args.limit]
    pos = [p for p in imgs if p.stem.endswith("_p")]
    neg = [p for p in imgs if p.stem.endswith("_n")]
    model = YOLO(args.teacher)
    kept, offs, confs = [], [], []
    t0 = time.perf_counter()
    for i in range(0, len(pos), args.batch):
        chunk = pos[i:i + args.batch]
        res = model.predict([str(p) for p in chunk], imgsz=args.imgsz, conf=0.1, device=0, half=True, verbose=False)
        for p, r in zip(chunk, res):
            lab = (ds / "labels" / "train" / (p.stem + ".txt")).read_text().split()
            lu, lv = float(lab[1]) * 640, float(lab[2]) * 640
            if r.boxes is None or len(r.boxes) == 0:
                continue
            xy = r.boxes.xywh.cpu().numpy()
            cf = r.boxes.conf.cpu().numpy()
            d = np.hypot(xy[:, 0] - lu, xy[:, 1] - lv)
            ok = np.flatnonzero((d <= args.radius_px) & (cf >= args.min_conf))
            if not len(ok):
                continue
            j = ok[np.argmax(cf[ok])]
            u, v = float(xy[j, 0]), float(xy[j, 1])
            (out / "labels" / "train" / (p.stem + ".txt")).write_text(
                f"0 {u / 640:.6f} {v / 640:.6f} {10 / 640:.6f} {10 / 640:.6f}\n")
            kept.append(p)
            offs.append(float(d[j]))
            confs.append(float(cf[j]))
    for p in neg:
        (out / "labels" / "train" / (p.stem + ".txt")).write_text("")
    lst = [(out / "images" / "train" / p.name).as_posix() for p in kept + neg]
    (out / "train_list.txt").write_text("\n".join(lst) + "\n")
    val_imgs = sorted((out / "images" / "val").glob("*.png"))[::4]
    (out / "val_subset.txt").write_text("\n".join(i.as_posix() for i in val_imgs) + "\n")
    (out / "data.yaml").write_text(
        f"path: {out.as_posix()}\ntrain: train_list.txt\nval: val_subset.txt\nnames:\n  0: ball\n")
    offs_a = np.asarray(offs)
    stats = {"positive_crops": len(pos), "kept_positive": len(kept),
             "kept_pct": round(100.0 * len(kept) / max(1, len(pos)), 2), "negatives": len(neg),
             "teacher_minus_atd_px_median_p90": [round(float(np.median(offs_a)), 2),
                                                 round(float(np.quantile(offs_a, 0.9)), 2)] if len(offs_a) else None,
             "teacher_conf_median": round(float(np.median(confs)), 3) if confs else None,
             "min_conf": args.min_conf, "radius_px": args.radius_px,
             "seconds": round(time.perf_counter() - t0, 1)}
    print(json.dumps(stats, indent=1))
    (out / "refine_stats.json").write_text(json.dumps(stats, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
