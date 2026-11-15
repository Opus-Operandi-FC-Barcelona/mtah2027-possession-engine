#!/usr/bin/env python
"""Fine-tune a small single-class YOLO ball detector on the self-supervised dataset (Issue #68).

Needs the separate vision environment (torch + CUDA, ultralytics; AGPL-3.0, optional extra).  Starts from a
public weights file (``--weights``), trains with a wall-clock budget (``--hours``) and early stopping on the
validation blocks (separate in time from the training blocks).  Trained weights stay in the cache directory
and are NEVER committed.

    python scripts/train_ball_detector.py --ds C:/Users/<me>/issue68-cache/ds --weights yolov8s.pt --imgsz 960
"""
# ruff: noqa: E501
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ds", required=True, help="dataset directory made by build_ball_dataset.py")
    ap.add_argument("--weights", default="yolov8s.pt")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--hours", type=float, default=2.5, help="wall-clock budget (ultralytics `time`)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--name", default="ball")
    ap.add_argument("--project", default=None, help="default: <ds>/../runs")
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-every", type=int, default=4, help="in-training validation uses every k-th val image")
    ap.add_argument("--data", default=None, help="ready data.yaml (e.g. from refine_ball_labels.py)")
    ap.add_argument("--lr0", type=float, default=0.0007)
    args = ap.parse_args(argv)

    from ultralytics import YOLO

    ds = Path(args.ds)
    project = Path(args.project) if args.project else ds.parent / "runs"
    imgs = sorted((ds / "images" / "val").glob("*.png"))[::max(1, args.val_every)]
    (ds / "val_subset.txt").write_text("\n".join(i.as_posix() for i in imgs) + "\n")
    (ds / "data_train.yaml").write_text(
        f"path: {ds.as_posix()}\ntrain: images/train\nval: val_subset.txt\nnames:\n  0: ball\n")
    model = YOLO(args.weights)
    model.train(data=args.data or str(ds / "data_train.yaml"), imgsz=args.imgsz, epochs=args.epochs, time=args.hours,
                batch=args.batch, workers=min(4, args.workers), patience=args.patience, seed=args.seed,
                deterministic=True, project=str(project), name=args.name, exist_ok=True, single_cls=True,
                fraction=args.fraction, device=0, cache=False, plots=False,
                # a 4-5 px ball: no shrinking, mild colour jitter, flips are safe, mosaic only for variety
                scale=0.25, mosaic=0.0, close_mosaic=0, mixup=0.0, degrees=0.0, translate=0.1,
                fliplr=0.5, flipud=0.0, hsv_h=0.01, hsv_s=0.4, hsv_v=0.3, cos_lr=True, amp=True, optimizer="AdamW", lr0=args.lr0,
                warmup_epochs=1.0)
    best = project / args.name / "weights" / "best.pt"
    print(json.dumps({"best_weights_exists": best.is_file()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
