#!/usr/bin/env python
"""Build the self-supervised ball dataset from the real tactical video (Issue #68; aggregates only).

Labels = the automatic ATD ball projected with the homography (no manual labels, no possession labels).
Temporal split by 2-min blocks, every 5th block validation, 10 s guard band.  Images and metadata are written
OUTSIDE git (default ``~/issue68-cache``, override with ``--out`` or ``MTAH_ISSUE68_CACHE``); only counts are
printed.

    python scripts/build_ball_dataset.py --data-dir <data> --out C:/Users/<me>/issue68-cache/ds
"""
# ruff: noqa: E501
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _extra in (REPO_ROOT / "src", REPO_ROOT, REPO_ROOT / "scripts"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--train-stride", type=int, default=4)
    ap.add_argument("--val-stride", type=int, default=4)
    ap.add_argument("--max-frame", type=int, default=None, help="only frames up to this one (smoke runs)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    from run_vision_recovery import load_inputs

    from mtah2027.config import set_global_seeds
    from mtah2027.pipeline import resolve_config
    from mtah2027.vision.attention_crops import VisionParams
    from mtah2027.vision.ball_dataset import DatasetParams, build_dataset, default_cache_dir

    set_global_seeds(0)
    cfg_path, _ = resolve_config(args.config)
    params = VisionParams.from_config(cfg_path)
    data_dir = args.data_dir or os.environ.get("MTAH_DATA_DIR") or str(REPO_ROOT / "data")
    ball, players, hom, blocks, fps, pitch, video = load_inputs(data_dir, cfg_path)
    out = Path(args.out) if args.out else default_cache_dir() / "ds"
    t0 = time.perf_counter()
    stats = build_dataset(video, ball, players, hom, blocks, params, out, fps=fps, pitch=pitch,
                          p=DatasetParams(train_stride=args.train_stride, val_stride=args.val_stride,
                                          seed=args.seed), max_frame=args.max_frame)
    stats["seconds"] = round(time.perf_counter() - t0, 1)
    print(json.dumps(stats, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
