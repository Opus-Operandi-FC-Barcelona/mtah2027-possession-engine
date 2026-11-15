#!/usr/bin/env python
"""Issue #57: train Specialist C (learned H<->A turnover gate) from a cached TrainingSet pickle.

The rows are those of the specialists' training set (``scripts/build_training_pickle.py``; stride-10 frames of
the degraded external matches, home / away swapped copies included); the target of each row is the turnover
class within ``+- half_window`` frames of its frame, derived from the ground truth of that match (external
data only, never BAR-ATH).  Jobs:

* ``--lomo``: one model per held-out IDSSE match (trained without that match) -> ``<out-dir>/<match>.json``;
* ``--pff-folds``: one model per PFF fold (fold = sorted index modulo ``--folds``; the held-out PFF matches are
  excluded, IDSSE rows are all used) -> ``<out-dir>/fold<k>.json``;
* ``--full``: one model on every row -> ``<out-dir>/unseen.json`` (the committed model for the real match).

    python scripts/train_specialist_c.py --training-pickle t_cal_aff33.pkl --pff-from t_legacy.pkl \
        --pff-weight 0.5 --lomo --pff-folds --full --out-dir c_pff
    python scripts/train_specialist_c.py --training-pickle t_cal_aff33.pkl --no-pff --lomo --out-dir c_idsse

Only aggregates are printed.
"""
# ruff: noqa: E501
from __future__ import annotations

import argparse
import gc
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for _extra in (REPO_ROOT / "src", REPO_ROOT):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

DEFAULT_BENCH = REPO_ROOT.parent / "mtah2027-external-benchmarks" / "cache"


def rows_with_targets(parts, bench: Path, half_window: int, pff_weight: float):
    """Stack the rows of ``parts``: ``(x, y, match, source, weight, names)`` with turnover targets."""
    from mtah2027.models.specialist_c import turnover_targets
    from mtah2027.models.specialist_data import swap_codes
    from mtah2027.validation import eval_bridge as eb

    gt_cache: dict = {}
    xs, ys, ms, ss, ws = [], [], [], [], []
    for p in parts:
        if p.match not in gt_cache:
            gt_cache[p.match] = eb.to_codes(np.load(bench / f"{p.match}.npz", allow_pickle=True)["gt"])
        gt = swap_codes(gt_cache[p.match]) if p.swap else gt_cache[p.match]
        target = turnover_targets(gt, half_window)
        frames = np.concatenate([p.frames_vis, p.frames_hid]).astype(np.int64)
        xs.append(np.concatenate([p.x_vis, p.x_hid]).astype(np.float32))
        ys.append(target[frames])
        ms.append(np.full(len(frames), p.match, dtype=object))
        ss.append(np.full(len(frames), p.source, dtype=object))
        ws.append(np.full(len(frames), pff_weight if p.source == "pff" else 1.0))
    return (np.concatenate(xs), np.concatenate(ys), np.concatenate(ms), np.concatenate(ss),
            np.concatenate(ws), parts[0].names)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bench-dir", default=str(DEFAULT_BENCH))
    ap.add_argument("--config", default=None)
    ap.add_argument("--training-pickle", required=True)
    ap.add_argument("--pff-from", default=None, help="pickle whose PFF parts are added (weight --pff-weight)")
    ap.add_argument("--no-pff", action="store_true")
    ap.add_argument("--pff-weight", type=float, default=None)
    ap.add_argument("--lomo", action="store_true")
    ap.add_argument("--pff-folds", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--max-pff", type=int, default=20)
    ap.add_argument("--neg-keep", type=int, default=4,
                    help="keep every k-th 'no change' row (weight x k) to speed up training; 1 = all rows")
    ap.add_argument("--jobs", nargs="*", default=None)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)

    from mtah2027.models.specialist_c import SpecialistCParams, save_specialist_c, train_specialist_c

    sp = SpecialistCParams.from_config(args.config)
    pff_w = sp.pff_weight if args.pff_weight is None else args.pff_weight
    bench = Path(args.bench_dir)
    ts = pickle.loads(Path(args.training_pickle).read_bytes())
    parts = [p for p in ts.parts if p.source == "idsse"]
    if args.pff_from and not args.no_pff:
        base = pickle.loads(Path(args.pff_from).read_bytes())
        keep = sorted({p.match for p in base.parts if p.source == "pff"})[:args.max_pff]
        parts += [p for p in base.parts if p.source == "pff" and p.match in keep]
        del base
    del ts
    gc.collect()
    x, y, match, source, w, names = rows_with_targets(parts, bench, sp.half_window, pff_w)
    del parts
    gc.collect()
    print(f"rows {len(y)}: classes {[int((y == k).sum()) for k in range(3)]}, "
          f"pff rows {int((source == 'pff').sum())}", flush=True)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    wt = None if (w == 1.0).all() else w

    def fit(mask, path):
        if path.is_file():
            return
        t0 = time.perf_counter()
        idx = np.flatnonzero(mask)
        neg = y[idx] == 0
        keep = ~neg | ((np.cumsum(neg) % max(args.neg_keep, 1)) == 0)
        idx = idx[keep]
        wrow = (np.ones(len(y)) if wt is None else wt.copy())
        wrow[(y == 0)] *= max(args.neg_keep, 1)
        model = train_specialist_c(x[idx], y[idx], match[idx], names, params=sp, weights=wrow[idx])
        save_specialist_c(model, path)
        print(f"{path.name}: {len(idx)} rows, T={model.classifier.temperature:.3f}, "
              f"{time.perf_counter() - t0:.0f} s", flush=True)

    if args.lomo:
        stems = args.jobs or [f.stem for f in sorted(bench.glob("idsse_*.npz"))]
        for stem in stems:
            fit(match != stem, out / f"{stem}.json")
    if args.pff_folds:
        pff = sorted(bench.glob("pff_*.npz"))[:args.max_pff]
        for fold in (int(j) for j in args.jobs) if (args.jobs and not args.lomo) else range(args.folds):
            held = [f.stem for i, f in enumerate(pff) if i % args.folds == fold]
            fit(~np.isin(match, held), out / f"fold{fold}.json")
    if args.full:
        fit(np.ones(len(y), bool), out / "unseen.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
