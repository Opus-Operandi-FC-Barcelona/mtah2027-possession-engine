#!/usr/bin/env python
"""Issue #44: train and cache the cross-validation specialists of one training variant (external data only).

Two kinds of jobs share the training recipe of ``eval_external.py --specialists lomo`` / ``eval_pff_heldout.py``:

* ``--lomo``: one model per held-out IDSSE match (trained on the other matches), saved to ``--out-dir/<match>.json``;
* ``--full``: one model trained on every match of the pickle (Issue #55: the all-data model scored on the unseen PFF);
* ``--pff-folds``: one model per fold of the first ``--max-matches`` PFF matches (fold = sorted index modulo
  ``--folds``), saved to ``--out-dir/fold<k>.json``.

    python scripts/train_cv_models.py --training-pickle t.pkl --pff-from t_legacy.pkl --pff-weight 0.5 \
        --lomo --jobs idsse_J03WMX idsse_J03WN1 --out-dir lomo_calib

The models are the inputs of ``scripts/decoder_sweep.py``.  Only aggregates are printed.  Nothing from BAR-ATH.
"""
# ruff: noqa: E501
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _extra in (REPO_ROOT / "src", REPO_ROOT):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

DEFAULT_BENCH = REPO_ROOT.parent / "mtah2027-external-benchmarks" / "cache"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bench-dir", default=str(DEFAULT_BENCH))
    ap.add_argument("--config", default=None)
    ap.add_argument("--training-pickle", required=True)
    ap.add_argument("--pff-from", default=None, help="pickle whose PFF rows replace those of the training pickle")
    ap.add_argument("--pff-weight", type=float, default=0.5)
    ap.add_argument("--lomo", action="store_true")
    ap.add_argument("--pff-folds", action="store_true")
    ap.add_argument("--full", action="store_true", help="Issue #55: one all-data model, saved to --out-dir/full.json")
    ap.add_argument("--seed", type=int, default=None, help="Issue #60: LightGBM seed of the models (bag member)")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--max-matches", type=int, default=20)
    ap.add_argument("--jobs", nargs="*", default=None, help="IDSSE match stems (lomo) or fold indices (pff-folds)")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)

    from mtah2027.models import specialist_fusion as sf

    ts0 = pickle.loads(Path(args.training_pickle).read_bytes())
    if args.pff_from:
        base = pickle.loads(Path(args.pff_from).read_bytes())
        ts0.parts = [p for p in ts0.parts if p.source != "pff"] + [p for p in base.parts if p.source == "pff"]
        ts0.weights = dict(base.weights)
    ts = ts0.select(["idsse", "pff"], {**ts0.weights, "pff": args.pff_weight}, ts0.drop_dead)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bench = Path(args.bench_dir)
    if args.lomo:
        jobs = args.jobs or [f.stem for f in sorted(bench.glob("idsse_*.npz"))]
        for stem in jobs:
            path = out / f"{stem}.json"
            if path.is_file():
                continue
            t0 = time.perf_counter()
            spec = sf.train_specialists(bench, config_path=args.config, training=ts, exclude_match=stem, seed=args.seed)
            sf.save_specialists(spec, path)
            print(f"lomo {stem}: {time.perf_counter() - t0:.0f} s", flush=True)
    if args.full and not (out / "full.json").is_file():
        t0 = time.perf_counter()
        spec = sf.train_specialists(bench, config_path=args.config, training=ts, seed=args.seed)
        sf.save_specialists(spec, out / "full.json")
        print(f"full model ({len(ts.matches)} matches): {time.perf_counter() - t0:.0f} s", flush=True)
    if args.pff_folds:
        pff = sorted(bench.glob("pff_*.npz"))[:args.max_matches]
        folds = [int(j) for j in args.jobs] if args.jobs else list(range(args.folds))
        for fold in folds:
            path = out / f"fold{fold}.json"
            if path.is_file():
                continue
            held = [f.stem for i, f in enumerate(pff) if i % args.folds == fold]
            t0 = time.perf_counter()
            sub = ts.select(["idsse", "pff"], ts.weights, ts.drop_dead, exclude_matches=held)
            spec = sf.train_specialists(bench, config_path=args.config, training=sub, seed=args.seed)
            sf.save_specialists(spec, path)
            print(f"pff fold {fold} ({len(held)} matches held out): {time.perf_counter() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
