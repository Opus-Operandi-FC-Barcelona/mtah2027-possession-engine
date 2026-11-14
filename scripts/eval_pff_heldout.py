#!/usr/bin/env python
"""Validation A of Issue #35 / #38: specialists scored on held-out PFF matches (real ball occlusions).

The first ``--max-matches`` PFF matches by name are split into ``--folds`` folds by sorted index modulo the
number of folds.  For every fold the specialists are trained on the training set minus the matches of that
fold (IDSSE degraded by the configured degrader + the other PFF matches, per ``--sources`` /
``--source-weight``) and the held-out matches are processed as unlabelled targets (no extra degradation: the
PFF ball is truly missing where it was not visible).  Only aggregates are printed.  Nothing from BAR-ATH.

    python scripts/eval_pff_heldout.py --sources idsse pff --source-weight pff=0.5 --training-pickle t.pkl
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for _extra in (REPO_ROOT / "src", REPO_ROOT, REPO_ROOT / "scripts"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import eval_external as ee  # noqa: E402

from mtah2027.validation import dropout_strata as ds  # noqa: E402


def _print_strata(rows) -> None:
    pooled = ds.pool([r["strata"] for r in rows])
    total = pooled["lt20"]["frames"] + pooled["ge20"]["frames"]
    for line in ds.format_table(ds.summarize(pooled), total):
        print("  " + line)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bench-dir", default=str(ee.DEFAULT_BENCH_DIR))
    ap.add_argument("--config", default=None)
    ap.add_argument("--degrader-section", default=None,
                    help="config section of the degrader of the IDSSE training rows (default: configuration)")
    ap.add_argument("--training-pickle", default=None)
    ap.add_argument("--sources", nargs="*", default=None)
    ap.add_argument("--source-weight", nargs="*", default=[], metavar="SOURCE=W")
    ap.add_argument("--max-matches", type=int, default=20, help="first N PFF matches by name")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--unseen-model", default=None,
                    help="Issue #42: also score this committed model file on the last --unseen-n PFF matches "
                         "(never used in any training)")
    ap.add_argument("--unseen-n", type=int, default=8, help="last N PFF matches by name")
    ap.add_argument("--no-folds", action="store_true", help="skip the 4-fold evaluation")
    ap.add_argument("--sim-tags", choices=("clean", "low", "medium", "high"), default=None,
                    help="Issue #43: feed a simulated Smart Tagging (derived from the true labels)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    from mtah2027.models import specialist_fusion as sf
    from mtah2027.models.specialist_data import TrainingDataParams, build_training_set

    pff = sorted(Path(args.bench_dir).glob("pff_*.npz"))[:args.max_matches]
    if not pff:
        print("no PFF matches found")
        return 1
    pkl = Path(args.training_pickle) if args.training_pickle else None
    if pkl is not None and pkl.is_file():
        ts = pickle.loads(pkl.read_bytes())
    else:
        dp = TrainingDataParams.from_config(args.config)
        if args.degrader_section:
            dp = dataclasses.replace(dp, degrader_section=args.degrader_section)
        ts = build_training_set(args.bench_dir, params=dp, config_path=args.config, log=lambda m: None)
        if pkl is not None:
            pkl.write_bytes(pickle.dumps(ts))
    weights = {k: float(v) for k, _, v in (i.partition("=") for i in args.source_weight)}
    sources = args.sources if args.sources is not None else sorted({p.source for p in ts.parts})
    fusion = sf.FusionParams.from_config(args.config)
    rows = []
    for fold in range(0 if args.no_folds else args.folds):
        held = [f for i, f in enumerate(pff) if i % args.folds == fold]
        sub = ts.select(sources, weights or ts.weights, ts.drop_dead, exclude_matches=[f.stem for f in held])
        trained = sf.train_specialists(args.bench_dir, config_path=args.config, training=sub)
        spec = dataclasses.replace(trained, fusion=fusion)
        for f in held:
            row = ee.evaluate_match(f, 0.0, 0, args.config, False, False, spec, sim_tags=args.sim_tags)
            rows.append({"match": f.stem, "fold": fold, **row["specialists"]})
        print(f"fold {fold} done ({len(held)} matches)", flush=True)
    if rows:
        sp = np.mean([r["spell_f1"] for r in rows])
        mf = np.mean([r["macro_f1"] for r in rows])
        print(f"A (PFF held-out, {len(rows)} matches): spellF1 {sp:.4f}  macroF1 {mf:.4f}  possessions "
              f"predicted {sum(r['n_pred_spells'] for r in rows)} "
              f"(true {sum(r['n_gt_spells'] for r in rows)})")
        _print_strata(rows)
    if args.unseen_model:
        unseen = sorted(Path(args.bench_dir).glob("pff_*.npz"))[-args.unseen_n:]
        spec = sf.load_specialists(args.unseen_model, args.config)
        urows = [ee.evaluate_match(f, 0.0, 0, args.config, False, False, spec)["specialists"] for f in unseen]
        sp_u = np.mean([r["spell_f1"] for r in urows])
        mf_u = np.mean([r["macro_f1"] for r in urows])
        print(f"A-unseen ({len(urows)} PFF matches never used in training): spellF1 {sp_u:.4f}  "
              f"macroF1 {mf_u:.4f}")
        _print_strata(urows)
        if args.json:
            Path(str(args.json) + ".unseen.json").write_text(json.dumps(urows, indent=1), encoding="utf-8")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
