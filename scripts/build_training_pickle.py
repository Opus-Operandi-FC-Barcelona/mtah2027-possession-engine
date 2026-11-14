#!/usr/bin/env python
"""Issue #48: build a TrainingSet pickle for one degrader setting, reusing the PFF rows of another pickle.

    python scripts/build_training_pickle.py --section external_degrader --affinity 7 --out t_cal_or7.pkl \
        --pff-from t_legacy.pkl

The IDSSE rows are degraded with ``--section`` / ``--affinity`` (degrader of the training rows); the PFF rows do
not depend on the degrader, so they are copied from ``--pff-from`` (otherwise they are built).  The pickles are
inputs of ``eval_external.py --training-pickle`` and ``eval_pff_heldout.py --training-pickle``.  Nothing from
BAR-ATH is read.
"""
# ruff: noqa: E501
from __future__ import annotations

import argparse
import dataclasses
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _extra in (REPO_ROOT / "src", REPO_ROOT):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from mtah2027.models.specialist_data import TrainingDataParams, build_training_set  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bench-dir", default=str(REPO_ROOT.parent / "mtah2027-external-benchmarks" / "cache"))
    ap.add_argument("--section", default="external_degrader_legacy")
    ap.add_argument("--affinity", type=float, default=1.0)
    ap.add_argument("--pff-from", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--idsse-only", action="store_true", help="no PFF rows (merge them later: gap_curve --pff-from)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    dp = dataclasses.replace(TrainingDataParams.from_config(args.config), degrader_section=args.section,
                             dead_affinity=args.affinity)
    if args.pff_from or args.idsse_only:
        ts = build_training_set(args.bench_dir, params=dataclasses.replace(dp, extra_sources=()),
                                config_path=args.config, log=print)
        if args.pff_from:
            base = pickle.loads(Path(args.pff_from).read_bytes())
            ts.parts += [p for p in base.parts if p.source == "pff"]
            ts.weights = dict(base.weights)
    else:
        ts = build_training_set(args.bench_dir, params=dp, config_path=args.config, log=print)
    Path(args.out).write_bytes(pickle.dumps(ts))
    print(f"wrote {args.out}: {len(ts.parts)} parts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
