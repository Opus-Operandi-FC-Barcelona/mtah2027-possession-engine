#!/usr/bin/env python
"""Train Specialists A and B on the degraded EXTERNAL open matches and save them (Issues #20-#22).

Training data: the open Bundesliga (IDSSE / DFL, Bassek et al. 2025) matches of the benchmark repository
``mtah2027-external-benchmarks`` (``cache/idsse_*.npz``); their per-frame H / A / D labels are produced
algorithmically from the provider's ball-status and possession channels (see that repository's
``docs/LABELLING_RULES.md``).  Each match is degraded with the realistic degrader of Issue #19 for several
seeds and also used with home / away swapped.  Nothing from BAR-ATH is read.

Usage::

    python scripts/train_specialists.py --bench-dir ../mtah2027-external-benchmarks/cache \\
        --out models/specialists.json

The run is deterministic (LightGBM ``deterministic=True``, ``num_threads=1``, ``seed=0``): equal inputs give
a byte-identical file.  ``generate_submission.py`` retrains at run time when the external matches are on
disk and otherwise loads this file.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _extra in (REPO_ROOT / "src", REPO_ROOT):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bench-dir", default=None, help="directory with the benchmark .npz files")
    ap.add_argument("--out", default=str(REPO_ROOT / "models" / "specialists.json"))
    ap.add_argument("--config", default=None)
    ap.add_argument("--extra-sources", nargs="*", default=None, metavar="PREFIX",
                    help="Issue #35: extra training matches, e.g. skillcorner_ pff_ (default: config)")
    ap.add_argument("--pff-max-matches", type=int, default=None, help="first N PFF matches by name (0 = all)")
    ap.add_argument("--sources", nargs="*", default=None,
                    help="train only on these sources (idsse skillcorner pff), e.g. pff for a PFF-only model")
    ap.add_argument("--source-weight", nargs="*", default=[], metavar="SOURCE=W",
                    help="sample weight per source")
    ap.add_argument("--drop-dead", nargs="*", default=[], help="sources whose D rows are left out")
    args = ap.parse_args(argv)
    if os.environ.get("PYTHONHASHSEED") != "0" and not os.environ.get("MTAH_NO_REEXEC"):
        env = dict(os.environ, PYTHONHASHSEED="0", MTAH_NO_REEXEC="1")
        return subprocess.call([sys.executable] + sys.argv, env=env)

    from mtah2027.config import set_global_seeds
    from mtah2027.models import specialist_fusion as sf

    set_global_seeds(0)
    bench = sf.find_bench_dir(args.bench_dir)
    if bench is None:
        print("no external benchmark matches found (--bench-dir / $MTAH_BENCH_DIR)")
        return 1
    t0 = time.perf_counter()
    import dataclasses

    from mtah2027.models.specialist_data import TrainingDataParams, build_training_set

    dp = TrainingDataParams.from_config(args.config)
    if args.extra_sources is not None:
        dp = dataclasses.replace(dp, extra_sources=tuple(args.extra_sources))
    if args.pff_max_matches is not None:
        dp = dataclasses.replace(dp, pff_max_matches=args.pff_max_matches)
    log = lambda m: print(m, flush=True)  # noqa: E731
    ts = build_training_set(bench, params=dp, config_path=args.config, log=log)
    if args.sources is not None or args.source_weight or args.drop_dead:
        weights = {k: float(v) for k, _, v in (i.partition("=") for i in args.source_weight)}
        ts = ts.select(args.sources if args.sources is not None else sorted({p.source for p in ts.parts}),
                       weights or ts.weights, args.drop_dead or ts.drop_dead)
    spec = sf.train_specialists(bench, config_path=args.config, training=ts, log=log)
    out = sf.save_specialists(spec, args.out)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    t = spec.training
    print(f"trained on {len(t['matches'])} matches x {len(t['degradation_seeds'])} seeds (+ swapped copies): "
          f"{t['rows_visible']} visible-ball rows, {t['rows_hidden']} hidden-ball rows")
    print(f"temperatures: A {t['temperature_a']:.3f}  B {t['temperature_b']:.3f}")
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KiB)   SHA256: {digest}   "
          f"({time.perf_counter() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
