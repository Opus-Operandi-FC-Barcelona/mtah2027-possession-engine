#!/usr/bin/env python
"""KPI dashboard CLI (Issue #27): seven quality KPIs, aggregate numbers only, table + JSON.

Examples::

    python scripts/run_kpi_dashboard.py --synthetic                       # full-length synthetic match
    python scripts/run_kpi_dashboard.py --data-dir data --submission "outputs/Opus Operandi.csv"
    python scripts/run_kpi_dashboard.py --data-dir data --external-lomo   # also the leave-one-match-out route

Only counts, shares, scores, hashes and timings are printed and written to ``outputs/kpi_report.json``
(safe to paste).  KPI 5 writes ``outputs/audit_transitions.csv`` (git-ignored derived data; only the
count is printed).
KPI 4 (public 5 minutes) runs only on real data and is labelled
"pipeline check only - not for model selection".
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _extra in (REPO_ROOT / "src", REPO_ROOT, REPO_ROOT / "scripts"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

SEED = 0


def _pin_hash_seed() -> None:
    if os.environ.get("PYTHONHASHSEED") != str(SEED) and not os.environ.get("MTAH_NO_REEXEC"):
        env = dict(os.environ, PYTHONHASHSEED=str(SEED), MTAH_NO_REEXEC="1")
        raise SystemExit(subprocess.call([sys.executable] + sys.argv, env=env))


def add_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--data-dir", default=None, help="bundle root (default: $MTAH_DATA_DIR or ./data)")
    ap.add_argument("--synthetic", action="store_true", help="run on the full-length synthetic match")
    ap.add_argument(
        "--synthetic-frames", type=int, default=None, help="with --synthetic: short match (tests)"
    )
    ap.add_argument("--submission", default=None, help="existing submission CSV (default: run the pipeline)")
    ap.add_argument(
        "--bench-dir",
        default=None,
        help="external benchmark .npz directory (default: $MTAH_BENCH_DIR or the sibling "
        "mtah2027-external-benchmarks/cache if it exists)",
    )
    ap.add_argument("--skip-external", action="store_true", help="skip KPI 3 (no external benchmark)")
    ap.add_argument(
        "--external-lomo",
        action="store_true",
        help="KPI 3/7: also score the specialists trained leave-one-match-out (slower)",
    )
    ap.add_argument("--specialists", choices=("auto", "train", "load", "off"), default="auto")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out-json", default=None, help="default: outputs/kpi_report.json")
    ap.add_argument("--audit-csv", default=None, help="default: outputs/audit_transitions.csv")


def run(args: argparse.Namespace) -> int:
    from mtah2027.config import MatchConfig
    from mtah2027.models.specialist_fusion import find_bench_dir, resolve_specialists
    from mtah2027.pipeline import resolve_config
    from mtah2027.validation import eval_bridge as eb
    from mtah2027.validation.kpi_dashboard import format_report, run_dashboard

    cfg_path, _ = resolve_config(args.config)
    outputs = REPO_ROOT / "outputs"
    out_json = Path(args.out_json) if args.out_json else outputs / "kpi_report.json"
    audit = Path(args.audit_csv) if args.audit_csv else outputs / "audit_transitions.csv"
    bench = find_bench_dir(args.bench_dir)
    t0 = time.perf_counter()
    tmp = None
    truth = None
    try:
        if args.synthetic:
            import generate_synthetic_fixtures as synth

            tmp = tempfile.TemporaryDirectory(prefix="mtah_synth_")
            data_dir = Path(tmp.name)
            if args.synthetic_frames:
                info = synth.generate_synthetic_match(data_dir, seed=SEED, n_frames=args.synthetic_frames)
            else:
                info = synth.generate_synthetic_match(data_dir, seed=SEED, full_length=True)
            spec = info["spec"]
            blocks = MatchConfig(
                n_video_frames=spec.n_frames,
                half1=spec.half1,
                half2_dead=spec.half2_dead,
                half2_play=spec.half2_play,
            )
            truth = eb.read_submission_codes(data_dir / "synthetic_gt.csv", spec.n_frames)
            cache_dir = None
            real = False
            print(f"synthetic match: {spec.n_frames} frames (seed {SEED}), {time.perf_counter() - t0:.1f} s")
        else:
            data_dir = args.data_dir or os.environ.get("MTAH_DATA_DIR") or str(REPO_ROOT / "data")
            from mtah2027.config import load_config

            blocks = load_config(cfg_path).match
            cache_dir = REPO_ROOT / "cache"
            real = True
        specialists = resolve_specialists(
            args.specialists,
            bench_dir=bench,
            model_path=args.model_path,
            config_path=cfg_path,
            train_fallback=not args.synthetic,
            log=lambda m: print(f"[specialists] {m}"),
        )
        print(f"specialists: {specialists.source if specialists else 'none (heuristic emissions)'}")
        report = run_dashboard(
            data_dir=data_dir,
            blocks=blocks,
            specialists=specialists,
            submission_path=args.submission,
            config_path=cfg_path,
            bench_dir=bench,
            skip_external=args.skip_external,
            external_lomo=args.external_lomo,
            real_data=real,
            out_json=out_json,
            audit_path=audit,
            cache_dir=cache_dir,
            truth=truth,
            log=print,
        )
        print(format_report(report))
        print(f"report: {out_json}")
        failed = [k for k, v in report["kpis"].items() if v.get("status") == "FAIL"]
        return 1 if failed else 0
    finally:
        if tmp is not None:
            tmp.cleanup()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_arguments(ap)
    args = ap.parse_args(argv)
    _pin_hash_seed()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
