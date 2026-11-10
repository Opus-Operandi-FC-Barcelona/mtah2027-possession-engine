#!/usr/bin/env python
"""Deterministic end-to-end submission generator: tracking bundle -> ``<team>.csv`` (Issue #28, waves 1-2).

Examples::

    python scripts/generate_submission.py --data-dir data --team OpusOperandi
    python scripts/generate_submission.py --synthetic --out /tmp/sub_run1.csv     # CI determinism check

Only aggregate information is printed (row count, label shares, number and mean duration of possessions,
SHA256, runtime) -- never per-frame rows -- so the command is safe to run on the real match.  Without any
optional module or input the route degrades step by step (see ``mtah2027.pipeline``) and still writes a
valid file.  The output is byte-identical between runs: seeds are fixed, ``PYTHONHASHSEED`` is pinned (the
script re-executes itself once if needed), and every decoder input is rounded to 1e-6.
"""
from __future__ import annotations

import argparse
import hashlib
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
    """Re-run this script once with ``PYTHONHASHSEED=0`` so that hashing is reproducible in-process."""
    if os.environ.get("PYTHONHASHSEED") != str(SEED) and not os.environ.get("MTAH_NO_REEXEC"):
        env = dict(os.environ, PYTHONHASHSEED=str(SEED), MTAH_NO_REEXEC="1")
        raise SystemExit(subprocess.call([sys.executable] + sys.argv, env=env))


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default=None,
                    help="bundle root with tactical/ (default: $MTAH_DATA_DIR or ./data)")
    ap.add_argument("--out", default=None, help="output CSV (default: outputs/<team>.csv)")
    ap.add_argument("--team", default=None, help="team name (default: team_name of the configuration)")
    ap.add_argument("--config", default=None, help="YAML configuration (default: configs/default.yaml)")
    ap.add_argument("--synthetic", action="store_true",
                    help="generate the full-length synthetic match (fixed seed) in a temp dir and run on it")
    ap.add_argument("--synthetic-frames", type=int, default=None,
                    help="with --synthetic: use a short match of this many frames instead (tests)")
    ap.add_argument("--specialists", choices=("auto", "train", "load", "off"), default="auto",
                    help="learned emissions (Issues #20-#22): train = fit on the external matches at run "
                         "time; load = committed models/specialists.json; auto = train if the external "
                         "matches are on disk, else load, else heuristic emissions (with --synthetic, auto "
                         "never trains); off = wave-1 heuristic emissions")
    ap.add_argument("--bench-dir", default=None,
                    help="external benchmark .npz directory (default: $MTAH_BENCH_DIR or the sibling "
                         "mtah2027-external-benchmarks/cache)")
    ap.add_argument("--model-path", default=None, help="specialists file (default: models/specialists.json)")
    ap.add_argument("--no-cache", action="store_true", help="do not cache the parsed ATD under cache/")
    ap.add_argument("--quiet", action="store_true", help="do not print the per-stage log")
    return ap.parse_args(argv)


def _summary(labels, blocks, seconds: float, out: Path, digest: str) -> None:
    import numpy as np

    from mtah2027.pipeline import summarize_labels

    s = summarize_labels(labels, [blocks.half1, blocks.half2_play], blocks.fps)
    codes = np.asarray(labels)
    print(f"rows: {len(codes)} (+ header)   file: {out}")
    print("labels over all rows: " + "  ".join(f"{name}={(codes == k).mean():.3f}"
                                               for k, name in enumerate("HADX")))
    print(f"graded play frames: {s['graded_frames']}   H={s['share_H']:.3f} A={s['share_A']:.3f} "
          f"D={s['share_D']:.3f}")
    print(f"possessions: {s['n_possessions']}   mean duration: {s['mean_possession_s']:.2f} s")
    print(f"runtime: {seconds:.1f} s")
    print(f"SHA256: {digest}")


def main(argv=None) -> int:
    args = parse_args(argv)
    _pin_hash_seed()

    from mtah2027.config import MatchConfig, load_config, set_global_seeds
    from mtah2027.models.specialist_fusion import resolve_specialists
    from mtah2027.pipeline import resolve_config, run_pipeline
    from mtah2027.postprocessing.submission_contract import write_submission_csv
    from mtah2027.validation import eval_bridge as eb

    set_global_seeds(SEED)
    cfg_path, _ = resolve_config(args.config)
    config = load_config(cfg_path)
    team = args.team or config.team_name
    out = Path(args.out) if args.out else REPO_ROOT / "outputs" / f"{team}.csv"
    t0 = time.perf_counter()

    tmp = None
    gt_path = None
    blocks = config.match
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
            blocks = MatchConfig(n_video_frames=spec.n_frames, half1=spec.half1, half2_dead=spec.half2_dead,
                                 half2_play=spec.half2_play)
            gt_path = data_dir / "synthetic_gt.csv"
            print(f"synthetic match: {spec.n_frames} frames (seed {SEED}), generated in "
                  f"{time.perf_counter() - t0:.1f} s")
            cache_dir = None
        else:
            data_dir = args.data_dir or os.environ.get("MTAH_DATA_DIR") or str(REPO_ROOT / "data")
            cache_dir = None if args.no_cache else REPO_ROOT / "cache"

        say = (lambda m: None) if args.quiet else print
        # --synthetic (CI / tests): never train at run time, so the run is fast and independent of local data
        t_models = time.perf_counter()
        specialists = resolve_specialists(
            args.specialists, bench_dir=args.bench_dir, model_path=args.model_path, config_path=cfg_path,
            prefer_file=args.synthetic, log=lambda m: say(f"[specialists] {m}"))
        print(f"specialists: {specialists.source if specialists else 'none (heuristic emissions)'} "
              f"({time.perf_counter() - t_models:.1f} s)")
        if specialists is not None and specialists.training:
            t = specialists.training
            print(f"  trained on {len(t.get('matches', []))} external matches, "
                  f"{t.get('rows_visible', 0)} visible-ball / {t.get('rows_hidden', 0)} hidden-ball rows")
        result = run_pipeline(data_dir, cfg_path, blocks, cache_dir=cache_dir,
                              log=(lambda m: None) if args.quiet else None, specialists=specialists)
        write_submission_csv(out, result.labels, n_expected=blocks.n_video_frames)
        check = eb.validate_submission_file(out, blocks.n_video_frames)
        digest = hashlib.sha256(out.read_bytes()).hexdigest()
        core = result.diagnostics.get("core", {})
        print(f"emissions: {core.get('emission_source', 'n/a')}")
        pseudo = core.get("specialists", {}).get("pseudo_labels", {}) if core else {}
        if pseudo:
            print(f"pseudo-labels (target match, algorithmic): {pseudo.get('blocks', 0)} masked blocks, "
                  f"{pseudo.get('rows', 0)} rows, adapted={pseudo.get('adapted')}")
        print(f"route: {result.diagnostics['route']}   validate_submission: "
              f"{'OK' if check else 'FAIL: ' + str(check.reason)}")
        _summary(result.labels, blocks, time.perf_counter() - t0, out, digest)
        if gt_path is not None:
            gt = eb.read_submission_codes(gt_path, blocks.n_video_frames)
            r = eb.evaluate_arrays(gt, result.labels)
            print(f"synthetic ground truth: spellF1={r['spell_f1_mean']:.4f}  macroF1={r['macro_f1']:.4f}  "
                  f"possessions pred/gt={r['n_pred_spells']}/{r['n_gt_spells']}")
        return 0 if check else 1
    finally:
        if tmp is not None:
            tmp.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
