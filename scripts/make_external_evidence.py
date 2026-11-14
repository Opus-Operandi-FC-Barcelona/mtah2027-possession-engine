#!/usr/bin/env python
"""Write ``reports/external_evidence.json``: the external (open data) evidence behind KPI 7's route decision.

Scores the heuristic route and the leave-one-match-out specialists on the external matches with a complete
ball channel (never BAR-ATH) in two settings -- the realistic Issue #19 degradation over several seeds and the
undegraded complete ball -- and stores only aggregate numbers, the seeds, the date and the git sha.  The KPI
dashboard reads this file whenever ``--external-lomo`` is not run, so the route decision does not need a
retraining of tens of minutes.

    python scripts/make_external_evidence.py --bench-dir ../mtah2027-external-benchmarks/cache
    python scripts/make_external_evidence.py --cache models_cache/    # reuse/save the LOMO specialists
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT / "src", REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import eval_external as ee  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "reports" / "external_evidence.json"


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def summarize(rows: list[dict], key: str) -> dict:
    return {
        "spell_f1": float(np.mean([r[key]["spell_f1"] for r in rows])),
        "macro_f1": float(np.mean([r[key]["macro_f1"] for r in rows])),
        "possessions_pred_mean": float(np.mean([r[key]["n_pred_spells"] for r in rows])),
        "possessions_true_mean": float(np.mean([r[key]["n_gt_spells"] for r in rows])),
        "n_runs": len(rows),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bench-dir", default=str(ee.DEFAULT_BENCH_DIR))
    ap.add_argument("--config", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2], help="degrader seeds")
    ap.add_argument("--cache", default=None, help="directory caching the leave-one-match-out specialists")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args(argv)

    from mtah2027.models import specialist_fusion as sf
    from mtah2027.models.specialist_data import build_training_set

    files = sorted(Path(args.bench_dir).glob("*.npz"))
    files = [f for f in files if np.isfinite(np.load(f, allow_pickle=True)["ball_xy"]).all()]
    if not files:
        print(f"no complete-ball benchmark files in {args.bench_dir}")
        return 1
    fusion = sf.FusionParams.from_config(args.config)
    training = []                      # rows are only built when a leave-one-match-out model is not cached
    cache = Path(args.cache) if args.cache else None
    models: dict = {}

    def specialists_for(stem: str):
        if stem not in models:
            path = cache / f"{stem}.json" if cache else None
            if path is not None and path.is_file():
                models[stem] = dataclasses.replace(sf.load_specialists(path, args.config), fusion=fusion)
                if models[stem].durations is None:
                    from mtah2027.decoding.duration_model import fit_from_bench

                    models[stem] = dataclasses.replace(models[stem], durations=fit_from_bench(
                        args.bench_dir, exclude=stem))
            else:
                if not training:
                    training.append(build_training_set(args.bench_dir, config_path=args.config,
                                                       log=lambda m: None))
                spec = sf.train_specialists(args.bench_dir, config_path=args.config, training=training[0],
                                            exclude_match=stem)
                models[stem] = dataclasses.replace(spec, fusion=fusion)
                if path is not None:
                    sf.save_specialists(models[stem], path)
            print(f"  specialists without {stem} ready", flush=True)
        return models[stem]

    settings = {}
    for name, degrade, seeds in (("degraded", True, list(args.seeds)), ("complete_ball", False, [0])):
        rows = []
        for seed in seeds:
            for i, f in enumerate(files):
                rows.append(ee.evaluate_match(f, 0.0, seed + i, args.config, False, degrade,
                                              specialists_for(f.stem)))
        settings[name] = {
            "degrader": "realistic (Issue #19)" if degrade else "none",
            "seeds": seeds, "n_matches": len(files),
            "routes": {"heuristic": summarize(rows, "engine"),
                       "specialists_lomo": summarize(rows, "specialists")},
        }
        r = settings[name]["routes"]
        print(f"{name}: heuristic {r['heuristic']['spell_f1']:.4f}/{r['heuristic']['macro_f1']:.4f}  "
              f"specialists_lomo {r['specialists_lomo']['spell_f1']:.4f}/"
              f"{r['specialists_lomo']['macro_f1']:.4f}",
              flush=True)

    evidence = {
        "schema": 1,
        "note": "external open data only (no BAR-ATH); aggregates of leave-one-match-out evaluations",
        "date": dt.date.today().isoformat(),
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--", "src", "configs")),
        "matches": [f.stem for f in files],
        "settings": settings,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
