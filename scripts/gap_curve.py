#!/usr/bin/env python
"""Issue #48: B' (7 IDSSE leave-one-match-out, learned route only) as a function of the degrader's dead_affinity.

For one training configuration (training pickle + sources / weights) the leave-one-match-out specialists are
trained ONCE and the held-out matches are degraded at every ``--eval-affs`` value and every seed; it prints,
per affinity, the mean spellF1 / macroF1 (all matches x seeds), the mean per seed, the predicted possessions
(seed 0) against the true ones and the share of D among the graded frames whose ball is hidden (predicted vs
true).  Only aggregates.  Nothing from BAR-ATH.

    python scripts/gap_curve.py --training-pickle t.pkl --sources idsse pff \
        --eval-affs 1 7 33 --seeds 0 1 2 --combos 0.5:0.5 0:0 --label legacy --json out.json
"""
# ruff: noqa: E501
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for _extra in (REPO_ROOT / "src", REPO_ROOT, REPO_ROOT / "scripts"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import eval_external as ee  # noqa: E402

from mtah2027.decoding.duration_model import DurationParams, fit_from_bench  # noqa: E402
from mtah2027.models import specialist_fusion as sf  # noqa: E402
from mtah2027.models.external_degrader import DegraderParams, degrade_match  # noqa: E402
from mtah2027.pipeline import run_core  # noqa: E402
from mtah2027.validation import eval_bridge as eb  # noqa: E402


def summarise(label: str, key: str, rows: list, seeds) -> str:
    per_seed = {int(s): float(np.mean([r["spell_f1"] for r in rows if r["seed"] == s])) for s in seeds}
    s0 = [r for r in rows if r["seed"] == seeds[0]]
    m = lambda k: float(np.mean([r[k] for r in rows]))  # noqa: E731
    return (f"[{label}] {key}: spellF1 {m('spell_f1'):.4f} macroF1 {m('macro_f1'):.4f} per seed "
            f"{[round(v, 4) for v in per_seed.values()]} poss {sum(r['n_pred_spells'] for r in s0)}/"
            f"{sum(r['n_gt_spells'] for r in s0)} D(hidden) pred {100 * m('pred_d_hidden'):.1f}% "
            f"true {100 * m('true_d_hidden'):.1f}%  D(all) pred {100 * m('pred_d_all'):.1f}% "
            f"true {100 * m('true_d_all'):.1f}%")


def merge(paths, seeds) -> None:
    rows: dict = {}
    for p in paths:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        for k, v in d["curve"].items():
            rows.setdefault(k, []).extend(v)
    for k, v in rows.items():
        print(summarise(Path(paths[0]).stem.split(".")[0], k, v, seeds))


def run_config(args) -> dict:
    import pickle

    from mtah2027.models.frame_classifier import ClassifierParams
    from mtah2027.models.specialist_ball_absent import SpecialistBParams, train_specialist_b
    from mtah2027.models.specialist_ball_present import SpecialistAParams, train_specialist_a

    ts0 = pickle.loads(Path(args.training_pickle).read_bytes())
    if args.pff_from:             # PFF rows do not depend on the degrader: take them from another pickle
        base = pickle.loads(Path(args.pff_from).read_bytes())
        ts0.parts = [p for p in ts0.parts if p.source != "pff"] + [p for p in base.parts if p.source == "pff"]
        ts0.weights = dict(base.weights)
    sources = args.sources if args.sources else sorted({p.source for p in ts0.parts})
    combos = [tuple(float(v) for v in c.split(":")) for c in args.combos]      # (PFF weight A, PFF weight B)
    fusion = sf.FusionParams.from_config(args.config)
    if args.fusion:
        fusion = dataclasses.replace(fusion, **{k: json.loads(v) for k, _, v in (i.partition("=") for i in args.fusion)})
    files = sorted(Path(args.bench_dir).glob("idsse_*.npz"))
    if args.held:
        files = [f for f in files if f.stem in set(args.held)]
    cp = ClassifierParams.from_config(args.config)
    models = {}
    for f in files:
        t0 = time.perf_counter()
        a_cache, b_cache = {}, {}
        dur = fit_from_bench(args.bench_dir, exclude=f.stem, params=DurationParams.from_config(args.config))
        for wa, wb in combos:
            if wa not in a_cache:
                ts = ts0.select([x for x in sources if wa > 0 or x != "pff"], {**ts0.weights, "pff": wa}, ts0.drop_dead)
                a_cache[wa] = train_specialist_a(ts, params=SpecialistAParams.from_config(args.config),
                                                 classifier_params=cp, exclude_match=f.stem)
            if wb not in b_cache:
                ts = ts0.select([x for x in sources if wb > 0 or x != "pff"], {**ts0.weights, "pff": wb}, ts0.drop_dead)
                b_cache[wb] = train_specialist_b(ts, params=SpecialistBParams.from_config(args.config),
                                                 classifier_params=cp, exclude_match=f.stem)
            models[(f.stem, wa, wb)] = sf.SpecialistSet(a=a_cache[wa], b=b_cache[wb], fusion=fusion,
                                                        source="trained", durations=dur)
        print(f"  [{args.label}] trained without {f.stem} ({time.perf_counter() - t0:.0f} s)", flush=True)
    base_p = DegraderParams.from_config(args.config)
    out = {"label": args.label, "args": {k: v for k, v in vars(args).items() if k != "json"}, "curve": {}}
    for aff in args.eval_affs:
        for wa, wb in combos:
            rows = []
            for seed in args.seeds:
                for i, f in enumerate(files):
                    z = np.load(f, allow_pickle=True)
                    players, team = z["players_xy"].astype(np.float32), z["team_of_slot"].astype(np.int8)
                    ball = z["ball_xy"].astype(np.float32)
                    fps, pitch = float(z["fps"]), tuple(float(v) for v in z["pitch"])
                    periods = [(int(a), int(b)) for a, b in z["period_bounds"]]
                    gt = eb.to_codes(z["gt"])
                    dm = degrade_match(players, team, ball, gt, fps=fps, seed=seed + i,
                                       params=dataclasses.replace(base_p, dead_affinity=float(aff)))
                    spec = models[(f.stem, wa, wb)]
                    core = run_core(dm.players_xy, team, dm.ball_xy, play_blocks=periods, fps=fps, pitch=pitch,
                                    config_path=args.config, log=lambda m: None, specialists=spec,
                                    durations=spec.durations)
                    sc = ee.score(gt, core.codes)
                    g = gt != 3
                    hid = g & ~np.isfinite(dm.ball_xy).all(axis=1)
                    rows.append({"seed": seed, **sc, "pred_d_hidden": float((core.codes[hid] == 2).mean()),
                                 "true_d_hidden": float((gt[hid] == 2).mean()),
                                 "pred_d_all": float((core.codes[g] == 2).mean()),
                                 "true_d_all": float((gt[g] == 2).mean())})
            key = f"aff={aff}|wa={wa}|wb={wb}"
            for i, r in enumerate(rows):
                r["match"] = files[i % len(files)].stem
            out["curve"][key] = rows
            if not args.held:
                print(summarise(args.label, key, rows, args.seeds), flush=True)
            if args.json:
                Path(args.json).write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bench-dir", default=str(ee.DEFAULT_BENCH_DIR))
    ap.add_argument("--config", default=None)
    ap.add_argument("--training-pickle")
    ap.add_argument("--held", nargs="*", default=None, help="only these held-out matches (raw rows saved)")
    ap.add_argument("--merge", nargs="*", default=None, help="merge raw-row JSON files and print the table")
    ap.add_argument("--pff-from", default=None, help="pickle whose PFF rows are added")
    ap.add_argument("--sources", nargs="*", default=None)
    ap.add_argument("--combos", nargs="+", default=["0.5:0.5"], metavar="WA:WB",
                    help="PFF weight of Specialist A : of Specialist B (0:0 = IDSSE only, 0.5:0.5 = adopted mixed)")
    ap.add_argument("--fusion", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--eval-affs", type=float, nargs="+", default=[1.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--label", default="run")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    if args.merge:
        merge(args.merge, args.seeds)
    else:
        run_config(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
