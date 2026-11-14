#!/usr/bin/env python
"""Why does a route under- or over-segment possessions?  Aggregate diagnostics on the external matches.

For every external match (``--degrade`` as in ``eval_external.py``) and route (``engine`` = heuristic,
``specialists`` = leave-one-match-out learned) it reports, with the official metric's own helpers:

* predicted vs true possession counts and the spellF1 / macroF1;
* missed vs spurious **transitions** (tolerance ``--tol`` frames, matched on the destination label like the
  official scorer) by type: ``H->A``, ``A->H``, ``->D`` (an alive team loses the ball to a stoppage) and
  ``D->`` (restart);
* how the true possessions are lost: *merged* (the prediction holds one same-team spell across a true stoppage
  or across a true opposing spell), *absorbed fast turnover* (true spell shorter than 3 s not matched), and
  the share of the true stoppages ``D`` shorter than 3 s that the prediction does not reproduce.

Only external open data is used and only aggregates are printed.  The learned route re-trains the
leave-one-match-out specialists (cached as JSON under ``--cache`` to make repeated runs cheap).
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT / "src", REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import eval_external as ee  # noqa: E402
import possession_metrics as pm  # noqa: E402

NAMES = {0: "H", 1: "A", 2: "D", 3: "X"}


def transition_type(frm: int, to: int) -> str:
    if to == 2:
        return "->D"
    if frm == 2:
        return "D->"
    return f"{NAMES[frm]}->{NAMES[to]}"


def diagnose(gt: np.ndarray, pred: np.ndarray, fps: float, tol: int,
             visible: np.ndarray | None = None) -> dict:
    runs = pm.valid_runs(gt)
    filled = pm.fill_abstentions(pred, runs)
    gt_tr, pr_tr = pm.transitions(gt, runs), pm.transitions(filled, runs)
    _, missed, spurious = pm.match_transitions(gt_tr, pr_tr, tol, "to")
    by_type = {"missed": collections.Counter(transition_type(f, t) for _, f, t in missed),
               "spurious": collections.Counter(transition_type(f, t) for _, f, t in spurious),
               "gt": collections.Counter(transition_type(f, t) for _, f, t in gt_tr),
               "pred": collections.Counter(transition_type(f, t) for _, f, t in pr_tr)}

    gt_sp, pr_sp = pm.possession_spells(gt, runs), pm.possession_spells(filled, runs)
    overlaps = pm.spell_overlaps(gt_sp, pr_sp)
    matched = pm.matched_gt_flags(overlaps, len(gt_sp), 0.5)
    dur = np.array([(b - a) / fps for a, b, _ in gt_sp])
    # a predicted spell "merges" when it overlaps >= 2 true spells by at least half of each one's length
    cover = collections.defaultdict(list)
    for iou, i, j in overlaps:
        a, b, _ = gt_sp[i]
        pa, pb, _ = pr_sp[j]
        if min(b, pb) - max(a, pa) >= 0.5 * (b - a):
            cover[j].append(i)
    merged_preds = [j for j, ids in cover.items() if len(ids) >= 2]
    merged_gt = sorted({i for j in merged_preds for i in cover[j]})
    # true stoppages (D runs between two alive spells) and whether the prediction reproduces them
    seg = pm.segments(gt, runs)
    short_d = [(a, b) for a, b, k in seg if k == 2 and (b - a) / fps < 3.0]
    reproduced = sum(1 for a, b in short_d if (filled[a:b] == 2).mean() >= 0.5)
    hid = {"total": 0, "missed": 0, "vis_total": 0, "vis_missed": 0}
    if visible is not None:
        miss_set = {int(f) for f, _, _ in missed}
        for f, frm, to in gt_tr:
            if frm in (0, 1) and to in (0, 1) and frm != to:
                key = "vis_" if visible[int(f)] else ""
                hid[key + "total"] += 1
                hid[key + "missed"] += int(int(f) in miss_set)
    out = {
        "hidden_turnovers": hid,
        "n_gt": len(gt_sp), "n_pred": len(pr_sp),
        "gt_short_lt3s": int((dur < 3.0).sum()),
        "gt_short_lt3s_unmatched": int(((dur < 3.0) & ~matched).sum()),
        "gt_unmatched": int((~matched).sum()),
        "pred_spells_merging_2plus_gt": len(merged_preds),
        "gt_spells_inside_a_merged_pred": len(merged_gt),
        "gt_short_D_lt3s": len(short_d), "gt_short_D_lt3s_reproduced": int(reproduced),
        "transitions": {k: dict(v) for k, v in by_type.items()},
    }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bench-dir", default=str(ee.DEFAULT_BENCH_DIR))
    ap.add_argument("--config", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0])
    ap.add_argument("--tol", type=int, default=25, help="transition matching tolerance in frames")
    ap.add_argument("--routes", nargs="*", default=["engine", "specialists"])
    ap.add_argument("--no-degrade", action="store_true")
    ap.add_argument("--cache", default=None, help="directory caching the leave-one-match-out specialists")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    files = sorted(Path(args.bench_dir).glob("*.npz"))
    files = [f for f in files if np.isfinite(np.load(f, allow_pickle=True)["ball_xy"]).all()]
    models = {}
    training = None
    if "specialists" in args.routes:
        import dataclasses

        from mtah2027.decoding.duration_model import DurationParams
        from mtah2027.models import specialist_fusion as sf
        from mtah2027.models.specialist_data import build_training_set

        fusion = sf.FusionParams.from_config(args.config)
        cache = Path(args.cache) if args.cache else None
        training = []          # the training rows are only built when a leave-one-out model is not cached

        def specialists_for(stem: str):
            if stem not in models:
                path = cache / f"{stem}.json" if cache else None
                if path is not None and path.is_file():
                    models[stem] = sf.load_specialists(path, args.config)
                else:
                    if not training:
                        training.append(build_training_set(args.bench_dir, config_path=args.config,
                                                           log=lambda m: None))
                    spec = sf.train_specialists(args.bench_dir, config_path=args.config, training=training[0],
                                                exclude_match=stem)
                    models[stem] = dataclasses.replace(spec, fusion=fusion)
                    if path is not None:
                        cache.mkdir(parents=True, exist_ok=True)
                        sf.save_specialists(models[stem], path)
            if models[stem].durations is None:     # Issue #34: duration table without the held-out match
                from mtah2027.decoding.duration_model import fit_from_bench

                models[stem] = dataclasses.replace(models[stem], durations=fit_from_bench(
                    args.bench_dir, exclude=stem, params=DurationParams.from_config(args.config)))
            return models[stem]
    total = {r: collections.defaultdict(float) for r in args.routes}
    rows = []
    for seed in args.seeds:
        for i, f in enumerate(files):
            spec = specialists_for(f.stem) if training is not None else None
            row = ee.evaluate_match(f, 0.0, seed + i, args.config, False, not args.no_degrade, spec,
                                    return_codes=True)
            codes = row["_codes"]
            for route in args.routes:
                pred = codes["engine"] if route == "engine" else codes["specialists"]
                d = diagnose(codes["gt"], pred, codes["fps"], args.tol, codes.get("visible"))
                d.update({"match": row["match"], "seed": seed, "route": route,
                          "spell_f1": row[route]["spell_f1"], "macro_f1": row[route]["macro_f1"]})
                rows.append(d)
                t = total[route]
                for key in ("n_gt", "n_pred", "gt_short_lt3s", "gt_short_lt3s_unmatched", "gt_unmatched",
                            "pred_spells_merging_2plus_gt", "gt_spells_inside_a_merged_pred",
                            "gt_short_D_lt3s", "gt_short_D_lt3s_reproduced", "spell_f1", "macro_f1"):
                    t[key] += d[key]
                for kind in ("missed", "spurious", "gt", "pred"):
                    for typ, n in d["transitions"][kind].items():
                        t[f"{kind}:{typ}"] += n
                for k, v in d["hidden_turnovers"].items():
                    t["hid_" + k] += v
                t["n"] += 1
    for route in args.routes:
        t = total[route]
        n = t["n"]
        print(f"== {route}: {int(n)} match-seed runs; mean spellF1 {t['spell_f1'] / n:.4f} "
              f"macroF1 {t['macro_f1'] / n:.4f}")
        print(f"   possessions predicted {int(t['n_pred'])} vs true {int(t['n_gt'])}; "
              f"true spells unmatched@0.5 {int(t['gt_unmatched'])}, "
              f"of which <3 s {int(t['gt_short_lt3s_unmatched'])} "
              f"(true <3 s spells: {int(t['gt_short_lt3s'])})")
        print(f"   predicted spells merging >=2 true spells {int(t['pred_spells_merging_2plus_gt'])} "
              f"(covering {int(t['gt_spells_inside_a_merged_pred'])} true spells); true D stoppages <3 s "
              f"reproduced {int(t['gt_short_D_lt3s_reproduced'])}/{int(t['gt_short_D_lt3s'])}")
        ht, hm = t["hid_total"], t["hid_missed"]
        vt, vm = t["hid_vis_total"], t["hid_vis_missed"]
        print(f"   H<->A turnovers with the ball hidden: {int(ht)} true, {int(hm)} missed "
              f"({100 * hm / max(ht, 1):.1f} %); ball visible: {int(vt)} true, {int(vm)} missed "
              f"({100 * vm / max(vt, 1):.1f} %)")
        for typ in ("H->A", "A->H", "->D", "D->"):
            print(f"   {typ:>5}: true {int(t['gt:' + typ]):5d} pred {int(t['pred:' + typ]):5d} "
                  f"missed {int(t['missed:' + typ]):5d} spurious {int(t['spurious:' + typ]):5d}")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
