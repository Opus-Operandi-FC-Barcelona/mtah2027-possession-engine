#!/usr/bin/env python
# ruff: noqa: E501
"""Issue #72: score the sequence model (TCN) in the B'(33) / B'(1) / A cells of #63 / #66 / #67 / #71.

Same cells and production models as ``decoder_sweep.py`` (7 IDSSE leave-one-match-out matches degraded with the
calibrated degrader at ``dead_affinity`` 33 / 1, eval seeds 0..2; A = the first 20 PFF matches with their real
occlusions), official metric.  Every cell runs the production route (``base``) and the fixed candidate grid of
the preregistration (comment on #72), with the same cached front end / specialist probabilities.  Per variant it
also records the H<->A turnover recall by stratum (ball visible / hidden at the true change) and the number of
predicted turnovers.  External matches only, aggregates only.

    python scripts/sequence_eval.py run --world b33 --seeds 0 1 2 --models-root W44 --seq-root S --out-dir D --shard 0/4
    python scripts/sequence_eval.py agg --out-dirs D
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for _extra in (REPO_ROOT / "src", REPO_ROOT, REPO_ROOT / "scripts"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import decoder_sweep as ds  # noqa: E402
from eval_external import score  # noqa: E402

from mtah2027.models.specialist_c import change_events, effective_changes  # noqa: E402

# the fixed candidate grid of the preregistration: name -> config overrides ("sequence_model.*")
SEQ = "sequence_model."
VARIANTS: dict[str, dict | None] = {
    "base": None,
    "E50": {SEQ + "enabled": True, SEQ + "mode": "emissions", SEQ + "emission_weight": 0.5},
    "E100": {SEQ + "enabled": True, SEQ + "mode": "emissions", SEQ + "emission_weight": 1.0},
    # production-model choice: the same E100 with the float64 inference of the JSON loader (E100f64) and with the
    # mean of the softmax probabilities of the training-seed-0 and -1 models (E100ens; two --seq-root values)
    "E100f64": {SEQ + "enabled": True, SEQ + "mode": "emissions", SEQ + "emission_weight": 1.0},
    "E100ens": {SEQ + "enabled": True, SEQ + "mode": "emissions", SEQ + "emission_weight": 1.0},
    "T-hard": {SEQ + "enabled": True, SEQ + "mode": "turnover", SEQ + "turnover_gate_mode": "hard",
               SEQ + "turnover_threshold": 0.5},
    "T-max": {SEQ + "enabled": True, SEQ + "mode": "turnover", SEQ + "turnover_gate_mode": "max"},
    "T-blend": {SEQ + "enabled": True, SEQ + "mode": "turnover", SEQ + "turnover_gate_mode": "blend",
                SEQ + "turnover_blend": 0.5},
    # Issue #73 (final preregistered grid): visible-frame emissions; hidden frames keep E100
    "V0": {SEQ + "enabled": True, SEQ + "mode": "emissions", SEQ + "emission_weight": 1.0,
           SEQ + "visible_weight": 0.0, SEQ + "emission_scale": 0.8},
    "V25": {SEQ + "enabled": True, SEQ + "mode": "emissions", SEQ + "emission_weight": 1.0,
            SEQ + "visible_weight": 0.25, SEQ + "emission_scale": 0.5},
    "V50": {SEQ + "enabled": True, SEQ + "mode": "emissions", SEQ + "emission_weight": 1.0,
            SEQ + "visible_weight": 0.5, SEQ + "emission_scale": 0.5},
    "V25s": {SEQ + "enabled": True, SEQ + "mode": "emissions", SEQ + "emission_weight": 1.0,
             SEQ + "visible_weight": 0.25, SEQ + "emission_scale": 0.8},
    "V50s": {SEQ + "enabled": True, SEQ + "mode": "emissions", SEQ + "emission_weight": 1.0,
             SEQ + "visible_weight": 0.5, SEQ + "emission_scale": 0.8},
    "E50+T-max": {SEQ + "enabled": True, SEQ + "mode": "both", SEQ + "emission_weight": 0.5,
                  SEQ + "turnover_gate_mode": "max"},
}


def turnover_strata(gt, pred, hidden, tol: int = 25) -> dict:
    """True direct H<->A changes matched by a predicted change of the same direction (as Specialist C), by
    the ball visibility at the true change; ``n_pred`` / ``n_spurious`` over all predicted changes."""
    ha, ah = change_events(gt)
    truth = sorted([(int(t), 1) for t in ha] + [(int(t), 2) for t in ah])
    preds = effective_changes(pred)
    used: set[int] = set()
    out = {"visible": [0, 0], "hidden": [0, 0]}          # [n_true, matched]
    last = len(hidden) - 1
    for t, d in truth:
        key = "hidden" if hidden[min(t, last)] else "visible"
        out[key][0] += 1
        best, best_gap = None, tol + 1
        for k, (tp, dp) in enumerate(preds):
            if dp == d and k not in used and abs(tp - t) < best_gap:
                best, best_gap = k, abs(tp - t)
        if best is not None:
            used.add(best)
            out[key][1] += 1
    return {"visible": out["visible"], "hidden": out["hidden"], "n_pred": len(preds),
            "n_spurious": len(preds) - len(used)}


def seq_path(root: Path, cell: dict) -> Path:
    root = Path(root)
    if cell["world"] in ("b33", "b1"):
        return root / "lomo" / f"{cell['match']}.npz"
    return root / "full.npz"


def run_cell(cell, models_root, seq_roots, config_raw, variants):
    from mtah2027.models.sequence_model import SequenceEnsemble, SequenceModel
    from mtah2027.pipeline import run_core

    players, team, ball, fps, pitch, periods, gt = ds.load_cell_data(cell, config_raw)
    visible = np.isfinite(ball).all(axis=1)
    members = [SequenceModel.load(seq_path(r, cell)) for r in seq_roots]
    seq = members[0] if len(members) == 1 else SequenceEnsemble(members)
    spec = ds.build_specialists({"name": "base"}, cell, models_root, {}, config_raw)
    out = {"cell": {k: v for k, v in cell.items() if k != "path"}, "results": {}}
    t0 = time.perf_counter()
    cache = ds.CellCache(ball)
    cache.install()
    try:
        for name in variants:
            cfg = ds.apply_config(config_raw, VARIANTS[name])
            spec.seq = None if name == "base" else seq
            core = run_core(players, team, ball, play_blocks=periods, fps=fps, pitch=pitch, config_path=cfg,
                            log=lambda m: None, specialists=spec, durations=spec.durations)
            row = score(gt, core.codes)
            row.pop("lat_by_type", None)
            row["turn"] = turnover_strata(gt, core.codes, ~visible)
            out["results"][name] = row
    finally:
        cache.uninstall()
        spec.seq = None
    out["seconds"] = round(time.perf_counter() - t0, 1)
    return out


def cmd_run(a) -> int:
    import yaml

    config_raw = yaml.safe_load((REPO_ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    cells = ds.cells_for(a.world, Path(a.bench_dir), a.seeds, n_pff=20)
    k, _, m = a.shard.partition("/")
    cells = [c for i, c in enumerate(cells) if i % int(m) == int(k)]
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = a.variants.split(",") if a.variants else list(VARIANTS)
    for cell in cells:
        path = out_dir / f"{ds.cell_id(cell)}.json"
        have = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"results": {}}
        todo = [n for n in names if n not in have["results"]]
        if not todo:
            continue
        res = run_cell(cell, Path(a.models_root), [Path(r) for r in a.seq_root], config_raw, todo)
        res["results"] = {**have["results"], **res["results"]}
        path.write_text(json.dumps(res), encoding="utf-8")
        print(f"{ds.cell_id(cell)}: {len(todo)} variants in {res['seconds']} s", flush=True)
    return 0


def cmd_agg(a) -> int:
    table = ds.load_results(a.out_dirs)
    ref = a.ref
    names = [n for n in VARIANTS if any(n in m for by in table.values() for m in by.values())]
    worlds = sorted({w for w, _ in table})
    print(f"{'variant':<12}" + "".join(f"{w:>10}{'d':>9}" for w in worlds) + "   per-seed delta")
    summary = {}
    for n in names:
        parts, per_world = [], {}
        for w in worlds:
            seeds = sorted(s for ww, s in table if ww == w)
            sm = ds.summarise(table, w, seeds, n)
            bm = ds.summarise(table, w, seeds, ref)
            d = {s: sm["per_seed"][s] - bm["per_seed"][s] for s in sm["per_seed"] if s in bm["per_seed"]}
            per_world[w] = {"mean": sm["mean"], "delta": sm["mean"] - bm["mean"], "per_seed_delta": d}
            parts.append(f"{sm['mean']:10.4f}{sm['mean'] - bm['mean']:+9.4f}")
        summary[n] = per_world
        print(f"{n:<12}" + "".join(parts) + "   " + "  ".join(
            f"{w}:" + "/".join(f"{v:+.4f}" for v in per_world[w]["per_seed_delta"].values()) for w in worlds))
    print("\nturnover recall pooled over cells (matched / true; stratum = ball visible or hidden at the true change)")
    turn = {}
    for n in names:
        row = {}
        parts = []
        for w in worlds:
            t = [m[n]["turn"] for (ww, _), by in table.items() if ww == w for m in by.values() if n in m]
            agg = {k: [sum(x[k][0] for x in t), sum(x[k][1] for x in t)] for k in ("visible", "hidden")}
            npred, nspur = sum(x["n_pred"] for x in t), sum(x["n_spurious"] for x in t)
            row[w] = {**agg, "n_pred": npred, "n_spurious": nspur}
            tv, th = agg["visible"], agg["hidden"]
            parts.append(f"{w}: vis {tv[1]}/{tv[0]}={tv[1] / max(tv[0], 1):.3f} hid {th[1]}/{th[0]}={th[1] / max(th[0], 1):.3f} "
                         f"pred {npred} spurious {nspur}")
        turn[n] = row
        print(f"  {n:<11}" + "   ".join(parts))
    print("\npossessions predicted / true (pooled)")
    for n in names:
        parts = []
        for w in worlds:
            r = [m[n] for (ww, _), by in table.items() if ww == w for m in by.values() if n in m]
            parts.append(f"{w}: {sum(x['n_pred_spells'] for x in r)}/{sum(x['n_gt_spells'] for x in r)}")
        print(f"  {n:<11}" + "   ".join(parts))
    if a.json:
        Path(a.json).write_text(json.dumps({"summary": summary, "turnover": turn}, indent=1), encoding="utf-8")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--world", choices=("b33", "b1", "pff"), required=True)
    r.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    r.add_argument("--models-root", required=True)
    r.add_argument("--seq-root", required=True, nargs="+", help="one TCN root, or several (mean of softmax)")
    r.add_argument("--out-dir", required=True)
    r.add_argument("--bench-dir", default=str(ds.DEFAULT_BENCH))
    r.add_argument("--shard", default="0/1")
    r.add_argument("--variants", default=None, help="comma-separated subset (default: the whole grid)")
    g = sub.add_parser("agg")
    g.add_argument("--out-dirs", nargs="+", required=True)
    g.add_argument("--json", default=None)
    g.add_argument("--ref", default="base", help="reference variant of the deltas (Issue #73: E100ens)")
    args = ap.parse_args(argv)
    return cmd_run(args) if args.cmd == "run" else cmd_agg(args)


if __name__ == "__main__":
    raise SystemExit(main())
