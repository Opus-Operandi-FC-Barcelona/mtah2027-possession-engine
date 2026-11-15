#!/usr/bin/env python
# ruff: noqa: E501
"""Issue #75: decoder re-tuning (H<->A switch cost, micro-possession pruning) for the sequence-model emissions.

Scores the preregistered 36-cell grid ``ha_closed x ha_occluded x micro_poss_s`` (body of #75) on the cells of
#72-#74 (B'(33), B'(1) = 7 IDSSE leave-one-match-out matches degraded with the calibrated degrader, eval seeds
0-2; A = the first 20 PFF matches), with the production V50s emissions (#72 two-seed ensemble, #73 mixing).
The emissions do not depend on the decoder cell, so the specialist fusion and the TCN inference are computed once
per match and cell (memoised) and only the transitions, Viterbi and spell optimiser are re-run per grid cell.

The grid values are written where the learned route reads them (``route_overrides.specialists``); the top-level
sections are overridden there and would make the grid a no-op.  The values the learned route actually ships with
(-8.76, -7.5746, 0.9847) are not a grid point and are scored as the extra reference cell ``ship``.

    python scripts/decoder_retune_eval.py run --world b33 --seeds 0 --cells grid --out-dir D ...
    python scripts/decoder_retune_eval.py agg --out-dirs D
External data only, aggregates only.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
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
import sequence_eval as se  # noqa: E402
from eval_external import score  # noqa: E402

HA_CLOSED = (-11.5, -9.5, -7.5, -5.5)
HA_OCCLUDED = (-6.0, -4.5, -3.0)
MICRO_POSS_S = (1.2, 0.6, 0.0)
NOMINAL = (-11.5, -6.0, 1.2)            # "production cell" named by the preregistration (top-level defaults)
SHIPPED = (-8.76, -7.5746, 0.9847)      # what the learned route actually decodes with (route_overrides.specialists)
GRID = list(itertools.product(HA_CLOSED, HA_OCCLUDED, MICRO_POSS_S))   # 36 cells, preregistered order


def cell_name(c: tuple) -> str:
    return f"hc{c[0]}_ho{c[1]}_mp{c[2]}"


def decoder_config(raw: dict, cell: tuple, base_overrides: dict | None) -> dict:
    """Config with the V50s sequence-model settings and the decoder cell set in the learned route."""
    cfg = ds.apply_config(raw, base_overrides)
    spec = cfg.setdefault("route_overrides", {}).setdefault("specialists", {})
    spec.setdefault("transition_matrix", {}).update({"ha_closed": float(cell[0]), "ha_occluded": float(cell[1])})
    spec.setdefault("spell_optimizer", {}).update({"micro_poss_s": float(cell[2])})
    return cfg


class EmissionMemo:
    """Memoises the decoder-independent learned emissions of one cell (specialist fusion and TCN inference)."""

    def __init__(self, spec):
        from mtah2027.models import specialist_fusion as sf

        self.sf, self.spec = sf, spec
        self.orig_fuse, self.orig_predict = sf.fuse_emissions, spec.seq.predict
        self.fused: dict = {}
        self.seq: dict = {}

    def install(self):
        def fuse(specialists, fe, em, **k):
            key = id(fe)
            if key not in self.fused:
                self.fused[key] = (fe, self.orig_fuse(specialists, fe, em, **k))
            return self.fused[key][1]

        def predict(x, names, blocks):
            key = (hashlib.sha1(np.ascontiguousarray(x).tobytes()).hexdigest(), tuple(names), tuple(blocks))
            if key not in self.seq:
                self.seq[key] = self.orig_predict(x, names, blocks)
            return self.seq[key]

        self.sf.fuse_emissions = fuse
        self.spec.seq.predict = predict

    def uninstall(self):
        self.sf.fuse_emissions = self.orig_fuse
        self.spec.seq.__dict__.pop("predict", None)


def run_cell(cell, models_root, seq_roots, config_raw, decoders, save_emissions: Path | None = None):
    from mtah2027.models.sequence_model import SequenceEnsemble, SequenceModel
    from mtah2027.pipeline import run_core

    players, team, ball, fps, pitch, periods, gt = ds.load_cell_data(cell, config_raw)
    visible = np.isfinite(ball).all(axis=1)
    members = [SequenceModel.load(se.seq_path(r, cell)) for r in seq_roots]
    seq = members[0] if len(members) == 1 else SequenceEnsemble(members)
    spec = ds.build_specialists({"name": "base"}, cell, models_root, {}, config_raw)
    spec.seq = seq
    out = {"cell": {k: v for k, v in cell.items() if k != "path"}, "results": {}}
    t0 = time.perf_counter()
    cache, memo = ds.CellCache(ball), EmissionMemo(spec)
    cache.install()
    memo.install()
    first = True
    try:
        for name, dcell in decoders.items():
            cfg = decoder_config(config_raw, dcell, se.VARIANTS["V50s"])
            keep = save_emissions is not None and first
            core = run_core(players, team, ball, play_blocks=periods, fps=fps, pitch=pitch, config_path=cfg,
                            log=lambda m: None, specialists=spec, durations=spec.durations, keep_features=keep)
            if keep:
                np.savez_compressed(save_emissions / f"{ds.cell_id(cell)}.npz",
                                    log_probs=np.asarray(core.features["log_probs"], np.float32), visible=visible)
            first = False
            row = score(gt, core.codes)
            row.pop("lat_by_type", None)
            row["turn"] = se.turnover_strata(gt, core.codes, ~visible)
            out["results"][name] = row
    finally:
        memo.uninstall()
        cache.uninstall()
    out["seconds"] = round(time.perf_counter() - t0, 1)
    return out


def named_decoders(which: str) -> dict:
    if which == "grid":
        return {cell_name(c): c for c in GRID} | {"ship": SHIPPED}
    out = {}
    for tok in which.split(","):
        if tok == "ship":
            out["ship"] = SHIPPED
        elif tok == "nominal":
            out["nominal"] = NOMINAL
        else:
            hc, ho, mp = (float(v) for v in tok.split("/"))
            out[cell_name((hc, ho, mp))] = (hc, ho, mp)
    return out


def cmd_run(a) -> int:
    import yaml

    config_raw = yaml.safe_load((REPO_ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    cells = ds.cells_for(a.world, Path(a.bench_dir), a.seeds, n_pff=20)
    k, _, m = a.shard.partition("/")
    cells = [c for i, c in enumerate(cells) if i % int(m) == int(k)]
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    emis = Path(a.save_emissions) if a.save_emissions else None
    if emis:
        emis.mkdir(parents=True, exist_ok=True)
    decoders = named_decoders(a.cells)
    for cell in cells:
        path = out_dir / f"{ds.cell_id(cell)}.json"
        have = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"results": {}}
        todo = {n: c for n, c in decoders.items() if n not in have["results"]}
        if not todo:
            continue
        res = run_cell(cell, Path(a.models_root), [Path(r) for r in a.seq_root], config_raw, todo, emis)
        res["results"] = {**have["results"], **res["results"]}
        path.write_text(json.dumps(res), encoding="utf-8")
        print(f"{ds.cell_id(cell)}: {len(todo)} cells in {res['seconds']} s", flush=True)
    return 0


def pooled(table: dict, world: str, seeds, name: str) -> dict:
    rows = [m[name] for (w, s), by in table.items() if w == world and s in seeds for m in by.values() if name in m]
    t = [r["turn"] for r in rows]
    n_true = [sum(x[k][0] for x in t) for k in ("visible", "hidden")]
    n_hit = [sum(x[k][1] for x in t) for k in ("visible", "hidden")]
    return {"n_pred_spells": sum(r["n_pred_spells"] for r in rows), "n_gt_spells": sum(r["n_gt_spells"] for r in rows),
            "n_pred_turnovers": sum(x["n_pred"] for x in t), "n_spurious_turnovers": sum(x["n_spurious"] for x in t),
            "recall_visible": n_hit[0] / max(n_true[0], 1), "recall_hidden": n_hit[1] / max(n_true[1], 1),
            "n_true_visible": n_true[0], "n_true_hidden": n_true[1]}


def select_cell(table: dict) -> tuple[str, dict]:
    """Preregistered selection: best grid cell on B'(33) eval seed 0 (ties: first in grid order)."""
    scores = {cell_name(c): ds.summarise(table, "b33", [0], cell_name(c))["mean"] for c in GRID}
    best = max(scores, key=lambda n: (round(scores[n], 12), -list(scores).index(n)))
    return best, scores


def adoption(delta_b33: dict, delta_b1: dict, delta_a: float) -> dict:
    """Preregistered rule: mean B'(33) delta (seeds 1-2) >= +0.010 and no seed < -0.003; B'(1) mean drop <= 0.010;
    A drop <= 0.030."""
    mean33 = float(np.mean(list(delta_b33.values())))
    b1 = float(np.mean(list(delta_b1.values())))
    checks = {"b33_mean_ge_0.010": mean33 >= 0.010, "b33_no_seed_below_-0.003": min(delta_b33.values()) >= -0.003,
              "b1_drop_le_0.010": b1 >= -0.010, "a_drop_le_0.030": delta_a >= -0.030}
    return {"b33_mean": mean33, "b1_mean": b1, "a": delta_a, "checks": checks, "adopted": all(checks.values())}


def cmd_agg(a) -> int:
    table = ds.load_results(a.out_dirs)
    best, scores = select_cell(table)
    print("seed-0 B'(33) spellF1 of the 36 cells")
    for n, v in sorted(scores.items(), key=lambda kv: -kv[1]):
        print(f"  {n:<26}{v:.4f}")
    print(f"ship {ds.summarise(table, 'b33', [0], 'ship')['mean']:.4f}  nominal-cell {scores[cell_name(NOMINAL)]:.4f}\n"
          f"selected: {best}")
    return 0


def cmd_report(a) -> int:
    """Aggregates of #75: 36 seed-0 scores, selected cell, held-out deltas against the shipped values and the nominal
    cell, possessions / spurious turnovers / H<->A recall, adoption verdict."""
    table = ds.load_results(a.out_dirs)
    best, scores = select_cell(table)
    seeds = {"b33": [1, 2], "b1": [1, 2], "pff": [0]}

    def mean(world, s, n):
        return ds.summarise(table, world, s, n)["mean"]

    out = {"selected": best, "seed0_b33_spellF1": scores,
           "seed0_b33_spellF1_ship": mean("b33", [0], "ship"), "seed0_b33_spellF1_nominal": mean("b33", [0], "nominal")}
    for ref in ("ship", "nominal"):
        per = {}
        for w, ss in seeds.items():
            per[w] = {str(s): ds.summarise(table, w, [s], best)["mean"] - ds.summarise(table, w, [s], ref)["mean"]
                      for s in ss}
        d33 = per["b33"]
        out[f"delta_vs_{ref}"] = {
            "b33_seeds_1_2": d33, "b33_mean_seeds_1_2": float(np.mean(list(d33.values()))),
            "b33_seed0": scores[best] - mean("b33", [0], ref),
            "b33_mean_3_seeds": float(np.mean([scores[best] - mean("b33", [0], ref)] + list(d33.values()))),
            "b1_seeds_1_2": per["b1"], "b1_mean_seeds_1_2": float(np.mean(list(per["b1"].values()))),
            "a": per["pff"]["0"], "adoption": adoption(d33, per["b1"], per["pff"]["0"])}
    out["absolute"] = {n: {w: ds.summarise(table, w, seeds[w], n)["mean"] for w in seeds} for n in (best, "ship", "nominal")}
    out["absolute"]["b33_seed0"] = {n: mean("b33", [0], n) for n in (best, "ship", "nominal")}
    out["pooled"] = {n: {w: pooled(table, w, seeds[w] + ([0] if w == "b33" else []), n) for w in seeds}
                     for n in (best, "ship", "nominal")}
    out["adopted"] = out["delta_vs_ship"]["adoption"]["adopted"]
    Path(a.json).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k not in ("seed0_b33_spellF1", "pooled")}, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--world", choices=("b33", "b1", "pff"), required=True)
    r.add_argument("--seeds", type=int, nargs="+", default=[0])
    r.add_argument("--models-root", required=True)
    r.add_argument("--seq-root", required=True, nargs="+")
    r.add_argument("--out-dir", required=True)
    r.add_argument("--bench-dir", default=str(ds.DEFAULT_BENCH))
    r.add_argument("--shard", default="0/1")
    r.add_argument("--cells", default="grid", help="grid | comma list of 'ship', 'nominal', hc/ho/mp")
    r.add_argument("--save-emissions", default=None)
    g = sub.add_parser("agg")
    g.add_argument("--out-dirs", nargs="+", required=True)
    p = sub.add_parser("report")
    p.add_argument("--out-dirs", nargs="+", required=True)
    p.add_argument("--json", required=True)
    args = ap.parse_args(argv)
    return {"run": cmd_run, "agg": cmd_agg, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
