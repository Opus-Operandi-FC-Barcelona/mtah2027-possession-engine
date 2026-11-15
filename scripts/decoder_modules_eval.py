#!/usr/bin/env python
# ruff: noqa: E501
"""Issue #77: the decoder modules rejected under the LightGBM emissions (explicit-duration decoder #34, expected-spellF1
decoding #62, boundary snap #61) revisited with the V50s sequence-model emissions (#72/#73).

Adapted from scripts/decoder_retune_eval.py of Issue #75 (branch feat/issue-75-decoder-retune, commit 502f014, not
merged): same per-cell memoisation of the decoder-independent learned emissions (specialist fusion + TCN inference), so
only the decoder is re-run per variant.  Generalised from the 36-cell grid to arbitrary ``section.key`` overrides.

``duration``, ``expected_spellf1`` and ``boundary_snap`` are NOT in ``route_overrides.specialists`` (which holds only
transition_matrix, spell_optimizer and specialist_fusion), so the top-level sections are what the shipped learned
route reads (``route_config`` merges the route sections over the top level) and the variants write there.  ``report``
checks that every variant changes the decoded output (predicted possessions differ from ``ship``).

    python scripts/decoder_modules_eval.py run --world b33 --seeds 0 --variants ship,D1,D05,X,S --out-dir D \
        --models-root M --seq-root S0 S1
    python scripts/decoder_modules_eval.py report --out-dirs D --json reports/issue77_decoder_modules.json
Other sequence nets: pass their directories to ``--seq-root`` (one = single net, several = ensemble); extra config
switches (e.g. a different ``sequence_model.visible_weight``) go in ``--extra "section.key=value,..."``.
External data only, aggregates only.
"""
from __future__ import annotations

import argparse
import hashlib
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

# Preregistered variants of Issue #77 (switches on top of the shipped V50s route; "ship" = nothing changed).
VARIANTS = {
    "ship": {},
    "D1": {"duration.enabled": True, "duration.weight": 1.0},
    "D05": {"duration.enabled": True, "duration.weight": 0.5},
    "X": {"expected_spellf1.enabled": True, "expected_spellf1.k": 100, "expected_spellf1.temperature": 1.0,
          "expected_spellf1.seed": 0},
    "S": {"boundary_snap.enabled": True},
}
VARIANTS["D1S"] = {**VARIANTS["D1"], **VARIANTS["S"]}      # D-best + S: only the one for the better of D1/D05 is run
VARIANTS["D05S"] = {**VARIANTS["D05"], **VARIANTS["S"]}
VARIANTS["XS"] = {**VARIANTS["X"], **VARIANTS["S"]}
CANDIDATES = ("D1", "D05", "X", "S", "D1S", "D05S", "XS")


def parse_extra(text: str) -> dict:
    out = {}
    for tok in filter(None, (text or "").split(",")):
        k, _, v = tok.partition("=")
        out[k] = json.loads(v)
    return out


def variant_config(raw: dict, name: str, extra: dict | None = None) -> dict:
    """Config with the V50s sequence-model settings, the optional extra overrides and the variant switches."""
    return ds.apply_config(raw, {**se.VARIANTS["V50s"], **(extra or {}), **VARIANTS[name]})


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


def run_cell(cell, models_root, seq_roots, config_raw, names, extra):
    from mtah2027.models.sequence_model import SequenceEnsemble, SequenceModel
    from mtah2027.pipeline import run_core

    players, team, ball, fps, pitch, periods, gt = ds.load_cell_data(cell, config_raw)
    visible = np.isfinite(ball).all(axis=1)
    members = [SequenceModel.load(se.seq_path(r, cell)) for r in seq_roots]
    seq = members[0] if len(members) == 1 else SequenceEnsemble(members)
    spec = ds.build_specialists({"name": "base"}, cell, models_root, {}, config_raw)
    spec.seq = seq
    out = {"cell": {k: v for k, v in cell.items() if k != "path"}, "results": {},
           "has_duration_table": spec.durations is not None}
    t0 = time.perf_counter()
    cache, memo = ds.CellCache(ball), EmissionMemo(spec)
    cache.install()
    memo.install()
    try:
        for name in names:
            cfg = variant_config(config_raw, name, extra)
            core = run_core(players, team, ball, play_blocks=periods, fps=fps, pitch=pitch, config_path=cfg,
                            log=lambda m: None, specialists=spec, durations=spec.durations)
            row = score(gt, core.codes)
            row.pop("lat_by_type", None)
            row["turn"] = se.turnover_strata(gt, core.codes, ~visible)
            row["sha"] = hashlib.sha1(np.ascontiguousarray(core.codes).tobytes()).hexdigest()[:12]
            row["stages"] = {k: str(v)[:80] for k, v in core.diagnostics["stages"].items()
                             if k in ("duration", "viterbi", "boundary_snap")}
            out["results"][name] = row
    finally:
        memo.uninstall()
        cache.uninstall()
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
    names = [n for n in a.variants.split(",") if n]
    extra = parse_extra(a.extra)
    for cell in cells:
        path = out_dir / f"{ds.cell_id(cell)}.json"
        have = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"results": {}}
        todo = [n for n in names if n not in have["results"]]
        if not todo:
            continue
        res = run_cell(cell, Path(a.models_root), [Path(r) for r in a.seq_root], config_raw, todo, extra)
        res["results"] = {**have["results"], **res["results"]}
        path.write_text(json.dumps(res), encoding="utf-8")
        print(f"{ds.cell_id(cell)}: {len(todo)} variants in {res['seconds']} s", flush=True)
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


def n_changed(table: dict, name: str) -> dict:
    """Cells where the variant's decoded labels differ from ``ship`` (hash of the codes) / total cells."""
    rows = [m for by in table.values() for m in by.values() if name in m and "ship" in m]
    return {"cells": len(rows), "changed": sum(m[name]["sha"] != m["ship"]["sha"] for m in rows),
            "possessions_ship": sum(m["ship"]["n_pred_spells"] for m in rows),
            "possessions_variant": sum(m[name]["n_pred_spells"] for m in rows)}


def select_variant(table: dict, names=CANDIDATES) -> tuple[str, dict]:
    """Preregistered selection on B'(33) eval seed 0: best of D1/D05/X/S, then the D-best+S and X+S combinations
    (D-best = the better of D1/D05); ties resolved in the order listed."""
    sc = {n: ds.summarise(table, "b33", [0], n)["mean"] for n in names}
    d_best = "D1" if not sc["D1"] < sc["D05"] else "D05"
    allowed = [n for n in ("D1", "D05", "X", "S", d_best + "S", "XS") if sc[n] == sc[n]]
    best = max(allowed, key=lambda n: (round(sc[n], 12), -allowed.index(n)))
    return best, sc


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
    best, sc = select_variant(table)
    print("seed-0 B'(33) spellF1")
    print(f"  ship {ds.summarise(table, 'b33', [0], 'ship')['mean']:.7f}")
    for n, v in sc.items():
        print(f"  {n:<6}{v:.7f}   {n_changed(table, n)}")
    print(f"selected: {best}")
    return 0


def cmd_report(a) -> int:
    table = ds.load_results(a.out_dirs)
    best, sc = select_variant(table)
    seeds = {"b33": [1, 2], "b1": [1, 2], "pff": [0]}

    def mean(world, s, n):
        return ds.summarise(table, world, s, n)["mean"]

    per = {w: {str(s): mean(w, [s], best) - mean(w, [s], "ship") for s in ss} for w, ss in seeds.items()}
    d33 = per["b33"]
    s0 = sc[best] - mean("b33", [0], "ship")
    out = {"selected": best, "seed0_b33_spellF1": {"ship": mean("b33", [0], "ship"), **sc},
           "decoded_output_changed": {n: n_changed(table, n) for n in CANDIDATES if sc[n] == sc[n]},
           "delta_vs_ship": {"b33_seeds_1_2": d33, "b33_mean_seeds_1_2": float(np.mean(list(d33.values()))),
                             "b33_seed0": s0, "b33_mean_3_seeds": float(np.mean([s0] + list(d33.values()))),
                             "b1_seeds_1_2": per["b1"], "b1_mean_seeds_1_2": float(np.mean(list(per["b1"].values()))),
                             "a": per["pff"]["0"], "adoption": adoption(d33, per["b1"], per["pff"]["0"])},
           "absolute": {n: {w: mean(w, seeds[w], n) for w in seeds} for n in (best, "ship")},
           "pooled": {n: {w: pooled(table, w, seeds[w] + ([0] if w == "b33" else []), n) for w in seeds}
                      for n in (best, "ship")}}
    out["adopted"] = out["delta_vs_ship"]["adoption"]["adopted"]
    Path(a.json).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "pooled"}, indent=1))
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
    r.add_argument("--variants", default="ship,D1,D05,X,S", help="comma list of " + ",".join(VARIANTS))
    r.add_argument("--extra", default="", help="extra top-level overrides, e.g. sequence_model.visible_weight=0.5")
    g = sub.add_parser("agg")
    g.add_argument("--out-dirs", nargs="+", required=True)
    p = sub.add_parser("report")
    p.add_argument("--out-dirs", nargs="+", required=True)
    p.add_argument("--json", required=True)
    args = ap.parse_args(argv)
    return {"run": cmd_run, "agg": cmd_agg, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
