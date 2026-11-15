#!/usr/bin/env python
# ruff: noqa: E501
"""Issue #79: post-decoding boundary refinement with the TCN turnover head, preregistered 12-cell grid.

Reuses ``scripts/decoder_modules_eval.py`` (#77: decoder-independent emissions memoised per match, decoder re-run per
cell).  Grid: ``boundary_refine.peak_threshold`` in {0.5, 0.7, 0.9} x ``window_s`` in {1, 2} x ``margin`` in {0.5, 1.0}
on top of production V50s; ``ship`` = nothing changed.  ``boundary_refine`` is read at top level (it is not in
``route_overrides.specialists``) and ``report`` checks that every cell changes the decoded output vs ``ship``.

    python scripts/boundary_refine_eval.py run --world b33 --seeds 0 --variants ship,<cells> --out-dir D \
        --models-root M --seq-root S0 S1
    python scripts/boundary_refine_eval.py agg --out-dirs D
    python scripts/boundary_refine_eval.py report --out-dirs D --json reports/issue79_boundary_refinement.json
External data only, aggregates only.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for _extra in (REPO_ROOT / "src", REPO_ROOT, REPO_ROOT / "scripts"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import decoder_modules_eval as dm  # noqa: E402
import decoder_sweep as ds  # noqa: E402

THRESHOLDS, WINDOWS, MARGINS = (0.5, 0.7, 0.9), (1.0, 2.0), (0.5, 1.0)


def cell_name(thr: float, win: float, margin: float) -> str:
    return f"T{thr}_W{win:g}_M{margin}"


GRID = {cell_name(t, w, m): {"boundary_refine.enabled": True, "boundary_refine.peak_threshold": t,
                             "boundary_refine.window_s": w, "boundary_refine.margin": m}
        for t, w, m in itertools.product(THRESHOLDS, WINDOWS, MARGINS)}
CELLS = tuple(GRID)


def variant_config(raw: dict, name: str) -> dict:
    """Config of a cell (V50s route + the cell's overrides) without touching ``decoder_modules_eval.VARIANTS``."""
    return ds.apply_config(raw, {**dm.se.VARIANTS["V50s"], **({} if name == "ship" else GRID[name])})


def select_cell(table: dict, names=CELLS) -> tuple[str, dict]:
    """Preregistered selection on B'(33) eval seed 0: best spellF1; ties resolved in grid order."""
    sc = {n: ds.summarise(table, "b33", [0], n)["mean"] for n in names}
    allowed = [n for n in names if sc[n] == sc[n]]
    return max(allowed, key=lambda n: (round(sc[n], 12), -allowed.index(n))), sc


def adoption(delta_b33: dict, delta_b1: dict, delta_a: float) -> dict:
    """Preregistered rule: mean B'(33) delta (seeds 1-2) >= +0.010, no seed < -0.003, B'(1) mean delta >= 0, A delta >= 0."""
    mean33 = float(np.mean(list(delta_b33.values())))
    b1 = float(np.mean(list(delta_b1.values())))
    checks = {"b33_mean_ge_0.010": mean33 >= 0.010, "b33_no_seed_below_-0.003": min(delta_b33.values()) >= -0.003,
              "b1_delta_ge_0": b1 >= 0.0, "a_delta_ge_0": delta_a >= 0.0}
    return {"b33_mean": mean33, "b1_mean": b1, "a": delta_a, "checks": checks, "adopted": all(checks.values())}


def cmd_agg(a) -> int:
    table = ds.load_results(a.out_dirs)
    best, sc = select_cell(table)
    print(f"seed-0 B'(33) spellF1: ship {ds.summarise(table, 'b33', [0], 'ship')['mean']:.7f}")
    for n, v in sc.items():
        print(f"  {n:<14}{v:.7f}   {dm.n_changed(table, n)}")
    print(f"selected: {best}")
    return 0


def cmd_report(a) -> int:
    table = ds.load_results(a.out_dirs)
    best, sc = select_cell(table)
    seeds = {"b33": [1, 2], "b1": [1, 2], "pff": [0]}

    def mean(world, s, n):
        return ds.summarise(table, world, s, n)["mean"]

    per = {w: {str(s): mean(w, [s], best) - mean(w, [s], "ship") for s in ss} for w, ss in seeds.items()}
    d33 = per["b33"]
    s0 = sc[best] - mean("b33", [0], "ship")
    out = {"selected": best, "overrides": GRID[best], "seed0_b33_spellF1": {"ship": mean("b33", [0], "ship"), **sc},
           "decoded_output_changed": {n: dm.n_changed(table, n) for n in CELLS if sc[n] == sc[n]},
           "delta_vs_ship": {"b33_seeds_1_2": d33, "b33_mean_seeds_1_2": float(np.mean(list(d33.values()))),
                             "b33_seed0": s0, "b33_mean_3_seeds": float(np.mean([s0] + list(d33.values()))),
                             "b1_seeds_1_2": per["b1"], "b1_mean_seeds_1_2": float(np.mean(list(per["b1"].values()))),
                             "a": per["pff"]["0"], "adoption": adoption(d33, per["b1"], per["pff"]["0"])},
           "absolute": {n: {w: mean(w, seeds[w], n) for w in seeds} for n in (best, "ship")},
           "pooled": {n: {w: dm.pooled(table, w, seeds[w] + ([0] if w == "b33" else []), n) for w in seeds}
                      for n in (best, "ship")}}
    out["adopted"] = out["delta_vs_ship"]["adoption"]["adopted"]
    Path(a.json).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "pooled"}, indent=1))
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "run":
        dm.VARIANTS.update({"ship": {}, **GRID})        # only for the run (importing this module must not alter #77's)
        return dm.main(argv)
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("agg")
    g.add_argument("--out-dirs", nargs="+", required=True)
    p = sub.add_parser("report")
    p.add_argument("--out-dirs", nargs="+", required=True)
    p.add_argument("--json", required=True)
    args = ap.parse_args(argv)
    return {"agg": cmd_agg, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
