#!/usr/bin/env python
# ruff: noqa: E501
"""Issue #80: per-player possession zones, preregistered 12-cell grid (R x C x usage) on top of the shipped V50s route.

Reuses the per-cell emission memoisation of scripts/decoder_modules_eval.py (Issue #77): the specialist fusion and the
TCN inference are computed once per cell, the possession signal (a cheap vectorised function of the front end of the
cell, computed inside ``run_core`` from the player/ball positions) and the decoder are re-run per variant.

    python scripts/player_possession_eval.py run --world b33 --seeds 0 --variants ship,P1.0_0.4_e,... --out-dir D \
        --models-root M --seq-root S0 S1
    python scripts/player_possession_eval.py select --out-dirs D            # selection on B'(33) seed 0 only
    python scripts/player_possession_eval.py report --out-dirs Db33 Db1 Dpff --json reports/issue80_player_possession.json

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

RADII, CONFIRMS = (1.0, 1.5), (0.4, 0.8)
USAGE_CODE = {"e": "emission", "t": "transition", "b": "both"}
GRID = {f"P{r}_{c}_{u}": {"player_possession.enabled": True, "player_possession.radius_m": r,
                          "player_possession.confirm_s": c, "player_possession.usage": USAGE_CODE[u]}
        for r, c, u in itertools.product(RADII, CONFIRMS, "etb")}
dm.VARIANTS.update(GRID)
CANDIDATES = tuple(GRID)


def select_cell(table: dict) -> tuple[str, dict]:
    sc = {n: ds.summarise(table, "b33", [0], n)["mean"] for n in CANDIDATES}
    allowed = [n for n in CANDIDATES if sc[n] == sc[n]]
    return max(allowed, key=lambda n: (round(sc[n], 12), -allowed.index(n))), sc


def cmd_select(a) -> int:
    table = ds.load_results(a.out_dirs)
    best, sc = select_cell(table)
    print(f"seed-0 B'(33) spellF1: ship {ds.summarise(table, 'b33', [0], 'ship')['mean']:.7f}")
    for n, v in sc.items():
        print(f"  {n:<12}{v:.7f}   {dm.n_changed(table, n)}")
    print(f"selected: {best}   overrides: {GRID[best]}")
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
    rule = {"b33_mean_ge_0.010": float(np.mean(list(d33.values()))) >= 0.010,
            "b33_no_seed_below_-0.003": min(d33.values()) >= -0.003,
            "b1_delta_ge_0": float(np.mean(list(per["b1"].values()))) >= 0.0,
            "a_delta_ge_0": per["pff"]["0"] >= 0.0}
    out = {"selected": best, "overrides": GRID[best], "seed0_b33_spellF1": {"ship": mean("b33", [0], "ship"), **sc},
           "decoded_output_changed": {n: dm.n_changed(table, n) for n in CANDIDATES if sc[n] == sc[n]},
           "delta_vs_ship": {"b33_seeds_1_2": d33, "b33_mean_seeds_1_2": float(np.mean(list(d33.values()))),
                             "b33_seed0": s0, "b33_mean_3_seeds": float(np.mean([s0] + list(d33.values()))),
                             "b1_seeds_1_2": per["b1"], "b1_mean_seeds_1_2": float(np.mean(list(per["b1"].values()))),
                             "a": per["pff"]["0"], "rule": rule},
           "absolute": {n: {w: mean(w, seeds[w], n) for w in seeds} for n in (best, "ship")},
           "pooled": {n: {w: dm.pooled(table, w, seeds[w] + ([0] if w == "b33" else []), n) for w in seeds}
                      for n in (best, "ship")}}
    out["adopted"] = all(rule.values())
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
    r.add_argument("--variants", default="ship," + ",".join(GRID))
    r.add_argument("--extra", default="")
    s = sub.add_parser("select")
    s.add_argument("--out-dirs", nargs="+", required=True)
    p = sub.add_parser("report")
    p.add_argument("--out-dirs", nargs="+", required=True)
    p.add_argument("--json", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "run":
        return dm.cmd_run(args)
    return {"select": cmd_select, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
