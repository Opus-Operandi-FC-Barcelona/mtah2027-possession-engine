#!/usr/bin/env python
# ruff: noqa: E501
"""Issue #63: would the sv_ball640 detector raise spellF1 in the hidden-ball gaps?  (aggregates only)

Same machinery as ``simulate_vision_value.py`` (#36): the degraded external match, one planned crop every
``vision.every_k`` frames of each hidden gap, ``vision.recovery.inject_points``, then the production
specialist route (``decoder_sweep`` cells: B'(33) / B'(1), leave-one-match-out ``lomo_gap`` models) scored with
the official metric.  The detector is simulated from the aggregates measured on the real video
(``vision.sim_detector``): top-confidence candidate within 1 / 2 / 5 m, false positives, no-detection share;
scenarios fixed a priori: gap detection rate 100 / 50 / 25 % of the measured rate, plus the measured ``motion``
profile as a control.  Never reads BAR-ATH.

    python scripts/simulate_sv_ball_value.py run --world b33 --seeds 0 1 2 --models-root W --out-dir D --shard 0/3
    python scripts/simulate_sv_ball_value.py agg --out-dir D
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

from mtah2027.pipeline import run_core  # noqa: E402
from mtah2027.vision.attention_crops import (  # noqa: E402
    VisionParams,
    hidden_gap_frames,
    subsample_gap_frames,
)
from mtah2027.vision.recovery import inject_points  # noqa: E402
from mtah2027.vision.sim_detector import MOTION, SV_BALL640, simulate_detections  # noqa: E402

SCENARIOS = (("none", None, 0.0), ("sv100", SV_BALL640, 1.0), ("sv50", SV_BALL640, 0.5),
             ("sv25", SV_BALL640, 0.25), ("motion100", MOTION, 1.0))


def true_ball(cell: dict) -> np.ndarray:
    z = np.load(cell["path"], allow_pickle=True)
    ball = z["ball_xy"].astype(np.float32)
    if "ball_visible" in z.files:
        ball = np.where(z["ball_visible"][:, None], ball, np.float32(np.nan)).astype(np.float32)
    return ball


def scenario_ball(ball_degraded, ball_true, periods, profile, rate, *, fps, seed, params):
    hidden = hidden_gap_frames(ball_degraded, periods)
    frames, _ = subsample_gap_frames(hidden, params.every_k)
    frames = frames[np.isfinite(ball_true[frames]).all(axis=1)]
    if profile is None or not len(frames):
        return ball_degraded, {"planned": int(len(frames)), "hidden": int(hidden.sum()), "returned": 0,
                               "fill": 0.0}
    got, xy = simulate_detections(ball_true[frames], profile, rate=rate, seed=seed)
    ball, source = inject_points(ball_degraded, frames[got], xy[got], fps=fps, params=params)
    return ball, {"planned": int(len(frames)), "hidden": int(hidden.sum()), "returned": int(got.sum()),
                  "fill": float((hidden & (source >= 2)).sum() / max(hidden.sum(), 1))}


def run_cell(cell, root, config_raw, params):
    players, team, ball, fps, pitch, periods, gt = ds.load_cell_data(cell, config_raw)
    truth = true_ball(cell)
    spec = ds.build_specialists({"name": "base"}, cell, root, {}, config_raw)
    seed = 1000 * cell["seed"] + cell["index"] + 17
    out = {"cell": {k: v for k, v in cell.items() if k != "path"}, "results": {}}
    t0 = time.perf_counter()
    for name, profile, rate in SCENARIOS:
        b, info = scenario_ball(ball, truth, periods, profile, rate, fps=fps, seed=seed, params=params)
        core = run_core(players, team, b, play_blocks=periods, fps=fps, pitch=pitch, config_path=config_raw,
                        log=lambda m: None, specialists=spec, durations=spec.durations)
        out["results"][name] = {**score(gt, core.codes), **info}
    out["seconds"] = round(time.perf_counter() - t0, 1)
    return out


def cmd_run(a) -> int:
    import yaml
    cfg = REPO_ROOT / "configs" / "default.yaml"
    config_raw = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    params = VisionParams.from_config()
    cells = ds.cells_for(a.world, Path(a.bench_dir), a.seeds)
    k, _, m = a.shard.partition("/")
    cells = [c for i, c in enumerate(cells) if i % int(m) == int(k)]
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for cell in cells:
        path = out_dir / f"{ds.cell_id(cell)}.json"
        if path.is_file():
            continue
        res = run_cell(cell, Path(a.models_root), config_raw, params)
        path.write_text(json.dumps(res), encoding="utf-8")
        print(f"{ds.cell_id(cell)}: {res['seconds']} s  " +
              "  ".join(f"{n} {r['spell_f1']:.4f}" for n, r in res["results"].items()), flush=True)
    return 0


def cmd_agg(a) -> int:
    table: dict = {}
    for p in sorted(Path(a.out_dir).glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        c = r["cell"]
        table.setdefault((c["world"], c["seed"]), {})[c["match"]] = r["results"]
    summary = {}
    for world in sorted({w for w, _ in table}):
        seeds = sorted(s for w, s in table if w == world)
        print(f"world {world}, seeds {seeds}")
        print(f"{'scenario':<11}{'spellF1':>9}{'d':>9}{'macroF1':>9}{'fill%':>7}{'ret/plan':>9}  per-seed d | per-seed")
        for name, _, _ in SCENARIOS:
            per, mac, fill, rp = {}, [], [], []
            for s in seeds:
                rows = list(table[(world, s)].values())
                per[s] = float(np.mean([r[name]["spell_f1"] for r in rows]))
                mac += [r[name]["macro_f1"] for r in rows]
                fill += [100 * r[name]["fill"] for r in rows]
                rp += [r[name]["returned"] / max(r[name]["planned"], 1) for r in rows]
            base = {s: float(np.mean([r["none"]["spell_f1"] for r in table[(world, s)].values()]))
                    for s in seeds}
            d = {s: per[s] - base[s] for s in seeds}
            mean = float(np.mean(list(per.values())))
            dm = float(np.mean(list(d.values())))
            summary[f"{world}/{name}"] = {"mean": mean, "delta": dm, "per_seed": per, "seed_delta": d,
                                          "macro": float(np.mean(mac)), "fill": float(np.mean(fill)),
                                          "ret_plan": float(np.mean(rp))}
            print(f"{name:<11}{mean:9.4f}{dm:+9.4f}{np.mean(mac):9.4f}{np.mean(fill):7.1f}{np.mean(rp):9.2f}  "
                  + " ".join(f"{d[s]:+.4f}" for s in seeds) + "  | " + " ".join(f"{per[s]:.4f}" for s in seeds))
        pl = [r["none"]["planned"] for s in seeds for r in table[(world, s)].values()]
        hd = [r["none"]["hidden"] for s in seeds for r in table[(world, s)].values()]
        print(f"planned crops per match: mean {np.mean(pl):.0f} (hidden graded frames {np.mean(hd):.0f})\n")
    if a.json:
        Path(a.json).write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--world", choices=("b33", "b1"), required=True)
    r.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    r.add_argument("--models-root", required=True)
    r.add_argument("--out-dir", required=True)
    r.add_argument("--bench-dir", default=str(ds.DEFAULT_BENCH))
    r.add_argument("--shard", default="0/1")
    g = sub.add_parser("agg")
    g.add_argument("--out-dir", required=True)
    g.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    return cmd_run(a) if a.cmd == "run" else cmd_agg(a)


if __name__ == "__main__":
    raise SystemExit(main())
