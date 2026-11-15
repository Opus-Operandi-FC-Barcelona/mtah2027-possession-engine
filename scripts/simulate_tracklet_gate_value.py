#!/usr/bin/env python
# ruff: noqa: E501
"""Issue #66: does a tracklet gate / emission tempering make the sv_ball640 detections useful?  (aggregates only)

Same machinery as ``simulate_sv_ball_value.py`` (#63) with a detector that has temporal structure
(``vision.sim_detector.simulate_detections_temporal``: persistent true hits, mostly isolated but sometimes
persistent false positives).  Scenarios are (rate, detector, tracklet gate min length, emission scale).  The
``iid`` detector is the #63 one (reproduces its -0.074 at 50 %).  Never reads BAR-ATH.

    python scripts/simulate_tracklet_gate_value.py run --world b33 --seeds 0 1 2 --models-root W --out-dir D --shard 0/2
    python scripts/simulate_tracklet_gate_value.py agg --out-dir D
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
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
from eval_external import score  # noqa: E402
from simulate_sv_ball_value import true_ball  # noqa: E402

from mtah2027.pipeline import run_core  # noqa: E402
from mtah2027.vision.attention_crops import (  # noqa: E402
    VisionParams,
    hidden_gap_frames,
    subsample_gap_frames,
)
from mtah2027.vision.recovery import inject_points  # noqa: E402
from mtah2027.vision.sim_detector import (  # noqa: E402
    PERSIST_FP,
    SV_BALL640,
    TemporalModel,
    simulate_detections,
    simulate_detections_temporal,
)
from mtah2027.vision.tracklet_gate import tracklet_mask  # noqa: E402

# name: (detector, rate, gate_min_len or 0, emission scale)
SCENARIOS = {
    "none": (None, 0.0, 0, 1.0),
    "iid50": ("iid", 0.5, 0, 1.0),
    "t50": ("temporal", 0.5, 0, 1.0),
    "t50_g3": ("temporal", 0.5, 3, 1.0),
    "t50_g5": ("temporal", 0.5, 5, 1.0),
    "t50_s05": ("temporal", 0.5, 0, 0.5),
    "t50_g3s05": ("temporal", 0.5, 3, 0.5),
    "pf50": ("persist", 0.5, 0, 1.0),
    "pf50_g3": ("persist", 0.5, 3, 1.0),
    "pf50_g5": ("persist", 0.5, 5, 1.0),
    "t25": ("temporal", 0.25, 0, 1.0),
    "t25_g3": ("temporal", 0.25, 3, 1.0),
    "t25_g3s05": ("temporal", 0.25, 3, 0.5),
    "t100": ("temporal", 1.0, 0, 1.0),
    "t100_g3": ("temporal", 1.0, 3, 1.0),
    "t100_g3s05": ("temporal", 1.0, 3, 0.5),
}
MODELS = {"temporal": TemporalModel(), "persist": PERSIST_FP}


def make_points(truth, ball_degraded, periods, kind, rate, seed, params):
    hidden = hidden_gap_frames(ball_degraded, periods)
    frames, _ = subsample_gap_frames(hidden, params.every_k)
    frames = frames[np.isfinite(truth[frames]).all(axis=1)]
    info = {"planned": int(len(frames)), "hidden": int(hidden.sum())}
    if kind is None or not len(frames):
        return frames[:0], np.zeros((0, 2)), np.zeros(0, bool), info
    if kind == "iid":
        got, xy = simulate_detections(truth[frames], SV_BALL640, rate=rate, seed=seed)
        good = got & (np.hypot(*(xy - truth[frames]).T) <= 5.0)
    else:
        got, xy, good = simulate_detections_temporal(truth[frames], frames, SV_BALL640, MODELS[kind],
                                                     rate=rate, seed=seed)
    return frames[got], xy[got], good[got], info


def run_cell(cell, root, config_raw, params):
    players, team, ball, fps, pitch, periods, gt = ds.load_cell_data(cell, config_raw)
    truth = true_ball(cell)
    spec = ds.build_specialists({"name": "base"}, cell, root, {}, config_raw)
    seed = 1000 * cell["seed"] + cell["index"] + 17
    out = {"cell": {k: v for k, v in cell.items() if k != "path"}, "results": {}}
    t0 = time.perf_counter()
    cache: dict = {}
    hidden = hidden_gap_frames(ball, periods)
    for name, (kind, rate, min_len, scale) in SCENARIOS.items():
        fr, xy, good, info = make_points(truth, ball, periods, kind, rate, seed, params)
        n_ret = len(fr)
        if min_len and n_ret:
            gp = dataclasses.replace(params, tracklet_gate=True, tracklet_min_len=min_len)
            keep = tracklet_mask(fr, xy, fps=fps, params=gp)
            fr, xy, good = fr[keep], xy[keep], good[keep]
        err = np.hypot(*(xy - truth[fr]).T) if len(fr) else np.zeros(0)
        info.update({"returned": int(n_ret), "kept": int(len(fr)),
                     "kept_good": int(good.sum()), "kept_le1": int((err <= 1.0).sum()),
                     "kept_gt5": int((err > 5.0).sum()),
                     "err_median": float(np.median(err)) if len(err) else float("nan")})
        b, source = inject_points(ball, fr, xy, fps=fps, params=params)
        key = hashlib.sha1(np.ascontiguousarray(b).tobytes() + source.tobytes() + str(scale).encode()).hexdigest()
        if key not in cache:
            cfg = copy.deepcopy(config_raw)
            cfg.setdefault("vision", {})["injected_emission_scale"] = scale
            core = run_core(players, team, b, play_blocks=periods, fps=fps, pitch=pitch, config_path=cfg,
                            log=lambda m: None, specialists=spec, durations=spec.durations, ball_source=source)
            cache[key] = score(gt, core.codes)
        info["fill"] = float((hidden & (source >= 2)).sum() / max(hidden.sum(), 1))
        out["results"][name] = {**cache[key], **info}
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
    nan = float("nan")
    for world in sorted({w for w, _ in table}):
        seeds = sorted(s for w, s in table if w == world)
        print(f"world {world}, seeds {seeds}")
        print(f"{'scenario':<11}{'spellF1':>9}{'d':>9}{'macroF1':>9}{'fill%':>7}{'kept%':>7}{'<=1m%':>7}{'>5m%':>6}{'medErr':>7}  per-seed d")
        for name in SCENARIOS:
            per, mac, fill = {}, [], []
            ret = kept = le1 = gt5 = 0
            errs = []
            for s in seeds:
                rows = list(table[(world, s)].values())
                per[s] = float(np.mean([r[name]["spell_f1"] for r in rows]))
                mac += [r[name]["macro_f1"] for r in rows]
                fill += [100 * r[name]["fill"] for r in rows]
                for r in rows:
                    x = r[name]
                    ret += x["returned"]
                    kept += x["kept"]
                    le1 += x["kept_le1"]
                    gt5 += x["kept_gt5"]
                    if x["kept"]:
                        errs.append(x["err_median"])
            base = {s: float(np.mean([r["none"]["spell_f1"] for r in table[(world, s)].values()])) for s in seeds}
            d = {s: per[s] - base[s] for s in seeds}
            mean = float(np.mean(list(per.values())))
            dm = float(np.mean(list(d.values())))
            kp = 100 * kept / max(ret, 1)
            med = float(np.median(errs)) if errs else nan
            summary[f"{world}/{name}"] = {"mean": mean, "delta": dm, "per_seed": per, "seed_delta": d,
                                          "macro": float(np.mean(mac)), "fill": float(np.mean(fill)),
                                          "returned": ret, "kept": kept, "kept_pct": kp,
                                          "le1_pct": 100 * le1 / max(kept, 1), "gt5_pct": 100 * gt5 / max(kept, 1),
                                          "err_median": med}
            print(f"{name:<11}{mean:9.4f}{dm:+9.4f}{np.mean(mac):9.4f}{np.mean(fill):7.1f}{kp:7.1f}"
                  f"{100 * le1 / max(kept, 1):7.1f}{100 * gt5 / max(kept, 1):6.1f}{med:7.2f}  "
                  + " ".join(f"{d[s]:+.4f}" for s in seeds))
        print()
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
