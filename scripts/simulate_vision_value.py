#!/usr/bin/env python
"""How much would video ball recovery be worth?  A simulated detector on the degraded external matches (#11).

For every external match (never BAR-ATH) the realistic degrader of Issue #19 hides the ball; then a
*simulated* detector recovers the true ball on a fraction ``X`` of the planned crops (one every
``--every-k`` frames of each hidden gap) with Gaussian position noise ``--noise`` metres.  The recovered
points go through the very same injection code as the real chain (``vision.recovery.inject_points``:
speed-plausible interpolation between points, tracking never overwritten) and the pipeline is scored again
with the official metric.  The degraded matches are identical for every ``X`` (paired comparison);
``X = 0`` is the reference.

``--specialists lomo`` also scores the learned route, trained leave-one-match-out on the *un-assisted*
degraded matches (the specialists are not retrained with vision).  Aggregates only.

    python scripts/simulate_vision_value.py --fracs 0 0.2 0.4 0.6 --noise 1.0 --every-k 5 --specialists lomo
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT / "src", REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import eval_external as ee  # noqa: E402

from mtah2027.models.external_degrader import degrade_match  # noqa: E402
from mtah2027.pipeline import run_core  # noqa: E402
from mtah2027.validation import eval_bridge as eb  # noqa: E402
from mtah2027.vision.attention_crops import (  # noqa: E402
    VisionParams,
    hidden_gap_frames,
    subsample_gap_frames,
)
from mtah2027.vision.recovery import inject_points  # noqa: E402


def simulate_recovery(ball_degraded: np.ndarray, ball_true: np.ndarray, periods, *, frac: float,
                      noise_m: float, every_k: int, fps: float, seed: int,
                      params: VisionParams | None = None) -> tuple[np.ndarray, dict]:
    """Ball track after a simulated detector recovered ``frac`` of the planned hidden frames."""
    p = params or VisionParams.from_config()
    hidden = hidden_gap_frames(ball_degraded, periods)
    frames, _ = subsample_gap_frames(hidden, every_k)
    frames = frames[np.isfinite(ball_true[frames]).all(axis=1)]
    rng = np.random.default_rng(seed)
    hit = rng.random(len(frames)) < frac
    frames = frames[hit]
    xy = ball_true[frames].astype(np.float64) + rng.normal(0.0, noise_m, (len(frames), 2))
    ball, source = inject_points(ball_degraded, frames, xy, fps=fps, params=p)
    filled = hidden & (source >= 2)
    return ball, {"detections": int(len(frames)), "fill_rate": float(filled.sum() / max(hidden.sum(), 1))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bench-dir", default=str(ee.DEFAULT_BENCH_DIR))
    ap.add_argument("--matches", nargs="*", default=None)
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.0, 0.2, 0.4, 0.6])
    ap.add_argument("--noise", type=float, default=1.0, help="position noise of a recovered point (m)")
    ap.add_argument("--every-k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default=None)
    ap.add_argument("--specialists", choices=("off", "lomo"), default="off")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    files = sorted(Path(args.bench_dir).glob("*.npz"))
    if args.matches:
        files = [f for f in files if f.stem in set(args.matches)]
    if not files:
        print(f"no benchmark files in {args.bench_dir}")
        return 1
    training = None
    if args.specialists == "lomo":
        from mtah2027.models import specialist_fusion as sf
        from mtah2027.models.specialist_data import build_training_set

        training = build_training_set(args.bench_dir, config_path=args.config, log=lambda m: None)
        fusion = sf.FusionParams.from_config(args.config)
    rows = []
    for i, f in enumerate(files):
        z = np.load(f, allow_pickle=True)
        players, team = z["players_xy"].astype(np.float32), z["team_of_slot"].astype(np.int8)
        truth_ball = z["ball_xy"].astype(np.float32)
        fps, pitch = float(z["fps"]), tuple(float(v) for v in z["pitch"])
        periods = [(int(a), int(b)) for a, b in z["period_bounds"]]
        gt = eb.to_codes(z["gt"])
        dm = degrade_match(players, team, truth_ball, gt, fps=fps, seed=args.seed + i)
        spec = None
        if training is not None:
            t0 = time.perf_counter()
            spec = dataclasses.replace(sf.train_specialists(args.bench_dir, config_path=args.config,
                                                            training=training, exclude_match=f.stem),
                                       fusion=fusion)
            print(f"  trained specialists without {f.stem} ({time.perf_counter() - t0:.0f} s)", flush=True)
        for frac in args.fracs:
            ball, info = simulate_recovery(dm.ball_xy, truth_ball, periods, frac=frac, noise_m=args.noise,
                                           every_k=args.every_k, fps=fps, seed=1000 * i + 7)
            row = {"match": f.stem, "frac": frac, **info}
            core = run_core(dm.players_xy, team, ball, play_blocks=periods, fps=fps, pitch=pitch,
                            config_path=args.config, log=lambda m: None)
            row["engine"] = ee.score(gt, core.codes)
            if spec is not None:
                lc = run_core(dm.players_xy, team, ball, play_blocks=periods, fps=fps, pitch=pitch,
                              config_path=args.config, log=lambda m: None, specialists=spec)
                row["specialists"] = ee.score(gt, lc.codes)
            rows.append(row)
            print(f"{f.stem:<16} X={frac:.1f} fill={100 * info['fill_rate']:5.1f}% "
                  f"engine {row['engine']['spell_f1']:.4f}"
                  + (f"  specialists {row['specialists']['spell_f1']:.4f}" if spec is not None else ""),
                  flush=True)
    print(f"\nnoise {args.noise} m, one crop every {args.every_k} frames, {len(files)} matches")
    head = f"{'X':>5}{'fill%':>8}{'engine':>9}{'d_eng':>8}"
    if training is not None:
        head += f"{'spec':>9}{'d_spec':>8}"
    print(head)
    base = {}
    summary = []
    for frac in args.fracs:
        sel = [r for r in rows if r["frac"] == frac]
        e = float(np.mean([r["engine"]["spell_f1"] for r in sel]))
        fill = 100 * float(np.mean([r["fill_rate"] for r in sel]))
        item = {"frac": frac, "fill_pct": fill, "engine": e}
        base.setdefault("engine", e)
        line = f"{frac:5.2f}{fill:8.1f}{e:9.4f}{e - base['engine']:+8.4f}"
        if training is not None:
            s = float(np.mean([r["specialists"]["spell_f1"] for r in sel]))
            base.setdefault("spec", s)
            item["specialists"] = s
            line += f"{s:9.4f}{s - base['spec']:+8.4f}"
        summary.append(item)
        print(line)
    if args.json:
        Path(args.json).write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
