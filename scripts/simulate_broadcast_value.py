#!/usr/bin/env python
"""How much would broadcast synchronisation be worth?  A simulated TV view on the degraded external matches (#10).

For every external match (never BAR-ATH) the realistic degrader of Issue #19 hides the ball in the *tactical*
tracking.  A fake broadcast feed is then built from the clean truth: the camera follows the ball (only players
and ball within ``--window`` metres along the pitch are visible), shots last 3-12 s, are separated by cuts
(invalid homography), start at a random offset and drift by replays (the tactical clock moves on while the TV
shows a replay), the slots are permuted at every cut and the positions carry noise.  The blocks are
synchronised with ``broadcast_sync.sync_broadcast`` and the TV ball is injected where the tactical one is NaN.

Reported (aggregates only): offset-recovery accuracy per accepted block, the share of hidden graded frames
filled, the error of the injected ball, and spellF1 of the heuristic route (and, with ``--specialists lomo``, of
the leave-one-match-out learned route) with and without injection on the same degraded matches (paired).

    python scripts/simulate_broadcast_value.py --seed 0 --specialists lomo
"""
# ruff: noqa: E501  (long docstring / comment lines)
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
from mtah2027.preprocessing import broadcast_sync as bs  # noqa: E402
from mtah2027.validation import eval_bridge as eb  # noqa: E402


def fake_broadcast(players: np.ndarray, team: np.ndarray, ball: np.ndarray, periods, *, fps: float = 25.0,
                   window_m: float = 35.0, shot_s=(3.0, 12.0), cut_s=(1.0, 5.0), replay_prob: float = 0.3,
                   replay_s=(8.0, 40.0), noise_player_m: float = 0.3, noise_ball_m: float = 0.5,
                   start_offset_s: float = 60.0, seed: int = 0) -> dict:
    """Broadcast view of clean tracking: ``players_xy [Nb, S, 2]``, ``ball_xy [Nb, 2]``, ``usable [Nb]`` and the
    true ``offset [Nb]`` (tactical frame - broadcast frame; NaN in cuts).  The camera follows the ball and only
    shows the part of the pitch within ``window_m`` of it."""
    rng = np.random.default_rng(seed)
    n, s = players.shape[:2]
    shots: list[tuple[int, int, int]] = []          # (tactical start, length, broadcast start)
    b_cursor = int(round(start_offset_s * fps * rng.random())) + 50
    for a, b in periods:
        t = int(a)
        while t < b:
            length = int(min(round(rng.uniform(*shot_s) * fps), b - t + 1))
            if length < 20:
                break
            shots.append((t, length, b_cursor))
            cut = int(round(rng.uniform(*cut_s) * fps))
            t += length + cut
            b_cursor += length + cut
            if rng.random() < replay_prob:           # a replay: the live clock moves on while TV shows it
                skip = int(round(rng.uniform(*replay_s) * fps))
                t += skip
                b_cursor += int(round(skip * 0.4))
        b_cursor += int(round(30 * fps))             # half-time
    nb = b_cursor + 50
    out_p = np.full((nb, s, 2), np.nan, dtype=np.float32)
    out_b = np.full((nb, 2), np.nan, dtype=np.float32)
    usable = np.zeros(nb, bool)
    offset = np.full(nb, np.nan)
    cx = ball[:, 0].astype(np.float64)               # camera centre: ball x, interpolated through its gaps
    ok = np.isfinite(cx)
    if ok.any():
        idx = np.arange(n)
        cx = np.interp(idx, idx[ok], cx[ok])
    else:
        cx = np.full(n, players[..., 0].mean())
    for t0, length, b0 in shots:
        t = np.arange(t0, min(t0 + length, n))
        j = b0 + np.arange(len(t))
        j_ok = j < nb
        t, j = t[j_ok], j[j_ok]
        perm = np.concatenate([rng.permutation(np.flatnonzero(team == k)) for k in (0, 1)])
        src = np.concatenate([np.flatnonzero(team == k) for k in (0, 1)])
        pos = players[t][:, perm] + rng.normal(0.0, noise_player_m, (len(t), s, 2))
        vis = np.abs(pos[..., 0] - cx[t][:, None]) <= window_m / 2
        pos = np.where(vis[..., None], pos, np.nan)
        out_p[j[:, None], src[None, :]] = pos
        bp = ball[t] + rng.normal(0.0, noise_ball_m, (len(t), 2))
        bvis = np.abs(ball[t, 0] - cx[t]) <= window_m / 2
        out_b[j] = np.where(bvis[:, None], bp, np.nan)
        usable[j] = True
        offset[j] = t0 - b0
    return {"players_xy": out_p, "ball_xy": out_b, "team_of_slot": team.copy(), "usable": usable,
            "offset": offset}


def offset_accuracy(blocks, truth_offset: np.ndarray, tol: int = 2) -> dict:
    """Share of accepted blocks whose offset is within ``tol`` frames of the true one (median over the block)."""
    acc = [b for b in blocks if b.accepted]
    errs = []
    for b in acc:
        tr = truth_offset[b.start:b.stop]
        tr = tr[np.isfinite(tr)]
        errs.append(abs(b.offset - float(np.median(tr))) if len(tr) else np.inf)
    errs = np.array(errs)
    return {"blocks": len(blocks), "accepted": len(acc),
            "accepted_frac": float(len(acc) / max(len(blocks), 1)),
            "within_tol": float((errs <= tol).mean()) if len(errs) else float("nan"),
            "median_abs_err": float(np.median(errs)) if len(errs) else float("nan")}


def injected_error(ball_after: np.ndarray, source: np.ndarray, truth: np.ndarray) -> float:
    sel = (source == bs.SRC_BROADCAST) & np.isfinite(truth).all(axis=1)
    if not sel.any():
        return float("nan")
    return float(np.median(np.linalg.norm(ball_after[sel] - truth[sel], axis=1)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bench-dir", default=str(ee.DEFAULT_BENCH_DIR))
    ap.add_argument("--matches", nargs="*", default=None)
    ap.add_argument("--window", type=float, default=35.0, help="visible pitch window along x (m)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="several seeds in one run (the lomo specialists of a held-out match are trained once)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--specialists", choices=("off", "lomo", "file"), default="off",
                    help="lomo: train leave-one-match-out (slow); file: load --model-path (use a file whose "
                         "training data excludes the simulated matches, e.g. models/specialists_pff.json)")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    files = sorted(Path(args.bench_dir).glob("*.npz"))
    if args.matches:
        files = [f for f in files if f.stem in set(args.matches)]
    if not files:
        print(f"no benchmark files in {args.bench_dir}")
        return 1
    params = bs.BroadcastSyncParams.from_config(args.config)
    training = fusion = None
    file_spec = None
    if args.specialists == "file":
        from mtah2027.models import specialist_fusion as sf

        file_spec = sf.load_specialists(args.model_path or sf.DEFAULT_MODEL_PATH, args.config)
    if args.specialists == "lomo":
        from mtah2027.models import specialist_fusion as sf
        from mtah2027.models.specialist_data import build_training_set

        training = build_training_set(args.bench_dir, config_path=args.config, log=lambda m: None)
        fusion = sf.FusionParams.from_config(args.config)
    rows = []
    seeds = list(args.seeds) if args.seeds else [args.seed]
    spec_cache: dict = {}
    jobs = [(seed, i, f) for seed in seeds for i, f in enumerate(files)]
    for seed, i, f in jobs:
        z = np.load(f, allow_pickle=True)
        players, team = z["players_xy"].astype(np.float32), z["team_of_slot"].astype(np.int8)
        truth_ball = z["ball_xy"].astype(np.float32)
        fps, pitch = float(z["fps"]), tuple(float(v) for v in z["pitch"])
        periods = [(int(a), int(b)) for a, b in z["period_bounds"]]
        gt = eb.to_codes(z["gt"])
        dm = degrade_match(players, team, truth_ball, gt, fps=fps, seed=seed + i)
        fb = fake_broadcast(players, team, truth_ball, periods, fps=fps, window_m=args.window,
                            seed=1000 * seed + i)
        t0 = time.perf_counter()
        res = bs.sync_broadcast(dm.players_xy, team, dm.ball_xy, fb["players_xy"], fb["team_of_slot"],
                                fb["ball_xy"], fb["usable"], fps=fps, pitch=pitch, graded=periods, params=params)
        sec = time.perf_counter() - t0
        acc = offset_accuracy(res.blocks, fb["offset"])
        err = injected_error(res.ball_xy, res.ball_source, truth_ball)
        spec = file_spec
        if training is not None:
            if f.stem not in spec_cache:
                spec_cache[f.stem] = dataclasses.replace(sf.train_specialists(
                    args.bench_dir, config_path=args.config, training=training, exclude_match=f.stem),
                    fusion=fusion)
            spec = spec_cache[f.stem]
        row = {"match": f.stem, "seed": seed, **acc, "fill_rate": res.stats["fill_rate"], "inject_err_m": err,
               "seconds": round(sec, 1)}
        for tag, ball in (("without", dm.ball_xy), ("with", res.ball_xy)):
            core = run_core(dm.players_xy, team, ball, play_blocks=periods, fps=fps, pitch=pitch,
                            config_path=args.config, log=lambda m: None)
            row[f"engine_{tag}"] = ee.score(gt, core.codes)["spell_f1"]
            if spec is not None:
                lc = run_core(dm.players_xy, team, ball, play_blocks=periods, fps=fps, pitch=pitch,
                              config_path=args.config, log=lambda m: None, specialists=spec)
                row[f"spec_{tag}"] = ee.score(gt, lc.codes)["spell_f1"]
        rows.append(row)
        print(f"{f.stem:<16} s{seed} blocks {acc['accepted']:3d}/{acc['blocks']:3d} within2f {acc['within_tol']:.2f} "
              f"fill {100 * row['fill_rate']:5.1f}% err {err:.2f} m  engine {row['engine_without']:.4f} -> "
              f"{row['engine_with']:.4f}"
              + (f"  spec {row['spec_without']:.4f} -> {row['spec_with']:.4f}" if spec is not None else "")
              + f"  ({sec:.1f} s)", flush=True)

    def mean(key):
        return float(np.nanmean([r[key] for r in rows]))

    def gain_stats(a, b):
        d = np.array([r[b] - r[a] for r in rows])
        return f"matches up {int((d > 0).sum())}, down {int((d < 0).sum())}, worst {d.min():+.4f}, best {d.max():+.4f}"

    print(f"\nwindow {args.window:.0f} m, seeds {seeds}, {len(rows)} runs")
    print(f"offset recovery: accepted blocks {100 * mean('accepted_frac'):.1f}%, within 2 frames "
          f"{100 * mean('within_tol'):.1f}%, median |err| {mean('median_abs_err'):.2f} frames")
    print(f"hidden graded frames filled {100 * mean('fill_rate'):.1f}%, injected ball error "
          f"{mean('inject_err_m'):.2f} m (median)")
    print(f"{'route':<12}{'without':>9}{'with':>9}{'delta':>9}")
    summary = {}
    for name, a, b in (("heuristic", "engine_without", "engine_with"), ("specialists", "spec_without", "spec_with")):
        if a in rows[0]:
            wa, wb = mean(a), mean(b)
            summary[name] = {"without": wa, "with": wb}
            print(f"{name:<12}{wa:9.4f}{wb:9.4f}{wb - wa:+9.4f}   ({gain_stats(a, b)})")
            if len(seeds) > 1:
                per = [np.mean([r_[b] - r_[a] for r_ in rows if r_["seed"] == sd]) for sd in seeds]
                print(f"{'':<12}per-seed mean delta: " + "  ".join(f"s{sd} {d:+.4f}" for sd, d in zip(seeds, per)))
                summary[name]["per_seed_delta"] = [float(d) for d in per]
    if args.json:
        Path(args.json).write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
