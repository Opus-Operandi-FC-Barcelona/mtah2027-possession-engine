#!/usr/bin/env python
"""Score the array-level core of the pipeline on the external benchmark matches (never on BAR-ATH).

For every ``*.npz`` in ``--bench-dir`` (open data of other matches; format ``players_xy[N,S,2]`` metres,
``team_of_slot[S]``, ``ball_xy[N,2]``, ``fps``, ``pitch``, ``period_bounds[P,2]`` inclusive, ``gt[N]`` of
H/A/D/X) it runs features -> emissions -> transitions -> Viterbi -> spell optimiser (no file loaders, no
Smart Tagging) and scores the result with the official metric through ``eval_bridge``; the graded frames are
those where ``gt != X``.  The official baseline recipe (nearest player + ``+-50`` majority window) is computed
on the same arrays for reference.

``--hide-ball-frac F`` blanks the ball in random contiguous 5-60 s blocks until a share ``F`` of the frames
inside the periods is hidden (a crude degradation; the realistic degrader is Issue #19).  Only aggregate
numbers are printed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT / "src", REPO_ROOT):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from mtah2027.models.baseline_labels import baseline_labels  # noqa: E402
from mtah2027.pipeline import run_core  # noqa: E402
from mtah2027.validation import eval_bridge as eb  # noqa: E402

DEFAULT_BENCH_DIR = REPO_ROOT.parent / "mtah2027-external-benchmarks" / "cache"


def hide_ball(ball_xy: np.ndarray, periods, frac: float, fps: float, rng: np.random.Generator) -> np.ndarray:
    """Copy of ``ball_xy`` with random contiguous 5-60 s blocks set to NaN (``frac`` of the period frames)."""
    out = ball_xy.copy()
    n = len(out)
    inside = np.zeros(n, bool)
    for a, b in periods:
        inside[a:b + 1] = True
    target = frac * inside.sum()
    hidden = np.zeros(n, bool)
    guard = 0
    while (hidden & inside).sum() < target and guard < 100000:
        guard += 1
        dur = int(np.exp(rng.uniform(np.log(5.0), np.log(60.0))) * fps)
        start = int(rng.integers(0, max(n - dur, 1)))
        if inside[start:start + dur].mean() < 0.9:
            continue
        hidden[start:start + dur] = True
    out[hidden] = np.nan
    return out


def score(gt: np.ndarray, pred: np.ndarray) -> dict:
    r = eb.evaluate_arrays(gt, pred)
    return {"spell_f1": float(r["spell_f1_mean"]), "macro_f1": float(r["macro_f1"]),
            "n_pred_spells": int(r["n_pred_spells"]), "n_gt_spells": int(r["n_gt_spells"]),
            "mean_spell_s": float(r["mean_spell_duration_s"]),
            "gt_mean_spell_s": float(r["gt_mean_spell_duration_s"]),
            "abstain_on_graded": int(r["n_abstain_on_graded"])}


def evaluate_match(path: Path, hide_frac: float, seed: int, config: str | None, with_baseline: bool) -> dict:
    z = np.load(path, allow_pickle=True)
    players, team = z["players_xy"].astype(np.float32), z["team_of_slot"].astype(np.int8)
    ball = z["ball_xy"].astype(np.float32)
    fps, pitch = float(z["fps"]), tuple(float(v) for v in z["pitch"])
    periods = [(int(a), int(b)) for a, b in z["period_bounds"]]
    gt = eb.to_codes(z["gt"])
    if hide_frac > 0:
        ball = hide_ball(ball, periods, hide_frac, fps, np.random.default_rng(seed))
    t0 = time.perf_counter()
    core = run_core(players, team, ball, play_blocks=periods, fps=fps, pitch=pitch, config_path=config,
                    log=lambda m: None)
    seconds = time.perf_counter() - t0
    row = {"match": path.stem, "seconds": round(seconds, 1),
           "ball_visible": float(np.isfinite(ball).all(axis=1)[gt != 3].mean())}
    row["engine"] = score(gt, core.codes)
    if with_baseline:
        base = np.full(len(gt), 3, dtype=np.int8)
        full = baseline_labels(players, team, ball)
        for a, b in periods:
            base[a:b + 1] = full[a:b + 1]
        row["baseline"] = score(gt, base)
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bench-dir", default=str(DEFAULT_BENCH_DIR))
    ap.add_argument("--hide-ball-frac", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default=None, help="YAML configuration (default configs/default.yaml)")
    ap.add_argument("--matches", nargs="*", default=None, help="subset of match names (file stems)")
    ap.add_argument("--only-complete-ball", action="store_true",
                    help="skip matches whose own ball channel has gaps (Metrica samples drop the ball "
                         "while it is dead, which leaks D into the visibility)")
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--json", default=None, help="write the aggregate rows to this JSON file")
    args = ap.parse_args(argv)

    files = sorted(Path(args.bench_dir).glob("*.npz"))
    if args.matches:
        files = [f for f in files if f.stem in set(args.matches)]
    if args.only_complete_ball:
        files = [f for f in files if np.isfinite(np.load(f, allow_pickle=True)["ball_xy"]).all()]
    if not files:
        print(f"no benchmark files in {args.bench_dir}")
        return 1
    rows = []
    print(f"hide_ball_frac={args.hide_ball_frac} seed={args.seed}")
    head = f"{'match':<18}{'spellF1':>8}{'macroF1':>8}{'poss':>6}{'gt_poss':>8}{'dur_s':>7}{'gt_s':>6}"
    if not args.no_baseline:
        head += f"{'| base_sF1':>11}{'base_mF1':>9}{'b_poss':>7}"
    print(head)
    for i, f in enumerate(files):
        row = evaluate_match(f, args.hide_ball_frac, args.seed + i, args.config, not args.no_baseline)
        rows.append(row)
        e = row["engine"]
        line = (f"{row['match']:<18}{e['spell_f1']:8.4f}{e['macro_f1']:8.4f}{e['n_pred_spells']:6d}"
                f"{e['n_gt_spells']:8d}{e['mean_spell_s']:7.1f}{e['gt_mean_spell_s']:6.1f}")
        if not args.no_baseline:
            b = row["baseline"]
            line += f"{'| ' + format(b['spell_f1'], '.4f'):>11}{b['macro_f1']:9.4f}{b['n_pred_spells']:7d}"
        print(line, flush=True)
    mean = {k: float(np.mean([r["engine"][k] for r in rows])) for k in ("spell_f1", "macro_f1")}
    line = f"{'MEAN':<18}{mean['spell_f1']:8.4f}{mean['macro_f1']:8.4f}"
    if not args.no_baseline:
        bm = {k: float(np.mean([r["baseline"][k] for r in rows])) for k in ("spell_f1", "macro_f1")}
        line += f"{'':<31}{'| ' + format(bm['spell_f1'], '.4f'):>11}{bm['macro_f1']:9.4f}"
        mean["baseline_spell_f1"], mean["baseline_macro_f1"] = bm["spell_f1"], bm["macro_f1"]
    print(line)
    if args.json:
        Path(args.json).write_text(json.dumps({"mean": mean, "rows": rows}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
