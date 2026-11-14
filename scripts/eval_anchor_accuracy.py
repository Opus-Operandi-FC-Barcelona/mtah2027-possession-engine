#!/usr/bin/env python
"""Accuracy of the restart-frame anchoring ``t*`` (Issue #14) on EXTERNAL matches with known labels (#40).

The true restart of an external match is the first alive frame after a dead-ball run of at least
``--min-dead-s``.  Pseudo set-piece windows of ``--window-s`` seconds (35 s: the length of every real corner,
throw-in, goal-kick and free-kick window of the Smart Tagging) are placed so that the restart lies at a random
offset in ``[--offset-min-s, --offset-max-s]`` from the window start (seeded), the anchoring is run with the
ball as tracked ("ball" cue) and once with the ball hidden everywhere (collective cue), and the error
``t*_found - t*_true`` is summarised in frames.  Only aggregates are printed.  Never touches BAR-ATH.

    python scripts/eval_anchor_accuracy.py --bench-dir ../mtah2027-external-benchmarks/cache
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for _extra in (REPO_ROOT / "src", REPO_ROOT):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from mtah2027.features.setpiece_anchor import AnchorParams, anchor_set_pieces  # noqa: E402
from mtah2027.validation import eval_bridge as eb  # noqa: E402

D, X = 2, 3


def true_restarts(gt: np.ndarray, fps: float, min_dead_s: float) -> list[int]:
    """First alive frame after each dead run of >= ``min_dead_s`` (inside graded frames)."""
    dead = gt == D
    d = np.diff(np.concatenate([[0], dead.astype(np.int8), [0]]))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    out = []
    for a, b in zip(starts, ends):
        if (b - a) / fps >= min_dead_s and b < len(gt) and gt[b] in (0, 1):
            out.append(int(b))
    return out


def match_errors(path: Path, params: AnchorParams, window_s: float, min_dead_s: float, offset_s=(3.0, 33.0),
                 skip_s: float = 10.0, seed: int = 0):
    z = np.load(path, allow_pickle=True)
    players, team = z["players_xy"].astype(np.float32), z["team_of_slot"].astype(np.int8)
    ball = z["ball_xy"].astype(np.float32)
    fps, pitch = float(z["fps"]), tuple(float(v) for v in z["pitch"])
    gt = eb.to_codes(z["gt"])
    ts = [t for t in true_restarts(gt, fps, min_dead_s) if t > skip_s * fps]
    rng = np.random.default_rng(seed)
    starts = [t - int(rng.uniform(*offset_s) * fps) for t in ts]
    sps = [SimpleNamespace(type="free kick", team=-1, start_frame=max(0, a),
                           end_frame=max(0, a) + int(window_s * fps), half=1) for a in starts]
    rows = []
    for cue, b in (("ball", ball), ("collective", np.full_like(ball, np.nan))):
        res = anchor_set_pieces(b, players, team, sps, (), fps=fps, pitch=pitch, params=params)
        for a, t in zip(res.anchors, ts):
            rows.append((cue, a.method, a.t_star - t if a.resolved else None, a.confidence))
    return rows


def summarise(rows) -> dict:
    out: dict = {}
    by = defaultdict(list)
    for cue, method, err, conf in rows:
        by[(cue, method)].append((err, conf))
    for (cue, method), items in sorted(by.items()):
        errs = np.array([e for e, _ in items if e is not None], float)
        out[f"{cue}->{method}"] = {
            "n": len(items), "resolved": int(len(errs)),
            **({"err_frames_q10_50_90": [round(float(np.quantile(errs, q)), 1) for q in (0.1, 0.5, 0.9)],
                "within_3f_pct": round(100.0 * float((np.abs(errs) <= 3).mean()), 1),
                "within_12f_pct": round(100.0 * float((np.abs(errs) <= 12).mean()), 1),
                "mean_abs_err_f": round(float(np.abs(errs).mean()), 1)} if len(errs) else {})}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bench-dir", required=True)
    ap.add_argument("--groups", nargs="+", default=["idsse", "metrica"])
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--window-s", type=float, default=35.0)
    ap.add_argument("--offset-min-s", type=float, default=3.0)
    ap.add_argument("--offset-max-s", type=float, default=33.0)
    ap.add_argument("--min-dead-s", type=float, default=5.0)
    ap.add_argument("--collective-only-incomplete", action="store_true",
                    help="matches with ball gaps (PFF, SkillCorner, Metrica): collective rows only")
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override AnchorParams (external tuning)")
    args = ap.parse_args(argv)
    import dataclasses
    import json

    params = AnchorParams.from_config(args.config)
    if args.set:
        kv = (item.partition("=") for item in args.set)
        params = dataclasses.replace(params, **{k: json.loads(v) for k, _, v in kv})
    for g in args.groups:
        files = sorted(Path(args.bench_dir).glob(f"{g}*.npz"))[: args.n]
        rows = []
        for f in files:
            z = np.load(f, allow_pickle=True)
            complete = bool(np.isfinite(z["ball_xy"]).all())
            if not complete and not args.collective_only_incomplete:
                continue
            got = match_errors(f, params, args.window_s, args.min_dead_s,
                               (args.offset_min_s, args.offset_max_s), seed=len(rows))
            rows += got if complete else [r for r in got if r[0] == "collective"]
        print(f"== {g}: {len(rows)} anchored windows")
        for k, v in summarise(rows).items():
            print(f"  {k:<22} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
