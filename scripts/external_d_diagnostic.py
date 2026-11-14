#!/usr/bin/env python
"""Where does the engine say ``D``?  Predicted vs true D share by ball visibility / tracked players on
EXTERNAL matches (Issue #37).  Same cuts as ``real_data_sanity.py`` so the BAR-ATH table can be read against
matches whose labels are known.  Aggregates only; never touches BAR-ATH.

    python scripts/external_d_diagnostic.py --bench-dir ../mtah2027-external-benchmarks/cache --groups pff
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for _extra in (REPO_ROOT / "src", REPO_ROOT, REPO_ROOT / "scripts"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from real_data_sanity import d_table, onpitch_counts  # noqa: E402

CUTS = ("all_graded", "ball_visible", "ball_hidden", "players_ge20", "players_lt20", "visible&players_ge20",
        "visible&players_lt20", "hidden&players_ge20", "hidden&players_lt20")


def match_tables(path: Path, config=None, degrade: bool = False, seed: int = 0, specialists=None) -> dict:
    from mtah2027.models.external_degrader import degrade_match
    from mtah2027.pipeline import run_core
    from mtah2027.validation import eval_bridge as eb

    z = np.load(path, allow_pickle=True)
    players, team = z["players_xy"].astype(np.float32), z["team_of_slot"].astype(np.int8)
    ball = z["ball_xy"].astype(np.float32)
    fps, pitch = float(z["fps"]), tuple(float(v) for v in z["pitch"])
    periods = [(int(a), int(b)) for a, b in z["period_bounds"]]
    gt = eb.to_codes(z["gt"])
    if "ball_visible" in z.files:
        ball = np.where(z["ball_visible"][:, None], ball, np.float32(np.nan)).astype(np.float32)
    if degrade:
        dm = degrade_match(players, team, ball, gt, fps=fps, seed=seed)
        players, ball = dm.players_xy, dm.ball_xy
    core = run_core(players, team, ball, play_blocks=periods, fps=fps, pitch=pitch, config_path=config,
                    log=lambda m: None, specialists=specialists)
    graded = gt != 3
    visible = np.isfinite(ball).all(axis=1)
    nh, na = onpitch_counts(players, team, pitch)
    n_pl = nh + na
    return {"gt": d_table(gt, graded, visible, n_pl, None),
            "pred": d_table(core.codes, graded, visible, n_pl, None)}


def pool(tables: list[dict]) -> dict:
    out = {}
    for cut in CUTS:
        rows = [(t["gt"][cut], t["pred"][cut]) for t in tables if cut in t["gt"]]
        if not rows:
            continue
        w = np.array([g["frames_pct_of_graded"] for g, _ in rows])
        avg = lambda vals: round(float(np.average(vals, weights=w + 1e-9)), 2)  # noqa: E731
        out[cut] = {"frames_pct": round(float(w.mean()), 2), "D_true_pct": avg([g["D_pct"] for g, _ in rows]),
                    "D_pred_pct": avg([p["D_pct"] for _, p in rows])}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bench-dir", required=True)
    ap.add_argument("--groups", nargs="+", default=["pff", "skillcorner", "idsse"])
    ap.add_argument("--n", type=int, default=4, help="matches per group (first n, sorted)")
    ap.add_argument("--degrade", action="store_true", help="apply the Issue #19 degrader (clean sources)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--specialists", choices=("off", "load"), default="off")
    args = ap.parse_args(argv)
    spec = None
    if args.specialists == "load":
        from mtah2027.models.specialist_fusion import resolve_specialists

        spec = resolve_specialists("load", config_path=args.config, prefer_file=True)
    for g in args.groups:
        files = sorted(Path(args.bench_dir).glob(f"{g}*.npz"))[: args.n]
        tabs = [match_tables(f, args.config, args.degrade, i, spec) for i, f in enumerate(files)]
        print(f"== {g} ({len(files)} matches, degrade={args.degrade}, specialists={args.specialists})")
        for cut, row in pool(tabs).items():
            print(f"  {cut:<22} frames {row['frames_pct']:6.2f}%  D true {row['D_true_pct']:6.2f}%  "
                  f"D pred {row['D_pred_pct']:6.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
