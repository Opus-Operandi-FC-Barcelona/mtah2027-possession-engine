#!/usr/bin/env python
"""Issue #70 probe: do the ATD ball channels carry information while the ball position is hidden?

Aggregates only.  For the graded frames of the real tactical (and optionally broadcast) ATD: share of frames with a
finite ball position, a finite native ball speed ``s``, and their overlap; percentiles of ``s`` where the position is
hidden vs visible.  Nothing per frame is printed and no label is read."""
# ruff: noqa: E501
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for extra in (ROOT / "src", ROOT):
    sys.path.insert(0, str(extra))

from mtah2027.config import load_config  # noqa: E402
from mtah2027.io import atd_loader as al  # noqa: E402


def _pct(x: np.ndarray, qs=(5, 25, 50, 75, 95)) -> str:
    if x.size == 0:
        return "n=0"
    return "n=%d " % x.size + " ".join(f"p{q}={np.percentile(x, q):.2f}" for q in qs)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--source", default="tactical", choices=["tactical", "broadcast"])
    args = ap.parse_args(argv)
    cfg = load_config()
    td = al.load_atd(args.data_dir, source=args.source)
    n = td.n_frames
    graded = np.zeros(n, dtype=bool)
    for a, b in cfg.match.graded_blocks:
        graded[a:min(b, n - 1) + 1] = True
    vis = td.ball_visible
    print(f"source={args.source} frames={n} graded={int(graded.sum())}")
    print("ball channels:", sorted(td.ball_channels))
    for name, arr in sorted(td.ball_channels.items()):
        fin = np.isfinite(arr)
        g = graded
        print(f"  channel {name}: finite on graded {100 * fin[g].mean():.2f}% | pos visible: "
              f"{100 * fin[g & vis].mean():.2f}% | pos hidden: {100 * fin[g & ~vis].mean():.2f}%")
    s = td.ball_speed
    fin = np.isfinite(s)
    print(f"graded: position visible {100 * vis[graded].mean():.2f}%  hidden {100 * (~vis[graded]).mean():.2f}%")
    print(f"graded hidden-position frames with finite s: {int((graded & ~vis & fin).sum())} "
          f"({100 * (fin[graded & ~vis]).mean():.2f}% of hidden)")
    print(f"graded visible-position frames with finite s: {int((graded & vis & fin).sum())} "
          f"({100 * fin[graded & vis].mean():.2f}% of visible)")
    for lab, m in (("hidden", graded & ~vis & fin), ("visible", graded & vis & fin)):
        v = s[m]
        print(f"s | {lab}: {_pct(v)}  exact-zero {100 * (v == 0).mean() if v.size else 0:.1f}%  "
              f"distinct {np.unique(np.round(v, 3)).size if v.size else 0}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
