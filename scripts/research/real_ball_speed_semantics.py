#!/usr/bin/env python
"""Issue #70 probe: what does the sparse native ball speed channel ``s`` of the real ATD mean?

Aggregates only (runs, shares, medians).  Compares frames where ``s`` is finite with the rest: ball speed derived
from positions, mean outfield speed of the tracked players (a label-free stoppage proxy), position visibility."""
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
from mtah2027.features._utils import speed_from_xy  # noqa: E402
from mtah2027.io import atd_loader as al  # noqa: E402


def runs(mask: np.ndarray) -> np.ndarray:
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    s, e = np.where(d == 1)[0], np.where(d == -1)[0]
    return e - s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args(argv)
    cfg = load_config()
    td = al.load_atd(args.data_dir)
    n = td.n_frames
    graded = np.zeros(n, dtype=bool)
    for a, b in cfg.match.graded_blocks:
        graded[a:min(b, n - 1) + 1] = True
    vis, s = td.ball_visible, td.ball_speed
    fin = np.isfinite(s) & graded
    print(f"graded {int(graded.sum())}  finite-s {int(fin.sum())} ({100 * fin.sum() / graded.sum():.2f}%)")
    r = runs(fin)
    print(f"runs of finite s: {r.size}  length frames p10/p50/p90/max = "
          f"{np.percentile(r, 10):.0f}/{np.percentile(r, 50):.0f}/{np.percentile(r, 90):.0f}/{r.max()}")
    r2 = runs(~np.isfinite(s) & graded)
    print(f"gaps between: median {np.median(r2):.0f} frames")
    bd = speed_from_xy(td.ball_xy[:, None, :], td.fps, 2)[:, 0]
    ok = vis & np.isfinite(bd) & graded
    for lab, m in (("finite-s", ok & fin), ("no-s", ok & ~fin)):
        v = bd[m]
        print(f"derived ball speed | {lab}: n={v.size} <0.4 m/s {100 * (v < 0.4).mean():.1f}%  "
              f"0.4-2 {100 * ((v >= 0.4) & (v < 2)).mean():.1f}%  >=2 {100 * (v >= 2).mean():.1f}%  "
              f"median {np.median(v):.2f}")
    # outfield tracked-player speed proxy
    ps = speed_from_xy(td.players_xy, td.fps, 1)
    with np.errstate(all="ignore"):
        pm = np.nanmedian(np.where(ps > 0, ps, np.nan), axis=1)
    for lab, m in (("finite-s", fin), ("no-s", graded & ~fin),
                   ("no-s & ball visible", graded & ~fin & vis), ("no-s & ball hidden", graded & ~fin & ~vis)):
        v = pm[m & np.isfinite(pm)]
        print(f"median player speed | {lab}: n={v.size} median {np.median(v):.2f} m/s  share<1.0 {100 * (v < 1.0).mean():.1f}%")
    # corr of finite-s indicator with slow players, within hidden frames
    hid = graded & ~vis & np.isfinite(pm)
    for lab, m in (("hidden & finite-s", hid & fin), ("hidden & no-s", hid & ~fin)):
        v = pm[m]
        print(f"median player speed | {lab}: n={v.size} median {np.median(v):.2f}  share<1.0 {100 * (v < 1.0).mean():.1f}%")
    # ball position of the visible finite-s frames: out of pitch / near lines share
    bx, by = td.ball_xy[:, 0], td.ball_xy[:, 1]
    L, W = td.pitch
    out = (bx < 0) | (bx > L) | (by < 0) | (by > W)
    for lab, m in (("finite-s", vis & fin), ("no-s", vis & graded & ~fin)):
        print(f"ball outside pitch | {lab}: {100 * out[m].mean():.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
