#!/usr/bin/env python
"""Issue #70 probe: is "the TV shows live play here" (a synchronised broadcast block) a liveness cue? (aggregates only)

Uses the label-free stoppage proxy "median tracked-player speed over 1 s < 1 m/s" (no label is read).  Compares
covered vs uncovered graded frames, split by tactical ball visibility, and the proxy just before / after the end of
every synchronised block (the TV cutting away to a replay / close-up)."""
# ruff: noqa: E501
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for extra in (ROOT / "src", ROOT, ROOT / "scripts", ROOT / "scripts" / "research"):
    sys.path.insert(0, str(extra))

from real_broadcast_beyond_ball import pct, q, synced_cover  # noqa: E402

from mtah2027.features._utils import speed_from_xy  # noqa: E402


def runs(mask):
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--cache-dir", default=str(ROOT / "cache"))
    args = ap.parse_args(argv)
    td, btd, bh, res, graded, bf, p = synced_cover(args.data_dir, args.cache_dir)
    fps = int(td.fps)
    cov = (bf >= 0) & graded
    unc = graded & ~cov
    ps = speed_from_xy(td.players_xy, td.fps, 1)
    with np.errstate(all="ignore"):
        pm = np.nanmedian(np.where(ps > 0, ps, np.nan), axis=1)
    pm = np.where(np.isfinite(pm), pm, 0.0)
    k = np.ones(fps) / fps
    pm = np.convolve(pm, k, mode="same")
    slow = pm < 1.0
    vt = td.ball_visible
    print(f"graded {int(graded.sum())}: covered {pct(cov[graded])}; slow-player proxy overall {pct(slow[graded])}")
    for lab, m in (("covered", cov), ("uncovered", unc), ("covered & ball visible", cov & vt),
                   ("covered & ball hidden", cov & ~vt), ("uncovered & ball visible", unc & vt),
                   ("uncovered & ball hidden", unc & ~vt)):
        print(f"slow proxy | {lab}: n={int(m.sum())}  slow {pct(slow[m])}  median player speed {np.median(pm[m]):.2f} m/s")
    s0, e0 = runs(unc)
    dur = (e0 - s0) / fps
    print(f"uncovered stretches: {len(dur)}; duration s p10/50/90 {q(dur)}; share of uncovered frames in stretches > 30 s "
          f"{100 * dur[dur > 30].sum() / dur.sum():.1f}%")
    cs, ce = runs(cov)
    w = 5 * fps
    before, after = [], []
    for e in ce:
        if e + w < len(slow) and e - w >= 0 and graded[e - w:e + w].all():
            before.append(slow[e - w:e].mean())
            after.append(slow[e:e + w].mean())
    print(f"block ends with a full +-5 s graded window: n={len(before)}; slow share 5 s before {100 * np.mean(before):.1f}% "
          f"-> 5 s after {100 * np.mean(after):.1f}%")
    before, after = [], []
    for s in cs:
        if s + w < len(slow) and s - w >= 0 and graded[s - w:s + w].all():
            before.append(slow[s - w:s].mean())
            after.append(slow[s:s + w].mean())
    print(f"block starts: n={len(before)}; slow share 5 s before {100 * np.mean(before):.1f}% -> 5 s after {100 * np.mean(after):.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
