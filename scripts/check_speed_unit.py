#!/usr/bin/env python
"""Issue #42: which unit does the native ATD speed channel really have?  Aggregate output only.

Prints the median ratio ``native speed (as loaded) / speed derived from positions`` over frames where both
exist, plus the unit the XML declares.  ~1.0 means the loader's unit handling is right; ~3.6 means the values
are really km/h but were not converted; ~0.28 means they are m/s but were divided by 3.6.  Players and ball
are checked separately.  No per-frame output."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "src", ROOT):
    sys.path.insert(0, str(extra))

from mtah2027.features._utils import speed_from_xy  # noqa: E402
from mtah2027.io import atd_loader as al  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args(argv)
    td = al.load_atd(args.data_dir)
    native = td.players_speed
    if not np.isfinite(native).any():
        print("no native speed channel in this ATD")
        return 0
    derived = speed_from_xy(td.players_xy, td.fps, 2)
    ok = np.isfinite(native) & np.isfinite(derived) & (derived > 0.5) & (derived < 9.0)
    ratio = native[ok] / derived[ok]
    print(f"declared channels: {td.report.get('player_channels')}")
    print(f"players: pairs {int(ok.sum())}  median native/derived ratio {np.median(ratio):.3f}  "
          f"(p25 {np.percentile(ratio, 25):.3f}, p75 {np.percentile(ratio, 75):.3f})  "
          f"unit factor applied {td.report.get('speed_unit_factor_applied')}")
    bs = td.ball_speed
    bd = speed_from_xy(td.ball_xy[:, None, :], td.fps, 2)[:, 0]
    bok = np.isfinite(bs) & np.isfinite(bd) & (bd > 0.5) & (bd < 30.0)
    if bok.any():
        med = np.median(bs[bok]) / np.median(bd[bok])
        print(f"ball: pairs {int(bok.sum())}  median(s)/median(derived) {med:.3f}"
              f"  median of ratios {np.median(bs[bok] / bd[bok]):.3f}  "
              f"unit factor applied {td.report.get('ball_speed_unit_factor_applied')}")
    else:
        print("ball: no native speed channel")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
