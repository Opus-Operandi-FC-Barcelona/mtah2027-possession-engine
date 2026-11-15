# ruff: noqa: E501
"""Issue #70 probe: which external world does the real match resemble? (aggregates only)

For the label-free stoppage proxy "median tracked-player speed over 1 s < 1 m/s", prints the share of such frames among
ball-visible and ball-hidden graded frames of (a) the 7 IDSSE matches degraded with the calibrated degrader at
dead_affinity 1 / 33 (seed 0) and (b) the PFF matches with their real occlusions.  The same two numbers for the real
BAR-ATH come from ``real_ball_speed_semantics.py`` / ``real_broadcast_liveness_cue.py`` (visible 10.9 %, hidden 50.3 %
for the 'no native speed' frames).  No label is used for the proxy (labels only define the graded frames)."""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for extra in (ROOT / "src", ROOT, ROOT / "scripts"):
    sys.path.insert(0, str(extra))

from mtah2027.features._utils import speed_from_xy  # noqa: E402

BENCH = ROOT.parent / "mtah2027-external-benchmarks" / "cache"


def slow_share(players, ball, graded, fps):
    ps = speed_from_xy(players, fps, 1)
    with np.errstate(all="ignore"):
        pm = np.nanmedian(np.where(ps > 0, ps, np.nan), axis=1)
    pm = np.convolve(np.where(np.isfinite(pm), pm, 0.0), np.ones(int(fps)) / int(fps), mode="same")
    slow = pm < 1.0
    vis = np.isfinite(ball).all(axis=1)
    return slow[graded & vis], slow[graded & ~vis], float((~vis[graded]).mean())


def main(argv=None) -> int:
    import yaml

    from mtah2027.models.external_degrader import DegraderParams, degrade_match
    from mtah2027.validation import eval_bridge as eb

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--pff", type=int, default=8)
    args = ap.parse_args(argv)
    cfg_raw = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    base = DegraderParams.from_config(cfg_raw)
    pools: dict[str, list] = {"idsse_aff1": [], "idsse_aff33": [], "pff": []}
    for i, f in enumerate(sorted(BENCH.glob("idsse_*.npz"))):
        z = np.load(f, allow_pickle=True)
        pl, tm = z["players_xy"].astype(np.float32), z["team_of_slot"].astype(np.int8)
        ball, fps = z["ball_xy"].astype(np.float32), float(z["fps"])
        gt = eb.to_codes(z["gt"])
        graded = gt != eb.X
        for aff, key in ((1.0, "idsse_aff1"), (33.0, "idsse_aff33")):
            dm = degrade_match(pl, tm, ball, gt, fps=fps, seed=i, params=dataclasses.replace(base, dead_affinity=aff))
            pools[key].append(slow_share(dm.players_xy, dm.ball_xy, graded, fps))
    for f in sorted(BENCH.glob("pff_*.npz"))[:args.pff]:
        z = np.load(f, allow_pickle=True)
        ball = z["ball_xy"].astype(np.float32)
        if "ball_visible" in z.files:
            ball = np.where(z["ball_visible"][:, None], ball, np.float32(np.nan))
        gt = eb.to_codes(z["gt"])
        pools["pff"].append(slow_share(z["players_xy"].astype(np.float32), ball, gt != eb.X, float(z["fps"])))
    for k, rows in pools.items():
        v = np.concatenate([r[0] for r in rows])
        h = np.concatenate([r[1] for r in rows])
        hid = np.mean([r[2] for r in rows])
        print(f"{k}: hidden share {100 * hid:.1f}%  slow | visible {100 * v.mean():.1f}%  slow | hidden {100 * h.mean():.1f}%  ({len(rows)} matches)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
