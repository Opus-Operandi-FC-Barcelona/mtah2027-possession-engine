#!/usr/bin/env python
"""Issue #70 probe (external matches only): how much does a COARSE ball position help the possession team?

A coarse proxy (for instance the pitch point under the TV camera centre) is far less accurate than the ball
but available where the ball is hidden.  This probe measures its information content, not a pipeline route: on
the 7 IDSSE matches (complete ball, calibrated degrader dead_affinity 33, seed 0, alive frames inside the
degrader's hidden-ball blocks) a leave-one-match-out gradient-boosting classifier predicts the possessing team from

* ``base``  : physics-only team-shape features (centroids, spread, line advance, mean velocity, tracked counts);
* ``+proxy``: ``base`` plus features relative to a proxy point = true ball + isotropic Gaussian noise of s metres
  (s = 0 is the oracle ball), smoothed over 1 s as a camera would be.

It prints the accuracy of the team on alive hidden frames per noise level (aggregates only).  Frame accuracy is a
proxy for spellF1, not a substitute: see the report for the mapping.

    python scripts/research/coarse_ball_proxy_curve.py --sigmas 0 5 10 15 25
"""
# ruff: noqa: E501
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[2]
for extra in (ROOT / "src", ROOT, ROOT / "scripts"):
    sys.path.insert(0, str(extra))

BENCH = ROOT.parent / "mtah2027-external-benchmarks" / "cache"


def load(path: Path, cfg_raw: dict, seed: int, index: int):
    from mtah2027.models.external_degrader import DegraderParams, degrade_match
    from mtah2027.validation import eval_bridge as eb

    z = np.load(path, allow_pickle=True)
    players, team = z["players_xy"].astype(np.float32), z["team_of_slot"].astype(np.int8)
    ball = z["ball_xy"].astype(np.float32)
    fps, pitch = float(z["fps"]), tuple(float(v) for v in z["pitch"])
    periods = [(int(a), int(b)) for a, b in z["period_bounds"]]
    gt = eb.to_codes(z["gt"])
    base = DegraderParams.from_config(cfg_raw)
    dm = degrade_match(players, team, ball, gt, fps=fps, seed=seed + index,
                       params=dataclasses.replace(base, dead_affinity=33.0))
    return dm.players_xy, team, ball, dm.ball_xy, gt, fps, pitch, periods


def smooth(a: np.ndarray, w: int) -> np.ndarray:
    k = np.ones(w) / w
    return np.stack([np.convolve(np.pad(a[:, i], (w // 2, w - 1 - w // 2), mode="edge"), k, "valid")
                     for i in range(a.shape[1])], axis=1)


def orient(players, team, periods, pitch):
    """Mirror each period so that the home team attacks +x (home keeper side from the min/max x of its slots)."""
    P = players.copy()
    flip = np.zeros(len(P), bool)
    L = pitch[0]
    for a, b in periods:
        h = P[a:b + 1][:, team == 0, 0]
        lo, hi = np.nanmedian(np.nanmin(h, axis=1)), np.nanmedian(np.nanmax(h, axis=1))
        home_left = (lo + hi) / 2 < L / 2
        if not home_left:
            flip[a:b + 1] = True
    P[flip, :, 0] = L - P[flip, :, 0]
    return P, flip


def base_features(P, team, fps, pitch):
    f = {}
    vel = np.full_like(P, np.nan)
    h = int(round(fps * 0.4))
    vel[h:-h] = (P[2 * h:] - P[:-2 * h]) / (2 * h / fps)
    for t, nm in ((0, "h"), (1, "a")):
        X = P[:, team == t]
        V = vel[:, team == t]
        with np.errstate(all="ignore"):
            f[f"{nm}_cx"], f[f"{nm}_cy"] = np.nanmean(X[..., 0], 1), np.nanmean(X[..., 1], 1)
            f[f"{nm}_sx"], f[f"{nm}_sy"] = np.nanstd(X[..., 0], 1), np.nanstd(X[..., 1], 1)
            f[f"{nm}_vx"], f[f"{nm}_vy"] = np.nanmean(V[..., 0], 1), np.nanmean(V[..., 1], 1)
            f[f"{nm}_sp"] = np.nanmean(np.linalg.norm(V, axis=-1), 1)
            f[f"{nm}_n"] = np.isfinite(X[..., 0]).sum(1).astype(float)
    f["dcx"] = f["h_cx"] - f["a_cx"]
    f["dvx"] = f["h_vx"] - f["a_vx"]
    return f


def proxy_features(P, team, proxy, pitch):
    f = {"px": proxy[:, 0], "py": proxy[:, 1]}
    with np.errstate(all="ignore"):
        for t, nm in ((0, "h"), (1, "a")):
            X = P[:, team == t]
            d = np.linalg.norm(X - proxy[:, None, :], axis=-1)
            f[f"{nm}_dmin"] = np.nanmin(np.where(np.isfinite(d), d, np.inf), axis=1)
            for r in (8.0, 15.0):
                f[f"{nm}_in{int(r)}"] = np.nansum(d < r, axis=1).astype(float)
            near = np.where(d < 15.0, 1.0, np.nan)[..., None] * X
            f[f"{nm}_loc_dx"] = np.nanmean(near[..., 0], 1) - proxy[:, 0]
        f["d_dmin"] = f["h_dmin"] - f["a_dmin"]
        f["d_in15"] = f["h_in15"] - f["a_in15"]
    f["d_dmin"] = np.where(np.isfinite(f["d_dmin"]), f["d_dmin"], 0.0)
    return f


def stack(f):
    return np.stack([np.nan_to_num(v, nan=-999.0, posinf=999.0, neginf=-999.0) for v in f.values()], axis=1)


def main(argv=None) -> int:
    import yaml

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0, 5, 10, 15, 25])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--matches", type=int, default=7)
    args = ap.parse_args(argv)
    cfg_raw = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    files = sorted(BENCH.glob("idsse_*.npz"))[:args.matches]
    data = []
    rng = np.random.default_rng(args.seed)
    for i, p in enumerate(files):
        print(f"loading {p.stem}", flush=True)
        players, team, ball_true, ball_deg, gt, fps, pitch, periods = load(p, cfg_raw, args.seed, i)
        P, flip = orient(players, team, periods, pitch)
        bt = ball_true.copy()
        bt[flip, 0] = pitch[0] - bt[flip, 0]
        hidden = ~np.isfinite(ball_deg).all(axis=1)
        codes = gt
        data.append((P, team, bt, hidden, codes, fps, pitch, periods, flip))
    from mtah2027.validation import eval_bridge as eb

    H, A = eb.H, eb.A
    rows = {s: [] for s in ["base"] + [f"s{int(s)}" for s in args.sigmas]}
    labels, groups = [], []
    for mi, (P, team, bt, hidden, codes, fps, pitch, periods, flip) in enumerate(data):
        idx = np.zeros(len(P), bool)
        for a, b in periods:
            idx[a:b + 1] = True
        sel = np.where(idx & hidden & np.isin(codes, [H, A]))[0][::args.stride]
        bf = base_features(P, team, fps, pitch)
        xb = stack(bf)
        y = (codes == H).astype(int)
        rows["base"].append(xb[sel])
        for s in args.sigmas:
            noise = rng.normal(0.0, 1.0, size=(len(P), 2)).astype(np.float32)
            w = int(fps) * 3
            noisy = bt + s * smooth(noise, w) * np.sqrt(w) if s > 0 else bt
            xp = stack(proxy_features(P, team, np.where(np.isfinite(noisy), noisy, 0.0), pitch))
            rows[f"s{int(s)}"].append(np.hstack([xb[sel], xp[sel]]))
        labels.append(y[sel])
        groups.append(np.full(len(sel), mi))
    y = np.concatenate(labels)
    g = np.concatenate(groups)
    print(f"matches {len(data)}, alive hidden-ball frames used {len(y)}; home share {100 * y.mean():.1f}%")
    for name, parts in rows.items():
        X = np.vstack(parts)
        acc = []
        for m in range(len(data)):
            tr, te = g != m, g == m
            clf = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, max_leaf_nodes=15, random_state=0)
            clf.fit(X[tr], y[tr])
            acc.append(float((clf.predict(X[te]) == y[te]).mean()))
        print(f"{name:>5}: LOMO team accuracy mean {100 * np.mean(acc):.2f}%  (per match min {100 * min(acc):.1f} max {100 * max(acc):.1f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
