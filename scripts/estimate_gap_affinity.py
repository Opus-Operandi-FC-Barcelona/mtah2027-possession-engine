#!/usr/bin/env python
"""Issue #48: label-free estimate of the hidden-ball <-> dead-ball affinity, with its validation.

    python scripts/estimate_gap_affinity.py calibrate --targets 1 3 7 18       # dead_affinity -> realised OR
    python scripts/estimate_gap_affinity.py validate                           # IDSSE planted OR, PFF, Metrica
    python scripts/estimate_gap_affinity.py barath --data-dir data              # aggregates only

The estimator (``mtah2027.validation.gap_affinity``) uses ball-free player features, a classifier trained on
other matches with known state and prior-shift EM, separately for the <20 / >=20 tracked-player strata.
``validate`` reports estimate vs truth for the three kinds of other matches; ``barath`` applies the classifier
(trained on the IDSSE matches degraded by the calibrated degrader) to the competition ATD and prints only
aggregates (never per-frame or per-gap rows).  Nothing here reads the public ground truth.
"""
# ruff: noqa: E501
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for _extra in (REPO_ROOT / "src", REPO_ROOT):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from mtah2027.models.external_degrader import DegraderParams, degrade_match  # noqa: E402
from mtah2027.validation import eval_bridge as eb  # noqa: E402
from mtah2027.validation import gap_affinity as ga  # noqa: E402

BENCH = REPO_ROOT.parent / "mtah2027-external-benchmarks" / "cache"


def load(path):
    z = np.load(path, allow_pickle=True)
    return (z["players_xy"].astype(np.float32), z["team_of_slot"].astype(np.int8), z["ball_xy"].astype(np.float32),
            eb.to_codes(z["gt"]), float(z["fps"]), z)


def idsse_files(bench):
    return sorted(Path(bench).glob("idsse_*.npz"))


def realised_or(files, aff, seed=0, params=None):
    """Pooled realised odds ratio (and missing share) of the degrader on ``files`` (ball occlusion only)."""
    p = dataclasses.replace(params or DegraderParams.from_config(), dead_affinity=float(aff))
    hd = hn = vd = vn = hid = tot = 0
    for f in files:
        pl, tm, ball, gt, fps, _ = load(f)
        dm = degrade_match(pl, tm, ball, gt, fps=fps, seed=seed, params=p, artefacts=False)
        g = gt != 3
        h = dm.hidden & g
        v = ~dm.hidden & g
        hd += int((gt[h] == 2).sum()); hn += int(h.sum()); vd += int((gt[v] == 2).sum()); vn += int(v.sum())  # noqa: E702
        hid += int(h.sum()); tot += int(g.sum())  # noqa: E702
    return ga.odds_ratio(hd / hn, vd / vn), hid / tot


def calibrate(files, targets, params=None):
    out = {}
    for t in targets:
        if abs(t - 1.0) < 1e-9:
            out[t] = (1.0, realised_or(files, 1.0, params=params)[0])
            continue
        lo, hi = 0.0, 8.0                       # log affinity
        for _ in range(14):
            mid = (lo + hi) / 2
            r = realised_or(files, float(np.exp(mid)), params=params)[0]
            lo, hi = (mid, hi) if r < t else (lo, mid)
        aff = float(np.exp((lo + hi) / 2))
        out[t] = (aff, realised_or(files, aff, params=params))
        print(f"target OR {t}: dead_affinity {aff:.3f} -> realised OR {out[t][1][0]:.2f}, "
              f"missing {100 * out[t][1][1]:.1f} %", flush=True)
    return out


def reference_features(files, params, seed=0):
    """(features, is_dead, graded) of each match degraded (players only matter) by the calibrated degrader."""
    out = {}
    for f in files:
        pl, tm, ball, gt, fps, z = load(f)
        dm = degrade_match(pl, tm, ball, gt, fps=fps, seed=seed, params=params)
        g = gt != 3
        out[f.stem] = (ga.player_features(dm.players_xy, fps), gt == 2, g)
    return out


def fit_on(ref, names):
    x = np.concatenate([ref[n][0][ref[n][2]] for n in names])
    y = np.concatenate([ref[n][1][ref[n][2]] for n in names])
    return ga.fit_dead_classifier(x, y)


def _fmt(e, t):
    return (f"hidden D {100 * e['d_share_hidden']:5.1f}% (true {100 * t['d_share_hidden']:5.1f}%)  "
            f"visible D {100 * e['d_share_visible']:5.1f}% (true {100 * t['d_share_visible']:5.1f}%)  "
            f"OR {e['odds_ratio']:6.2f} (true {t['odds_ratio']:6.2f})")


def validate(args):
    base = DegraderParams.from_config()
    files = idsse_files(args.bench_dir)
    print("reference features (IDSSE, calibrated degrader, aff 1) ...", flush=True)
    ref = reference_features(files, base)
    rows = []
    # (i) IDSSE with planted OR
    cal = ({t: (a, None) for t, a in zip(args.targets, args.affs)} if args.affs
           else calibrate(files, args.targets, base))
    for target in args.targets:
        aff = cal[target][0]
        pooled_e, pooled_t = [], []
        for f in files:
            clf = fit_on(ref, [x.stem for x in files if x != f])
            pl, tm, ball, gt, fps, z = load(f)
            dm = degrade_match(pl, tm, ball, gt, fps=fps, seed=11, params=dataclasses.replace(base, dead_affinity=aff))
            g = gt != 3
            inter, vis = ga.gap_masks(dm.ball_xy, g, fps)
            feats = ga.player_features(dm.players_xy, fps)
            e, t = ga.estimate(clf, feats, inter, vis), ga.true_shares(gt, inter, vis)
            pooled_e.append(e); pooled_t.append(t)  # noqa: E702
            rows.append({"case": f"idsse_planted_{target}", "match": f.stem, "est": e, "true": t})
        w = np.array([e["n_hidden"] for e in pooled_e], float)
        mean = lambda k, L: float(np.average([d[k] for d in L], weights=w))  # noqa: E731
        err = [abs(e["d_share_hidden"] - t["d_share_hidden"]) for e, t in zip(pooled_e, pooled_t)]
        print(f"IDSSE planted OR {target:>3} (aff {aff:.2f}): est hidden D {100 * mean('d_share_hidden', pooled_e):5.1f}% "
              f"true {100 * mean('d_share_hidden', pooled_t):5.1f}%   est visible D "
              f"{100 * mean('d_share_visible', pooled_e):5.1f}% true {100 * mean('d_share_visible', pooled_t):5.1f}%   "
              f"est OR {np.exp(np.average(np.log([e['odds_ratio'] for e in pooled_e]), weights=w)):6.2f} "
              f"true OR {np.exp(np.average(np.log([t['odds_ratio'] for t in pooled_t]), weights=w)):6.2f}   "
              f"max |err hidden D| {100 * max(err):.1f} pts", flush=True)
    # (ii) PFF and (iii) Metrica: classifier trained on the 7 IDSSE matches
    clf = fit_on(ref, [f.stem for f in files])
    for kind, glob_, n in (("pff", "pff_*.npz", args.max_pff), ("metrica", "metrica_*.npz", 3)):
        es, ts = [], []
        for f in sorted(Path(args.bench_dir).glob(glob_))[:n]:
            pl, tm, ball, gt, fps, z = load(f)
            if "ball_visible" in z.files:
                ball = np.where(z["ball_visible"][:, None], ball, np.float32(np.nan)).astype(np.float32)
            g = gt != 3
            inter, vis = ga.gap_masks(ball, g, fps)
            if inter.sum() < 500:
                continue
            e = ga.estimate(clf, ga.player_features(pl, fps), inter, vis)
            t = ga.true_shares(gt, inter, vis)
            es.append(e); ts.append(t)  # noqa: E702
            rows.append({"case": kind, "match": f.stem, "est": e, "true": t})
        if not es:
            continue
        w = np.array([e["n_hidden"] for e in es], float)
        avg = lambda k, L: float(np.average([d[k] for d in L], weights=w))  # noqa: E731
        e_pool = {k: avg(k, es) for k in ("d_share_hidden", "d_share_visible")}
        t_pool = {k: avg(k, ts) for k in ("d_share_hidden", "d_share_visible")}
        e_pool["odds_ratio"] = ga.odds_ratio(e_pool["d_share_hidden"], e_pool["d_share_visible"])
        t_pool["odds_ratio"] = ga.odds_ratio(t_pool["d_share_hidden"], t_pool["d_share_visible"])
        print(f"{kind.upper()} ({len(es)} matches, pooled): {_fmt(e_pool, t_pool)}", flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1, default=float), encoding="utf-8")
    return 0


def barath(args):
    from mtah2027.config import load_config
    from mtah2027.io.atd_loader import load_atd

    base = DegraderParams.from_config()
    files = idsse_files(args.bench_dir)
    ref = reference_features(files, base)
    clf = fit_on(ref, [f.stem for f in files])
    match = load_config().match
    td = load_atd(args.data_dir, use_cache=True)
    n = len(td.ball_xy)
    graded = np.zeros(n, bool)
    for a, b in (match.half1, match.half2_play):          # the dead start of half 2 is known and excluded
        graded[int(a):int(b) + 1] = True
    inter, vis = ga.gap_masks(td.ball_xy, graded, td.fps)
    feats = ga.player_features(td.players_xy, td.fps)
    e = ga.estimate(clf, feats, inter, vis)
    print(f"BAR-ATH (label-free): hidden-gap frames {e['n_hidden']}, visible {e['n_visible']}")
    print(f"  estimated D share in hidden gaps {100 * e['d_share_hidden']:.1f} %, in visible frames "
          f"{100 * e['d_share_visible']:.1f} %, odds ratio {e['odds_ratio']:.2f}")
    for which in ("hidden", "visible"):
        for name, v in e["strata"][which].items():
            print(f"  stratum {name:<4} {which:<7}: n={v['n']:>7}  D share {100 * v['d_share']:.1f} %")
    lo = {w: e["strata"][w]["low"]["d_share"] for w in ("hidden", "visible")}
    hi = {w: e["strata"][w]["high"]["d_share"] for w in ("hidden", "visible")}
    for name, s in (("<20 players", lo), (">=20 players", hi)):
        if np.isfinite(s["hidden"]) and np.isfinite(s["visible"]):
            print(f"  OR within stratum {name}: {ga.odds_ratio(s['hidden'], s['visible']):.2f}")
    bf = ga.boundary_features(td.ball_xy, td.players_xy, graded, td.fps)
    print("  gap borders (aggregate):", json.dumps({k: {a: round(b, 3) for a, b in v.items()} for k, v in bf.items()}))
    if args.json:
        Path(args.json).write_text(json.dumps({"estimate": e, "boundary": bf}, default=float), encoding="utf-8")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("calibrate", "validate", "barath"))
    ap.add_argument("--bench-dir", default=str(BENCH))
    ap.add_argument("--data-dir", default=str(REPO_ROOT.parent / "mtah2027-possession-engine" / "data"))
    ap.add_argument("--targets", type=float, nargs="*", default=[1, 5, 18])
    ap.add_argument("--affs", type=float, nargs="*", default=None,
                    help="dead_affinity for each target (skips the bisection; see the grid in the report)")
    ap.add_argument("--max-pff", type=int, default=10)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    if args.mode == "calibrate":
        calibrate(idsse_files(args.bench_dir), args.targets)
        return 0
    return validate(args) if args.mode == "validate" else barath(args)


if __name__ == "__main__":
    raise SystemExit(main())
