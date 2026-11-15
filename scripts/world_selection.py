#!/usr/bin/env python
"""Issue #71: label-free world selection (adversarial validation) for the hidden-ball frames.

    python scripts/world_selection.py features --data-dir data            # ball-free features per world (cache/)
    python scripts/world_selection.py fit      --out reports/issue71_world_selection.json
    python scripts/world_selection.py rescore  --fit reports/issue71_domain_fit.json --out reports/issue71_world_selection.json

``features`` writes per-frame feature matrices of the hidden-ball graded frames to ``--work-dir`` (git-ignored ``cache/``;
the real-match file never leaves the machine and is never printed).  ``fit`` trains one cross-fitted domain classifier
per external world (real BAR-ATH vs IDSSE degraded at ``dead_affinity`` 33 / 1 vs PFF) and prints AUCs, feature usage,
two-sample statistics, the implied mixture of worlds and the match weights.  ``rescore`` aggregates the existing
per-cell results of Issues #44 / #55 / #57 under the weighted metric (no model is retrained).  Aggregates only.
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

from mtah2027.validation import world_selection as ws  # noqa: E402

BENCH = REPO_ROOT.parent / "mtah2027-external-benchmarks" / "cache"
WORLDS = ("idsse33", "idsse1", "pff")
STRIDE_EXT, STRIDE_REAL = 10, 5


def cmd_features(args) -> int:
    import yaml

    from mtah2027.models.external_degrader import DegraderParams, degrade_match
    from mtah2027.validation import eval_bridge as eb

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    if args.data_dir:
        from mtah2027.config import load_config
        from mtah2027.io import atd_loader as al

        cfg = load_config()
        td = al.load_atd(args.data_dir)
        graded = np.zeros(td.n_frames, bool)
        for a, b in cfg.match.graded_blocks:
            graded[a:min(b, td.n_frames - 1) + 1] = True
        x, idx = ws.hidden_ball_features(td.players_xy, td.team_of_slot, td.ball_xy, td.fps, td.pitch, graded,
                                         STRIDE_REAL)
        np.savez_compressed(work / "real.npz", x=x.astype(np.float32), frame=idx)
        print(f"real: {len(idx)} hidden-ball graded rows (stride {STRIDE_REAL}), {int(graded.sum())} graded frames")
    cfg_raw = yaml.safe_load((REPO_ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    base = DegraderParams.from_config(cfg_raw)
    for i, f in enumerate(sorted(BENCH.glob("idsse_*.npz"))):
        z = np.load(f, allow_pickle=True)
        pl, tm = z["players_xy"].astype(np.float32), z["team_of_slot"].astype(np.int8)
        ball, fps = z["ball_xy"].astype(np.float32), float(z["fps"])
        gt = eb.to_codes(z["gt"])
        graded = gt != eb.X
        for aff, key in ((33.0, "idsse33"), (1.0, "idsse1")):
            dm = degrade_match(pl, tm, ball, gt, fps=fps, seed=i, params=dataclasses.replace(base, dead_affinity=aff))
            x, idx = ws.hidden_ball_features(dm.players_xy, tm, dm.ball_xy, fps, tuple(float(v) for v in z["pitch"]),
                                             graded, STRIDE_EXT)
            np.savez_compressed(work / f"{key}__{f.stem}.npz", x=x.astype(np.float32), frame=idx)
        print(f"{f.stem}: done", flush=True)
    for f in sorted(BENCH.glob("pff_*.npz")):
        z = np.load(f, allow_pickle=True)
        ball = np.where(z["ball_visible"][:, None], z["ball_xy"].astype(np.float32), np.float32(np.nan))
        gt = eb.to_codes(z["gt"])
        x, idx = ws.hidden_ball_features(z["players_xy"].astype(np.float32), z["team_of_slot"].astype(np.int8), ball,
                                         float(z["fps"]), tuple(float(v) for v in z["pitch"]), gt != eb.X, STRIDE_EXT)
        np.savez_compressed(work / f"pff__{f.stem}.npz", x=x.astype(np.float32), frame=idx)
    print("features done")
    return 0


def _load_world(work: Path, key: str):
    xs, ids = [], []
    for f in sorted(work.glob(f"{key}__*.npz")):
        x = np.load(f)["x"]
        xs.append(x)
        ids.append(np.full(len(x), f.stem.split("__", 1)[1], dtype=object))
    return np.concatenate(xs), np.concatenate(ids)


def cmd_fit(args) -> int:
    work = Path(args.work_dir)
    real = np.load(work / "real.npz")
    xr, fr = real["x"], real["frame"]
    block = (fr // (25 * 60)).astype(np.int64)          # one-minute blocks never split between folds
    names = ws.FEATURE_NAMES
    report: dict = {"features": list(names), "n_real_rows": int(len(xr)), "worlds": {}}
    ratio_real = np.zeros((len(xr), len(WORLDS)))
    match_w: dict = {}
    for wi, key in enumerate(WORLDS):
        xe, ide = _load_world(work, key)
        names_m = list(dict.fromkeys(ide.tolist()))
        gid_e = np.array([names_m.index(m) for m in ide]) + 10_000
        x = np.concatenate([xr, xe])
        y = np.concatenate([np.ones(len(xr)), np.zeros(len(xe))])
        groups = np.concatenate([block, gid_e])
        p, imp = ws.cross_fit_probabilities(x, y, groups, n_folds=args.folds, seed=args.seed, max_iter=args.max_iter)
        ok = np.isfinite(p)
        wgt = np.where(y == 1, 0.5 / (y == 1).sum(), 0.5 / (y == 0).sum())
        auc = ws.auc_score(y[ok], p[ok], wgt[ok])
        pm_auc = [ws.auc_score(np.concatenate([np.ones(len(xr)), np.zeros((ide == m).sum())]),
                               np.concatenate([p[:len(xr)], p[len(xr):][ide == m]])) for m in names_m[:5]]
        ratio_real[:, wi] = ws.density_ratio(p[:len(xr)])
        r_e = ws.density_ratio(p[len(xr):])
        mw = ws.match_weights(r_e, ide)
        match_w[key] = mw
        top = np.argsort(-imp)[:8]
        stats = {r["feature"]: r for r in ws.two_sample_stats(xr, xe)}
        report["worlds"][key] = {
            "n_rows": int(len(xe)), "n_matches": len(names_m), "auc": round(auc, 4),
            "auc_vs_single_match_first5": [round(v, 4) for v in pm_auc],
            "top_features": [{"feature": names[j], "auc_drop": round(float(imp[j]), 4), **{k: round(v, 3) for k, v in stats[names[j]].items() if k != "feature"}} for j in top],
            "ess_frames": round(ws.effective_sample_size(np.minimum(r_e, np.quantile(r_e, 0.99))), 1),
            "ess_matches": round(ws.effective_sample_size(np.array(list(mw.values()))), 2),
            "match_weights": {m: round(v, 5) for m, v in mw.items()},
        }
        print(f"{key}: AUC {auc:.4f} (rows {len(xe)}, matches {len(names_m)}); ESS frames {report['worlds'][key]['ess_frames']}, matches {report['worlds'][key]['ess_matches']}")
        for t in report["worlds"][key]["top_features"][:5]:
            print(f"   {t['feature']:<24} auc_drop {t['auc_drop']:.4f}  median real {t['median_a']:.3f} vs {t['median_b']:.3f}  smd {t['smd']:+.2f}")
    pi = ws.fit_mixture(ratio_real)
    rng = np.random.default_rng(args.seed)
    ub = np.unique(block)
    boots = []
    for _ in range(args.boot):
        pick = rng.choice(ub, len(ub), replace=True)
        rows = np.concatenate([np.flatnonzero(block == b) for b in pick])
        boots.append(ws.fit_mixture(ratio_real[rows], n_iter=200))
    boots = np.array(boots)
    nearest = np.bincount(np.argmax(1.0 / ratio_real, axis=1), minlength=len(WORLDS)) / len(xr)
    report["mixture"] = {k: round(float(v), 4) for k, v in zip(WORLDS, pi)}
    report["mixture_boot_ci90"] = {k: [round(float(np.quantile(boots[:, i], q)), 4) for q in (0.05, 0.95)] for i, k in enumerate(WORLDS)}
    report["nearest_world_share"] = {k: round(float(v), 4) for k, v in zip(WORLDS, nearest)}
    print("implied mixture pi:", report["mixture"], "CI90:", report["mixture_boot_ci90"])
    print("share of real hidden frames whose density is highest in each world:", report["nearest_world_share"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


# ---------------------------------------------------------------------------------------------
# re-score of the existing per-cell results under the weighted metric (Issue #71, no retraining)
# ---------------------------------------------------------------------------------------------
def _cells(dirs, cand, first_n=None):
    """``{(world, seed): {match: spellF1}}`` of candidate ``cand`` from decoder_sweep cell files in ``dirs``."""
    out: dict = {}
    for d in dirs:
        for p in sorted(Path(d).glob("*.json")):
            r = json.loads(p.read_text(encoding="utf-8"))
            if cand not in r["results"]:
                continue
            c = r["cell"]
            if first_n and c["world"] == "pff" and c["index"] >= first_n:
                continue
            out.setdefault((c["world"], c["seed"]), {})[c["match"]] = r["results"][cand]["spell_f1"]
    return out


def _world_scores(cells: dict, seed: int) -> dict:
    """Scores per weighted world for one seed (single-seed worlds use seed 0, as preregistered)."""
    g = lambda w, s: cells.get((w, s)) or cells[(w, 0)]            # noqa: E731
    return {"idsse33": g("b33", seed), "idsse1": g("b1", seed), "pff": g("pff", 0)}


def cmd_rescore(args) -> int:
    root = REPO_ROOT.parent
    w55, w57, w44 = root / "engine-wt-55-work", root / "engine-wt-57-work", root / "engine-wt-44-work"
    fit = json.loads(Path(args.fit).read_text(encoding="utf-8"))
    pi = {k: float(fit["mixture"][k]) for k in WORLDS}
    mw = {k: fit["worlds"][k]["match_weights"] for k in WORLDS}
    # variant -> directories per world and candidate name (existing cell results of Issues #44 / #55 / #57)
    variants = {
        "production": {"cand": "default", "dirs": [w55 / "base_b33", w55 / "base_b1", w55 / "base_pff", w55 / "base_unseen"]},
        "v55_pff56": {"cand": "default", "dirs": [w55 / "cand_b33", w55 / "cand_b1", w55 / "cand_pff", w55 / "cand_unseen"]},
        "v57_hard_idsse": {"cand": "c_idsse_hard", "dirs": [w57 / "res_b33", w57 / "res_b1", w57 / "res_pff", w57 / "res_unseen"]},
        "v57_hard_pff": {"cand": "c_pff_hard", "dirs": [w57 / "res_b33p", w57 / "res_b1", w57 / "res_pff", w57 / "res_unseen"]},
        "deg_legacy_aff1": {"cand": "mixed_only", "dirs": [w44 / "f3_b33", w44 / "f3_b1", w44 / "f3_pff", w44 / "f3_unseen"]},
    }
    seeds = (0, 1, 2)
    res: dict = {}
    for name, v in variants.items():
        cells = _cells(v["dirs"], v["cand"], first_n=20)
        row = {"per_seed": {}, "b33": {}, "b1": {}}
        for s in seeds:
            sc = _world_scores(cells, s)
            row["per_seed"][s] = ws.weighted_metric(sc, pi, mw)
            row["b33"][s] = float(np.mean(list(cells[("b33", s)].values())))
            g1 = cells.get(("b1", s))
            if g1:
                row["b1"][s] = float(np.mean(list(g1.values())))
        row["A"] = float(np.mean(list(cells[("pff", 0)].values())))
        row["A_unseen"] = float(np.mean(list(cells[("unseen", 0)].values())))
        row["weighted_mean"] = float(np.mean(list(row["per_seed"].values())))
        row["sens"] = {k: float(np.mean([ws.weighted_metric(_world_scores(cells, s), {kk: float(kk == k) for kk in WORLDS}, mw)
                                         for s in seeds])) for k in WORLDS}
        res[name] = row
    ref = res["production"]
    print(f"pi = {pi}")
    print(f"{'variant':<18}{'Bp33':>8}{'Bp1':>8}{'A':>8}{'Aunseen':>9}{'weighted':>10}{'d_w':>8}   weighted per seed / delta per seed")
    summary = {}
    for name, r in res.items():
        d = {s: r["per_seed"][s] - ref["per_seed"][s] for s in seeds}
        common1 = [s for s in r["b1"] if s in ref["b1"]]
        b1 = float(np.mean([r["b1"][s] for s in common1])) if common1 else float("nan")
        b1_ref = float(np.mean([ref["b1"][s] for s in common1])) if common1 else float("nan")
        dw = float(np.mean(list(d.values())))
        rule = {"weighted_ge_0.010": dw >= 0.010, "no_seed_below_-0.003": min(d.values()) >= -0.003,
                "b1_drop_le_0.01": (b1 - b1_ref) >= -0.01, "A_drop_le_0.03": (r["A"] - ref["A"]) >= -0.03}
        summary[name] = {"b33": r["b33"], "b33_mean": float(np.mean(list(r["b33"].values()))), "b1_seeds": r["b1"],
                         "b1_mean_common": b1, "b1_delta_common": b1 - b1_ref, "A": r["A"], "A_delta": r["A"] - ref["A"],
                         "A_unseen": r["A_unseen"], "weighted_per_seed": r["per_seed"], "weighted_mean": r["weighted_mean"],
                         "delta_per_seed": d, "delta_mean": dw, "sens_pure_world": r["sens"], "rule": rule,
                         "adopt": all(rule.values())}
        print(f"{name:<18}{summary[name]['b33_mean']:8.4f}{b1:8.4f}{r['A']:8.4f}{r['A_unseen']:9.4f}{r['weighted_mean']:10.4f}{dw:+8.4f}   "
              f"{[round(r['per_seed'][s], 4) for s in seeds]} / {[round(d[s], 4) for s in seeds]}   adopt={summary[name]['adopt']}")
    out = {"pi": pi, "fit": {k: fit[k] for k in ("mixture", "mixture_boot_ci90", "nearest_world_share")},
           "auc": {k: fit["worlds"][k]["auc"] for k in WORLDS}, "ess_matches": {k: fit["worlds"][k]["ess_matches"] for k in WORLDS},
           "variants": summary}
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("features")
    f.add_argument("--data-dir", default=None)
    f.add_argument("--work-dir", default=str(REPO_ROOT / "cache" / "world_selection"))
    g = sub.add_parser("fit")
    g.add_argument("--work-dir", default=str(REPO_ROOT / "cache" / "world_selection"))
    g.add_argument("--out", default=str(REPO_ROOT / "reports" / "issue71_domain_fit.json"))
    g.add_argument("--folds", type=int, default=5)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--max-iter", type=int, default=150)
    g.add_argument("--boot", type=int, default=200)
    r = sub.add_parser("rescore")
    r.add_argument("--fit", default=str(REPO_ROOT / "reports" / "issue71_domain_fit.json"))
    r.add_argument("--out", default=str(REPO_ROOT / "reports" / "issue71_world_selection.json"))
    args = ap.parse_args(argv)
    return {"features": cmd_features, "fit": cmd_fit, "rescore": cmd_rescore}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
