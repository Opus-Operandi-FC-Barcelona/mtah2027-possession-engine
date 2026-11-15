#!/usr/bin/env python
# ruff: noqa: E501
"""Issue #67: ceiling of the hidden-ball value for any video detector, accuracy curve, error decomposition and
soft-evidence variant (external matches only, aggregates only; never reads BAR-ATH).

Same cells as #63 / #66 (B'(33) / B'(1): 7 IDSSE leave-one-match-out, calibrated degrader, ``lomo_gap`` models,
production specialist route, official metric).  Scenarios (see ``SCENARIOS``):

* ``orc<cov>_<k|all>``  the true ball restored in the hidden frames at coverage 100/75/50/25 % (nested per seed),
  with the production crop cadence (one every ``vision.every_k`` hidden frames, ``inject_points`` interpolates) or at
  every hidden frame;
* ``n<sigma>_f<fp>_c<cov>``  true ball + isotropic Gaussian error sigma (m) + false positives (uniform 5-30 m away) at
  the production cadence, injected as the ball (hard evidence);
* ``s<w>_n<sigma>_f<fp>_c<cov>``  the same detections as *soft evidence* (``vision.soft_evidence_weight``): the ball
  stays missing and the H/A emission is tilted towards the team nearest to the detection.

    python scripts/oracle_ceiling.py run --world b33 --seeds 0 1 2 --models-root W --out-dir D --shard 0/4
    python scripts/oracle_ceiling.py agg --out-dir D --json reports/issue67_oracle_ceiling.json
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for _extra in (REPO_ROOT / "src", REPO_ROOT, REPO_ROOT / "scripts"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import decoder_sweep as ds  # noqa: E402
from eval_external import score  # noqa: E402
from simulate_sv_ball_value import true_ball  # noqa: E402

from mtah2027.pipeline import run_core  # noqa: E402
from mtah2027.validation import eval_bridge as eb  # noqa: E402
from mtah2027.vision.attention_crops import (  # noqa: E402
    VisionParams,
    hidden_gap_frames,
    subsample_gap_frames,
)
from mtah2027.vision.recovery import inject_points  # noqa: E402

pm = eb.pm
COVS = (100, 75, 50, 25)
SIGMAS = (0.1, 0.3, 0.5, 1.0, 2.0)
FP_RATES = (0.0, 0.10, 0.25)
ACC_COVS = (100, 50)
SOFT_MODELS = ((0.0, 0.0), (0.3, 0.0), (0.5, 0.10), (1.0, 0.10), (2.0, 0.25))   # (sigma, fp)
SOFT_WEIGHTS = (1.0, 2.0, 4.0)
DECOMP_ROWS = ("none", "orc100_all", "orc50_k")


def _tag(x: float) -> str:
    return f"{x:g}".replace(".", "p")


def build_scenarios() -> dict[str, dict]:
    sc: dict[str, dict] = {"none": {"kind": "none"}}
    for cad in ("k", "all"):
        for cov in COVS:
            sc[f"orc{cov}_{cad}"] = {"kind": "hard", "cov": cov, "cadence": cad, "sigma": 0.0, "fp": 0.0}
    for cov in ACC_COVS:
        for fp in FP_RATES:
            for sg in SIGMAS:
                sc[f"n{_tag(sg)}_f{int(fp * 100)}_c{cov}"] = {"kind": "hard", "cov": cov, "cadence": "k",
                                                              "sigma": sg, "fp": fp}
    for cov in ACC_COVS:
        for sg, fp in SOFT_MODELS:
            for w in SOFT_WEIGHTS:
                sc[f"s{_tag(w)}_n{_tag(sg)}_f{int(fp * 100)}_c{cov}"] = {
                    "kind": "soft", "cov": cov, "cadence": "k", "sigma": sg, "fp": fp, "weight": w}
    # refinement of the accuracy curve around the +0.010 crossing (run on seed 0 only)
    for sg, fp in ((0.2, 0.0), (0.1, 0.02), (0.1, 0.05), (0.2, 0.02)):
        sc[f"n{_tag(sg)}_f{int(round(fp * 100))}_c50"] = {"kind": "hard", "cov": 50, "cadence": "k",
                                                          "sigma": sg, "fp": fp}
    return sc


SCENARIOS = build_scenarios()
# rows run for every seed (the others only for ``--full-seed``)
KEY_ROWS = {"none", "orc100_k", "orc100_all", "orc50_k", "orc50_all", "orc25_k",
            "n0p5_f0_c100", "n1_f10_c100", "n1_f10_c50", "n0p3_f0_c50", "n2_f25_c50",
            "n0p1_f0_c100", "n0p1_f0_c50", "n0p5_f0_c50", "n1_f0_c50", "n2_f0_c50"}


def draws(n: int, seed: int) -> dict[str, np.ndarray]:
    """Per-frame uniforms / normals: coverage and fp rate are nested per seed and independent of the cadence."""
    rng = np.random.default_rng(seed)
    return {"gate": rng.random(n), "fp_u": rng.random(n), "gauss": rng.standard_normal((n, 2)),
            "ang": rng.random(n), "rad": rng.random(n)}


def detections(truth, frames, d, *, cov, sigma, fp):
    """Detections at ``frames`` (true ball + error); returns ``(frames, xy, is_fp)`` of the covered ones."""
    keep = d["gate"][frames] < cov / 100.0
    f = frames[keep]
    xy = truth[f].astype(np.float64) + sigma * d["gauss"][f]
    is_fp = d["fp_u"][f] < fp
    r = 5.0 + 25.0 * d["rad"][f]
    th = 2.0 * np.pi * d["ang"][f]
    xy = np.where(is_fp[:, None], truth[f] + np.stack([r * np.cos(th), r * np.sin(th)], axis=1), xy)
    return f, xy, is_fp


def scenario_inputs(spec, ball_deg, truth, periods, d, params, fps):
    """``(ball, soft_evidence, info)`` of a scenario."""
    hidden = hidden_gap_frames(ball_deg, periods)
    if spec["kind"] == "none":
        return ball_deg, None, {"hidden": int(hidden.sum()), "planned": 0, "returned": 0, "fill": 0.0}
    if spec["cadence"] == "all":
        frames = np.flatnonzero(hidden)
    else:
        frames, _ = subsample_gap_frames(hidden, params.every_k)
    frames = frames[np.isfinite(truth[frames]).all(axis=1)]
    f, xy, is_fp = detections(truth, frames, d, cov=spec["cov"], sigma=spec["sigma"], fp=spec["fp"])
    info = {"hidden": int(hidden.sum()), "planned": int(len(frames)), "returned": int(len(f)),
            "fp_share": float(is_fp.mean()) if len(f) else 0.0}
    if spec["kind"] == "soft":
        info["fill"] = 0.0
        return ball_deg, (f, xy, np.ones(len(f))), info
    ball, source = inject_points(ball_deg, f, xy, fps=fps, params=params)
    info["fill"] = float((hidden & (source >= 2)).sum() / max(hidden.sum(), 1))
    return ball, None, info


# ---------------------------------------------------------------------------------------------
# error decomposition (Issue #67, item 3)
# ---------------------------------------------------------------------------------------------
def _spell_f1(gt, pred) -> float:
    return float(eb.evaluate_arrays(gt, pred)["spell_f1_mean"])


def decompose(gt: np.ndarray, pred: np.ndarray, hidden: np.ndarray, tol: int = 25) -> dict:
    """Where the spellF1 is lost, with the official functions of ``possession_metrics``.

    ``spell_f1_fix_hidden`` / ``_fix_visible``: the score if the prediction were replaced by the truth in the
    hidden-ball (resp. ball-visible) graded frames, so ``fix - actual`` is the loss attributable to that part.
    ``spell_fate``: what happened to every ground-truth possession at IoU 0.5 (matched / split in >= 2 predicted
    spells / merged into a predicted spell that also covers another one / boundary error / no same-team overlap).
    ``transitions``: H<->A turnovers, stops (to D) and restarts (from D) matched within ``tol`` frames.
    """
    gt, pred = np.asarray(gt, np.int8), np.asarray(pred, np.int8)
    graded = gt != pm.X
    runs = pm.valid_runs(gt)
    out: dict = {"spell_f1": _spell_f1(gt, pred)}
    fix_h, fix_v = pred.copy(), pred.copy()
    fix_h[hidden & graded] = gt[hidden & graded]
    fix_v[~hidden & graded] = gt[~hidden & graded]
    out["spell_f1_fix_hidden"] = _spell_f1(gt, fix_h)
    out["spell_f1_fix_visible"] = _spell_f1(gt, fix_v)
    for name, m in (("hidden", hidden & graded), ("visible", ~hidden & graded)):
        out[f"frames_{name}"] = int(m.sum())
        out[f"acc_{name}"] = float((pred[m] == gt[m]).mean()) if m.any() else None
        for code, lab in ((pm.H, "H"), (pm.A, "A"), (pm.D, "D")):
            sel = m & (gt == code)
            out[f"recall_{lab}_{name}"] = float((pred[sel] == code).mean()) if sel.any() else None
    gs, ps = pm.possession_spells(gt, runs), pm.possession_spells(pred, runs)
    ov = pm.spell_overlaps(gs, ps)
    matched = pm.matched_gt_flags(ov, len(gs), 0.5)
    by_gt: dict[int, list] = {}
    by_pr: dict[int, list] = {}
    for iou, i, j in ov:
        by_gt.setdefault(i, []).append((iou, j))
        by_pr.setdefault(j, []).append((iou, i))
    cause = {"matched": 0, "split": 0, "merged": 0, "boundary": 0, "missed": 0}
    cause_hidden = {k: 0 for k in cause}
    hid_cum = np.concatenate(([0], np.cumsum(hidden)))
    for i, (a, b, _) in enumerate(gs):
        hid = (hid_cum[b] - hid_cum[a]) / max(b - a, 1) >= 0.5
        if matched[i]:
            c = "matched"
        else:
            cand = by_gt.get(i, [])
            big = [j for _, j in cand if (min(b, ps[j][1]) - max(a, ps[j][0])) >= 0.2 * (b - a)]
            if not cand:
                c = "missed"
            elif len(big) >= 2:
                c = "split"
            else:
                best_j = max(cand)[1]
                pa, pb, _ = ps[best_j]
                other = [k for _, k in by_pr[best_j] if k != i and
                         (min(gs[k][1], pb) - max(gs[k][0], pa)) >= 0.2 * (pb - pa)]
                c = "merged" if other else "boundary"
        cause[c] += 1
        cause_hidden[c] += int(hid)
    out["gt_spells"], out["pred_spells"] = len(gs), len(ps)
    out["spell_fate"], out["spell_fate_hidden"] = cause, cause_hidden
    gt_t, pr_t = pm.transitions(gt, runs), pm.transitions(pred, runs)
    pairs, missed, spurious = pm.match_transitions(gt_t, pr_t, tol)
    kinds = {"turnover": lambda s, t: s != pm.D and t != pm.D, "to_D": lambda s, t: t == pm.D,
             "from_D": lambda s, t: s == pm.D}
    last = len(hidden) - 1
    tr = {}
    for k, fn in kinds.items():
        g = [t for t in gt_t if fn(t[1], t[2])]
        mis = [t for t in missed if fn(t[1], t[2])]
        pr_ = [(a, b) for a, b in pairs if fn(a[1], a[2])]
        spur = [t for t in spurious if fn(t[1], t[2])]
        lat = np.array([b[0] - a[0] for a, b in pr_], dtype=float)
        tr[k] = {"n_gt": len(g), "n_gt_in_hidden": sum(1 for t in g if hidden[min(t[0], last)]),
                 "missed": len(mis), "missed_in_hidden": sum(1 for t in mis if hidden[min(t[0], last)]),
                 "spurious": len(spur), "n_matched": int(len(lat)),
                 "lat_abs_median": float(np.median(np.abs(lat))) if len(lat) else None}
    out["transitions"] = tr
    return out


# ---------------------------------------------------------------------------------------------
def run_cell(cell, root, config_raw, params, rows):
    players, team, ball, fps, pitch, periods, gt = ds.load_cell_data(cell, config_raw)
    truth = true_ball(cell)
    models = ds.build_specialists({"name": "base"}, cell, root, {}, config_raw)
    seed = 1000 * cell["seed"] + cell["index"] + 17
    d = draws(len(ball), seed)
    hidden_mask = hidden_gap_frames(ball, periods)
    out = {"cell": {k: v for k, v in cell.items() if k != "path"}, "results": {}}
    t0 = time.perf_counter()
    for name in rows:
        sc = SCENARIOS[name]
        cfg = config_raw
        if sc["kind"] == "soft":
            cfg = copy.deepcopy(config_raw)
            cfg.setdefault("vision", {})["soft_evidence_weight"] = sc["weight"]
        b, ev, info = scenario_inputs(sc, ball, truth, periods, d, params, fps)
        core = run_core(players, team, b, play_blocks=periods, fps=fps, pitch=pitch, config_path=cfg,
                        log=lambda m: None, specialists=models, durations=models.durations, soft_evidence=ev)
        res = {**score(gt, core.codes), **info}
        res.pop("lat_by_type", None)
        if name in DECOMP_ROWS:
            res["decomp"] = decompose(gt, core.codes, hidden_mask)
        out["results"][name] = res
    out["seconds"] = round(time.perf_counter() - t0, 1)
    return out


def cmd_run(a) -> int:
    import yaml
    config_raw = yaml.safe_load((REPO_ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    params = VisionParams.from_config()
    cells = ds.cells_for(a.world, Path(a.bench_dir), a.seeds)
    k, _, m = a.shard.partition("/")
    cells = [c for i, c in enumerate(cells) if i % int(m) == int(k)]
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for cell in cells:
        path = out_dir / f"{ds.cell_id(cell)}.json"
        if path.is_file():
            continue
        if a.rows == "all":
            rows = list(SCENARIOS) if cell["seed"] == a.full_seed else [r for r in SCENARIOS if r in KEY_ROWS]
        else:
            rows = a.rows.split(",")
        res = run_cell(cell, Path(a.models_root), config_raw, params, rows)
        path.write_text(json.dumps(res), encoding="utf-8")
        print(f"{ds.cell_id(cell)}: {res['seconds']} s, {len(rows)} rows, {rows[0]} {res['results'][rows[0]]['spell_f1']:.4f}",
              flush=True)
    return 0


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def cmd_agg(a) -> int:
    table: dict = {}
    for d in a.out_dir:
        for p in sorted(Path(d).glob("*.json")):
            r = json.loads(p.read_text(encoding="utf-8"))
            c = r["cell"]
            table.setdefault((c["world"], c["seed"]), {}).setdefault(c["match"], {}).update(r["results"])
    summary: dict = {}
    for world in sorted({w for w, _ in table}):
        seeds = sorted(s for w, s in table if w == world)
        names = [n for n in SCENARIOS if any(n in r for s in seeds for r in table[(world, s)].values())]
        base = {s: float(np.mean([r["none"]["spell_f1"] for r in table[(world, s)].values()])) for s in seeds}
        basem = {s: float(np.mean([r["none"]["macro_f1"] for r in table[(world, s)].values()])) for s in seeds}
        print(f"world {world}, seeds {seeds}; none spellF1 " + " / ".join(f"{base[s]:.4f}" for s in seeds))
        print(f"{'scenario':<22}{'seeds':>6}{'spellF1':>9}{'d':>9}{'macroF1':>9}{'dM':>9}{'fill%':>7}  per-seed d")
        for name in names:
            ss = [s for s in seeds if all(name in r for r in table[(world, s)].values())]
            if not ss:
                continue
            per = {s: float(np.mean([r[name]["spell_f1"] for r in table[(world, s)].values()])) for s in ss}
            mac = {s: float(np.mean([r[name]["macro_f1"] for r in table[(world, s)].values()])) for s in ss}
            fill = float(np.mean([100 * r[name].get("fill", 0.0) for s in ss for r in table[(world, s)].values()]))
            dd = {s: per[s] - base[s] for s in ss}
            dm = float(np.mean([mac[s] - basem[s] for s in ss]))
            summary[f"{world}/{name}"] = {"seeds": ss, "spell_f1": float(np.mean(list(per.values()))),
                                          "delta": float(np.mean(list(dd.values()))), "delta_macro": dm,
                                          "macro_f1": float(np.mean(list(mac.values()))), "fill_pct": fill,
                                          "per_seed_delta": {str(s): dd[s] for s in ss}}
            print(f"{name:<22}{len(ss):>6}{np.mean(list(per.values())):9.4f}{np.mean(list(dd.values())):+9.4f}"
                  f"{np.mean(list(mac.values())):9.4f}{dm:+9.4f}{fill:7.1f}  " + " ".join(f"{dd[s]:+.4f}" for s in ss))
        for name in DECOMP_ROWS:
            rows = [r[name]["decomp"] for s in seeds for r in table[(world, s)].values()
                    if name in r and "decomp" in r[name]]
            if not rows:
                continue
            agg = {"n_cells": len(rows)}
            for k in ("spell_f1", "spell_f1_fix_hidden", "spell_f1_fix_visible", "acc_hidden", "acc_visible",
                      "recall_H_hidden", "recall_A_hidden", "recall_D_hidden", "recall_H_visible",
                      "recall_A_visible", "recall_D_visible", "gt_spells", "pred_spells", "frames_hidden",
                      "frames_visible"):
                agg[k] = _mean([r.get(k) for r in rows])
            agg["spell_fate"] = {c: _mean([r["spell_fate"][c] for r in rows]) for c in rows[0]["spell_fate"]}
            agg["spell_fate_hidden"] = {c: _mean([r["spell_fate_hidden"][c] for r in rows])
                                        for c in rows[0]["spell_fate_hidden"]}
            agg["transitions"] = {k: {kk: _mean([r["transitions"][k][kk] for r in rows])
                                      for kk in rows[0]["transitions"][k]} for k in rows[0]["transitions"]}
            summary[f"{world}/decomp_{name}"] = agg
            print(f"decomp {name}: " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                                    for k, v in agg.items() if k != "transitions"}))
            print("   transitions " + json.dumps({k: {kk: (round(v, 2) if v is not None else None)
                                                      for kk, v in t.items()} for k, t in agg["transitions"].items()}))
        print()
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--world", choices=("b33", "b1"), required=True)
    r.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    r.add_argument("--models-root", required=True)
    r.add_argument("--out-dir", required=True)
    r.add_argument("--bench-dir", default=str(ds.DEFAULT_BENCH))
    r.add_argument("--shard", default="0/1")
    r.add_argument("--rows", default="all",
                   help="'all' (every row for --full-seed, key rows otherwise) or a comma list of scenario names")
    r.add_argument("--full-seed", type=int, default=0)
    g = sub.add_parser("agg")
    g.add_argument("--out-dir", nargs="+", required=True)
    g.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    return cmd_run(a) if a.cmd == "run" else cmd_agg(a)


if __name__ == "__main__":
    raise SystemExit(main())
