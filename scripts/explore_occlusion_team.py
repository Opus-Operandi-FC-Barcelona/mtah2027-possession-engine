#!/usr/bin/env python
"""Learnability check: can the TEAM in possession be predicted while the ball is hidden? (exploration for #22)

Question: on realistically degraded external matches (Issue #19), how accurately can a small model predict
the ground-truth team (H / A) on the **alive frames whose ball is hidden**, using only the collective-physics
(#17) and ball-memory (#18) features -- compared with the current rule "last toucher carried forward /
next toucher backwards, change point in the middle of the gap" and with the engine's decoded output?

Pipeline per (match, seed)
--------------------------
1. ``degrade_match`` (Issue #19) hides the ball on 41 % of the alive frames and damages the tracking.
2. ``clean_tracks`` -> ``compute_kinematics_and_orientation`` (home attacks +x) -> ``compute_ball_kinematics``
   on the degraded ball (touches / delta_d / possession evidence only exist where the ball is visible) ->
   ``compute_collective_physics`` + ``compute_ball_memory``.
3. Target ``y = (gt == H)`` on the hidden-ball alive frames; features are the #17 + #18 matrices plus
   home-minus-away differences.  Nothing from the hidden ball itself is used.
4. Leave-one-match-out ``HistGradientBoostingClassifier`` (training frames sub-sampled every ``--stride``
   frames to decorrelate), accuracy on all hidden alive frames of the held-out match, together with
   * ``rule``    : last/next touch team with the change point in the middle of the gap (falls back to the
     clear possessor and then to the other side of the gap),
   * ``engine``  : labels decoded by ``pipeline.run_core`` on the same degraded input (D counts as wrong),
   * feature-group ablations (``physics`` #17 only, ``memory`` #18 only) and permutation importances.

Only aggregate numbers are printed.  The helpers ``build_occlusion_dataset`` and ``leave_one_match_out`` are
importable (``sys.path`` to ``scripts/``) so that Issue #22 can reuse the dataset construction.

Usage::

    python scripts/explore_occlusion_team.py --seeds 0 1 2 --json occlusion_team.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT / "src", REPO_ROOT):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402

from mtah2027.features import ball_memory as bmem  # noqa: E402
from mtah2027.features import collective_physics as cphys  # noqa: E402
from mtah2027.features.ball_kinematics import compute_ball_kinematics  # noqa: E402
from mtah2027.models.external_degrader import degrade_match  # noqa: E402
from mtah2027.pipeline import run_core  # noqa: E402
from mtah2027.preprocessing.clean_tracks import clean_tracks  # noqa: E402
from mtah2027.preprocessing.kinematics_and_orientation import compute_kinematics_and_orientation  # noqa: E402
from mtah2027.validation import eval_bridge as eb  # noqa: E402

DEFAULT_BENCH_DIR = REPO_ROOT.parent / "mtah2027-external-benchmarks" / "cache"
H, A, D = 0, 1, 2
PHYSICS_DIFFS = ["hull_area", "spectral_area", "d_hull", "d_spectral", "depth", "width", "centroid_vx"]
MEMORY_DIFFS = ["phase_R", "line_vx", "press_last", "press_next", "dist_last", "dist_next", "adv_press",
                "adv_x"]


def middle_rule(mem: bmem.BallMemory) -> np.ndarray:
    """Current heuristic as a team code (0 = home, 1 = away) per frame: last toucher before the middle of the
    gap, next toucher after it; falls back to the other side, then to the clear possessor, then to home."""
    first_half = np.where(np.isfinite(mem.gap_frac), mem.gap_frac < 0.5, np.isfinite(mem.since_last_touch_s))
    last_t, next_t = mem.last_touch_team.astype(int), mem.next_touch_team.astype(int)
    last_c, next_c = mem.last_clear_team.astype(int), mem.next_clear_team.astype(int)
    pick = np.where(first_half, last_t, next_t)
    pick = np.where(pick < 0, np.where(first_half, next_t, last_t), pick)
    clear = np.where(first_half, last_c, next_c)
    clear = np.where(clear < 0, np.where(first_half, next_c, last_c), clear)
    pick = np.where(pick < 0, clear, pick)
    return np.where(pick < 0, H, pick).astype(np.int8)


def feature_matrix(phys: cphys.CollectivePhysics, mem: bmem.BallMemory) -> tuple[np.ndarray, list[str], dict]:
    """``(X[N, F], names, groups)``: #17 and #18 features plus home-minus-away differences.  ``groups`` maps
    ``physics`` / ``memory`` to the column indices of that family."""
    xp, np_ = phys.matrix()
    xm, nm = mem.matrix()
    diffs, nd = [], []
    for base in PHYSICS_DIFFS + MEMORY_DIFFS:
        src = phys if base in PHYSICS_DIFFS else mem
        diffs.append(getattr(src, f"{base}_home") - getattr(src, f"{base}_away"))
        nd.append(f"diff:{base}")
    x = np.concatenate([xp, xm, np.stack(diffs, axis=1).astype(np.float32)], axis=1)
    names = [f"cp:{k}" for k in np_] + [f"bm:{k}" for k in nm] + nd
    groups = {"physics": [i for i, k in enumerate(names) if k.startswith("cp:") or k[5:] in PHYSICS_DIFFS],
              "memory": [i for i, k in enumerate(names) if k.startswith("bm:") or k[5:] in MEMORY_DIFFS]}
    return x, names, groups


def build_occlusion_dataset(path: Path, seed: int, *, config: str | None = None,
                            with_engine: bool = True) -> dict:
    """Features and targets of one degraded external match (see the module docstring)."""
    z = np.load(path, allow_pickle=True)
    players, team = z["players_xy"].astype(np.float32), z["team_of_slot"].astype(np.int8)
    ball = z["ball_xy"].astype(np.float32)
    fps, pitch = float(z["fps"]), tuple(float(v) for v in z["pitch"])
    periods = [(int(a), int(b)) for a, b in z["period_bounds"]]
    gt = eb.to_codes(z["gt"])
    dm = degrade_match(players, team, ball, gt, fps=fps, seed=seed)
    halves = (periods[0], periods[1] if len(periods) > 1 else periods[0])

    pxy = clean_tracks(dm.players_xy, team, pitch=pitch).players_xy
    kin = compute_kinematics_and_orientation(pxy, dm.ball_xy, team, fps=fps, pitch=pitch, halves=halves)
    bk = compute_ball_kinematics(kin.players_xy, team, kin.ball_xy, players_vel=kin.players_vel,
                                 ball_vel=kin.ball_vel, ball_acc=kin.ball_acc, fps=fps)
    phys = cphys.compute_collective_physics(kin.players_xy, team, players_vel=kin.players_vel, fps=fps)
    mem = bmem.compute_ball_memory(kin.players_xy, team, kin.ball_xy, players_vel=kin.players_vel,
                                   ball_vel=kin.ball_vel, touch_mask=bk.touch_mask, delta_d=bk.delta_d,
                                   possession_evidence=bk.possession_evidence, fps=fps)
    x, names, groups = feature_matrix(phys, mem)

    visible = np.isfinite(kin.ball_xy).all(axis=1)
    target = ((gt == H) | (gt == A)) & ~visible
    out = {"match": path.stem, "seed": seed, "X": x[target], "names": names, "groups": groups,
           "y": (gt[target] == H).astype(np.int8), "frame": np.flatnonzero(target),
           "rule_home": (middle_rule(mem) == H)[target], "hidden_share": dm.report.hidden_alive_share,
           "gap_s": np.where(np.isfinite(mem.since_seen_s + mem.until_seen_s),
                             mem.since_seen_s + mem.until_seen_s, np.nan)[target]}
    if with_engine:
        core = run_core(dm.players_xy, team, dm.ball_xy, play_blocks=periods, fps=fps, pitch=pitch,
                        config_path=config, log=lambda m: None)
        out["engine"] = core.codes[target]
    return out


def _fit(x: np.ndarray, y: np.ndarray, seed: int) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, max_leaf_nodes=15,
                                           min_samples_leaf=200, l2_regularization=1.0, random_state=seed)
    return model.fit(x, y)


def leave_one_match_out(datasets: list[dict], *, stride: int = 10, columns: list[int] | None = None,
                        importance: bool = False, seed: int = 0) -> dict:
    """Train on every other *match* (all its seeds, every ``stride``-th frame) and score on the held-out one.

    Returns per-dataset accuracies (``acc``), the mean over datasets and optionally mean permutation
    importances (drop in accuracy) per feature name.
    """
    matches = sorted({d["match"] for d in datasets})
    cols = columns if columns is not None else list(range(datasets[0]["X"].shape[1]))
    names = [datasets[0]["names"][c] for c in cols]
    acc, prior, imp = {}, {}, np.zeros(len(cols))
    for held in matches:
        train = [d for d in datasets if d["match"] != held]
        xtr = np.concatenate([d["X"][::stride][:, cols] for d in train])
        ytr = np.concatenate([d["y"][::stride] for d in train])
        model = _fit(xtr, ytr, seed)
        for d in datasets:
            if d["match"] != held:
                continue
            xte, yte = d["X"][:, cols], d["y"]
            acc[(d["match"], d["seed"])] = float((model.predict(xte) == yte).mean())
            prior[(d["match"], d["seed"])] = float((yte == int(ytr.mean() > 0.5)).mean())
            if importance:
                sub = np.random.default_rng(seed).choice(len(yte), size=min(15000, len(yte)), replace=False)
                res = permutation_importance(model, xte[sub], yte[sub], n_repeats=3, random_state=seed,
                                             n_jobs=1)
                imp += res.importances_mean / len(datasets)
    out = {"acc": acc, "mean": float(np.mean(list(acc.values()))),
           "prior_mean": float(np.mean(list(prior.values())))}
    if importance:
        out["importance"] = dict(sorted(zip(names, imp), key=lambda kv: -kv[1]))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bench-dir", default=str(DEFAULT_BENCH_DIR))
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--matches", nargs="*", default=None)
    ap.add_argument("--stride", type=int, default=10, help="training sub-sampling (frames)")
    ap.add_argument("--top", type=int, default=15, help="number of importances printed")
    ap.add_argument("--no-engine", action="store_true", help="skip the (slow) engine comparison")
    ap.add_argument("--config", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    files = sorted(Path(args.bench_dir).glob("*.npz"))
    files = [f for f in files if np.isfinite(np.load(f, allow_pickle=True)["ball_xy"]).all()]   # full ball
    if args.matches:
        files = [f for f in files if f.stem in set(args.matches)]
    if len(files) < 2:
        print("need at least two benchmark matches with a complete ball channel")
        return 1
    datasets = []
    for seed in args.seeds:
        for i, f in enumerate(files):
            t0 = time.perf_counter()
            datasets.append(build_occlusion_dataset(f, seed + i, config=args.config,
                                                    with_engine=not args.no_engine))
            d = datasets[-1]
            print(f"built {d['match']} seed {d['seed']}: {len(d['y'])} hidden alive frames "
                  f"({time.perf_counter() - t0:.1f}s)", flush=True)

    full = leave_one_match_out(datasets, stride=args.stride, importance=True)
    physics = leave_one_match_out(datasets, stride=args.stride, columns=datasets[0]["groups"]["physics"])
    memory = leave_one_match_out(datasets, stride=args.stride, columns=datasets[0]["groups"]["memory"])
    home = {(d["match"], d["seed"]): d["y"] == 1 for d in datasets}
    rule = {(d["match"], d["seed"]): float((d["rule_home"] == home[(d["match"], d["seed"])]).mean())
            for d in datasets}
    rows = [("model (#17+#18)", full["mean"]), ("model physics only (#17)", physics["mean"]),
            ("model memory only (#18)", memory["mean"]),
            ("constant: train majority team", full["prior_mean"]),
            ("rule: middle change point", float(np.mean(list(rule.values()))))]
    if not args.no_engine:
        eng = {(d["match"], d["seed"]): float((d["engine"] == np.where(d["y"] == 1, H, A)).mean())
               for d in datasets}
        rows.append(("engine (decoded, D = wrong)", float(np.mean(list(eng.values())))))
        dead = float(np.mean([(d["engine"] == D).mean() for d in datasets]))
        print(f"engine labels D on {100 * dead:.1f}% of the hidden alive frames")
    print("\naccuracy of the team on hidden-ball alive frames (leave-one-match-out, mean over runs)")
    for name, value in rows:
        print(f"  {name:<32}{value:8.4f}")
    print("\nper held-out match (model / rule" + ("" if args.no_engine else " / engine") + ")")
    for key in sorted(full["acc"]):
        line = f"  {key[0]:<16} seed {key[1]:>2}  {full['acc'][key]:.4f}  {rule[key]:.4f}"
        if not args.no_engine:
            line += f"  {eng[key]:.4f}"
        print(line)
    # accuracy by gap length (model vs rule), pooled
    print("\naccuracy by hidden-gap length (pooled)")
    print("  gap_s        frames   rule    (model: see mean above)")
    gap = np.concatenate([d["gap_s"] for d in datasets])
    ok_rule = np.concatenate([d["rule_home"] == (d["y"] == 1) for d in datasets])
    for lo, hi in ((0, 10), (10, 20), (20, 40), (40, 80), (80, 1e9)):
        sel = (gap >= lo) & (gap < hi)
        if sel.any():
            print(f"  {lo:>3}-{hi if hi < 1e8 else 'inf':<5}{int(sel.sum()):>10}{ok_rule[sel].mean():8.4f}")
    print(f"\ntop {args.top} features by permutation importance (drop in accuracy)")
    for name, value in list(full["importance"].items())[:args.top]:
        print(f"  {name:<34}{value:9.4f}")
    if args.json:
        Path(args.json).write_text(json.dumps({
            "summary": dict(rows), "importance": full["importance"],
            "per_match": {f"{k[0]}|{k[1]}": {"model": full["acc"][k], "rule": rule[k]} for k in full["acc"]}},
            indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
