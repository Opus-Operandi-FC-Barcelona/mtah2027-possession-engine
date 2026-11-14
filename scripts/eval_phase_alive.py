# ruff: noqa: E501
#!/usr/bin/env python
"""Issue #43: value of the Smart Tagging alive-ball prior and transition cue on external matches (never BAR-ATH).

The external matches have no Smart Tagging, so a tagging is *simulated* from the true labels
(``mtah2027.io.smart_tagging_sim``) at several noise levels.  The simulator is derived from the truth, hence
every gain is optimistic; read the medium and high levels.

Protocols (specialists trained without the target, targets are unlabelled inputs):

* ``b`` -- B': the 7 IDSSE matches, leave-one-match-out, realistic degrader (``external_degrader``), seeds ``--seeds``;
* ``a`` -- A: PFF held-out (``--folds`` folds over the first ``--max-matches`` PFF matches by name), real occlusions.

For every (noise, variant, seed, match) the specialists route is decoded; a *variant* is a set of config
overrides (``name:section.key=value,section.key=value``; ``off`` = the configuration as is).  Per run the output
holds spellF1 / macroF1 / possessions, the predicted vs true ``D`` share of the hidden-ball frames and the same per
tracked-player stratum (< 16, < 20, >= 20 players in the frame).  Only aggregates are written.

    python scripts/eval_phase_alive.py --protocol b --noise none medium high \
        --variants off alive1:heuristic_emissions.w_phase_alive=1.0 --seeds 0 1 2 --json out.json
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for _extra in (REPO_ROOT / "src", REPO_ROOT, REPO_ROOT / "scripts"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import eval_external as ee  # noqa: E402

from mtah2027.config import _load_raw  # noqa: E402
from mtah2027.io.smart_tagging_sim import simulate_smart_tagging  # noqa: E402
from mtah2027.models.external_degrader import DegraderParams, degrade_match  # noqa: E402
from mtah2027.pipeline import run_core  # noqa: E402
from mtah2027.validation import eval_bridge as eb  # noqa: E402

STRATA = (("lt16", 0, 16), ("lt20", 0, 20), ("ge20", 20, 999))


def parse_variant(text: str, base: dict) -> tuple[str, dict]:
    name, _, rest = text.partition(":")
    cfg = copy.deepcopy(base)
    for item in filter(None, rest.split(",")):
        key, _, value = item.partition("=")
        section, _, field = key.partition(".")
        cfg.setdefault(section, {})[field] = json.loads(value)
    return name, cfg


def hidden_stats(gt: np.ndarray, pred: np.ndarray, visible: np.ndarray, n_players: np.ndarray) -> dict:
    graded = gt != 3
    out = {}
    hid = graded & ~visible
    out["hidden"] = {"n": int(hid.sum()), "pred_d": int((pred[hid] == 2).sum()), "true_d": int((gt[hid] == 2).sum()),
                     "correct": int((pred[hid] == gt[hid]).sum())}
    for name, lo, hi in STRATA:
        m = hid & (n_players >= lo) & (n_players < hi)
        out[name] = {"n": int(m.sum()), "pred_d": int((pred[m] == 2).sum()), "true_d": int((gt[m] == 2).sum()),
                     "correct": int((pred[m] == gt[m]).sum())}
    return out


def run_one(players, team, ball, periods, fps, pitch, gt, cfg, spec, smart) -> dict:
    t0 = time.perf_counter()
    core = run_core(players, team, ball, play_blocks=periods, fps=fps, pitch=pitch, config_path=cfg,
                    log=lambda m: None, specialists=spec, smart_tagging=smart)
    row = ee.score(gt, core.codes)
    vis = np.isfinite(ball).all(axis=1)
    npl = np.isfinite(players).all(axis=2).sum(axis=1)
    row.update(hidden_stats(gt, core.codes, vis, npl))
    row["seconds"] = round(time.perf_counter() - t0, 1)
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--protocol", choices=("a", "b"), required=True)
    ap.add_argument("--bench-dir", default=str(ee.DEFAULT_BENCH_DIR))
    ap.add_argument("--noise", nargs="+", default=["none", "medium", "high"],
                    help="none | clean | low | medium | high (none = no Smart Tagging at all)")
    ap.add_argument("--variants", nargs="+", default=["off"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--training-pickle", default=None)
    ap.add_argument("--sources", nargs="*", default=None, help="training sources (default: all in the pickle)")
    ap.add_argument("--source-weight", nargs="*", default=[], metavar="SOURCE=W")
    ap.add_argument("--max-matches", type=int, default=20)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--matches", nargs="*", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    from mtah2027.models import specialist_fusion as sf
    from mtah2027.models.specialist_data import TrainingDataParams, build_training_set

    base = _load_raw(None)
    variants = [parse_variant(v, base) for v in args.variants]
    bench = Path(args.bench_dir)
    pkl = Path(args.training_pickle) if args.training_pickle else None
    if pkl is not None and pkl.is_file():
        ts = pickle.loads(pkl.read_bytes())
    else:
        ts = build_training_set(args.bench_dir, params=TrainingDataParams.from_config(None), config_path=None,
                                log=lambda m: None)
        if pkl is not None:
            pkl.write_bytes(pickle.dumps(ts))
    fusion = sf.FusionParams.from_config(None)
    if args.protocol == "b":
        files = sorted(bench.glob("idsse_*.npz"))
        groups = [([f], [f.stem]) for f in files]
    else:
        pff = sorted(bench.glob("pff_*.npz"))[:args.max_matches]
        groups = [([f for i, f in enumerate(pff) if i % args.folds == k],
                   [f.stem for i, f in enumerate(pff) if i % args.folds == k]) for k in range(args.folds)]
    if args.matches:
        groups = [([f for f in fs if f.stem in set(args.matches)], names) for fs, names in groups]
    rows = []
    for gi, (fs, names) in enumerate(groups):
        if not fs:
            continue
        weights = {k: float(v) for k, _, v in (i.partition("=") for i in args.source_weight)}
        sources = args.sources if args.sources is not None else sorted({p.source for p in ts.parts})
        sub = ts.select(sources, weights or ts.weights, ts.drop_dead, exclude_matches=names)
        t0 = time.perf_counter()
        spec = dataclasses.replace(sf.train_specialists(args.bench_dir, config_path=None, training=sub),
                                   fusion=fusion)
        print(f"group {gi}: trained without {names} ({time.perf_counter() - t0:.0f} s)", flush=True)
        for f in fs:
            z = np.load(f, allow_pickle=True)
            players0, team = z["players_xy"].astype(np.float32), z["team_of_slot"].astype(np.int8)
            ball0 = z["ball_xy"].astype(np.float32)
            fps, pitch = float(z["fps"]), tuple(float(v) for v in z["pitch"])
            periods = [(int(a), int(b)) for a, b in z["period_bounds"]]
            gt = eb.to_codes(z["gt"])
            if "ball_visible" in z.files:
                ball0 = np.where(z["ball_visible"][:, None], ball0, np.float32(np.nan)).astype(np.float32)
            seeds = args.seeds if args.protocol == "b" else [0]
            for seed in seeds:
                players, ball = players0, ball0
                if args.protocol == "b":
                    dm = degrade_match(players0, team, ball0, gt, fps=fps, seed=seed + gi,
                                       params=DegraderParams.from_config(None))
                    players, ball = dm.players_xy, dm.ball_xy
                for noise in args.noise:
                    smart = None if noise == "none" else simulate_smart_tagging(
                        gt, periods, fps, noise=noise, seed=seed, n_frames=len(gt))
                    for name, cfg in variants:
                        r = run_one(players, team, ball, periods, fps, pitch, gt, cfg, spec, smart)
                        r.update({"match": f.stem, "seed": seed, "noise": noise, "variant": name})
                        rows.append(r)
                        print(f"{f.stem:<14} s{seed} {noise:<7} {name:<12} spell {r['spell_f1']:.4f} "
                              f"macro {r['macro_f1']:.4f} poss {r['n_pred_spells']:5d}/{r['n_gt_spells']} "
                              f"hiddenD {r['hidden']['pred_d']}/{r['hidden']['true_d']} of {r['hidden']['n']} "
                              f"({r['seconds']} s)", flush=True)
                        if args.json:
                            Path(args.json).write_text(json.dumps(rows), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
