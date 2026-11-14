#!/usr/bin/env python
"""Score the array-level core of the pipeline on the external benchmark matches (never on BAR-ATH).

For every ``*.npz`` in ``--bench-dir`` (open data of other matches; format ``players_xy[N,S,2]`` metres,
``team_of_slot[S]``, ``ball_xy[N,2]``, ``fps``, ``pitch``, ``period_bounds[P,2]`` inclusive, ``gt[N]`` of
H/A/D/X) it runs features -> emissions -> transitions -> Viterbi -> spell optimiser (no file loaders, no
Smart Tagging) and scores the result with the official metric through ``eval_bridge``; the graded frames are
those where ``gt != X``.  The official baseline recipe (nearest player + ``+-50`` majority window) is computed
on the same arrays for reference.

``--degrade`` applies the realistic degrader of Issue #19 (``models/external_degrader.py``): the ball is
hidden in crowding-weighted blocks (5-20 s 33 %, 20-60 s 48 %, > 60 s 19 %) covering 41 % of the alive frames
plus tracking noise, identity cuts, duplicate tracks and dropped players.  ``--seeds 0 1 2`` repeats it per
seed and also prints the mean over seeds.

``--specialists lomo`` adds the learned route of Issues #20-#22 (Specialists A and B + fusion), evaluated
**leave-one-match-out**: for every held-out match the specialists are trained on the *other* matches only
(degraded with several seeds, home / away swapped copies), and the held-out match is processed as the
unlabelled target (its pseudo-labels, Issue #21, come from its own degraded tracking, never from its ground
truth).  ``--no-pseudo`` switches the Issue #21 adaptation off (ablation).  ``--fusion key=value ...``
overrides ``specialist_fusion`` parameters for tuning sweeps (external matches only).

``--hide-ball-frac F`` is the older crude degradation (random
5-60 s blocks, no artefacts) and still works.  Only aggregate numbers are printed.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT / "src", REPO_ROOT):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from mtah2027.models.baseline_labels import baseline_labels  # noqa: E402
from mtah2027.models.external_degrader import DegraderParams, degrade_match  # noqa: E402
from mtah2027.pipeline import run_core  # noqa: E402
from mtah2027.validation import dropout_strata as ds  # noqa: E402
from mtah2027.validation import eval_bridge as eb  # noqa: E402

DEFAULT_BENCH_DIR = REPO_ROOT.parent / "mtah2027-external-benchmarks" / "cache"


def hide_ball(ball_xy: np.ndarray, periods, frac: float, fps: float, rng: np.random.Generator) -> np.ndarray:
    """Copy of ``ball_xy`` with random contiguous 5-60 s blocks set to NaN (``frac`` of the period frames)."""
    out = ball_xy.copy()
    n = len(out)
    inside = np.zeros(n, bool)
    for a, b in periods:
        inside[a:b + 1] = True
    target = frac * inside.sum()
    hidden = np.zeros(n, bool)
    guard = 0
    while (hidden & inside).sum() < target and guard < 100000:
        guard += 1
        dur = int(np.exp(rng.uniform(np.log(5.0), np.log(60.0))) * fps)
        start = int(rng.integers(0, max(n - dur, 1)))
        if inside[start:start + dur].mean() < 0.9:
            continue
        hidden[start:start + dur] = True
    out[hidden] = np.nan
    return out


LATENCY_TAU = 25  # frames: tolerance of the transition matching used for the signed-latency diagnostic


def score(gt: np.ndarray, pred: np.ndarray) -> dict:
    r = eb.evaluate_arrays(gt, pred)
    trans = r["transitions"].get(LATENCY_TAU, {})
    # Signed latency (pred - gt, frames) of the matched transitions: a stable sign across matches and
    # seeds is a systematic timing bias of the boundaries, measurable on other matches only.
    return {"spell_f1": float(r["spell_f1_mean"]), "macro_f1": float(r["macro_f1"]),
            "n_pred_spells": int(r["n_pred_spells"]), "n_gt_spells": int(r["n_gt_spells"]),
            "mean_spell_s": float(r["mean_spell_duration_s"]),
            "gt_mean_spell_s": float(r["gt_mean_spell_duration_s"]),
            "abstain_on_graded": int(r["n_abstain_on_graded"]),
            "lat_mean": trans.get("mean_latency"), "lat_median": trans.get("median_latency"),
            "lat_n_matched": int(trans.get("n_matched", 0) or 0),
            "lat_by_type": _latency_by_type(trans.get("pairs", []))}


def _latency_by_type(pairs) -> dict:
    """Signed latencies (pred - gt) of the matched transitions, split by the true transition type:
    ``ha`` = H<->A, ``to_d`` = alive -> D, ``from_d`` = D -> alive (restarts)."""
    out = {"ha": [], "to_d": [], "from_d": []}
    for gt_t, pred_t in pairs:
        _, src, dst = gt_t
        key = "to_d" if dst == 2 else ("from_d" if src == 2 else "ha")
        out[key].append(int(pred_t[0] - gt_t[0]))
    return out


def latency_summary(rows: list, key: str) -> str | None:
    """One line: per-run signed median latency, its spread, and the share of runs with each sign."""
    meds = [r[key]["lat_median"] for r in rows if key in r and r[key].get("lat_median") is not None]
    if not meds:
        return None
    meds = np.asarray(meds, dtype=float)
    return (f"signed latency {key} (tau={LATENCY_TAU}, pred-gt frames, {len(meds)} runs): "
            f"median of medians {np.median(meds):+.1f}  mean {meds.mean():+.2f}  "
            f"min {meds.min():+.1f}  max {meds.max():+.1f}  "
            f"runs early {np.mean(meds < 0):.0%} / late {np.mean(meds > 0):.0%}")


def evaluate_match(path: Path, hide_frac: float, seed: int, config: str | None, with_baseline: bool,
                   degrade: bool = False, specialists=None, return_codes: bool = False,
                   dead_affinity: float | None = None, sim_tags: str | None = None) -> dict:
    z = np.load(path, allow_pickle=True)
    players, team = z["players_xy"].astype(np.float32), z["team_of_slot"].astype(np.int8)
    ball = z["ball_xy"].astype(np.float32)
    fps, pitch = float(z["fps"]), tuple(float(v) for v in z["pitch"])
    periods = [(int(a), int(b)) for a, b in z["period_bounds"]]
    gt = eb.to_codes(z["gt"])
    if "ball_visible" in z.files:       # PFF: the file's ball includes positions estimated from the players
        ball = np.where(z["ball_visible"][:, None], ball, np.float32(np.nan)).astype(np.float32)
    if degrade:
        dparams = DegraderParams.from_config(config)
        if dead_affinity is not None:
            dparams = dataclasses.replace(dparams, dead_affinity=float(dead_affinity))
        dm = degrade_match(players, team, ball, gt, fps=fps, seed=seed, params=dparams)
        players, ball = dm.players_xy, dm.ball_xy
    elif hide_frac > 0:
        ball = hide_ball(ball, periods, hide_frac, fps, np.random.default_rng(seed))
    t0 = time.perf_counter()
    durations = getattr(specialists, "durations", None)    # Issue #34 (used only when duration.enabled)
    smart = None
    if sim_tags:       # Issue #43: simulated Smart Tagging from the true labels (optimistic by design)
        from mtah2027.io.smart_tagging_sim import simulate_smart_tagging

        smart = simulate_smart_tagging(gt, periods, fps, noise=sim_tags, seed=seed, n_frames=len(gt))
    core = run_core(players, team, ball, play_blocks=periods, fps=fps, pitch=pitch, config_path=config,
                    log=lambda m: None, durations=durations, smart_tagging=smart)
    seconds = time.perf_counter() - t0
    row = {"match": path.stem, "seconds": round(seconds, 1),
           "ball_visible": float(np.isfinite(ball).all(axis=1)[gt != 3].mean())}
    row["engine"] = score(gt, core.codes)
    n_tracked = ds.tracked_counts(players)
    visible = np.isfinite(ball).all(axis=1)
    row["engine"]["strata"] = ds.stratum_counts(gt, core.codes, n_tracked, visible)    # Issue #42
    codes_spec = None
    if specialists is not None:
        t0 = time.perf_counter()
        learned = run_core(players, team, ball, play_blocks=periods, fps=fps, pitch=pitch, config_path=config,
                           log=lambda m: None, specialists=specialists, durations=durations,
                           smart_tagging=smart)
        row["specialists"] = score(gt, learned.codes)
        row["specialists"]["seconds"] = round(time.perf_counter() - t0, 1)
        row["specialists"]["source"] = learned.diagnostics["emission_source"]
        row["specialists"]["strata"] = ds.stratum_counts(gt, learned.codes, n_tracked, visible)
        codes_spec = learned.codes
    if with_baseline:
        base = np.full(len(gt), 3, dtype=np.int8)
        full = baseline_labels(players, team, ball)
        for a, b in periods:
            base[a:b + 1] = full[a:b + 1]
        row["baseline"] = score(gt, base)
    if return_codes:     # in-memory only (used by the KPI dashboard); never serialised
        row["_codes"] = {"gt": gt, "engine": core.codes, "specialists": codes_spec, "fps": fps,
                         "visible": visible, "n_tracked": n_tracked}
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bench-dir", default=str(DEFAULT_BENCH_DIR))
    ap.add_argument("--hide-ball-frac", type=float, default=0.0)
    ap.add_argument("--degrade", action="store_true",
                    help="realistic degradation (Issue #19): 41%% occluded alive frames + tracking artefacts")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="several seeds (mean reported); overrides --seed")
    ap.add_argument("--config", default=None, help="YAML configuration (default configs/default.yaml)")
    ap.add_argument("--degrader-section", default=None,
                    help="Issue #38: config section of the degrader of the IDSSE training rows "
                         "(external_degrader_legacy = adopted models, external_degrader = calibrated); "
                         "the held-out matches are always degraded with external_degrader")
    ap.add_argument("--eval-dead-affinity", type=float, default=None,
                    help="Issue #48: dead_affinity of the degrader of the HELD-OUT matches (default: config)")
    ap.add_argument("--train-dead-affinity", type=float, default=None,
                    help="Issue #48: dead_affinity of the degrader of the IDSSE training rows")
    ap.add_argument("--source-weight-b", nargs="*", default=None, metavar="SOURCE=W",
                    help="Issue #48: source weights of the hidden-ball rows (Specialist B) if different")
    ap.add_argument("--training-pickle", default=None,
                    help="Issue #38: load / save the training rows (valid for one --training-config)")
    ap.add_argument("--sources", nargs="*", default=None,
                    help="Issue #38: train only on these sources (idsse skillcorner pff)")
    ap.add_argument("--source-weight", nargs="*", default=[], metavar="SOURCE=W",
                    help="sample weight per source")
    ap.add_argument("--matches", nargs="*", default=None, help="subset of match names (file stems)")
    ap.add_argument("--only-complete-ball", action="store_true",
                    help="skip matches whose own ball channel has gaps (Metrica samples drop the ball "
                         "while it is dead, which leaks D into the visibility)")
    ap.add_argument("--specialists", choices=("off", "lomo"), default="off",
                    help="lomo: also score the learned specialists, trained leave-one-match-out")
    ap.add_argument("--no-pseudo", action="store_true", help="with --specialists: no Issue #21 adaptation")
    ap.add_argument("--fusion", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override specialist_fusion parameters (tuning on external matches only)")
    ap.add_argument("--lomo-cache", default=None,
                    help="Issue #42: LOMO specialists cache (JSON); valid for one "
                         "training set / configuration")
    ap.add_argument("--sim-tags", choices=("clean", "low", "medium", "high"), default=None,
                    help="Issue #43: feed a simulated Smart Tagging (from the true labels)")
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--strata", action="store_true",
                    help="Issue #42: pooled breakdown by number of tracked players (<16, <20, >=20)")
    ap.add_argument("--json", default=None, help="write the aggregate rows to this JSON file")
    args = ap.parse_args(argv)

    files = sorted(Path(args.bench_dir).glob("*.npz"))
    if args.matches:
        files = [f for f in files if f.stem in set(args.matches)]
    if args.only_complete_ball:
        files = [f for f in files if np.isfinite(np.load(f, allow_pickle=True)["ball_xy"]).all()]
    if not files:
        print(f"no benchmark files in {args.bench_dir}")
        return 1
    seeds = list(args.seeds) if args.seeds else [args.seed]
    models: dict = {}
    training = None
    if args.specialists == "lomo":
        from mtah2027.decoding.duration_model import DurationParams
        from mtah2027.models import specialist_fusion as sf
        from mtah2027.models.specialist_data import TrainingDataParams, build_training_set

        overrides = {}
        for item in args.fusion:
            key, _, value = item.partition("=")
            overrides[key] = json.loads(value)
        if args.no_pseudo:
            overrides["use_pseudo_labels"] = False
        fusion = dataclasses.replace(sf.FusionParams.from_config(args.config), **overrides)
        pkl = Path(args.training_pickle) if args.training_pickle else None
        if pkl is not None and pkl.is_file():
            training = pickle.loads(pkl.read_bytes())
        else:
            dp = TrainingDataParams.from_config(args.config)
            if args.degrader_section:
                dp = dataclasses.replace(dp, degrader_section=args.degrader_section)
            if args.train_dead_affinity is not None:
                dp = dataclasses.replace(dp, dead_affinity=args.train_dead_affinity)
            training = build_training_set(args.bench_dir, params=dp, config_path=args.config, log=print)
            if pkl is not None:
                pkl.write_bytes(pickle.dumps(training))
        if args.sources is not None or args.source_weight or args.source_weight_b:
            weights = {k: float(v) for k, _, v in (i.partition("=") for i in args.source_weight)}
            wb = ({k: float(v) for k, _, v in (i.partition("=") for i in args.source_weight_b)}
                  if args.source_weight_b else None)
            training = training.select(args.sources if args.sources is not None
                                       else sorted({p.source for p in training.parts}),
                                       weights or training.weights, training.drop_dead, weights_b=wb)

        def specialists_for(stem: str):
            if stem not in models:
                t0 = time.perf_counter()
                cached = Path(args.lomo_cache) / f"{stem}.json" if args.lomo_cache else None
                if cached is not None and cached.is_file():
                    loaded = sf.load_specialists(cached, args.config)
                    models[stem] = dataclasses.replace(loaded, fusion=fusion)
                    print(f"  loaded cached specialists without {stem}", flush=True)
                else:
                    spec = sf.train_specialists(args.bench_dir, config_path=args.config, training=training,
                                                exclude_match=stem)
                    models[stem] = dataclasses.replace(spec, fusion=fusion)
                    if cached is not None:
                        sf.save_specialists(models[stem], cached)
                    dt = time.perf_counter() - t0
                    print(f"  trained specialists without {stem} ({dt:.0f} s)", flush=True)
            if models[stem].durations is None:     # Issue #34: duration table without the held-out match
                from mtah2027.decoding.duration_model import fit_from_bench

                models[stem] = dataclasses.replace(models[stem], durations=fit_from_bench(
                    args.bench_dir, exclude=stem, params=DurationParams.from_config(args.config)))
            return models[stem]
    all_rows = []
    print(f"degrade={args.degrade} hide_ball_frac={args.hide_ball_frac} seeds={seeds}")
    head = (f"{'match':<18}{'seed':>5}{'spellF1':>8}{'macroF1':>8}{'poss':>6}{'gt_poss':>8}"
            f"{'dur_s':>7}{'gt_s':>6}")
    if not args.no_baseline:
        head += f"{'| base_sF1':>11}{'base_mF1':>9}{'b_poss':>7}"
    print(head)
    for seed in seeds:
        for i, f in enumerate(files):
            row = evaluate_match(f, args.hide_ball_frac, seed + i, args.config, not args.no_baseline,
                                 args.degrade, specialists_for(f.stem) if training is not None else None,
                                 dead_affinity=args.eval_dead_affinity, sim_tags=args.sim_tags)
            row["seed"] = seed
            all_rows.append(row)
            e = row["engine"]
            line = (f"{row['match']:<18}{seed:5d}{e['spell_f1']:8.4f}{e['macro_f1']:8.4f}"
                    f"{e['n_pred_spells']:6d}{e['n_gt_spells']:8d}{e['mean_spell_s']:7.1f}"
                    f"{e['gt_mean_spell_s']:6.1f}")
            if not args.no_baseline:
                b = row["baseline"]
                line += (f"{'| ' + format(b['spell_f1'], '.4f'):>11}{b['macro_f1']:9.4f}"
                         f"{b['n_pred_spells']:7d}")
            print(line, flush=True)
            if "specialists" in row:
                sp = row["specialists"]
                print(f"{'  + specialists':<18}{seed:5d}{sp['spell_f1']:8.4f}{sp['macro_f1']:8.4f}"
                      f"{sp['n_pred_spells']:6d}{sp['n_gt_spells']:8d}{sp['mean_spell_s']:7.1f}"
                      f"{sp['gt_mean_spell_s']:6.1f}   ({sp['source']}, {sp['seconds']} s)", flush=True)
    rows = all_rows
    mean = {}
    for key in ("engine", "baseline"):
        if key == "baseline" and args.no_baseline:
            continue
        for metric in ("spell_f1", "macro_f1"):
            mean[metric if key == "engine" else f"baseline_{metric}"] = float(
                np.mean([r[key][metric] for r in rows]))
    if training is not None:
        for metric in ("spell_f1", "macro_f1"):
            mean[f"specialists_{metric}"] = float(np.mean([r["specialists"][metric] for r in rows]))
    line = f"{'MEAN':<18}{'':>5}{mean['spell_f1']:8.4f}{mean['macro_f1']:8.4f}"
    if not args.no_baseline:
        line += (f"{'':<26}{'| ' + format(mean['baseline_spell_f1'], '.4f'):>11}"
                 f"{mean['baseline_macro_f1']:9.4f}")
    print(line)
    if training is not None:
        print(f"{'MEAN + specialists':<23}{mean['specialists_spell_f1']:8.4f}"
              f"{mean['specialists_macro_f1']:8.4f}")
    for key in ("engine", "specialists"):
        summary = latency_summary(rows, key)
        if summary:
            print(summary)
    if args.strata:
        for key in ("engine", "specialists"):
            if key == "specialists" and training is None:
                continue
            pooled = ds.pool([r[key]["strata"] for r in rows])
            print(f"strata ({key}, pooled over {len(rows)} runs):")
            total = pooled["lt20"]["frames"] + pooled["ge20"]["frames"]
            for line in ds.format_table(ds.summarize(pooled), total):
                print("  " + line)
    if args.json:
        Path(args.json).write_text(json.dumps({"mean": mean, "rows": rows}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
