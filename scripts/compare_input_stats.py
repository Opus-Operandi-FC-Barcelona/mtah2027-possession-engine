#!/usr/bin/env python
"""Aggregate, unlabelled input statistics: ball-gap lengths and tracked players of an ATD vs external matches.

Used to decide which specialists file fits the competition input (``models/specialists_idsse.json`` trained on
IDSSE degraded by our synthetic degrader vs ``models/specialists_mixed.json`` that also saw real occlusions
of PFF / SkillCorner).  It only needs the *tracking* (no labels) and prints aggregates, never per-frame rows:

* share of the graded frames whose ball is missing, number of ball gaps (>= ``--min-gap-s``) per 10 graded
  minutes, gap-length quantiles (50 / 90 / 99 %), and the share of the hidden frames lying in gaps of
  < 5 s, 5-20 s, 20-60 s and > 60 s (the classes of the degrader of Issue #19);
* mean number of tracked players with a position per frame and the share of frames with fewer than 20.

Groups: ``--data-dir`` (the competition ATD, graded blocks of the configuration), and from ``--bench-dir``:
``idsse`` (raw), ``idsse_degraded`` (Issue #19 degrader, seed ``--seed``), ``pff`` (ball only where truly
visible), ``skillcorner`` (broadcast).  Examples::

    python scripts/compare_input_stats.py --data-dir data --bench-dir ../mtah2027-external-benchmarks/cache
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for _extra in (REPO_ROOT / "src", REPO_ROOT):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

CLASS_EDGES_S = (5.0, 20.0, 60.0)


def gap_lengths(missing: np.ndarray, blocks) -> np.ndarray:
    """Lengths in frames of the runs of ``missing`` inside the inclusive ``blocks`` (cut at block borders)."""
    out = []
    for a, b in blocks:
        seg = np.asarray(missing[int(a):int(b) + 1], dtype=np.int8)
        if not len(seg):
            continue
        d = np.diff(np.concatenate([[0], seg, [0]]))
        starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        out.append(ends - starts)
    return np.concatenate(out) if out else np.zeros(0, np.int64)


def input_stats(ball_xy: np.ndarray, players_xy: np.ndarray, blocks, fps: float, min_gap_s: float = 0.2,
                graded_mask: np.ndarray | None = None) -> dict:
    """Raw counts of one match (summed over matches by :func:`pool`)."""
    missing = ~np.isfinite(np.asarray(ball_xy)).all(axis=1)
    inside = np.zeros(len(missing), bool)
    for a, b in blocks:
        inside[int(a):int(b) + 1] = True
    if graded_mask is not None:
        inside &= graded_mask
    gaps = gap_lengths(missing & inside, blocks)
    gaps = gaps[gaps >= max(1, int(round(min_gap_s * fps)))]
    n_players = np.isfinite(np.asarray(players_xy)[..., 0]).sum(axis=1)
    return {"frames": int(inside.sum()), "missing": int((missing & inside).sum()), "fps": float(fps),
            "gaps_s": (gaps / fps).tolist(), "players_sum": float(n_players[inside].sum()),
            "players_lt20": int((n_players[inside] < 20).sum())}


def pool(items: list[dict]) -> dict:
    """Aggregate several matches into the printed statistics."""
    frames = sum(i["frames"] for i in items)
    gaps = np.concatenate([np.asarray(i["gaps_s"]) for i in items]) if items else np.zeros(0)
    minutes = sum(i["frames"] / i["fps"] for i in items) / 60.0
    hidden = float(gaps.sum())
    edges = (0.0,) + CLASS_EDGES_S + (np.inf,)
    mass = [float(gaps[(gaps >= lo) & (gaps < hi)].sum() / max(hidden, 1e-9))
            for lo, hi in zip(edges[:-1], edges[1:])]
    return {"matches": len(items), "graded_minutes": round(minutes, 1),
            "ball_missing_pct": round(100.0 * sum(i["missing"] for i in items) / max(frames, 1), 1),
            "gaps_per_10min": round(10.0 * len(gaps) / max(minutes, 1e-9), 2),
            "gap_s_p50_p90_p99": [round(float(np.quantile(gaps, q)), 1) if len(gaps) else 0.0
                                  for q in (0.5, 0.9, 0.99)],
            "hidden_mass_pct_lt5_5to20_20to60_gt60": [round(100.0 * m, 1) for m in mass],
            "players_mean": round(sum(i["players_sum"] for i in items) / max(frames, 1), 2),
            "frames_lt20_players_pct": round(
                100.0 * sum(i["players_lt20"] for i in items) / max(frames, 1), 1)}


def bench_groups(bench_dir: Path, seed: int = 0, max_matches: int = 0,
                 config=None) -> dict[str, list[dict]]:
    from mtah2027.models.external_degrader import DegraderParams, degrade_match
    from mtah2027.validation import eval_bridge as eb

    out: dict[str, list[dict]] = {"idsse": [], "idsse_degraded": [], "pff": [], "skillcorner": []}
    for f in sorted(Path(bench_dir).glob("*.npz")):
        kind = next((k for k in ("idsse", "pff", "skillcorner") if f.stem.startswith(k + "_")), None)
        if kind is None or (max_matches and len(out[kind]) >= max_matches):
            continue
        z = np.load(f, allow_pickle=True)
        players, ball = z["players_xy"].astype(np.float32), z["ball_xy"].astype(np.float32)
        fps = float(z["fps"])
        blocks = [(int(a), int(b)) for a, b in z["period_bounds"]]
        gt = eb.to_codes(z["gt"])
        graded = gt != 3
        if kind == "pff":
            ball = np.where(z["ball_visible"][:, None], ball, np.float32(np.nan)).astype(np.float32)
        out[kind].append(input_stats(ball, players, blocks, fps, graded_mask=graded))
        if kind == "idsse":
            dm = degrade_match(players, z["team_of_slot"].astype(np.int8), ball, gt, fps=fps, seed=seed,
                               params=DegraderParams.from_config(config))
            out["idsse_degraded"].append(
                input_stats(dm.ball_xy, dm.players_xy, blocks, fps, graded_mask=graded))
    return {k: v for k, v in out.items() if v}


def print_table(groups: dict[str, dict]) -> None:
    for name, s in groups.items():
        print(f"== {name}: {s['matches']} match(es), {s['graded_minutes']} graded min")
        print(f"   ball missing {s['ball_missing_pct']} %   gaps/10 min {s['gaps_per_10min']}   "
              f"gap p50/p90/p99 {s['gap_s_p50_p90_p99']} s")
        mass = s["hidden_mass_pct_lt5_5to20_20to60_gt60"]
        print(f"   hidden time in gaps <5 / 5-20 / 20-60 / >60 s: {mass} %")
        print(f"   tracked players/frame {s['players_mean']}   "
              f"frames with <20 players {s['frames_lt20_players_pct']} %")


def target_from_config(path=None) -> dict | None:
    """Issue #38: the label-free aggregates of the competition ATD stored in ``external_degrader_target``."""
    from mtah2027.config import load_section

    t = load_section("external_degrader_target", path)
    return dict(t) if t else None


def print_target(target: dict, stats: dict) -> None:
    """Side by side: configured target vs the measured degraded group."""
    mass = stats["hidden_mass_pct_lt5_5to20_20to60_gt60"]
    rows = [("ball missing %", target["ball_missing_pct"], stats["ball_missing_pct"]),
            ("gaps / 10 min", target["gaps_per_10min"], stats["gaps_per_10min"]),
            ("players / frame", target["players_mean"], stats["players_mean"]),
            ("frames < 20 players %", target["frames_lt20_players_pct"], stats["frames_lt20_players_pct"])]
    rows += [(f"gap {n} (s)", t, m) for n, t, m in zip(("p50", "p90", "p99"), target["gap_s_p50_p90_p99"],
                                                     stats["gap_s_p50_p90_p99"])]
    rows += [(f"mass {n} %", t, m) for n, t, m in zip(("<5 s", "5-20 s", "20-60 s", ">60 s"),
                                                      target["hidden_mass_pct_lt5_5to20_20to60_gt60"], mass)]
    print("== calibration target (competition ATD, label-free) vs idsse_degraded")
    for name, t, m in rows:
        print(f"   {name:<24} target {t:>7}   degraded {m:>7}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default=None, help="competition ATD directory (graded blocks of config)")
    ap.add_argument("--bench-dir",
                    default=str(REPO_ROOT.parent / "mtah2027-external-benchmarks" / "cache"))
    ap.add_argument("--seed", type=int, default=0, help="degrader seed of idsse_degraded")
    ap.add_argument("--max-matches", type=int, default=0, help="per external group (0 = all)")
    ap.add_argument("--min-gap-s", type=float, default=0.2)
    ap.add_argument("--config", default=None, help="YAML (degrader parameters and Issue #38 target)")
    args = ap.parse_args(argv)
    groups: dict[str, dict] = {}
    if args.data_dir:
        from mtah2027.config import load_config
        from mtah2027.io.atd_loader import load_atd

        match = load_config().match
        td = load_atd(args.data_dir, use_cache=True)
        blocks = [match.half1, match.half2_dead, match.half2_play]
        groups["competition_atd"] = pool([input_stats(td.ball_xy, td.players_xy, blocks, td.fps,
                                                      args.min_gap_s)])
    if Path(args.bench_dir).is_dir():
        groups_ = bench_groups(Path(args.bench_dir), args.seed, args.max_matches, args.config)
        for name, items in groups_.items():
            groups[name] = pool(items)
    if not groups:
        print("nothing to describe: pass --data-dir and / or an existing --bench-dir")
        return 1
    print_table(groups)
    target = target_from_config(args.config)
    if target and "idsse_degraded" in groups:
        print_target(target, groups["idsse_degraded"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
