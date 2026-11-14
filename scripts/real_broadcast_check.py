#!/usr/bin/env python
"""Aggregate-only coverage report of the broadcast synchronisation on the real bundle (Issue #39).

Runs ``broadcast_sync`` (Issue #10) on ``data/broadcast`` against the tactical tracking and prints counts, shares and
quantiles only (never a frame, a time or a position):

* blocks found / synchronised, why the others were rejected, fingerprint-correlation and position-residual
  distributions, offset spread, ordering consistency of the placed blocks;
* tactical hidden-ball graded frames covered by a synchronised block and filled with a broadcast ball;
* physical consistency of the injected ball: speed inside the filled runs, implied speed of the jump between the
  last tactical ball and the first injected one (and the way back), share of impossible values (> 35 m/s), points
  off the pitch;
* with ``--effect``: how many graded labels change when the injection is switched on (diagnostic only).

    python scripts/real_broadcast_check.py --data-dir data
    python scripts/real_broadcast_check.py --synthetic            # tiny self-check on the synthetic bundle
"""
# ruff: noqa: E501  (long report lines)
from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for _extra in (REPO_ROOT / "src", REPO_ROOT, REPO_ROOT / "scripts"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

SPEED_LIMIT = 35.0


def _q(values, qs=(0.1, 0.5, 0.9), nd=2):
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    return [round(float(np.quantile(v, q)), nd) for q in qs] if len(v) else None


def _graded(n, blocks):
    g = np.zeros(n, bool)
    for a, b in blocks:
        g[int(a):int(b) + 1] = True
    return g


def runs(mask: np.ndarray):
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)       # [start, end)


def physical_consistency(ball_before: np.ndarray, ball_after: np.ndarray, source: np.ndarray, fps: float,
                         pitch, injected_code: int = 4, margin: float = 1.0) -> dict:
    """Speed and jump checks of the injected points (aggregates)."""
    inj = (source == injected_code) & ~np.isfinite(ball_before).all(axis=1)
    out: dict = {"injected_frames": int(inj.sum())}
    if not inj.any():
        return out
    s0, e0 = runs(inj)
    inside = ((ball_after[inj, 0] >= -margin) & (ball_after[inj, 0] <= pitch[0] + margin) &
              (ball_after[inj, 1] >= -margin) & (ball_after[inj, 1] <= pitch[1] + margin))
    out["injected_runs"] = int(len(s0))
    out["run_length_s_q10_50_90"] = _q((e0 - s0) / fps)
    out["off_pitch_pct"] = round(100.0 * float((~inside).mean()), 3)
    speeds = []
    for a, b in zip(s0, e0):
        if b - a >= 3:
            step = np.linalg.norm(np.diff(ball_after[a:b], axis=0), axis=1) * fps
            speeds.append(step)
    allv = np.concatenate(speeds) if speeds else np.zeros(0)
    out["in_run_speed_mps_q50_90_99"] = _q(allv, (0.5, 0.9, 0.99))
    out["in_run_speed_gt35_pct"] = round(100.0 * float((allv > SPEED_LIMIT).mean()), 3) if len(allv) else None
    # jumps at the borders: last tactical ball before the run / first after, implied speed over the gap frames
    known = np.isfinite(ball_before).all(axis=1)
    last_idx = np.maximum.accumulate(np.where(known, np.arange(len(known)), -1))
    next_idx = np.minimum.accumulate(np.where(known, np.arange(len(known)), 10 ** 9)[::-1])[::-1]
    jumps_in, jumps_out, dist_in, dist_out = [], [], [], []
    for a, b in zip(s0, e0):
        p = last_idx[a - 1] if a > 0 else -1
        if p >= 0 and a - p <= 10 * fps:
            dist_in.append(float(np.linalg.norm(ball_after[a] - ball_before[p])))
            jumps_in.append(dist_in[-1] / max((a - p) / fps, 1.0 / fps))
        q = next_idx[b] if b < len(known) else 10 ** 9
        if q < 10 ** 9 and q - (b - 1) <= 10 * fps:
            dist_out.append(float(np.linalg.norm(ball_before[q] - ball_after[b - 1])))
            jumps_out.append(dist_out[-1] / max((q - (b - 1)) / fps, 1.0 / fps))
    out["border_jump_into_run"] = {"n": len(jumps_in), "implied_mps_q50_90_99": _q(jumps_in, (0.5, 0.9, 0.99)),
                                   "distance_m_q50_90_99": _q(dist_in, (0.5, 0.9, 0.99)),
                                   "gt35_pct": round(100.0 * float((np.array(jumps_in) > SPEED_LIMIT).mean()), 2)
                                   if jumps_in else None}
    out["border_jump_out_of_run"] = {"n": len(jumps_out), "implied_mps_q50_90_99": _q(jumps_out, (0.5, 0.9, 0.99)),
                                     "distance_m_q50_90_99": _q(dist_out, (0.5, 0.9, 0.99)),
                                     "gt35_pct": round(100.0 * float((np.array(jumps_out) > SPEED_LIMIT).mean()), 2)
                                     if jumps_out else None}
    return out


def block_report(res, graded: np.ndarray) -> dict:
    blocks = res.blocks
    acc = [b for b in blocks if b.accepted]
    rep = {"stats": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.stats.items()},
           "block_length_s_q10_50_90_all": _q([b.length / 25.0 for b in blocks], nd=1),
           "block_length_s_q10_50_90_accepted": _q([b.length / 25.0 for b in acc], nd=1),
           "corr_q10_50_90_all": _q([b.corr for b in blocks]), "corr_q10_50_90_accepted": _q([b.corr for b in acc]),
           "cost_m_q10_50_90_accepted": _q([b.cost for b in acc]),
           "ball_disagree_m_q10_50_90_accepted": _q([b.ball_disagree for b in acc]),
           "offset_s_q10_50_90_accepted": _q([b.offset / 25.0 for b in acc], nd=1),
           "accepted_by_corr_band_pct": {
               ">=0.9": round(100.0 * sum(b.corr >= 0.9 for b in acc) / max(len(acc), 1), 1),
               "0.65-0.9": round(100.0 * sum(0.65 <= b.corr < 0.9 for b in acc) / max(len(acc), 1), 1),
               "<0.65 (local pass)": round(100.0 * sum(b.corr < 0.65 for b in acc) / max(len(acc), 1), 1)}}
    if len(acc) > 2:
        order_b = np.array([b.start for b in acc], float)
        order_t = np.array([b.start + b.offset for b in acc], float)
        i = np.argsort(order_b)
        d = np.diff(order_t[i])
        rep["placed_blocks_in_broadcast_order_monotone_pct"] = round(100.0 * float((d > 0).mean()), 1)
    cover = np.zeros(len(graded), bool)
    for b in acc:
        lo, hi = max(b.start + b.offset, 0), min(b.stop + b.offset, len(graded))
        if hi > lo:
            cover[lo:hi] = True
    rep["graded_frames_covered_by_a_synchronised_block_pct"] = round(
        100.0 * float((cover & graded).sum()) / max(graded.sum(), 1), 2)
    return rep


def build_report(data_dir, blocks, *, cache_dir=None, config_path=None, effect: bool = False,
                 specialists=None) -> dict:
    from mtah2027.io.atd_loader import load_atd
    from mtah2027.pipeline import resolve_config
    from mtah2027.preprocessing import broadcast_sync as bs

    cfg_path, _ = resolve_config(config_path)
    n = blocks.n_video_frames
    td = load_atd(data_dir, cache_dir=cache_dir, use_cache=cache_dir is not None, n_frames=n)
    params = dataclasses.replace(bs.BroadcastSyncParams.from_config(cfg_path), enabled=True)
    play = [blocks.half1, blocks.half2_play]
    res = bs.sync_broadcast_files(td, data_dir, play, params=params, cache_dir=cache_dir)
    graded = _graded(n, play)
    hidden = graded & ~np.isfinite(td.ball_xy).all(axis=1)
    rep: dict = {"hidden_graded_pct_of_graded": round(100.0 * hidden.sum() / graded.sum(), 2)}
    rep["blocks"] = block_report(res, graded)
    rep["fill"] = {"hidden_graded_frames_filled_pct": round(100.0 * float(res.stats["fill_rate"]), 2),
                   "ball_missing_after_pct_of_graded": round(
                       100.0 * float((graded & ~np.isfinite(res.ball_xy).all(axis=1)).sum()) / graded.sum(), 2)}
    rep["physical"] = physical_consistency(td.ball_xy, res.ball_xy, res.ball_source, float(td.fps), td.pitch)
    if effect:
        import shutil

        from mtah2027.pipeline import run_pipeline

        tmp_cfg = Path(tempfile.mkdtemp(prefix="mtah_bs_")) / "cfg.yaml"
        text = Path(cfg_path).read_text(encoding="utf-8")
        marker = "broadcast_sync:\n  enabled: false"
        if marker not in text.replace("\r\n", "\n"):
            raise RuntimeError("broadcast_sync.enabled marker not found in the configuration")
        tmp_cfg.write_text(text.replace("\r\n", "\n").replace(marker, "broadcast_sync:\n  enabled: true"),
                           encoding="utf-8")
        off = run_pipeline(data_dir, cfg_path, blocks, cache_dir=cache_dir, log=lambda m: None,
                           specialists=specialists)
        on = run_pipeline(data_dir, tmp_cfg, blocks, cache_dir=cache_dir, log=lambda m: None,
                          specialists=specialists)
        shutil.rmtree(tmp_cfg.parent, ignore_errors=True)
        diff = (off.labels != on.labels) & graded
        rep["effect_of_injection"] = {
            "graded_labels_changed_pct": round(100.0 * float(diff.sum()) / graded.sum(), 2),
            "off": {k: round(v, 3) for k, v in off.diagnostics["final"].items()},
            "on": {k: round(v, 3) for k, v in on.diagnostics["final"].items()}}
    return rep


def main(argv=None) -> int:
    import json

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--effect", action="store_true", help="also compare the labels with / without injection")
    ap.add_argument("--specialists", choices=("load", "off"), default="load")
    args = ap.parse_args(argv)
    from mtah2027.config import MatchConfig, load_config, set_global_seeds
    from mtah2027.models.specialist_fusion import resolve_specialists
    from mtah2027.pipeline import resolve_config

    set_global_seeds(0)
    cfg_path, _ = resolve_config(args.config)
    blocks, tmp = load_config(cfg_path).match, None
    try:
        if args.synthetic:
            import generate_synthetic_fixtures as synth

            tmp = tempfile.TemporaryDirectory(prefix="mtah_bscheck_")
            data_dir = Path(tmp.name)
            spec = synth.generate_synthetic_match(data_dir, seed=0, n_frames=12000)["spec"]
            blocks = MatchConfig(n_video_frames=spec.n_frames, half1=spec.half1, half2_dead=spec.half2_dead,
                                 half2_play=spec.half2_play)
            cache_dir = None
        else:
            data_dir = args.data_dir or os.environ.get("MTAH_DATA_DIR") or str(REPO_ROOT / "data")
            cache_dir = Path(args.cache_dir or REPO_ROOT / "cache")
        specs = resolve_specialists(args.specialists, config_path=cfg_path, prefer_file=True) if args.effect else None
        print(json.dumps(build_report(data_dir, blocks, cache_dir=cache_dir, config_path=cfg_path,
                                      effect=args.effect, specialists=specs), indent=1))
        return 0
    finally:
        if tmp is not None:
            tmp.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
