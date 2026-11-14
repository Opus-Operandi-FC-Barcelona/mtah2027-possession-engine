#!/usr/bin/env python
"""Aggregate-only check of how the Smart Tagging reaches the decoder on a tracking bundle (Issue #40).

Reports (counts / shares / quantiles, never times of individual events or per-frame rows):

* set-piece anchors by type x method (ball / collective / unresolved), resolved share, confidence quantiles,
  offset of ``t*`` from the tag-window start (seconds, quantiles), share of the graded frames forced to ``D``;
* the two published kick-offs (``Kick Off (Start)`` anchors vs the published half-start frames, in frames);
* throw-ins with Team = N/A: resolved share, method counts, confidence quantiles;
* the phase prior: share of graded frames with a non-zero prior, and for every phase code the share of the
  alive frames predicted for the *possession side* (tag team, flipped for the defending phases) versus the
  tag team itself -- a consistency diagnostic of the ``Team`` semantics, not a selection criterion.

    python scripts/smart_tagging_check.py --data-dir data
    python scripts/smart_tagging_check.py --synthetic
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for _extra in (REPO_ROOT / "src", REPO_ROOT, REPO_ROOT / "scripts"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

H, A, D, X = 0, 1, 2, 3


def _q(values, qs=(0.1, 0.5, 0.9)):
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    return [round(float(np.quantile(v, q)), 2) for q in qs] if len(v) else None


def _graded(n: int, blocks) -> np.ndarray:
    g = np.zeros(n, bool)
    for a, b in (blocks.half1, blocks.half2_play):
        g[a:b + 1] = True
    return g


def anchor_report(anchors, throwins, blocks, fps: float = 25.0) -> dict:
    """Aggregates of the :class:`AnchorResult` and of the throw-in resolutions."""
    rep: dict = {"n_anchors": len(anchors.anchors), "n_resolved": anchors.n_resolved}
    rep["resolved_pct"] = round(100.0 * anchors.n_resolved / max(len(anchors.anchors), 1), 1)
    by: dict = defaultdict(Counter)
    for a in anchors.anchors:
        by[a.type][a.method] += 1
    rep["by_type_method"] = {t: dict(c) for t, c in sorted(by.items())}
    res = [a for a in anchors.anchors if a.resolved]
    rep["confidence_q10_50_90"] = _q([a.confidence for a in res])
    rep["t_star_minus_tag_start_s_q10_50_90"] = _q([(a.t_star - a.tag_start) / fps for a in res])
    rep["t_star_inside_tag_window_pct"] = round(
        100.0 * sum(a.tag_start <= a.t_star <= a.tag_end for a in res) / max(len(res), 1), 1)
    graded = _graded(len(anchors.forced_dead_mask), blocks)
    rep["forced_dead_pct_of_graded"] = round(
        100.0 * (anchors.forced_dead_mask & graded).sum() / max(graded.sum(), 1), 2)
    rep["forced_dead_intervals"] = len(anchors.forced_dead_intervals)
    kick = {}
    for half, truth in ((1, blocks.half1[0]), (2, blocks.half2_play[0])):
        cands = [a for a in res if a.type == "kick off (start)" and a.tag_start <= truth <= a.tag_end]
        kick[f"half{half}_t_star_minus_published_frames"] = [int(a.t_star - truth) for a in cands] or None
    rep["kickoff_vs_published"] = kick
    if throwins is not None:
        done = sum(r.resolved for r in throwins)
        rep["throw_ins"] = {
            "total": len(throwins), "resolved": done,
            "resolved_pct": round(100.0 * done / max(len(throwins), 1), 1),
            "methods": dict(Counter(r.method for r in throwins)),
            "confidence_q10_50_90": _q([r.confidence for r in throwins if r.resolved])}
    return rep


def phase_report(smart, labels: np.ndarray, blocks) -> dict:
    """Prior coverage and, per phase code, agreement of the predicted possessor with the tag."""
    from mtah2027.io.smart_tagging_loader import GOAL_CODE, SET_PIECE_CODE, possession_side

    n = len(labels)
    graded = _graded(n, blocks)
    prior = np.asarray(smart.phase_prior_home)[:n]
    nz = (prior != 0) & graded
    rep: dict = {"flipped_events": int(smart.n_flipped),
                 "prior_nonzero_pct_of_graded": round(100.0 * nz.sum() / max(graded.sum(), 1), 1),
                 "prior_mean_abs_when_nonzero":
                 round(float(np.abs(prior[nz]).mean()), 3) if nz.any() else None}
    agree: dict = defaultdict(lambda: np.zeros(3))      # alive frames, = possession side, = raw tag team
    for e in smart.events:
        if e.code in (SET_PIECE_CODE, GOAL_CODE) or e.team < 0:
            continue
        lo, hi = max(e.start_frame, 0), min(e.end_frame, n)
        if hi <= lo:
            continue
        seg = labels[lo:hi]
        alive = ((seg == H) | (seg == A)) & graded[lo:hi]
        side = possession_side(e)
        agree[e.code] += [alive.sum(), (alive & (seg == side)).sum(), (alive & (seg == e.team)).sum()]
    rep["agreement_by_code"] = {
        c: {"alive_frames": int(v[0]),
            "pred_equals_possession_side_pct": round(100.0 * v[1] / max(v[0], 1), 1),
            "pred_equals_raw_tag_team_pct": round(100.0 * v[2] / max(v[0], 1), 1)}
        for c, v in sorted(agree.items())}
    return rep


def build_report(data_dir, blocks, *, cache_dir=None, specialists=None, config_path=None) -> dict:
    from mtah2027.io.atd_loader import load_atd
    from mtah2027.io.smart_tagging_loader import load_smart_tagging
    from mtah2027.pipeline import _Stages, compute_front_end, resolve_config, run_pipeline

    cfg_path, _ = resolve_config(config_path)
    n = blocks.n_video_frames
    td = load_atd(data_dir, cache_dir=cache_dir, use_cache=cache_dir is not None, n_frames=n)
    smart = load_smart_tagging(data_dir, n_frames=n)
    play = [blocks.half1, blocks.half2_play]
    fe = compute_front_end(td.players_xy[:n], td.team_of_slot, td.ball_xy[:n], play_blocks=play, fps=td.fps,
                           pitch=tuple(td.pitch), smart_tagging=smart, config_path=cfg_path,
                           stages=_Stages(lambda m: None))
    rep: dict = {"smart_events": len(smart.events), "set_pieces": len(smart.set_pieces)}
    rep["anchors"] = (anchor_report(fe.anchors, fe.throwins, blocks, td.fps) if fe.anchors is not None
                      else "none (stage failed or disabled)")
    res = run_pipeline(data_dir, cfg_path, blocks, cache_dir=cache_dir, log=lambda m: None,
                       specialists=specialists)
    core = res.diagnostics.get("core", {})
    rep["emissions"] = {"source": core.get("emission_source"),
                        "phase_prior_in_front_end": fe.phase is not None,
                        "throw_ins": core.get("throw_ins")}
    rep["phase"] = phase_report(smart, res.labels, blocks)
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--synthetic", action="store_true")
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

            tmp = tempfile.TemporaryDirectory(prefix="mtah_stcheck_")
            data_dir = Path(tmp.name)
            spec = synth.generate_synthetic_match(data_dir, seed=0, n_frames=12000)["spec"]
            blocks = MatchConfig(n_video_frames=spec.n_frames, half1=spec.half1, half2_dead=spec.half2_dead,
                                 half2_play=spec.half2_play)
            cache_dir = None
        else:
            data_dir = args.data_dir or os.environ.get("MTAH_DATA_DIR") or str(REPO_ROOT / "data")
            cache_dir = Path(args.cache_dir or REPO_ROOT / "cache")
        specs = resolve_specialists(args.specialists, config_path=cfg_path, prefer_file=True)
        print(json.dumps(build_report(data_dir, blocks, cache_dir=cache_dir, specialists=specs,
                                      config_path=cfg_path), indent=1))
        return 0
    finally:
        if tmp is not None:
            tmp.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
