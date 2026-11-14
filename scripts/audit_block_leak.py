#!/usr/bin/env python
"""Issue 51: how much of the graded front end depends on tracking OUTSIDE the play blocks?

Runs the front end twice -- ``pipeline.isolate_play_blocks`` off (legacy: all frames feed each other) and on
(frames outside the blocks masked) -- and counts, per feature family, the graded frames whose value
differs, plus the largest distance (seconds) of a differing frame from the nearest block edge.  With
``--run-labels`` the whole core is also run both ways (label changes, D share, possessions).  Aggregates
only: nothing per frame is printed.  Targets: external ``*.npz`` matches (``--bench-dir``) or the real match
(``--data-dir``).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT / "src", REPO_ROOT):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from mtah2027 import pipeline as pl  # noqa: E402
from mtah2027.models.specialist_features import build_feature_matrix  # noqa: E402

TOL = 1e-4


def _toggle(flag: bool):
    """Force ``isolate_play_blocks`` to ``flag``; returns the original classmethod function."""
    orig = pl.PipelineParams.from_config.__func__

    def patched(cls, path=None):
        return dataclasses.replace(orig(cls, path), isolate_play_blocks=flag)

    pl.PipelineParams.from_config = classmethod(patched)
    return orig


def _differs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per frame: any element differs (NaN vs finite counts, NaN vs NaN does not)."""
    a = np.asarray(a, np.float64).reshape(len(a), -1)
    b = np.asarray(b, np.float64).reshape(len(b), -1)
    fa, fb = np.isfinite(a), np.isfinite(b)
    with np.errstate(invalid="ignore"):
        d = (fa != fb) | (fa & fb & (np.abs(a - b) > TOL))
    return d.any(axis=1)


def _edge_distance(n: int, blocks) -> np.ndarray:
    dist = np.full(n, -1, np.int64)
    for a, b in blocks:
        i = np.arange(a, b + 1)
        dist[a:b + 1] = np.minimum(i - a, b - i)
    return dist


def front_end_pair(players, team, ball, blocks, fps, pitch, smart=None, channels=None, codes=None,
                   team_ids=(), team_names=(), cfg=None):
    out = {}
    for flag in (False, True):
        orig = _toggle(flag)
        try:
            fe = pl.compute_front_end(players, team, ball, play_blocks=blocks, fps=fps, pitch=pitch,
                                      ball_channels=channels, ball_channel_codes=codes, team_ids=team_ids,
                                      team_names=team_names, smart_tagging=smart, config_path=cfg,
                                      collective=True, stages=pl._Stages(lambda m: None))
            em = pl.heuristic_emissions(fe, fps, cfg)
            fm = build_feature_matrix(fe, em, play_blocks=blocks, config_path=cfg)
        finally:
            pl.PipelineParams.from_config = classmethod(orig)
        out[flag] = (fe, fm, em)
    return out


def compare(pair, blocks, fps) -> dict:
    (fe0, fm0, em0), (fe1, fm1, em1) = pair[False], pair[True]
    n = len(fe0.visible)
    dist = _edge_distance(n, blocks)
    graded = dist >= 0
    fam: dict[str, np.ndarray] = {}
    for key in ("pxy", "bxy", "pvel", "bvel", "bacc"):
        a, b = getattr(fe0, key), getattr(fe1, key)
        if a is not None and b is not None:
            fam[f"kin:{key}"] = _differs(a, b)
    cols = {name: _differs(fm0.x[:, j], fm1.x[:, j]) for j, name in enumerate(fm0.names)}
    groups: dict[str, list] = {}
    for name, m in cols.items():
        groups.setdefault(name.split(":")[0], []).append(m)
    for g, ms in groups.items():
        fam[f"feat:{g}:*"] = np.any(ms, axis=0)
    for name in ("sig:t_since_ball", "sig:t_to_ball"):
        if name in cols:
            fam[f"feat:{name}"] = cols[name]
    fam["feat:ANY"] = _differs(fm0.x, fm1.x)
    fam["emission:log_probs"] = _differs(em0.log_probs, em1.log_probs)
    res = {}
    for k, m in fam.items():
        m = m & graded
        res[k] = {"frames": int(m.sum()), "within_60s": int((m & (dist <= 60 * fps)).sum()),
                  "max_dist_s": round(float(dist[m].max() / fps), 2) if m.any() else 0.0}
    res["graded_frames"] = int(graded.sum())
    res["frames_outside_blocks"] = int((~graded).sum())
    return res


def label_effect(codes0, codes1, blocks) -> dict:
    g = np.zeros(len(codes0), bool)
    for a, b in blocks:
        g[a:b + 1] = True
    return {"labels_changed_pct": round(float((codes0[g] != codes1[g]).mean() * 100), 3),
            "D_pct_off": round(float((codes0[g] == 2).mean() * 100), 2),
            "D_pct_on": round(float((codes1[g] == 2).mean() * 100), 2),
            "possessions_off": pl._count_possessions(codes0, blocks),
            "possessions_on": pl._count_possessions(codes1, blocks)}


def run_labels(core_kwargs, blocks):
    codes = {}
    for flag in (False, True):
        orig = _toggle(flag)
        try:
            codes[flag] = pl.run_core(**core_kwargs, log=lambda m: None).codes
        finally:
            pl.PipelineParams.from_config = classmethod(orig)
    return label_effect(codes[False], codes[True], blocks)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bench-dir", default=None)
    ap.add_argument("--matches", nargs="*", default=None)
    ap.add_argument("--data-dir", default=None, help="real match bundle (aggregates only)")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--run-labels", action="store_true")
    ap.add_argument("--specialists", action="store_true", help="with --run-labels: committed specialists")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    report = {}
    if args.bench_dir:
        for f in sorted(Path(args.bench_dir).glob("*.npz")):
            if args.matches and f.stem not in args.matches:
                continue
            z = np.load(f, allow_pickle=True)
            players, team = z["players_xy"].astype(np.float32), z["team_of_slot"].astype(np.int8)
            ball, fps = z["ball_xy"].astype(np.float32), float(z["fps"])
            pitch = tuple(float(v) for v in z["pitch"])
            blocks = [(int(a), int(b)) for a, b in z["period_bounds"]]
            report[f.stem] = compare(front_end_pair(players, team, ball, blocks, fps, pitch, cfg=args.config),
                                     blocks, fps)
            if args.run_labels:
                report[f.stem]["labels"] = run_labels(dict(
                    players_xy=players, team_of_slot=team, ball_xy=ball, play_blocks=blocks, fps=fps,
                    pitch=pitch, config_path=args.config), blocks)
    if args.data_dir:
        from mtah2027.config import MatchConfig
        from mtah2027.io.atd_loader import load_atd
        from mtah2027.io.smart_tagging_loader import load_smart_tagging

        m = MatchConfig()
        n = m.n_video_frames
        td = load_atd(args.data_dir, cache_dir=args.cache_dir, use_cache=args.cache_dir is not None,
                      n_frames=n)
        smart = load_smart_tagging(args.data_dir, n_frames=n)
        blocks = [m.half1, m.half2_play]
        players, ball = pl._fit_length(td.players_xy, n), pl._fit_length(td.ball_xy, n)
        chans = {k: pl._fit_length(v, n) for k, v in td.ball_channels.items()}
        report["BAR-ATH"] = compare(front_end_pair(
            players, td.team_of_slot, ball, blocks, td.fps, tuple(td.pitch), smart=smart, channels=chans,
            codes=td.ball_channel_codes, team_ids=td.team_ids, team_names=td.team_names,
            cfg=args.config), blocks, td.fps)
        report["BAR-ATH"]["ball_frames_outside_blocks"] = int(
            np.isfinite(ball).all(1).sum() - sum(np.isfinite(ball[a:b + 1]).all(1).sum() for a, b in blocks))
        if args.run_labels:
            spec = None
            if args.specialists:
                from mtah2027.models.specialist_fusion import resolve_specialists
                spec = resolve_specialists("load", config_path=args.config, prefer_file=True)
            kw = dict(players_xy=players, team_of_slot=td.team_of_slot, ball_xy=ball, play_blocks=blocks,
                      dead_blocks=[m.half2_dead], fps=td.fps, pitch=tuple(td.pitch), ball_channels=chans,
                      ball_channel_codes=td.ball_channel_codes, team_ids=td.team_ids,
                      team_names=td.team_names, smart_tagging=smart, config_path=args.config,
                      specialists=spec)
            report["BAR-ATH"]["labels"] = run_labels(kw, blocks)
    text = json.dumps(report, indent=1)
    print(text)
    if args.json:
        Path(args.json).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
