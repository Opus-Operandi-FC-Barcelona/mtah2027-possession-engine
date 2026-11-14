#!/usr/bin/env python
"""Aggregate-only sanity report of the ingestion and the stages on a tracking bundle (Issue #37).

Meant for the real BAR-ATH bundle, whose content must never be printed: every number reported here is a count,
a share or a quantile, never a per-frame row, a time or a position.  It answers:

* ATD: declared channels (schema names), frames parsed vs expected, frames present in the file;
* tracking: ball-missing share per graded block, on-pitch players per team (distribution), slots used;
* homographies (tactical / broadcast): valid / usable share;
* Smart Tagging: instances in the XML vs parsed, counts by code and team, set pieces by type and team,
  distinct code and label-group names (schema metadata);
* which pipeline stages ran or were skipped, the anchors found, the throw-ins resolved;
* diagnostic of the predicted labels: share of ``D`` among graded frames with the ball visible / hidden, with
  >= 20 / < 20 tracked players, inside / outside the Smart Tagging set-piece windows.

Examples::

    python scripts/real_data_sanity.py --data-dir data
    python scripts/real_data_sanity.py --synthetic --synthetic-frames 12000     # no real data needed
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for _extra in (REPO_ROOT / "src", REPO_ROOT, REPO_ROOT / "scripts"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

H, A, D, X = 0, 1, 2, 3
MIN_CELL = 1          # cells with fewer frames are still printed (counts are aggregates), see ``d_table``


# ----------------------------------------------------------------------------------------------- helpers
def _pct(num, den) -> float:
    return round(100.0 * float(num) / max(float(den), 1.0), 2)


def _graded_mask(n: int, blocks) -> np.ndarray:
    m = np.zeros(n, bool)
    for a, b in blocks:
        m[int(a):int(b) + 1] = True
    return m


def onpitch_counts(players_xy: np.ndarray, team_of_slot: np.ndarray, pitch, margin: float = 2.0):
    """Tracked players inside the pitch (+ margin) per frame for the home and the away team."""
    x, y = players_xy[..., 0], players_xy[..., 1]
    with np.errstate(invalid="ignore"):
        ok = (x >= -margin) & (x <= pitch[0] + margin) & (y >= -margin) & (y <= pitch[1] + margin)
    team = np.asarray(team_of_slot)
    return ok[:, team == 0].sum(1), ok[:, team == 1].sum(1)


def tracking_report(td, blocks, names=("half1", "half2_play")) -> dict:
    n = len(td.ball_xy)
    visible = np.isfinite(td.ball_xy).all(axis=1)
    out: dict = {"frames_parsed": int(n), "frames_present": int(td.present.sum()),
                 "slots": int(td.players_xy.shape[1]), "fps": float(td.fps), "pitch": list(td.pitch),
                 "ball_missing_pct": {}, "players_onpitch": {}}
    nh, na = onpitch_counts(td.players_xy, td.team_of_slot, td.pitch)
    total = nh + na
    graded_all = _graded_mask(n, blocks)
    for name, (a, b) in zip(names, blocks):
        sl = slice(int(a), int(b) + 1)
        out["ball_missing_pct"][name] = _pct((~visible[sl]).sum(), b - a + 1)
    out["ball_missing_pct"]["graded"] = _pct((~visible[graded_all]).sum(), graded_all.sum())
    tg = total[graded_all]
    out["players_onpitch"] = {
        "mean_total": round(float(tg.mean()), 2) if len(tg) else 0.0,
        "mean_home": round(float(nh[graded_all].mean()), 2) if len(tg) else 0.0,
        "mean_away": round(float(na[graded_all].mean()), 2) if len(tg) else 0.0,
        "pct_total_lt20": _pct((tg < 20).sum(), len(tg)),
        "pct_total_lt16": _pct((tg < 16).sum(), len(tg)),
        "pct_total_ge22": _pct((tg >= 22).sum(), len(tg)),
        "hist_total": {str(k): _pct((tg == k).sum(), len(tg)) for k in range(0, 24)
                       if (tg == k).any()}}
    # channels: share of finite values per ball channel inside the graded blocks
    out["ball_channel_finite_pct"] = {
        k: _pct(np.isfinite(v[:n][graded_all]).sum(), graded_all.sum()) for k, v in td.ball_channels.items()}
    return out


def homography_report(data_dir, sources, blocks, n_frames: int) -> dict:
    from mtah2027.io.homography_loader import load_homography

    graded = _graded_mask(n_frames, blocks)
    res: dict = {}
    for source in sources:
        try:
            hom = load_homography(data_dir, source=source)
        except Exception as exc:  # noqa: BLE001 - a missing optional file is reported, not fatal
            res[source] = {"status": f"unavailable: {type(exc).__name__}"}
            continue
        valid = np.zeros(n_frames, bool)
        usable = np.zeros(n_frames, bool)
        m = min(n_frames, hom.n_frames)
        valid[:m], usable[:m] = hom.valid[:m], hom.usable()[:m]
        err = hom.reprojection_error[:m][graded[:m] & hom.valid[:m]]
        res[source] = {"rows": int(hom.n_frames), "valid_pct_all": _pct(valid.sum(), n_frames),
                       "valid_pct_graded": _pct(valid[graded].sum(), graded.sum()),
                       "usable_pct_graded(err<5)": _pct(usable[graded].sum(), graded.sum()),
                       "reproj_err_median": round(float(np.nanmedian(err)), 3) if len(err) else None}
    return res


def smart_tagging_report(smart, xml_path=None) -> dict:
    rep: dict = {"events": len(smart.events), "set_pieces": len(smart.set_pieces), "goals": len(smart.goals)}
    if xml_path is not None:
        try:
            rep["instances_in_xml"] = sum(1 for e in ET.parse(xml_path).getroot().iter()
                                          if e.tag.rsplit("}", 1)[-1].lower() == "instance")
        except Exception as exc:  # noqa: BLE001
            rep["instances_in_xml"] = f"unreadable: {type(exc).__name__}"
    team_name = {0: "H", 1: "A", -1: "N/A"}
    rep["by_code"] = dict(sorted(Counter(e.code for e in smart.events).items(), key=lambda kv: -kv[1]))
    rep["by_code_team"] = {f"{c}|{team_name[t]}": k for (c, t), k in sorted(
        Counter((e.code, e.team) for e in smart.events).items(), key=lambda kv: (kv[0][0], kv[0][1]))}
    rep["set_pieces_by_type_team"] = {f"{ty}|{team_name[t]}": k for (ty, t), k in sorted(
        Counter((s.type, s.team) for s in smart.set_pieces).items())}
    rep["goals_by_team"] = {team_name[t]: k for t, k in Counter(g.team for g in smart.goals).items()}
    rep["half_values"] = dict(sorted(Counter(e.half for e in smart.events).items()))
    groups = Counter(g for e in smart.events for g in e.tags)
    rep["label_groups"] = dict(sorted(groups.items(), key=lambda kv: -kv[1]))
    rep["phase_events_without_team_pct"] = _pct(
        sum(1 for e in smart.events if e.code not in ("set pieces", "goals") and e.team < 0),
        max(sum(1 for e in smart.events if e.code not in ("set pieces", "goals")), 1))
    rep["n_frames_with_phase_prior_pct"] = _pct((smart.phase_prior_home != 0).sum(), smart.n_frames)
    rep["n_frames_in_set_piece_pct"] = _pct(smart.in_set_piece.sum(), smart.n_frames)
    return rep


def stage_report(diag: dict) -> dict:
    stages = {}
    for src in (diag.get("loader_stages") or {}, (diag.get("core") or {}).get("stages") or {}):
        for name, info in src.items():
            stages[name] = str(info.get("status", "?"))[:110]
    core = diag.get("core") or {}
    return {"stages": stages, "route": diag.get("route"), "emission_source": core.get("emission_source"),
            "n_anchors": core.get("n_anchors"), "n_anchors_resolved": core.get("n_anchors_resolved"),
            "throw_ins": core.get("throw_ins"), "has_status_channel": core.get("has_status_channel"),
            "owner_channel": core.get("owner_channel"), "final": diag.get("final")}


def d_table(labels: np.ndarray, graded: np.ndarray, visible: np.ndarray, n_players: np.ndarray,
            in_set_piece: np.ndarray | None) -> dict:
    """Share of ``D`` among graded frames, cut by ball visibility, tracked players and set-piece windows."""
    lab = np.asarray(labels)
    dead = lab == D
    sp = np.zeros(len(lab), bool) if in_set_piece is None else np.asarray(in_set_piece, bool)
    dropout = n_players < 20
    cuts = {
        "all_graded": graded,
        "ball_visible": graded & visible, "ball_hidden": graded & ~visible,
        "players_ge20": graded & ~dropout, "players_lt20": graded & dropout,
        "visible&players_ge20": graded & visible & ~dropout,
        "visible&players_lt20": graded & visible & dropout,
        "hidden&players_ge20": graded & ~visible & ~dropout,
        "hidden&players_lt20": graded & ~visible & dropout,
        "in_set_piece_window": graded & sp, "outside_set_piece_window": graded & ~sp,
        "hidden&in_set_piece": graded & ~visible & sp, "hidden&outside_set_piece": graded & ~visible & ~sp,
        "visible&in_set_piece": graded & visible & sp, "visible&outside_set_piece": graded & visible & ~sp,
    }
    return {k: {"frames_pct_of_graded": _pct(m.sum(), graded.sum()), "D_pct": _pct(dead[m].sum(), m.sum()),
                "H_pct": _pct((lab[m] == H).sum(), m.sum()), "A_pct": _pct((lab[m] == A).sum(), m.sum())}
            for k, m in cuts.items() if m.sum() >= MIN_CELL}


# ------------------------------------------------------------------------------------------------- report
def build_report(data_dir, blocks, *, cache_dir=None, specialists=None, config_path=None,
                 run_pipeline_stages: bool = True, smart_xml=None, sources=("tactical", "broadcast")) -> dict:
    from mtah2027.io.atd_loader import load_atd
    from mtah2027.pipeline import resolve_config, run_pipeline

    cfg_path, _ = resolve_config(config_path)
    n = blocks.n_video_frames
    play = [blocks.half1, blocks.half2_play]
    td = load_atd(data_dir, cache_dir=cache_dir, use_cache=cache_dir is not None, n_frames=n)
    report: dict = {"atd": {
        "player_channels": list(td.report.get("player_channels", [])),
        "ball_channels": list(td.report.get("ball_channels", [])),
        "ball_extra_channels": list(td.report.get("ball_extra_channels", [])),
        "expected_frames": n, "frames_match_expected": bool(len(td.ball_xy) == n)}}
    report["tracking"] = tracking_report(td, play)
    report["homography"] = homography_report(data_dir, sources, play, n)
    smart = None
    try:
        from mtah2027.io.smart_tagging_loader import load_smart_tagging, smart_tagging_path
        smart = load_smart_tagging(data_dir, n_frames=n)
        report["smart_tagging"] = smart_tagging_report(smart, smart_xml or smart_tagging_path(data_dir))
    except Exception as exc:  # noqa: BLE001
        report["smart_tagging"] = {"status": f"unavailable: {type(exc).__name__}: {exc}"}
    if run_pipeline_stages:
        res = run_pipeline(data_dir, cfg_path, blocks, cache_dir=cache_dir, log=lambda m: None,
                           specialists=specialists)
        report["pipeline"] = stage_report(res.diagnostics)
        nh, na = onpitch_counts(td.players_xy, td.team_of_slot, td.pitch)
        visible = np.isfinite(td.ball_xy).all(axis=1)
        pad = lambda a, fill: np.concatenate([a, np.full(max(n - len(a), 0), fill, a.dtype)])[:n]  # noqa: E731
        report["d_diagnostic"] = d_table(
            res.labels, _graded_mask(n, play), pad(visible, False), pad(nh + na, 0),
            smart.in_set_piece if smart is not None else None)
    return report


def format_report(rep: dict) -> str:
    import json

    return json.dumps(rep, indent=1, sort_keys=False, ensure_ascii=True)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default=None, help="bundle root (default: $MTAH_DATA_DIR or ./data)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--synthetic", action="store_true", help="run on a generated synthetic match")
    ap.add_argument("--synthetic-frames", type=int, default=None)
    ap.add_argument("--specialists", choices=("load", "off"), default="load",
                    help="emissions of the diagnostic run: committed specialists (default) or heuristic")
    ap.add_argument("--no-pipeline", action="store_true", help="skip the pipeline run (ingestion only)")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--cache-dir", default=None, help="parsed-ATD cache directory (default: ./cache)")
    ap.add_argument("--out-json", default=None, help="also write the aggregate report here")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    from mtah2027.config import MatchConfig, load_config, set_global_seeds
    from mtah2027.models.specialist_fusion import resolve_specialists
    from mtah2027.pipeline import resolve_config

    set_global_seeds(0)
    cfg_path, _ = resolve_config(args.config)
    config = load_config(cfg_path)
    blocks, tmp = config.match, None
    try:
        if args.synthetic:
            import generate_synthetic_fixtures as synth

            tmp = tempfile.TemporaryDirectory(prefix="mtah_sanity_")
            data_dir = Path(tmp.name)
            info = synth.generate_synthetic_match(data_dir, seed=0, n_frames=args.synthetic_frames or 12000,
                                                  full_length=not args.synthetic_frames)
            spec = info["spec"]
            blocks = MatchConfig(n_video_frames=spec.n_frames, half1=spec.half1, half2_dead=spec.half2_dead,
                                 half2_play=spec.half2_play)
            cache_dir = None
        else:
            data_dir = args.data_dir or os.environ.get("MTAH_DATA_DIR") or str(REPO_ROOT / "data")
            cache_dir = None if args.no_cache else Path(args.cache_dir or REPO_ROOT / "cache")
        spec_set = resolve_specialists(args.specialists, config_path=cfg_path, prefer_file=True)
        rep = build_report(data_dir, blocks, cache_dir=cache_dir, specialists=spec_set, config_path=cfg_path,
                           run_pipeline_stages=not args.no_pipeline,
                           sources=("tactical",) if args.synthetic else ("tactical", "broadcast"))
        text = format_report(rep)
        print(text)
        if args.out_json:
            Path(args.out_json).write_text(text, encoding="utf-8")
        return 0
    finally:
        if tmp is not None:
            tmp.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
