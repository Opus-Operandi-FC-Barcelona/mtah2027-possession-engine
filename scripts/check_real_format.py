#!/usr/bin/env python
"""Run the whole pipeline on an explicit EPTS pair (e.g. Metrica's open Sample Game 3) and print aggregates.

    python scripts/check_real_format.py --meta Sample_Game_3_metadata.xml --raw Sample_Game_3_tracking.txt
    python scripts/check_real_format.py --meta m.xml --raw r.txt --smart-tagging st.xml --homography h.csv

Purpose: prove that the loaders cope with the real file layout (several ``DataFormatSpecification`` blocks,
no speed channel, ``NaN`` ball, ...).  Smart Tagging and homography are optional; without them the stages that
need them are skipped.  Only counts, shares and timings are printed (never per-frame data), so the script is
also safe on the real competition bundle.
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

from mtah2027.io.atd_loader import load_atd_files  # noqa: E402
from mtah2027.io.homography_loader import parse_homography_csv  # noqa: E402
from mtah2027.io.smart_tagging_loader import build_smart_tagging, parse_events  # noqa: E402
from mtah2027.pipeline import match_from_periods, run_pipeline_files  # noqa: E402


def _periods(text: str | None):
    if not text:
        return None
    return [tuple(int(v) for v in part.split(":")) for part in text.split(",")]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--meta", required=True, help="EPTS metadata XML")
    ap.add_argument("--raw", required=True, help="raw tracking text file")
    ap.add_argument("--smart-tagging", default=None, help="optional SportsCode XML")
    ap.add_argument("--homography", default=None, help="optional homography CSV")
    ap.add_argument("--periods", default=None,
                    help="video-frame periods 'a:b,c:d' (inclusive, 0-based); default: from the XML")
    ap.add_argument("--config", default=None)
    ap.add_argument("--json", default=None, help="write the aggregate report to this file")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    td = load_atd_files(args.meta, args.raw)
    r = td.report
    report: dict = {
        "atd": {
            "n_frames": td.n_frames, "n_slots": td.n_slots, "fps": td.fps, "pitch": list(td.pitch),
            "n_specifications": r["n_specifications"], "pitch_from_xml": r["pitch_from_xml"],
            "player_channels": r["player_channels"], "ball_channels": r["ball_channels"],
            "has_player_speed": bool(np.isfinite(td.players_speed).any()),
            "contiguous": r["contiguous"], "malformed_lines": r["malformed_lines"],
            "duplicate_frames": r["duplicate_frames"], "periods": [list(p) for p in td.periods],
            "ball_visible_share": round(float(td.ball_visible.mean()), 4),
            "players_on_pitch_mean": round(float(np.isfinite(td.players_xy[..., 0]).sum(1).mean()), 2),
            "home_slots": int((td.team_of_slot == 0).sum()), "away_slots": int((td.team_of_slot == 1).sum()),
            "seconds": round(time.perf_counter() - t0, 2),
        }}
    periods = _periods(args.periods) or list(td.periods)
    if len(periods) < 2:
        print("the XML declares no periods; pass --periods 'a:b,c:d'")
        return 2

    if args.smart_tagging:
        events = parse_events(args.smart_tagging, int(round(td.fps)))
        st = build_smart_tagging(events, td.n_frames, int(round(td.fps)))
        report["smart_tagging"] = {"events": len(events), "set_pieces": len(st.set_pieces),
                                   "goals": len(st.goals), "codes": st.codes}
    if args.homography:
        h = parse_homography_csv(args.homography, tuple(td.pitch))
        report["homography"] = {"frames": h.n_frames, "valid_share": round(float(h.valid.mean()), 4),
                                "usable_share": round(float(h.usable().mean()), 4)}

    messages: list[str] = []
    res = run_pipeline_files(args.meta, args.raw, smart_tagging_path=args.smart_tagging, config=args.config,
                             periods=periods, log=messages.append)
    match = match_from_periods(periods, td.n_frames, td.fps, tuple(td.pitch))
    labels = np.asarray(res.labels)
    graded = np.zeros(len(labels), bool)
    for a, b in ((match.half1), (match.half2_play)):
        graded[a:b + 1] = True
    d = res.diagnostics
    report["pipeline"] = {
        "route": d["route"], "x_in_graded": int((labels[graded] == 3).sum()),
        "x_outside_graded": int((labels[~graded] == 3).sum()), "outside_graded": int((~graded).sum()),
        "final": d["final"], "viterbi_raw": d["core"].get("viterbi_raw"),
        "emission_source": d["core"].get("emission_source"),
        "stages": {k: v["status"] for k, v in d["core"].get("stages", {}).items()},
        "loader_stages": {k: v["status"] for k, v in d["loader_stages"].items()},
        "seconds": d["seconds_total"], "messages": len(messages),
    }
    print(json.dumps(report, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    failed = [k for k, v in report["pipeline"]["stages"].items() if v.startswith("failed")]
    return 1 if failed or d["route"] == "baseline-fallback" or report["pipeline"]["x_in_graded"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
