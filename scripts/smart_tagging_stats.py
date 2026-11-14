# ruff: noqa: E501
#!/usr/bin/env python
"""Aggregate structure of a Smart Tagging file (Issue #43): input statistics to calibrate the simulator.

Prints per phase code the number of instances, duration quantiles and the share of graded frames covered; the
union coverage, the mean multiplicity of overlapping phases, the quantiles of the uncovered gaps between
phases and the set-piece window lengths.  Counts, shares and quantiles only: never times of individual
instances, no labels.

    python scripts/smart_tagging_stats.py --data-dir data [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for _extra in (REPO_ROOT / "src", REPO_ROOT):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from mtah2027.config import FPS, HALF1, HALF2_PLAY, N_VIDEO_FRAMES  # noqa: E402
from mtah2027.io.smart_tagging_loader import (  # noqa: E402
    GOAL_CODE,
    SET_PIECE_CODE,
    build_smart_tagging,
    parse_events,
    smart_tagging_path,
)


def _q(values, qs=(0.1, 0.25, 0.5, 0.75, 0.9)):
    v = np.asarray(values, dtype=float)
    return [round(float(np.quantile(v, q)), 2) for q in qs] if len(v) else None


def stats(events, n: int, blocks, fps: float = FPS) -> dict:
    graded = np.zeros(n, bool)
    for a, b in blocks:
        graded[a:b + 1] = True
    ng = int(graded.sum())
    phases = [e for e in events if e.code not in (SET_PIECE_CODE, GOAL_CODE)]
    cover = np.zeros(n, np.int16)
    by_code: dict = defaultdict(list)
    for e in phases:
        by_code[e.code].append(e)
        cover[max(e.start_frame, 0):min(e.end_frame, n)] += 1
    out: dict = {"n_graded": ng, "n_phases": len(phases), "codes": {}}
    for code, evs in sorted(by_code.items()):
        m = np.zeros(n, bool)
        for e in evs:
            m[max(e.start_frame, 0):min(e.end_frame, n)] = True
        out["codes"][code] = {
            "n": len(evs), "duration_s_q10_25_50_75_90": _q([(e.end_frame - e.start_frame) / fps for e in evs]),
            "graded_cover_pct": round(100.0 * (m & graded).sum() / ng, 2)}
    cov = (cover > 0) & graded
    out["union_cover_pct"] = round(100.0 * cov.sum() / ng, 2)
    out["mean_multiplicity_where_covered"] = round(float(cover[cov].mean()), 3) if cov.any() else None
    gaps, run = [], 0
    for a, b in blocks:
        seg = cover[a:b + 1] > 0
        run = 0
        for v in seg:
            if v:
                if run:
                    gaps.append(run / fps)
                run = 0
            else:
                run += 1
        if run:
            gaps.append(run / fps)
    out["uncovered_gap_s_q10_25_50_75_90"] = _q(gaps)
    out["n_uncovered_gaps"] = len(gaps)
    out["uncovered_gaps_over_5s_share_of_graded_pct"] = round(100.0 * sum(g for g in gaps if g > 5) * fps / ng, 2)
    out["phases_per_10_graded_min"] = round(len(phases) / (ng / fps / 600.0), 2)
    sets = [e for e in events if e.code == SET_PIECE_CODE]
    out["set_pieces"] = {"n": len(sets), "window_s_q10_50_90": _q([(e.end_frame - e.start_frame) / fps for e in sets],
                                                                  (0.1, 0.5, 0.9)),
                         "graded_cover_pct": round(100.0 * float(np.mean([
                             0])), 2)}
    m = np.zeros(n, bool)
    for e in sets:
        m[max(e.start_frame, 0):min(e.end_frame, n)] = True
    out["set_pieces"]["graded_cover_pct"] = round(100.0 * (m & graded).sum() / ng, 2)
    out["n_goals"] = sum(1 for e in events if e.code == GOAL_CODE)
    st = build_smart_tagging(events, n)
    out["phase_prior_nonzero_pct"] = round(100.0 * ((st.phase_prior_home != 0) & graded).sum() / ng, 2)
    out["phase_team_decided_pct"] = round(100.0 * ((st.phase_team >= 0) & graded).sum() / ng, 2)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    path = smart_tagging_path(args.data_dir)
    events = parse_events(path)
    res = stats(events, N_VIDEO_FRAMES, [HALF1, HALF2_PLAY])
    print(json.dumps(res, indent=1))
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
