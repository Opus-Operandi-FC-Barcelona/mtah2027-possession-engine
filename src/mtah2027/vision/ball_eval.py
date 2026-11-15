"""Aggregate evaluation of a ball detector on the validation crops of Issue #68 (no per-frame output).

Candidates are reprojected to pitch metres with ``H^-1`` and compared with the ATD ball (label-free: no H/A/D).
Everything returned is a scalar or a short list; :func:`assert_aggregate` enforces it.
"""
# ruff: noqa: E501
from __future__ import annotations

import numpy as np

RADII = (0.3, 0.5, 1.0, 2.0, 5.0)
HIST_EDGES = (0.15, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0)      # bins of the top-candidate error (m), plus ">10"
CONF_GATES = (0.1, 0.25, 0.4, 0.55, 0.7)
MAX_LIST = 12


def assert_aggregate(obj, _depth: int = 0) -> None:
    if _depth > 6:
        raise ValueError("report nested too deeply")
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(k, str):
                raise ValueError("report keys must be strings")
            assert_aggregate(v, _depth + 1)
        return
    if isinstance(obj, (list, tuple)):
        if len(obj) > MAX_LIST:
            raise ValueError(f"list of {len(obj)} entries in a report: per-frame data is not allowed")
        for v in obj:
            if isinstance(v, (list, tuple, dict)):
                raise ValueError("nested containers in lists are not allowed")
            assert_aggregate(v, _depth + 1)
        return
    raise ValueError(f"type {type(obj).__name__} is not allowed in a report")


def top1_errors(n_frames: int, cand_row, cand_xy, cand_conf, truth_xy):
    """Per crop ``(error of the top-confidence candidate (inf if none), its confidence (0 if none), count)``."""
    err = np.full(n_frames, np.inf)
    conf = np.zeros(n_frames)
    cnt = np.zeros(n_frames, np.int64)
    if len(cand_row):
        cand_row = np.asarray(cand_row)
        cc = np.asarray(cand_conf)
        order = np.lexsort((-cc, cand_row))
        r = cand_row[order]
        first = np.r_[True, r[1:] != r[:-1]]
        d = np.hypot(*(np.asarray(cand_xy)[order] - truth_xy[r]).T)
        err[r[first]] = d[first]
        conf[r[first]] = cc[order][first]
        np.add.at(cnt, cand_row, 1)
    return err, conf, cnt


def profile(err: np.ndarray, conf: np.ndarray, cnt: np.ndarray, n_false_positive: int,
            fp_radius: float = 5.0) -> dict:
    """Aggregate profile of one set of crops that all contain the ball (``err`` finite or inf)."""
    n = int(len(err))
    if n == 0:
        return {"crops": 0}
    fin = err[np.isfinite(err)]
    edges = (0.0,) + HIST_EDGES + (np.inf,)
    hist = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (err <= hi) if lo == 0.0 else ((err > lo) & (err <= hi))
        hist.append(round(100.0 * float(sel.mean()), 2))
    out = {"crops": n,
           "top1_within_pct": {f"{r:g}m": round(100.0 * float((err <= r).mean()), 2) for r in RADII},
           "top1_error_hist_pct": hist,
           "top1_error_hist_edges_m": list(HIST_EDGES),
           "top1_error_m_median_p90": [round(float(np.median(fin)), 3), round(float(np.quantile(fin, 0.9)), 3)]
           if len(fin) else None,
           "frames_with_detection_pct": round(100.0 * float((cnt > 0).mean()), 2),
           "detections_per_frame": round(float(cnt.mean()), 3),
           "false_positives_per_frame": round(float(n_false_positive) / n, 3),
           "false_positive_radius_m": float(fp_radius)}
    gates = {}
    for g in CONF_GATES:
        sel = conf >= g
        gates[f"conf>={g:g}"] = [round(100.0 * float(sel.mean()), 2),
                                 round(100.0 * float((err[sel] <= 0.5).mean()), 2) if sel.any() else None,
                                 round(100.0 * float((err[sel] <= 1.0).mean()), 2) if sel.any() else None]
    out["conf_gate_coverage_pct_prec0.5m_prec1m"] = gates
    return out
