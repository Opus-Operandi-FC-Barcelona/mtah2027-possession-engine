"""Stratified diagnostics by number of tracked players (Issue #42).

On the competition ATD 14-15 % of the frames have fewer than 20 tracked players.  The stratification splits
the graded frames of an (external, degraded) match by the number of players with a position in the tracking
that is fed to the pipeline: ``lt16`` (< 16), ``lt20`` (< 20, contains ``lt16``) and ``ge20`` (>= 20).  All
counters are additive, so matches and seeds are pooled exactly by summing; only aggregates are ever produced.

Per stratum :func:`summarize` reports the frame-level macro F1 of H / A / D (official definition, within
the stratum), the true and predicted share of ``D`` (also with the ball hidden / visible), the H/A accuracy on
the truly alive frames and the missed ``H <-> A`` turnovers (matched with ``tol`` frames on the destination
label, like the official transition scorer).
"""
from __future__ import annotations

import numpy as np

from mtah2027.validation import eval_bridge as eb

pm = eb.pm
STRATA = ("lt16", "lt20", "ge20")
H, A, D, X = 0, 1, 2, 3


def tracked_counts(players_xy: np.ndarray) -> np.ndarray:
    """Players with a position per frame (``players_xy[N, S, 2]``)."""
    return np.isfinite(np.asarray(players_xy)[..., 0]).sum(axis=1).astype(np.int16)


def stratum_masks(n_tracked: np.ndarray) -> dict[str, np.ndarray]:
    n = np.asarray(n_tracked)
    return {"lt16": n < 16, "lt20": n < 20, "ge20": n >= 20}


def missed_turnover_frames(gt: np.ndarray, pred: np.ndarray, tol: int) -> tuple[list[int], list[int]]:
    """Frames of the true ``H <-> A`` turnovers and of those the prediction misses."""
    runs = pm.valid_runs(gt)
    filled = pm.fill_abstentions(pred, runs)
    gt_tr, pr_tr = pm.transitions(gt, runs), pm.transitions(filled, runs)
    _, missed, _ = pm.match_transitions(gt_tr, pr_tr, tol, "to")
    miss = {int(f) for f, _, _ in missed}
    total = [int(f) for f, a, b in gt_tr if a in (H, A) and b in (H, A) and a != b]
    return total, [f for f in total if f in miss]


def stratum_counts(gt: np.ndarray, pred: np.ndarray, n_tracked: np.ndarray, visible: np.ndarray,
                   tol: int = 25) -> dict[str, dict[str, int]]:
    """Additive counters of every stratum (graded frames only; abstentions in graded frames are filled)."""
    gt = np.asarray(gt)
    runs = pm.valid_runs(gt)
    filled = pm.fill_abstentions(np.asarray(pred), runs)
    graded = gt != X
    vis = np.asarray(visible, dtype=bool)
    tot, mis = missed_turnover_frames(gt, pred, tol)
    tot, mis = np.asarray(tot, dtype=np.int64), np.asarray(mis, dtype=np.int64)
    out: dict[str, dict[str, int]] = {}
    for name, m in stratum_masks(n_tracked).items():
        sel = m & graded
        g, p = gt[sel], filled[sel]
        c = {"frames": int(sel.sum()), "gt_d": int((g == D).sum()), "pred_d": int((p == D).sum()),
             "alive": int(np.isin(g, (H, A)).sum()),
             "alive_ok": int(((g == p) & np.isin(g, (H, A))).sum()),
             "hid_frames": int((sel & ~vis).sum()), "hid_gt_d": int((g == D)[~vis[sel]].sum()),
             "hid_pred_d": int((p == D)[~vis[sel]].sum()),
             "vis_frames": int((sel & vis).sum()), "vis_gt_d": int((g == D)[vis[sel]].sum()),
             "vis_pred_d": int((p == D)[vis[sel]].sum()),
             "turnovers": int(m[tot].sum()) if len(tot) else 0,
             "turnovers_missed": int(m[mis].sum()) if len(mis) else 0}
        for k, cls in (("h", H), ("a", A), ("d", D)):
            c[f"tp_{k}"] = int(((g == cls) & (p == cls)).sum())
            c[f"fp_{k}"] = int(((g != cls) & (p == cls)).sum())
            c[f"fn_{k}"] = int(((g == cls) & (p != cls)).sum())
        out[name] = c
    return out


def pool(items: list[dict[str, dict[str, int]]]) -> dict[str, dict[str, int]]:
    """Sum the counters of several matches / seeds."""
    out: dict[str, dict[str, int]] = {}
    for it in items:
        for name, c in it.items():
            dst = out.setdefault(name, {})
            for k, v in c.items():
                dst[k] = dst.get(k, 0) + int(v)
    return out


def _f1(tp: int, fp: int, fn: int) -> float:
    return 2.0 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def summarize(counts: dict[str, dict[str, int]]) -> dict[str, dict[str, float]]:
    """Readable aggregates of pooled counters (percentages in 0-100)."""
    def pct(a, b):
        return 100.0 * a / b if b else float("nan")

    res = {}
    for name, c in counts.items():
        f1s = [_f1(c[f"tp_{k}"], c[f"fp_{k}"], c[f"fn_{k}"]) for k in "had"]
        res[name] = {
            "frames": c["frames"],
            "macro_f1": float(np.mean(f1s)),
            "true_d_pct": pct(c["gt_d"], c["frames"]), "pred_d_pct": pct(c["pred_d"], c["frames"]),
            "true_d_pct_hidden": pct(c["hid_gt_d"], c["hid_frames"]),
            "pred_d_pct_hidden": pct(c["hid_pred_d"], c["hid_frames"]),
            "true_d_pct_visible": pct(c["vis_gt_d"], c["vis_frames"]),
            "pred_d_pct_visible": pct(c["vis_pred_d"], c["vis_frames"]),
            "ha_acc_pct": pct(c["alive_ok"], c["alive"]),
            "turnovers": c["turnovers"], "turnovers_missed": c["turnovers_missed"],
            "turnovers_missed_pct": pct(c["turnovers_missed"], c["turnovers"])}
    return res


def format_table(summary: dict[str, dict[str, float]], total_frames: int | None = None) -> list[str]:
    """Text rows for the console (one line per stratum)."""
    head = (f"{'stratum':<8}{'frames%':>8}{'macroF1':>9}{'trueD%':>8}{'predD%':>8}{'predD%hid':>10}"
            f"{'predD%vis':>10}{'H/Aacc%':>9}{'turn':>6}{'missed%':>8}")
    rows = [head]
    tot = total_frames or sum(summary.get(k, {}).get("frames", 0) for k in ("ge20", "lt20"))
    for name in STRATA:
        if name not in summary:
            continue
        s = summary[name]
        share = 100.0 * s["frames"] / max(tot, 1)
        rows.append(f"{name:<8}{share:8.1f}{s['macro_f1']:9.4f}{s['true_d_pct']:8.1f}"
                    f"{s['pred_d_pct']:8.1f}{s['pred_d_pct_hidden']:10.1f}{s['pred_d_pct_visible']:10.1f}"
                    f"{s['ha_acc_pct']:9.1f}{s['turnovers']:6d}{s['turnovers_missed_pct']:8.1f}")
    return rows
