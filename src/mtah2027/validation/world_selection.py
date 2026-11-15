# ruff: noqa: E501
"""Issue #71: label-free world selection by adversarial validation (external data and aggregates only).

Question: in the frames without a ball, which validation world does the real tracking resemble: the clean IDSSE tracking
degraded by our degrader (``dead_affinity`` 33 or 1) or the real occlusions of PFF?  Nothing here uses a label or the ball
position: :func:`hidden_ball_features` describes a frame by collective physics, tracked counts, gap structure and
line advance only.  The rest are small numpy / scikit-learn helpers:

* :func:`cross_fit_probabilities`: out-of-fold P(real | features) of one binary domain classifier (grouped folds);
* :func:`auc_score`: rank AUC;
* :func:`fit_mixture`: EM weights ``pi`` such that ``sum_w pi_w p_w(x)`` approximates the real density (classifier-based
  density ratios ``p_real / p_w = P / (1 - P)`` under balanced classes);
* :func:`effective_sample_size`, :func:`weighted_metric`: importance weights and the weighted validation metric.
"""
from __future__ import annotations

import numpy as np

from mtah2027.features._utils import speed_from_xy

FEATURE_NAMES = (
    "n_tracked", "n_tracked_h", "n_tracked_a",
    "speed_med_1s", "speed_med_raw", "speed_mean", "speed_p80", "slow_frac_05", "slow_frac_10",
    "speed_mean_h", "speed_mean_a",
    "centroid_dist", "std_x_h", "std_y_h", "std_x_a", "std_y_a", "std_x_all", "std_y_all",
    "area_h", "area_a", "area_ratio",
    "adv_h", "adv_a", "mean_speed_all_centroid",
    "coh_h", "coh_a",
    "central_x", "central_y",
    "gap_len_s", "since_seen_s", "until_seen_s", "gap_pos",
)
GAP_CAP_S = 180.0


def _team_stats(xy: np.ndarray, v: np.ndarray, speed: np.ndarray):
    """Per-frame spread, spectral area, mean x velocity and coherence of one team (``xy[N, k, 2]``)."""
    ok = np.isfinite(xy).all(-1)
    n = ok.sum(1)
    with np.errstate(all="ignore"):
        cx = np.nanmean(np.where(ok, xy[..., 0], np.nan), axis=1)
        cy = np.nanmean(np.where(ok, xy[..., 1], np.nan), axis=1)
        dx = np.where(ok, xy[..., 0] - cx[:, None], 0.0)
        dy = np.where(ok, xy[..., 1] - cy[:, None], 0.0)
        den = np.maximum(n - 1, 1)
        sxx, syy, sxy = (dx * dx).sum(1) / den, (dy * dy).sum(1) / den, (dx * dy).sum(1) / den
        det = np.maximum(sxx * syy - sxy * sxy, 0.0)
        area = np.pi * np.sqrt(det)                      # pi * sqrt(l1 * l2), l1 * l2 = det of the 2x2 covariance
        vok = np.isfinite(v).all(-1) & ok
        vn = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-3)
        sv = np.where(vok[..., None], vn, 0.0).sum(1)
        coh = np.linalg.norm(sv, axis=-1) / np.maximum(vok.sum(1), 1)
        adv = np.nanmean(np.where(vok, v[..., 0], np.nan), axis=1)
        spd = np.nanmean(np.where(ok, speed, np.nan), axis=1)
    bad = n < 3
    for a in (sxx, syy, area, coh):
        a[bad] = np.nan
    return {"n": n, "cx": cx, "cy": cy, "sx": np.sqrt(sxx), "sy": np.sqrt(syy), "area": area, "coh": coh,
            "adv": adv, "spd": spd}


def gap_structure(visible: np.ndarray, fps: float):
    """``(gap_len_s, since_seen_s, until_seen_s)`` per frame of the run of hidden frames it belongs to (0 where visible).

    Distances are capped at ``GAP_CAP_S``; a gap cut by the array edge keeps the cap on the missing side."""
    n = len(visible)
    hid = ~visible
    d = np.diff(np.concatenate([[0], hid.astype(np.int8), [0]]))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)       # [start, end)
    length = np.zeros(n)
    since = np.zeros(n)
    until = np.zeros(n)
    cap = GAP_CAP_S * fps
    for s, e in zip(starts, ends):
        idx = np.arange(s, e)
        length[s:e] = e - s
        since[s:e] = (idx - s + 1) if s > 0 else cap
        until[s:e] = (e - idx) if e < n else cap
    return np.minimum(length / fps, GAP_CAP_S), np.minimum(since / fps, GAP_CAP_S), np.minimum(until / fps, GAP_CAP_S)


def hidden_ball_features(players_xy: np.ndarray, team_of_slot: np.ndarray, ball_xy: np.ndarray, fps: float,
                         pitch=(105.3, 68.0), graded: np.ndarray | None = None, stride: int = 5,
                         offset: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Ball-free feature matrix ``[M, len(FEATURE_NAMES)]`` of the hidden-ball graded frames (every ``stride``-th).

    Returns ``(features, frame_index)``.  ``ball_xy`` is only used to decide which frames have no ball and how long the
    gap is; its coordinates never enter a feature."""
    players = np.asarray(players_xy, dtype=np.float32)
    team = np.asarray(team_of_slot)
    n = len(players)
    visible = np.isfinite(np.asarray(ball_xy)).all(axis=1)
    graded = np.ones(n, bool) if graded is None else np.asarray(graded, bool)
    sel = np.flatnonzero(~visible & graded)
    sel = sel[(sel + offset) % max(stride, 1) == 0]
    if not len(sel):
        return np.zeros((0, len(FEATURE_NAMES))), sel
    ps_all = speed_from_xy(players, fps, 1)
    with np.errstate(all="ignore"):
        pm = np.nanmedian(np.where(ps_all > 0, ps_all, np.nan), axis=1)
    k = max(int(round(fps)), 1)
    pm_s = np.convolve(np.where(np.isfinite(pm), pm, 0.0), np.ones(k) / k, mode="same")
    # velocities over +-0.2 s for the direction features
    half = max(int(round(0.2 * fps)), 1)
    from mtah2027.features._utils import velocity_from_xy

    vel = velocity_from_xy(players, fps, half)
    glen, since, until = gap_structure(visible, fps)
    out = np.full((len(sel), len(FEATURE_NAMES)), np.nan)
    p, v, sp = players[sel].astype(np.float64), vel[sel].astype(np.float64), ps_all[sel].astype(np.float64)
    ok_all = np.isfinite(p).all(-1)
    st = {}
    for t, name in ((0, "h"), (1, "a")):
        m = team == t
        st[name] = _team_stats(p[:, m], v[:, m], sp[:, m])
    with np.errstate(all="ignore"):
        spd = np.where(ok_all & np.isfinite(sp), sp, np.nan)
        px = np.where(ok_all, p[..., 0], np.nan)
        py = np.where(ok_all, p[..., 1], np.nan)
        cxa, cya = np.nanmean(px, axis=1), np.nanmean(py, axis=1)
        vall = np.where(ok_all[..., None] & np.isfinite(v).all(-1)[..., None], v, np.nan)
        cols = {
            "n_tracked": ok_all.sum(1), "n_tracked_h": st["h"]["n"], "n_tracked_a": st["a"]["n"],
            "speed_med_1s": pm_s[sel], "speed_med_raw": np.nanmedian(spd, axis=1), "speed_mean": np.nanmean(spd, axis=1),
            "speed_p80": np.nanpercentile(spd, 80, axis=1),
            "slow_frac_05": np.nanmean(np.where(np.isfinite(spd), spd < 0.5, np.nan), axis=1),
            "slow_frac_10": np.nanmean(np.where(np.isfinite(spd), spd < 1.0, np.nan), axis=1),
            "speed_mean_h": st["h"]["spd"], "speed_mean_a": st["a"]["spd"],
            "centroid_dist": np.hypot(st["h"]["cx"] - st["a"]["cx"], st["h"]["cy"] - st["a"]["cy"]),
            "std_x_h": st["h"]["sx"], "std_y_h": st["h"]["sy"], "std_x_a": st["a"]["sx"], "std_y_a": st["a"]["sy"],
            "std_x_all": np.nanstd(px, axis=1), "std_y_all": np.nanstd(py, axis=1),
            "area_h": st["h"]["area"], "area_a": st["a"]["area"],
            "area_ratio": np.log((st["h"]["area"] + 1.0) / (st["a"]["area"] + 1.0)),
            "adv_h": np.abs(st["h"]["adv"]), "adv_a": np.abs(st["a"]["adv"]),
            "mean_speed_all_centroid": np.linalg.norm(np.nanmean(vall, axis=1), axis=-1),
            "coh_h": st["h"]["coh"], "coh_a": st["a"]["coh"],
            "central_x": np.abs(cxa - pitch[0] / 2.0), "central_y": np.abs(cya - pitch[1] / 2.0),
            "gap_len_s": glen[sel], "since_seen_s": since[sel], "until_seen_s": until[sel],
            "gap_pos": since[sel] / np.maximum(glen[sel], 1.0 / fps),
        }
    for i, name in enumerate(FEATURE_NAMES):
        out[:, i] = cols[name]
    return out, sel


def two_sample_stats(a: np.ndarray, b: np.ndarray, names=FEATURE_NAMES) -> list[dict]:
    """Aggregate per-feature statistics of two samples: medians and the standardised mean difference (aggregates)."""
    rows = []
    for i, nm in enumerate(names):
        x, y = a[:, i], b[:, i]
        x, y = x[np.isfinite(x)], y[np.isfinite(y)]
        if not len(x) or not len(y):
            continue
        sd = np.sqrt(0.5 * (x.var() + y.var())) + 1e-9
        rows.append({"feature": nm, "median_a": float(np.median(x)), "median_b": float(np.median(y)),
                     "smd": float((x.mean() - y.mean()) / sd)})
    return rows


def auc_score(y: np.ndarray, score: np.ndarray, weight: np.ndarray | None = None) -> float:
    """Rank AUC (ties averaged) of ``score`` for the positive class ``y == 1``."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(score, dtype=np.float64)
    w = np.ones(len(y)) if weight is None else np.asarray(weight, dtype=np.float64)
    order = np.argsort(s, kind="mergesort")
    s, y, w = s[order], y[order], w[order]
    # average ranks over ties, weighted
    uniq_start = np.concatenate([[0], np.flatnonzero(np.diff(s) != 0) + 1])
    grp = np.zeros(len(s), np.int64)
    grp[uniq_start] = 1
    grp = np.cumsum(grp) - 1
    wsum = np.bincount(grp, weights=w)
    wpos = np.bincount(grp, weights=w * y)
    neg_total = float((w * ~y).sum())
    pos_total = float((w * y).sum())
    if neg_total <= 0 or pos_total <= 0:
        return float("nan")
    wneg = wsum - wpos
    # pairs (pos, neg) with neg below pos + half of ties
    neg_before = np.cumsum(wneg) - wneg
    return float((wpos * (neg_before + 0.5 * wneg)).sum() / (pos_total * neg_total))


def _classifier(seed: int, max_iter: int):
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=0.1, max_leaf_nodes=15, min_samples_leaf=40,
                                          l2_regularization=1.0, random_state=seed)


def cross_fit_probabilities(x: np.ndarray, y: np.ndarray, groups: np.ndarray, n_folds: int = 5, seed: int = 0,
                            max_iter: int = 150) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold ``P(y = 1 | x)`` of a gradient-boosting classifier with balanced class mass.

    ``groups`` (integer ids: match, or time block for the real match) are never split between folds.  Returns
    ``(p, importance)`` where ``importance`` is the mean held-out AUC drop when one feature is shuffled; NaN is native."""
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y).astype(np.int8)
    ug = np.unique(groups)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ug))
    fold_of_group = {int(g): int(i % n_folds) for i, g in zip(range(len(ug)), ug[perm])}
    fold = np.fromiter((fold_of_group[int(g)] for g in groups), np.int64, len(groups))
    p = np.full(len(y), np.nan)
    imp = np.zeros(x.shape[1])
    n_used = 0
    for f in range(n_folds):
        tr, te = fold != f, fold == f
        if not te.any() or len(np.unique(y[tr])) < 2:
            continue
        w = np.where(y[tr] == 1, 0.5 / max((y[tr] == 1).sum(), 1), 0.5 / max((y[tr] == 0).sum(), 1))
        w = w * tr.sum()
        clf = _classifier(seed, max_iter)
        clf.fit(x[tr], y[tr], sample_weight=w)
        p[te] = clf.predict_proba(x[te])[:, 1]
        imp += _permutation_importance(clf, x[te], y[te], seed + f)
        n_used += 1
    return p, imp / max(n_used, 1)


def _permutation_importance(clf, x: np.ndarray, y: np.ndarray, seed: int, max_rows: int = 6000) -> np.ndarray:
    """Drop in AUC when one feature is shuffled (held-out rows), on a subsample."""
    rng = np.random.default_rng(seed)
    if len(x) > max_rows:
        idx = rng.choice(len(x), max_rows, replace=False)
        x, y = x[idx], y[idx]
    if len(np.unique(y)) < 2:
        return np.zeros(x.shape[1])
    base = auc_score(y, clf.predict_proba(x)[:, 1])
    out = np.zeros(x.shape[1])
    for j in range(x.shape[1]):
        xs = x.copy()
        xs[:, j] = xs[rng.permutation(len(xs)), j]
        out[j] = base - auc_score(y, clf.predict_proba(xs)[:, 1])
    return out


def density_ratio(p_real: np.ndarray, clip: float = 0.01) -> np.ndarray:
    """``p_real(x) / p_world(x)`` from a balanced-class classifier probability, with ``P`` clipped to ``[clip, 1-clip]``."""
    p = np.clip(np.asarray(p_real, dtype=np.float64), clip, 1.0 - clip)
    return p / (1.0 - p)


def fit_mixture(ratios: np.ndarray, n_iter: int = 500, tol: float = 1e-10) -> np.ndarray:
    """EM weights ``pi`` over worlds for the real frames.

    ``ratios[i, w] = p_real(x_i) / p_w(x_i)`` evaluated on the real frames.  The mixture density is
    ``sum_w pi_w p_w``, so ``p_mix / p_real = sum_w pi_w / ratios[i, w]``; maximising the mean log of that is the
    minimum of KL(p_real || p_mix)."""
    q = 1.0 / np.maximum(np.asarray(ratios, dtype=np.float64), 1e-12)
    k = q.shape[1]
    pi = np.full(k, 1.0 / k)
    prev = -np.inf
    for _ in range(n_iter):
        mix = q @ pi
        gam = q * pi[None, :] / np.maximum(mix[:, None], 1e-300)
        pi = gam.mean(0)
        ll = float(np.log(np.maximum(q @ pi, 1e-300)).mean())
        if abs(ll - prev) < tol:
            break
        prev = ll
    return pi / pi.sum()


def effective_sample_size(w: np.ndarray) -> float:
    w = np.asarray(w, dtype=np.float64)
    s = w.sum()
    return float(s * s / max((w * w).sum(), 1e-300)) if s > 0 else 0.0


def match_weights(ratio: np.ndarray, match_ids: np.ndarray, cap_quantile: float = 0.99) -> dict:
    """Per-match weight = mean of the frame density ratios (capped at the ``cap_quantile``), normalised to sum 1."""
    r = np.asarray(ratio, dtype=np.float64)
    r = np.minimum(r, np.quantile(r, cap_quantile))
    ids = np.asarray(match_ids)
    names = list(dict.fromkeys(ids.tolist()))
    raw = np.array([r[ids == n].mean() for n in names])
    return dict(zip(names, (raw / raw.sum()).tolist()))


def weighted_metric(scores: dict, world_weights: dict, match_w: dict) -> float:
    """``sum_w pi_w sum_m v_m score[w][m]`` with ``v`` the within-world match weights (summing to 1 per world).

    ``scores[w]`` maps match -> score; a match without weight in ``match_w[w]`` is ignored and the rest re-normalised."""
    total = 0.0
    for w, pi in world_weights.items():
        sc = scores[w]
        names = [m for m in sc if m in match_w[w]]
        v = np.array([match_w[w][m] for m in names], dtype=np.float64)
        total += float(pi) * float((v / v.sum()) @ np.array([sc[m] for m in names], dtype=np.float64))
    return total / max(sum(world_weights.values()), 1e-12)
