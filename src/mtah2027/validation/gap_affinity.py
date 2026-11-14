"""Label-free estimate of the hidden-ball <-> dead-ball affinity of an ATD (Issue #48).

Question: which share of the frames in which the ball is *not seen* is dead ball (``D``)?  The competition ATD
has 41 % hidden ball and nobody may label it, but the *players* are tracked in those frames and a dead ball
looks different from live play (collective speed, spread, stationary players).  The estimator therefore uses
**ball-free** features of the players only:

1. frame features (:func:`player_features`): number of tracked players, mean / 80th percentile speed, shares of
   slow and fast players, spread, and the same speeds averaged over 10 s; positions outside the pitch margins
   are discarded exactly as in the engine; the number of tracked players is a feature (stratification: tracking
   dropout mimics slow play, so the classifier learns the speed distribution *given* the player count);
2. a classifier ``P(D | features)`` trained on other matches with known state (:class:`DeadClassifier`;
   gradient boosting, the class-conditional histograms of the features in a discriminative form);
3. prior-shift EM (Saerens, Latinne & Decaestecker 2002; :func:`em_prior`): with the class-conditional
   ``p(x | class)`` assumed equal in reference and target, the prior ``P(D)`` of the target subset (hidden
   frames of gaps, or visible frames) is the fixed point of the EM on the classifier posteriors;
4. :func:`odds_ratio` combines the two estimates: ``OR = odds(D | hidden) / odds(D | visible)``.

Ball statistics at the borders of the gaps are described by :func:`boundary_features` (descriptive, aggregate).
Nothing here reads labels of the target; labels are only used for the reference matches.
"""
# ruff: noqa: E501
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter1d

PITCH = (105.3, 68.0)
MARGIN = 2.0
LOW_PLAYERS = 19.5      # (smoothed) tracked-player count below which a frame belongs to the "<20 players" stratum
FEATURE_NAMES = ("n_tracked", "mean_speed", "p80_speed", "frac_slow", "frac_fast", "sd_x", "sd_y",
                 "mean_speed_10s", "frac_slow_10s")


def _smooth(x: np.ndarray, frames: int) -> np.ndarray:
    return uniform_filter1d(x.astype(np.float64), size=max(int(frames), 1), mode="nearest")


def player_features(players_xy: np.ndarray, fps: float, pitch: tuple = PITCH) -> np.ndarray:
    """[N, 9] ball-free frame features (see module docstring); windows are expressed in seconds."""
    p = np.array(players_xy, dtype=np.float32, copy=True)
    out_of = ((p[..., 0] < -MARGIN) | (p[..., 0] > pitch[0] + MARGIN)
              | (p[..., 1] < -MARGIN) | (p[..., 1] > pitch[1] + MARGIN))
    p[out_of] = np.nan
    n = len(p)
    h = max(int(round(0.4 * fps)), 1)
    v = np.full(p.shape[:2], np.nan, dtype=np.float32)
    d = p[2 * h:] - p[:-2 * h]
    v[h:n - h] = np.hypot(d[..., 0], d[..., 1]) / np.float32(2 * h / fps)
    v[:h], v[n - h:] = v[h:h + 1], v[n - h - 1:n - h]
    v = np.where(v > 12.0, np.nan, v)                         # tracking glitches
    ok = np.isfinite(v)
    cnt = ok.sum(axis=1)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.where(cnt > 0, np.nansum(v, axis=1) / np.maximum(cnt, 1), np.nan)
        srt = np.sort(np.where(ok, v, np.nan), axis=1)             # NaN last
        k = np.clip(np.round(0.8 * (cnt - 1)).astype(int), 0, v.shape[1] - 1)
        p80 = np.where(cnt > 0, np.take_along_axis(srt, k[:, None], axis=1)[:, 0], np.nan)
        slow = np.where(cnt > 0, (ok & (v < 0.7)).sum(axis=1) / np.maximum(cnt, 1), np.nan)
        fast = np.where(cnt > 0, (ok & (v > 3.0)).sum(axis=1) / np.maximum(cnt, 1), np.nan)
        sdx = np.nanstd(p[..., 0], axis=1)
        sdy = np.nanstd(p[..., 1], axis=1)
    f = np.stack([np.isfinite(p[..., 0]).sum(axis=1), mean, p80, slow, fast, sdx, sdy], axis=1).astype(np.float64)
    for j in range(1, 7):                                       # fill empty frames, then smooth 2 s
        col = f[:, j]
        bad = ~np.isfinite(col)
        if bad.all():
            col[:] = 0.0
        elif bad.any():
            col[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), col[~bad])
        f[:, j] = _smooth(col, 2.0 * fps)
    ms10, sl10 = _smooth(f[:, 1], 10.0 * fps), _smooth(f[:, 3], 10.0 * fps)
    return np.column_stack([f, ms10, sl10])


def gap_masks(ball_xy: np.ndarray, graded: np.ndarray, fps: float, min_gap_s: float = 1.0):
    """``(interior, visible)``: hidden frames inside ball gaps of at least ``min_gap_s`` (graded frames only),
    and graded frames with a visible ball."""
    missing = ~np.isfinite(np.asarray(ball_xy)).all(axis=1)
    interior = np.zeros(len(missing), bool)
    d = np.diff(np.concatenate([[0], (missing & graded).astype(np.int8), [0]]))
    for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        if (b - a) >= min_gap_s * fps:
            interior[a:b] = True
    return interior, graded & ~missing


@dataclass
class DeadClassifier:
    model: object
    prior: float                  # P(D) of the (graded) reference frames
    prior_low: float = float("nan")    # ... of those with fewer than 20 tracked players (stratum "low")
    prior_high: float = float("nan")   # ... with 20 or more

    def proba(self, feats: np.ndarray) -> np.ndarray:
        return np.clip(self.model.predict_proba(np.nan_to_num(feats))[:, 1], 1e-4, 1 - 1e-4)


def fit_dead_classifier(feats: np.ndarray, is_dead: np.ndarray, *, seed: int = 0, stride: int = 5) -> DeadClassifier:
    """Binary D (1) vs alive (0) classifier on the graded reference frames (``stride`` thinning)."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    idx = np.arange(0, len(feats), stride)
    x, y = np.nan_to_num(feats[idx]), is_dead[idx].astype(int)
    model = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_leaf_nodes=15,
                                           min_samples_leaf=80, l2_regularization=1.0, random_state=seed)
    model.fit(x, y)
    low = x[:, 0] < LOW_PLAYERS
    return DeadClassifier(model=model, prior=float(y.mean()),
                          prior_low=float(y[low].mean()) if low.sum() > 50 else float(y.mean()),
                          prior_high=float(y[~low].mean()) if (~low).sum() > 50 else float(y.mean()))


def em_prior(post: np.ndarray, train_prior: float, iters: int = 200, tol: float = 1e-7) -> float:
    """Saerens et al. (2002): prior of the positive class of a target set from the posteriors ``post`` of a
    classifier trained under ``train_prior``."""
    post = np.asarray(post, dtype=np.float64)
    train_prior = min(max(float(train_prior), 1e-3), 1 - 1e-3)
    if len(post) == 0:
        return float("nan")
    pi = float(post.mean())
    for _ in range(iters):
        a = pi / train_prior * post
        b = (1 - pi) / (1 - train_prior) * (1 - post)
        new = float((a / (a + b)).mean())
        if abs(new - pi) < tol:
            pi = new
            break
        pi = new
    return pi


def odds(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return p / (1 - p)


def odds_ratio(p_hidden: float, p_visible: float) -> float:
    return odds(p_hidden) / odds(p_visible)


def _stratified_prior(clf: DeadClassifier, feats: np.ndarray, mask: np.ndarray) -> tuple[float, dict]:
    """EM per player-count stratum (<20 / >=20 tracked players) with the stratum's own reference prior, combined
    by frame counts.  Dropout slows the visible players down; conditioning on the count (feature + stratum
    prior) keeps missing players from reading as dead ball."""
    f = feats[mask]
    low = f[:, 0] < LOW_PLAYERS
    parts, total, acc = {}, 0, 0.0
    for name, sel, prior in (("low", low, clf.prior_low), ("high", ~low, clf.prior_high)):
        if sel.sum() < 30:
            parts[name] = {"n": int(sel.sum()), "d_share": float("nan")}
            continue
        lam = em_prior(clf.proba(f[sel]), prior)
        parts[name] = {"n": int(sel.sum()), "d_share": lam}
        acc += lam * sel.sum()
        total += int(sel.sum())
    return (acc / total if total else float("nan")), parts


def estimate(clf: DeadClassifier, feats: np.ndarray, interior: np.ndarray, visible: np.ndarray,
             stratified: bool = True) -> dict:
    """Estimated D share of the hidden-gap frames and of the visible frames, and the odds ratio.  With
    ``stratified`` (default) the EM runs separately for <20 and >=20 tracked players (reported in ``strata``)."""
    if stratified:
        lam_h, sh = _stratified_prior(clf, feats, interior)
        lam_v, sv = _stratified_prior(clf, feats, visible)
        strata = {"hidden": sh, "visible": sv}
    else:
        lam_h, lam_v = em_prior(clf.proba(feats[interior]), clf.prior), em_prior(clf.proba(feats[visible]), clf.prior)
        strata = {}
    return {"d_share_hidden": lam_h, "d_share_visible": lam_v,
            "odds_ratio": odds_ratio(lam_h, lam_v) if np.isfinite(lam_h) and np.isfinite(lam_v) else float("nan"),
            "n_hidden": int(interior.sum()), "n_visible": int(visible.sum()), "strata": strata}


def true_shares(codes: np.ndarray, interior: np.ndarray, visible: np.ndarray) -> dict:
    d = codes == 2
    lh = float(d[interior].mean()) if interior.any() else float("nan")
    lv = float(d[visible].mean()) if visible.any() else float("nan")
    return {"d_share_hidden": lh, "d_share_visible": lv,
            "odds_ratio": odds_ratio(lh, lv) if np.isfinite(lh) and np.isfinite(lv) else float("nan")}


def boundary_features(ball_xy: np.ndarray, players_xy: np.ndarray, graded: np.ndarray, fps: float,
                      pitch: tuple = PITCH, min_gap_s: float = 1.0) -> dict:
    """Aggregate description of the ball at the last-seen / first-seen frames of the gaps: share of gaps whose
    border position is within 1.5 m of (or outside) a line, share with a slow ball (< 1 m/s) and median distance
    to the nearest tracked player."""
    ball = np.asarray(ball_xy, dtype=np.float64)
    missing = ~np.isfinite(ball).all(axis=1)
    d = np.diff(np.concatenate([[0], (missing & graded).astype(np.int8), [0]]))
    rows = {"last": [], "first": []}
    k = max(int(round(0.2 * fps)), 1)
    for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        if (b - a) < min_gap_s * fps:
            continue
        for name, t in (("last", a - 1), ("first", b)):
            if t < k or t >= len(ball) - k or missing[t] or missing[t - k] and name == "last" \
                    or missing[min(t + k, len(ball) - 1)] and name == "first":
                continue
            t2 = t - k if name == "last" else t + k
            speed = float(np.hypot(*(ball[t] - ball[t2])) / (k / fps))
            x, y = ball[t]
            line = min(x, pitch[0] - x, y, pitch[1] - y)
            pd = np.hypot(players_xy[t, :, 0] - x, players_xy[t, :, 1] - y)
            near = float(np.nanmin(pd)) if np.isfinite(pd).any() else np.nan
            rows[name].append((line, speed, near))
    out = {}
    for name, r in rows.items():
        a = np.asarray(r, dtype=np.float64).reshape(-1, 3)
        out[name] = {"n": int(len(a)),
                     "near_line_1p5m": float((a[:, 0] < 1.5).mean()) if len(a) else float("nan"),
                     "slow_ball": float((a[:, 1] < 1.0).mean()) if len(a) else float("nan"),
                     "median_nearest_player_m": float(np.nanmedian(a[:, 2])) if len(a) else float("nan")}
    return out
