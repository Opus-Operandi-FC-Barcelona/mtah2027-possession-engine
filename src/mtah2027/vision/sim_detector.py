"""Simulated ball detector with measured characteristics (Issue #63).

The only thing taken from the real video is a set of aggregates (the cumulative distribution of the distance
between the top-confidence candidate and the tracked ball, and the share of frames without any detection).
:class:`DetectorProfile` turns them into a mixture over distance bins: a draw is "within 1 m", "1-2 m",
"2-5 m", "far" (a false positive, 5-``far_max_m`` metres from the truth) or "none".  Position error inside a
bin is uniform in radius and uniform in angle.  Draws are paired across detection-rate scenarios: a frame has
one gate uniform and one category uniform per seed, so a lower rate only removes frames.

Pure NumPy; no video, no weights, no per-frame data of any real match.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DetectorProfile:
    """Top-confidence candidate distance CDF (fractions of frames) and the no-detection share."""

    name: str
    within_1: float
    within_2: float
    within_5: float
    none: float = 0.0
    far_max_m: float = 30.0

    def probabilities(self) -> np.ndarray:
        """Probabilities of ``[<=1 m, 1-2 m, 2-5 m, far, none]`` (sum 1)."""
        if not 0.0 <= self.within_1 <= self.within_2 <= self.within_5 <= 1.0 - self.none:
            raise ValueError("CDF must be non-decreasing and leave room for the no-detection share")
        far = 1.0 - self.none - self.within_5
        return np.array([self.within_1, self.within_2 - self.within_1, self.within_5 - self.within_2, far,
                         self.none])


SV_BALL640 = DetectorProfile("sv_ball640", 0.410, 0.597, 0.757, none=0.05)
MOTION = DetectorProfile("motion", 0.051, 0.104, 0.216, none=0.0)

_BINS = ((0.0, 1.0), (1.0, 2.0), (2.0, 5.0))


def simulate_detections(truth_xy: np.ndarray, profile: DetectorProfile, *, rate: float, seed: int,
                        gate: np.ndarray | None = None, cat: np.ndarray | None = None,
                        angle: np.ndarray | None = None, radius_u: np.ndarray | None = None
                        ) -> tuple[np.ndarray, np.ndarray]:
    """One simulated top-confidence detection per row of ``truth_xy`` ``[M, 2]`` (metres).

    ``rate`` in [0, 1] scales the chance that the detector is asked at all and answers (1.0 = the measured
    profile, which already includes its own no-detection share).  Returns ``(returned[M] bool, xy[M, 2])``;
    rows not returned carry NaN.  The uniforms are drawn from ``seed`` (``default_rng``) unless given, so the
    same seed gives nested sets of returned frames for increasing ``rate``.
    """
    truth = np.asarray(truth_xy, dtype=np.float64).reshape(-1, 2)
    m = len(truth)
    rng = np.random.default_rng(seed)
    gate = rng.random(m) if gate is None else gate
    cat_u = rng.random(m) if cat is None else cat
    angle_u = rng.random(m) if angle is None else angle
    radius_u = rng.random(m) if radius_u is None else radius_u
    cum = np.cumsum(profile.probabilities())
    category = np.minimum(np.searchsorted(cum, cat_u, side="right"), 4)
    returned = (gate < rate) & (category < 4)
    lo = np.array([b[0] for b in _BINS] + [5.0])
    hi = np.array([b[1] for b in _BINS] + [float(profile.far_max_m)])
    r = lo[np.minimum(category, 3)] + radius_u * (hi[np.minimum(category, 3)] - lo[np.minimum(category, 3)])
    th = 2.0 * np.pi * angle_u
    xy = truth + np.stack([r * np.cos(th), r * np.sin(th)], axis=1)
    xy[~returned] = np.nan
    return returned, xy


@dataclass(frozen=True)
class TemporalModel:
    """Temporal structure of the simulated detector over consecutive planned crops (Issue #66).

    The per-crop marginals are those of the :class:`DetectorProfile` (a Markov chain over *good* = top
    candidate within 5 m of the ball, *far* = false positive and *none*, with the profile's stationary
    distribution).  ``persist_good`` / ``persist_far`` / ``persist_none`` are the chances the state of the
    previous crop (at most ``link_frames`` earlier) is kept.  A persistent false positive is a moving object
    (a boot, a line mark): it keeps its identity and drifts ``far_speed_mps`` m/s (Gaussian steps).  Position
    noise of good hits is drawn independently per crop from the profile bins.  These values are assumptions,
    not measurements: the real video was only scored per frame.
    """

    persist_good: float = 0.6
    persist_far: float = 0.2
    persist_none: float = 0.3
    far_speed_mps: float = 3.0
    link_frames: int = 10


PERSIST_FP = TemporalModel(persist_far=0.6)      # sensitivity variant: false positives are more persistent


def simulate_detections_temporal(truth_xy: np.ndarray, frames: np.ndarray, profile: DetectorProfile,
                                 model: TemporalModel, *, rate: float, seed: int, fps: float = 25.0
                                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Like :func:`simulate_detections` but with temporal structure over the sorted crop ``frames``.

    Returns ``(returned[M], xy[M, 2], is_good[M])`` (``is_good``: the draw was a true-ball hit, error <= 5 m).
    The same marginals per crop; ``rate`` removes crops independently (nested per seed) after the chain.
    """
    truth = np.asarray(truth_xy, dtype=np.float64).reshape(-1, 2)
    fr = np.asarray(frames, dtype=np.int64)
    m = len(truth)
    rng = np.random.default_rng(seed)
    gate, u_state, u_keep = rng.random(m), rng.random(m), rng.random(m)
    u_cat, u_ang, u_rad = rng.random(m), rng.random(m), rng.random(m)
    step = rng.standard_normal((m, 2))
    prob = profile.probabilities()                        # [<=1, 1-2, 2-5, far, none]
    pi = np.array([prob[:3].sum(), prob[3], prob[4]])     # good, far, none
    persist = np.array([model.persist_good, model.persist_far, model.persist_none])
    w = pi * (1.0 - persist)         # redraw weights that keep ``pi`` the stationary distribution
    cum_w = np.cumsum(w / w.sum())
    cum_pi = np.cumsum(pi)
    state = np.zeros(m, dtype=np.int64)
    far_xy = np.full((m, 2), np.nan)
    th = 2.0 * np.pi * u_ang
    for k in range(m):
        prev = state[k - 1] if k else -1
        linked = k > 0 and 0 < fr[k] - fr[k - 1] <= model.link_frames
        kept = linked and u_keep[k] < persist[prev]
        if kept:
            state[k] = prev
        else:
            state[k] = min(int(np.searchsorted(cum_w if linked else cum_pi, u_state[k], side="right")), 2)
        if state[k] == 1:
            if kept:
                dt = (fr[k] - fr[k - 1]) / fps
                far_xy[k] = far_xy[k - 1] + step[k] * model.far_speed_mps * dt
            else:
                r = 5.0 + u_rad[k] * (float(profile.far_max_m) - 5.0)
                far_xy[k] = truth[k] + r * np.array([np.cos(th[k]), np.sin(th[k])])
    good = state == 0
    sub = np.cumsum(prob[:3] / prob[:3].sum())
    cat = np.minimum(np.searchsorted(sub, u_cat, side="right"), 2)
    lo = np.array([b[0] for b in _BINS])
    hi = np.array([b[1] for b in _BINS])
    r = lo[cat] + u_rad * (hi[cat] - lo[cat])
    xy = np.where(good[:, None], truth + np.stack([r * np.cos(th), r * np.sin(th)], axis=1), far_xy)
    returned = (gate < rate) & (state < 2)
    xy[~returned] = np.nan
    return returned, xy, good & returned
