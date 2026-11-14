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
