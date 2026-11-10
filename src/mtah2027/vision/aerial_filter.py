"""Aerial-ball detector from the ball track alone (Issue #12): ``is_aerial[N]``.

A ball in the air cannot be played with the feet and its 2-D track is a poor proxy of the possession, so the
touches detected there are spurious.  Three independent cues, computed from the ball positions (metres, NaN
where missing) and OR-ed together:

1. **Apparent speed** -- ground-plane speed above ``speed_mps`` (35 m/s by default: a fast shot is ~30 m/s,
   so beyond that the 2-D point is the projection of a flying ball or a tracking glitch).  The speed is the
   distance between consecutive visible frames; a sample next to a hidden frame carries no speed.
2. **Reprojection jumps** (only when per-frame homographies are supplied) -- the positions are projected to
   pixels with ``H``; a displacement of more than ``jump_px`` pixels between two consecutive visible frames
   is not a ground-ball motion.
3. **Long straight flights** -- a maximal run of steps faster than ``flight_min_speed_mps`` along an almost
   straight line (heading change per frame below ``straight_deg``) whose path is longer than
   ``flight_min_length_m`` and whose speed does *not* decay like a ball rolling on grass (the least-squares
   slope of the speed is above ``-grass_decel_mps2``; rolling friction removes several m/s every second).
   The first and last ``edge_trim_frames`` of the run are not flagged: they contain the kick and the
   landing, which are genuine touch candidates.

An optional height channel ``ball_z`` (metres, from the ATD when it declares one) adds ``z > z_min_m``.
The speed / jump cues are grown by ``pad_frames`` on each side; flagged runs shorter than ``min_run_frames``
are dropped.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation

from mtah2027.config import FPS, PITCH, load_section, params_from_mapping
from mtah2027.features._utils import mask_runs
from mtah2027.io.homography_loader import project_pitch_to_pixels


@dataclass(frozen=True)
class AerialParams:
    speed_mps: float = 35.0              # apparent ground speed above which the ball is airborne
    jump_px: float = 90.0                # pixel displacement per frame above which the track jumps
    flight_min_speed_mps: float = 12.0   # a straight flight moves faster than this ...
    straight_deg: float = 8.0            # ... with a heading change per frame below this ...
    flight_min_length_m: float = 18.0    # ... over a path at least this long ...
    grass_decel_mps2: float = 1.5        # ... and a speed slope above -this (no rolling friction)
    edge_trim_frames: int = 3            # kick / landing frames of a flight that are not flagged
    pad_frames: int = 2                  # growth of the speed / jump cues
    min_run_frames: int = 2              # flagged runs shorter than this are dropped
    z_min_m: float = 1.5                 # ball height above which it is airborne (when a z channel exists)
    z_channels: tuple = ("z", "ball_z", "height", "ballz")   # candidate names of the ATD height channel

    @classmethod
    def from_config(cls, path=None) -> "AerialParams":
        return params_from_mapping(cls, load_section("aerial_filter", path), "aerial_filter")


def find_z_channel(ball_channels: dict | None, names, n: int) -> np.ndarray | None:
    """Height channel ``[n]`` (float, NaN where missing) of the ATD ball group, if one is declared."""
    if not ball_channels:
        return None
    lowered = {str(k).strip().lower(): k for k in ball_channels}
    for name in names:
        key = lowered.get(str(name).lower())
        if key is not None:
            z = np.asarray(ball_channels[key], dtype=np.float64)
            if z.shape == (n,) and np.isfinite(z).any():
                return z
    return None


def compute_is_aerial(ball_xy: np.ndarray, *, fps: float = FPS, ball_z: np.ndarray | None = None,
                      H: np.ndarray | None = None, pitch: tuple[float, float] = PITCH,
                      params: AerialParams | None = None) -> np.ndarray:
    """Boolean mask ``is_aerial[N]`` (see the module docstring).

    ``ball_xy[N, 2]`` in metres; ``H`` (optional) the per-frame ``[N, 3, 3]`` pitch -> pixel matrices.
    """
    p = params or AerialParams.from_config()
    ball = np.asarray(ball_xy, dtype=np.float64)
    n = len(ball)
    aerial = np.zeros(n, dtype=bool)
    visible = np.isfinite(ball).all(axis=1)
    if n < 3 or not visible.any():
        return aerial
    step = np.full_like(ball, np.nan)
    step[1:] = ball[1:] - ball[:-1]
    ok = np.isfinite(step).all(axis=1)
    dist = np.where(ok, np.hypot(step[:, 0], step[:, 1]), np.nan)
    speed = dist * fps

    # 1. apparent speed: the step ending at i and the one starting at i both involve frame i
    cue = np.nan_to_num(speed, nan=0.0) > p.speed_mps
    cue[:-1] |= cue[1:].copy()

    # 2. reprojection jumps
    if H is not None and np.shape(H) == (n, 3, 3):
        u, v = project_pitch_to_pixels(np.asarray(H, dtype=np.float64), ball[:, 0], ball[:, 1], pitch)
        with np.errstate(invalid="ignore"):
            jump = np.nan_to_num(np.hypot(np.diff(u, prepend=np.nan), np.diff(v, prepend=np.nan)),
                                 nan=0.0) > p.jump_px
        cue |= jump
        cue[:-1] |= jump[1:]
    if p.pad_frames > 0 and cue.any():
        cue = binary_dilation(cue, iterations=int(p.pad_frames))
    aerial |= cue & visible

    # 3. long straight flights without grass deceleration
    aerial |= _straight_flights(step, ok, dist, speed, fps, p)

    # optional height channel
    if ball_z is not None and np.shape(ball_z) == (n,):
        aerial |= np.nan_to_num(np.asarray(ball_z, dtype=np.float64), nan=0.0) > p.z_min_m

    if p.min_run_frames > 1 and aerial.any():
        for a, b in zip(*mask_runs(aerial)):
            if b - a < p.min_run_frames:
                aerial[a:b] = False
    return aerial


def _straight_flights(step, ok, dist, speed, fps, p: AerialParams) -> np.ndarray:
    n = len(step)
    out = np.zeros(n, dtype=bool)
    cross = np.full(n, np.nan)
    dot = np.full(n, np.nan)
    both = ok[1:] & ok[:-1]
    cross[1:][both] = (step[:-1, 0] * step[1:, 1] - step[:-1, 1] * step[1:, 0])[both]
    dot[1:][both] = (step[:-1, 0] * step[1:, 0] + step[:-1, 1] * step[1:, 1])[both]
    turn = np.degrees(np.abs(np.arctan2(cross, dot)))          # heading change from step i-1 to step i
    # step i continues a flight when it is fast and its heading differs little from step i-1
    cand = ok & (np.nan_to_num(speed, nan=0.0) >= p.flight_min_speed_mps)
    straight = np.nan_to_num(turn, nan=0.0) <= p.straight_deg
    straight[0] = True
    cand &= straight
    t = np.arange(n) / fps
    for a, b in zip(*mask_runs(cand)):
        if b - a < 3 or float(np.nansum(dist[a:b])) < p.flight_min_length_m:
            continue
        sp = speed[a:b]
        if np.polyfit(t[a:b], sp, 1)[0] < -p.grass_decel_mps2:
            continue
        lo, hi = a + int(p.edge_trim_frames), b - int(p.edge_trim_frames)
        if hi > lo:
            out[lo:hi] = True
    return out
