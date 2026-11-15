"""Attention crops: where to look for the ball in the tactical video when the tracking lost it (Issue #11).

Pure geometry, no video and no OpenCV.  For every ball-missing frame inside the graded play the plan holds

* a **search point** in pitch metres: the straight-line interpolation between the last and the next seen ball
  (the nearer one when the gap is longer than ``interp_max_s``), pulled towards the *pressure centroid* -- the
  centroid of the players around that point, weighted by a Gaussian of their distance and by how fast they
  close in on it (players converging on the ball are the best cue of where it is);
* its **pixel** position ``(u, v)``, projected with the per-frame tactical homography
  (:func:`~mtah2027.io.homography_loader.project_pitch_to_pixels`);
* a ``crop_size x crop_size`` **window** around it, clamped inside the ``frame_width x frame_height`` image.

To bound the cost only every ``every_k``-th frame of each gap is planned (and optionally at most
``max_crops`` of them).  Detections found in a crop are reprojected to pitch metres with ``H^-1``
(:func:`crop_points_to_pitch`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from mtah2027.config import FPS, PITCH, load_section, params_from_mapping
from mtah2027.features._utils import mask_runs
from mtah2027.io.homography_loader import (
    Homographies,
    project_pitch_to_pixels,
    project_pixels_to_pitch,
)


@dataclass(frozen=True)
class VisionParams:
    """Section ``vision`` of the configuration (everything of Issue #11; ``enabled`` is off by default)."""

    enabled: bool = False
    # --- crop planning
    crop_size: int = 640
    frame_width: int = 1920
    frame_height: int = 1080
    every_k: int = 5                  # one crop every k frames of each ball-missing gap
    max_crops: int = 0                # 0 = no cap on the number of crops
    max_homography_error: float = 5.0  # frames with a larger reprojection error are not planned
    offscreen_margin_px: float = 160.0  # search points farther than this outside the image are dropped
    interp_max_s: float = 20.0        # between seen balls farther apart in time: use the nearer one
    pressure_radius_m: float = 12.0   # players farther than this from the search point are ignored
    pressure_sigma_m: float = 6.0     # Gaussian scale of the distance weight
    closing_gain_mps: float = 3.0     # a player closing in at this speed doubles its weight
    pressure_weight: float = 0.5      # 0 = pure interpolation, 1 = pressure centroid
    vel_half_span: int = 2            # frames each side of a planned frame for the player velocity
    # --- detector
    detector: str = "hough"           # "yolo" | "hough" | "dryrun"
    weights: str = "yolov8n.pt"       # ultralytics weights (downloaded by the library)
    confidence: float = 0.25          # detections below this confidence are dropped
    batch_size: int = 16
    device: str = ""                  # ultralytics device ("" = automatic)
    seek_threshold_frames: int = 250  # skip by seeking instead of decoding when the jump is longer
    hough_min_radius_px: int = 4
    hough_max_radius_px: int = 14
    hough_param1: float = 120.0       # Canny high threshold
    hough_param2: float = 10.0        # accumulator threshold (lower = more circles)
    hough_min_brightness: float = 170.0   # mean grey level inside the circle (0-255)
    hough_max_candidates: int = 5
    # --- multi-frame motion detector (Issue #36; detector "motion")
    motion_threshold: float = 22.0    # grey-level difference (0-255) of the compensated temporal difference
    ball_diameter_m: float = 0.22     # real ball size: apparent size comes from the homography scale
    motion_size_tolerance: float = 3.0    # accepted blob area between ball_area / tol and ball_area * tol
    motion_max_candidates: int = 5
    motion_min_response: float = 0.1  # phase-correlation response below which the background is not shifted
    motion_blur_sigma: float = 1.0    # pre-blur against sub-pixel registration residue on sharp edges
    motion_neighbor_penalty: float = 0.6   # confidence factor when no t-1 / t+1 position of the blob is found
    # --- selection / injection
    pitch_margin_m: float = 1.0       # detections farther outside the pitch than this are dropped
    max_search_dist_m: float = 30.0   # candidates farther than this from the search point are dropped
    max_speed_mps: float = 40.0       # plausible ball speed between neighbouring points
    speed_slack_m: float = 1.5        # position noise allowed on top of max_speed * dt
    interp_max_frames: int = 25       # frames filled between two points when one of them is a detection
    cache_dir: str = "cache/vision"
    # --- Issue #66: tracklet gate on the detections + emission tempering on injected frames (both off)
    tracklet_gate: bool = False       # keep a detection only if it belongs to a smooth run of crops
    tracklet_min_len: int = 3         # consecutive crops (detections) a run needs
    tracklet_max_speed_mps: float = 35.0   # plausible speed between linked detections
    tracklet_speed_slack_m: float = 1.0    # position noise allowed on top of max_speed * dt
    tracklet_max_residual_m: float = 1.0   # distance to the smooth (local polynomial) fit of the run
    tracklet_max_link_frames: int = 10     # linked detections are at most this many frames apart
    injected_emission_scale: float = 1.0   # <1 tempers emissions where ball_source is 2/3
    # --- Issue #67: soft evidence (detections tilt the H/A emission instead of becoming the ball); 0 = off
    soft_evidence_weight: float = 0.0
    soft_evidence_scale_m: float = 3.0
    soft_evidence_hold_frames: int = 5

    @classmethod
    def from_config(cls, path=None) -> "VisionParams":
        return params_from_mapping(cls, load_section("vision", path), "vision")


@dataclass
class CropPlan:
    """The planned crops, sorted by frame (``M`` entries)."""

    frames: np.ndarray        # [M] int64 video frame indices
    search_xy: np.ndarray     # [M, 2] float64 predicted ball position (m)
    center_uv: np.ndarray     # [M, 2] float64 its pixel position
    origin: np.ndarray        # [M, 2] int64 top-left corner (x0, y0) of the window
    gap_id: np.ndarray        # [M] int64 index of the ball-missing run
    n_hidden: int             # ball-missing graded frames the plan was drawn from
    n_gaps: int

    def __len__(self) -> int:
        return int(len(self.frames))

    def head(self, n_frames: int) -> "CropPlan":
        """Only the crops of the first ``n_frames`` video frames."""
        keep = self.frames < n_frames
        return CropPlan(self.frames[keep], self.search_xy[keep], self.center_uv[keep], self.origin[keep],
                        self.gap_id[keep], self.n_hidden, self.n_gaps)


def crop_window(u, v, size: int = 640, width: int = 1920, height: int = 1080) -> np.ndarray:
    """Top-left corners ``[..., 2]`` (int) of ``size x size`` windows centred on ``(u, v)`` and clamped
    inside the image (the window is never smaller than ``size``)."""
    if size > width or size > height:
        raise ValueError(f"crop {size} does not fit in {width}x{height}")
    u, v = np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64)
    x0 = np.clip(np.rint(u - size / 2.0), 0, width - size)
    y0 = np.clip(np.rint(v - size / 2.0), 0, height - size)
    return np.stack([x0, y0], axis=-1).astype(np.int64)


def crop_points_to_pitch(points_px, origin, H_inv, pitch: tuple[float, float] = PITCH) -> np.ndarray:
    """Points ``[K, 2]`` in crop pixels -> pitch metres ``[K, 2]``.

    ``origin[K, 2]`` is the top-left corner of the crop of each point and ``H_inv`` either one ``[3, 3]``
    matrix or one per point ``[K, 3, 3]`` (pixels -> normalised pitch).
    """
    pts = np.asarray(points_px, dtype=np.float64).reshape(-1, 2)
    org = np.asarray(origin, dtype=np.float64).reshape(-1, 2)
    Hi = np.asarray(H_inv, dtype=np.float64)
    if Hi.ndim == 3:
        # one matrix per point: points are the "batch" axis, each with a single point
        x, y = project_pixels_to_pitch(Hi, (org[:, 0] + pts[:, 0])[:, None], (org[:, 1] + pts[:, 1])[:, None],
                                       pitch)
        return np.stack([x[:, 0], y[:, 0]], axis=1)
    x, y = project_pixels_to_pitch(Hi, org[:, 0] + pts[:, 0], org[:, 1] + pts[:, 1], pitch)
    return np.stack([x, y], axis=1)


def hidden_gap_frames(ball_xy: np.ndarray, play_blocks: Sequence[tuple[int, int]]) -> np.ndarray:
    """Mask of the graded frames in which the ball is missing."""
    n = len(ball_xy)
    graded = np.zeros(n, dtype=bool)
    for a, b in play_blocks:
        graded[max(int(a), 0):min(int(b), n - 1) + 1] = True
    return graded & ~np.isfinite(np.asarray(ball_xy)).all(axis=1)


def subsample_gap_frames(hidden: np.ndarray, every_k: int) -> tuple[np.ndarray, np.ndarray]:
    """Every ``every_k``-th frame of each hidden run: ``(frames, gap_id)``.

    The first sample is ``every_k // 2`` frames into the run (a frame next to the last seen ball adds little),
    and a run shorter than that still gets its middle frame.
    """
    k = max(1, int(every_k))
    starts, ends = mask_runs(hidden)
    frames, gaps = [], []
    for gid, (a, b) in enumerate(zip(starts, ends)):
        first = a + min(k // 2, (b - a - 1) // 2)
        idx = np.arange(first, b, k, dtype=np.int64)
        frames.append(idx)
        gaps.append(np.full(len(idx), gid, dtype=np.int64))
    if not frames:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.concatenate(frames), np.concatenate(gaps)


def _edge_indices(visible: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Index of the last / next visible frame at or before / after every frame (-1 / n when none)."""
    n = len(visible)
    idx = np.arange(n)
    last = np.maximum.accumulate(np.where(visible, idx, -1))
    nxt = np.minimum.accumulate(np.where(visible, idx, n)[::-1])[::-1]
    return last, nxt


def interpolated_search_points(frames: np.ndarray, ball_xy: np.ndarray, fps: float,
                               interp_max_s: float) -> np.ndarray:
    """Interpolation between the last and next seen ball for each frame (NaN if the ball is never seen)."""
    ball = np.asarray(ball_xy, dtype=np.float64)
    n = len(ball)
    visible = np.isfinite(ball).all(axis=1)
    last, nxt = _edge_indices(visible)
    lf, nf = last[frames], nxt[frames]
    has_l, has_n = lf >= 0, nf < n
    out = np.full((len(frames), 2), np.nan)
    both = has_l & has_n & ((nf - lf) / fps <= interp_max_s)
    span = np.maximum(nf - lf, 1)
    w = ((frames - lf) / span)[:, None]
    out[both] = (ball[np.clip(lf, 0, n - 1)] * (1 - w) + ball[np.clip(nf, 0, n - 1)] * w)[both]
    rest = ~both
    near_last = has_l & (~has_n | ((frames - lf) <= (nf - frames)))
    take_l = rest & near_last
    take_n = rest & ~near_last & has_n
    out[take_l] = ball[np.clip(lf, 0, n - 1)][take_l]
    out[take_n] = ball[np.clip(nf, 0, n - 1)][take_n]
    return out


def pressure_search_points(frames: np.ndarray, base_xy: np.ndarray, players_xy: np.ndarray, fps: float,
                           params: VisionParams) -> np.ndarray:
    """Pull ``base_xy`` towards the pressure centroid of the players at each frame.

    The weight of a player within ``pressure_radius_m`` of the base point is
    ``exp(-d^2 / (2 sigma^2)) * (1 + closing / closing_gain_mps)`` with ``closing`` the speed (m/s) at which
    it approaches the point (0 when moving away).  Frames without any player in range keep the base point.
    """
    n, s, _ = players_xy.shape
    h = max(1, int(params.vel_half_span))
    f_prev, f_next = np.clip(frames - h, 0, n - 1), np.clip(frames + h, 0, n - 1)
    p = players_xy[frames].astype(np.float64)                                  # [M, S, 2]
    dt = np.maximum(f_next - f_prev, 1)[:, None, None] / fps
    vel = (players_xy[f_next].astype(np.float64) - players_xy[f_prev].astype(np.float64)) / dt
    rel = base_xy[:, None, :] - p
    dist = np.hypot(rel[..., 0], rel[..., 1])
    with np.errstate(invalid="ignore", divide="ignore"):
        closing = np.maximum((vel * rel).sum(-1) / np.maximum(dist, 1e-6), 0.0)
    closing = np.nan_to_num(closing, nan=0.0)
    w = np.exp(-(np.nan_to_num(dist, nan=1e9) ** 2) / (2.0 * params.pressure_sigma_m ** 2))
    w = w * (1.0 + closing / params.closing_gain_mps)
    w = np.where(np.isfinite(dist) & (dist <= params.pressure_radius_m), w, 0.0)
    total = w.sum(axis=1)
    centroid = (np.nan_to_num(p) * w[..., None]).sum(axis=1) / np.maximum(total, 1e-12)[:, None]
    mix = np.where((total > 1e-9)[:, None], (1 - params.pressure_weight) * base_xy
                   + params.pressure_weight * centroid, base_xy)
    return mix


def plan_crops(ball_xy: np.ndarray, players_xy: np.ndarray, homographies: Homographies,
               play_blocks: Sequence[tuple[int, int]], *, fps: float = FPS,
               pitch: tuple[float, float] = PITCH, params: VisionParams | None = None,
               every_k: int | None = None) -> CropPlan:
    """Crop plan for the ball-missing graded frames (see the module docstring)."""
    p = params or VisionParams.from_config()
    n = len(ball_xy)
    hidden = hidden_gap_frames(ball_xy, play_blocks)
    frames, gap_id = subsample_gap_frames(hidden, p.every_k if every_k is None else every_k)
    n_hidden, n_gaps = int(hidden.sum()), int(len(mask_runs(hidden)[0]))
    empty = CropPlan(np.zeros(0, np.int64), np.zeros((0, 2)), np.zeros((0, 2)), np.zeros((0, 2), np.int64),
                     np.zeros(0, np.int64), n_hidden, n_gaps)
    if not len(frames):
        return empty
    usable = homographies.usable(p.max_homography_error)
    if len(usable) < n:
        usable = np.concatenate([usable, np.zeros(n - len(usable), bool)])
    base = interpolated_search_points(frames, ball_xy, fps, p.interp_max_s)
    ok = usable[frames] & np.isfinite(base).all(axis=1)
    frames, gap_id, base = frames[ok], gap_id[ok], base[ok]
    if not len(frames):
        return empty
    search = pressure_search_points(frames, base, np.asarray(players_xy), fps, p) \
        if p.pressure_weight > 0 else base
    u, v = project_pitch_to_pixels(homographies.H[frames], search[:, 0], search[:, 1], pitch)
    m = p.offscreen_margin_px
    inside = np.isfinite(u) & np.isfinite(v) & (u > -m) & (u < p.frame_width + m) & (v > -m) & \
        (v < p.frame_height + m)
    frames, gap_id, search, u, v = frames[inside], gap_id[inside], search[inside], u[inside], v[inside]
    if p.max_crops > 0 and len(frames) > p.max_crops:
        keep = np.unique(np.linspace(0, len(frames) - 1, p.max_crops).round().astype(np.int64))
        frames, gap_id, search, u, v = frames[keep], gap_id[keep], search[keep], u[keep], v[keep]
    origin = crop_window(u, v, p.crop_size, p.frame_width, p.frame_height)
    return CropPlan(frames=frames, search_xy=search, center_uv=np.stack([u, v], axis=1), origin=origin,
                    gap_id=gap_id, n_hidden=n_hidden, n_gaps=n_gaps)
