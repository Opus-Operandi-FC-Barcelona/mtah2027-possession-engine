"""Ball detectors for the attention crops (Issue #11).

A :class:`BallDetector` takes a batch of BGR ``uint8`` crops and returns, for each, an array ``[K, 3]`` of
``(cx, cy, confidence)`` in the pixel coordinates of that crop (best first).

* :class:`DryRunDetector` -- finds nothing: the whole planning / reading / caching chain runs ("tracking-only"
  mode) without any model.
* :class:`HoughCircleDetector` -- deterministic, dependency-light fallback (OpenCV only): bright circular
  blobs.  It is meant for tests and sanity checks, not as a competition-grade detector.
* :class:`MultiFrameMotionBallDetector` -- TrackNet-style temporal cue (Issue #36): the ball is the small
  round
  blob that moves in a frame triplet (t-1, t, t+1) once the camera pan is compensated; static distractors
  (pitch lines, boots, advertising) cancel out.  OpenCV only, deterministic, off unless selected.
* :class:`UltralyticsYoloDetector` -- COCO class 32 ("sports ball") of a pretrained public YOLO.  OPTIONAL
  (``pip install -e .[yolo]``).  Ultralytics is licensed **AGPL-3.0**: it is only imported when this
  detector is requested and it is never needed to produce the submission (see README, "Vision recovery").
"""
from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from mtah2027.vision.attention_crops import VisionParams

SPORTS_BALL_CLASS = 32       # COCO class id of "sports ball"
EMPTY = np.zeros((0, 3), dtype=np.float64)


@runtime_checkable
class BallDetector(Protocol):
    """Interface of every detector (see the module docstring)."""

    name: str

    def fingerprint(self) -> str:
        """Stable text identifying the detector and its settings (part of the cache key)."""

    def detect(self, crops: Sequence[np.ndarray]) -> list[np.ndarray]:
        """One ``[K, 3]`` array ``(cx, cy, confidence)`` per crop, in crop pixels, best first."""


def detector_fingerprint(params: VisionParams) -> str:
    """Fingerprint of the detector described by ``params`` (usable without constructing it)."""
    kind = params.detector.lower()
    if kind == "yolo":
        return f"yolo|{params.weights}|conf={params.confidence}|crop={params.crop_size}"
    if kind == "hough":
        return (f"hough|r={params.hough_min_radius_px}-{params.hough_max_radius_px}|p1={params.hough_param1}|"
                f"p2={params.hough_param2}|b={params.hough_min_brightness}|k={params.hough_max_candidates}")
    if kind == "motion":
        return (f"motion|thr={params.motion_threshold}|d={params.ball_diameter_m}|"
                f"tol={params.motion_size_tolerance}|k={params.motion_max_candidates}|"
                f"resp={params.motion_min_response}|blur={params.motion_blur_sigma}|"
                f"pen={params.motion_neighbor_penalty}|v={params.max_speed_mps}|crop={params.crop_size}")
    if kind == "dryrun":
        return "dryrun"
    raise ValueError(f"unknown detector '{params.detector}' (yolo | hough | motion | dryrun)")


class DryRunDetector:
    """Detects nothing (tracking-only mode)."""

    name = "dryrun"

    def fingerprint(self) -> str:
        return "dryrun"

    def detect(self, crops: Sequence[np.ndarray]) -> list[np.ndarray]:
        return [EMPTY.copy() for _ in crops]


class HoughCircleDetector:
    """Bright circles found with the Hough gradient transform (OpenCV), deterministic.

    The confidence is the mean grey level inside the circle divided by 255; circles darker than
    ``hough_min_brightness`` are dropped.
    """

    name = "hough"

    def __init__(self, params: VisionParams | None = None):
        import cv2  # noqa: F401  (fail early if OpenCV is missing)

        self.p = params or VisionParams()

    def fingerprint(self) -> str:
        return detector_fingerprint(_with_detector(self.p, "hough"))

    def detect(self, crops: Sequence[np.ndarray]) -> list[np.ndarray]:
        import cv2

        p = self.p
        out = []
        for crop in crops:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
            blur = cv2.GaussianBlur(gray, (5, 5), 1.0)
            circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.0, minDist=float(p.hough_max_radius_px),
                                       param1=p.hough_param1, param2=p.hough_param2,
                                       minRadius=int(p.hough_min_radius_px),
                                       maxRadius=int(p.hough_max_radius_px))
            if circles is None:
                out.append(EMPTY.copy())
                continue
            rows = []
            yy, xx = np.ogrid[:gray.shape[0], :gray.shape[1]]
            for cx, cy, r in circles[0]:
                inside = (xx - cx) ** 2 + (yy - cy) ** 2 <= max(r - 1.0, 1.0) ** 2
                if not inside.any():
                    continue
                level = float(gray[inside].mean())
                if level >= p.hough_min_brightness:
                    rows.append((float(cx), float(cy), level / 255.0))
            rows.sort(key=lambda t: (-t[2], t[1], t[0]))
            out.append(np.array(rows[:p.hough_max_candidates], dtype=np.float64).reshape(-1, 3))
        return out


class MultiFrameMotionBallDetector:
    """Ball = small round blob that moves between the frames t-1, t, t+1 of a crop (Issue #36).

    Input: each crop is a frame triplet ``[3, h, w, 3]`` (BGR, the same window in three consecutive frames;
    the reader produces it because ``temporal`` is true).  A single frame ``[h, w, 3]`` carries no motion and
    yields no candidate.  Optional ``geometry`` (one dict per crop, see :func:`motion_geometry`) gives the
    absolute crop origin and the homography inverses of the three frames; with it the apparent ball size is
    taken from the pixel-to-metre scale at the crop centre and the neighbouring positions are validated in
    metres against ``max_speed_mps``.

    Steps: grey + Gaussian blur; phase-correlation global-motion estimate of the t-1 and t+1 frames against
    t and sub-pixel warp (compensates the camera pan); ``m = min(|I_t - W(I_{t-1})|, |I_t - W(I_{t+1})|)`` (a
    moving ball is in both, a static pattern in neither); threshold; morphological opening (removes lines
    and registration residue thinner than the ball) and closing; connected components filtered by the
    expected ball area and compactness; the same components of each single difference give the positions at
    t-1 / t+1 for the speed check; the centroid is refined to sub-pixel with the difference intensity as
    weight.  A ball that does not move (or moves less than its own size, so that it merges with its previous
    position) is not detectable by construction.
    """

    name = "motion"
    temporal = True           # the reader must supply frame triplets
    context = 1               # frames of context on each side

    def __init__(self, params: VisionParams | None = None):
        import cv2  # noqa: F401  (fail early if OpenCV is missing)

        self.p = params or VisionParams()

    def fingerprint(self) -> str:
        return detector_fingerprint(_with_detector(self.p, "motion"))

    def detect(self, crops: Sequence[np.ndarray], geometry: Sequence[dict | None] | None = None
               ) -> list[np.ndarray]:
        out = []
        for i, crop in enumerate(crops):
            geo = geometry[i] if geometry is not None and i < len(geometry) else None
            out.append(self._detect_one(np.asarray(crop), geo))
        return out

    # -- helpers ---------------------------------------------------------------------------
    def _default_diameter(self) -> float:
        return float(self.p.hough_min_radius_px + self.p.hough_max_radius_px)

    def _gray(self, trip: np.ndarray):
        import cv2

        grays = []
        for k in range(3):
            g = trip[k]
            g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY) if g.ndim == 3 else g
            g = g.astype(np.float32)
            if self.p.motion_blur_sigma > 0:
                g = cv2.GaussianBlur(g, (0, 0), self.p.motion_blur_sigma)
            grays.append(g)
        return grays

    def _align(self, ref: np.ndarray, other: np.ndarray, window: np.ndarray) -> np.ndarray:
        """``other`` warped onto ``ref`` (global translation by phase correlation)."""
        import cv2

        (dx, dy), response = cv2.phaseCorrelate(other.astype(np.float64), ref.astype(np.float64), window)
        if response < self.p.motion_min_response or not (abs(dx) < 80 and abs(dy) < 80):
            return other
        M = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float64)
        return cv2.warpAffine(other, M, (other.shape[1], other.shape[0]), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)

    def _ball_scale(self, geo: dict | None, size: int) -> tuple[float, float | None]:
        """``(expected ball diameter in px, metres per pixel)`` at the crop centre."""
        if geo is None:
            return self._default_diameter(), None
        from mtah2027.io.homography_loader import project_pixels_to_pitch

        c = np.asarray(geo["origin"], dtype=np.float64) + size / 2.0
        u = np.array([c[0], c[0] + 1.0, c[0]])
        v = np.array([c[1], c[1], c[1] + 1.0])
        x, y = project_pixels_to_pitch(np.asarray(geo["H_inv"][1], dtype=np.float64), u, v,
                                       tuple(geo.get("pitch", (105.3, 68.0))))
        area = abs((x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]))
        if not (np.isfinite(area) and area > 0):
            return self._default_diameter(), None
        mpp = float(np.sqrt(area))
        return float(np.clip(self.p.ball_diameter_m / mpp, 3.0, 40.0)), mpp

    def _components(self, diff: np.ndarray, d_px: float) -> list[tuple[float, float, int, float, float]]:
        """Ball-sized compact blobs of a difference image: ``(cx, cy, area, mean diff, fill ratio)``."""
        import cv2

        p = self.p
        binary = (diff > p.motion_threshold).astype(np.uint8)
        k_open = max(2, int(round(0.35 * d_px)))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open)))
        k_close = max(3, int(round(0.5 * d_px)))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close)))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        ball_area = np.pi * (d_px / 2.0) ** 2
        lo, hi = ball_area / p.motion_size_tolerance, ball_area * p.motion_size_tolerance
        blobs = []
        for j in range(1, n):
            x, y, w, h, area = (int(v) for v in stats[j])
            if not (lo <= area <= hi):
                continue
            if max(w, h) > 2.6 * min(w, h) or area / float(w * h) < 0.35:
                continue
            roi = labels[y:y + h, x:x + w] == j
            weight = np.clip(diff[y:y + h, x:x + w] - p.motion_threshold, 0.0, None) * roi
            tot = float(weight.sum())
            if tot <= 0:
                continue
            yy, xx = np.mgrid[y:y + h, x:x + w]
            blobs.append((float((weight * xx).sum() / tot), float((weight * yy).sum() / tot), area,
                          float(diff[y:y + h, x:x + w][roi].mean()), area / float(w * h)))
        return blobs

    def _detect_one(self, trip: np.ndarray, geo: dict | None) -> np.ndarray:
        import cv2

        if trip.ndim < 3 or trip.shape[0] != 3 or (trip.ndim == 3 and trip.shape[-1] == 3):
            return EMPTY.copy()           # not a (t-1, t, t+1) triplet: no temporal information
        g0, g1, g2 = self._gray(trip)
        h, w = g1.shape
        window = cv2.createHanningWindow((w, h), cv2.CV_64F)
        d_prev = np.abs(g1 - self._align(g1, g0, window))
        d_next = np.abs(g1 - self._align(g1, g2, window))
        motion = np.minimum(d_prev, d_next)
        d_px, mpp = self._ball_scale(geo, max(h, w))
        p = self.p
        cands = self._components(motion, d_px)
        if not cands:
            return EMPTY.copy()
        side = {0: self._components(d_prev, d_px), 2: self._components(d_next, d_px)}
        fps = float(geo.get("fps", 25.0)) if geo else 25.0
        r_max = (p.max_speed_mps / fps / mpp + 2.0 * d_px) if mpp else 0.25 * max(h, w)
        rows = []
        for cx, cy, _, mean, fill in cands:
            found = 0
            for which, blobs in side.items():
                for bx, by, _, _, _ in blobs:
                    dist = float(np.hypot(bx - cx, by - cy))
                    if dist < 0.8 * d_px or dist > r_max:       # the blob itself / too far to be the ball
                        continue
                    if geo is None or self._speed_ok(geo, (cx, cy), (bx, by), which, fps):
                        found += 1
                        break
            conf = min(1.0, mean / (2.0 * p.motion_threshold)) * (0.5 + 0.5 * fill)
            if found == 0:
                conf *= p.motion_neighbor_penalty
            rows.append((cx, cy, float(conf)))
        rows.sort(key=lambda r: (-r[2], r[1], r[0]))
        return np.array(rows[:p.motion_max_candidates], dtype=np.float64).reshape(-1, 3)

    def _speed_ok(self, geo: dict, cur: tuple[float, float], other: tuple[float, float], which: int,
                  fps: float) -> bool:
        """Metres travelled between frame t and frame ``which`` of the triplet (via H^-1): plausible speed?"""
        from mtah2027.io.homography_loader import project_pixels_to_pitch

        org = np.asarray(geo["origin"], dtype=np.float64)
        pitch = tuple(geo.get("pitch", (105.3, 68.0)))
        x1, y1 = project_pixels_to_pitch(np.asarray(geo["H_inv"][1], dtype=np.float64),
                                         org[0] + cur[0], org[1] + cur[1], pitch)
        x2, y2 = project_pixels_to_pitch(np.asarray(geo["H_inv"][which], dtype=np.float64),
                                         org[0] + other[0], org[1] + other[1], pitch)
        dist = float(np.hypot(x2 - x1, y2 - y1))
        return bool(np.isfinite(dist) and dist * fps <= self.p.max_speed_mps + self.p.speed_slack_m * fps)


def motion_geometry(origin, H_inv_triplet, *, pitch=(105.3, 68.0), fps: float = 25.0) -> dict | None:
    """Geometry record of one crop for the motion detector (``None`` if a homography is not finite)."""
    Hi = np.asarray(H_inv_triplet, dtype=np.float64)
    if Hi.ndim != 3 or Hi.shape[1:] != (3, 3) or Hi.shape[0] < 3 or not np.isfinite(Hi).all():
        return None
    return {"origin": (float(origin[0]), float(origin[1])), "H_inv": Hi, "pitch": tuple(pitch),
            "fps": float(fps)}


class UltralyticsYoloDetector:
    """COCO "sports ball" (class 32) of a pretrained Ultralytics YOLO (optional extra ``vision``).

    ``weights`` is a public pretrained model name (``yolov8n.pt``, ``yolo11n.pt``, ...) that the library
    downloads itself; a local ``.pt`` path also works.  Ultralytics is AGPL-3.0 (see the module docstring).
    """

    name = "yolo"

    def __init__(self, params: VisionParams | None = None):
        try:
            from ultralytics import YOLO
        except ImportError as exc:    # pragma: no cover - exercised through a fake module in the tests
            raise ImportError("the 'yolo' detector needs ultralytics: pip install -e .[yolo]") from exc
        self.p = params or VisionParams()
        self.model = YOLO(self.p.weights)

    def fingerprint(self) -> str:
        return detector_fingerprint(_with_detector(self.p, "yolo"))

    def detect(self, crops: Sequence[np.ndarray]) -> list[np.ndarray]:
        if not len(crops):
            return []
        kwargs = {"imgsz": int(self.p.crop_size), "conf": float(self.p.confidence),
                  "classes": [SPORTS_BALL_CLASS], "verbose": False}
        if self.p.device:
            kwargs["device"] = self.p.device
        results = self.model.predict(list(crops), **kwargs)
        out = []
        for res in results:
            boxes = res.boxes
            if boxes is None or len(boxes) == 0:
                out.append(EMPTY.copy())
                continue
            xywh = np.asarray(boxes.xywh.cpu().numpy() if hasattr(boxes.xywh, "cpu") else boxes.xywh)
            conf = np.asarray(boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else boxes.conf)
            order = np.argsort(-conf, kind="stable")
            out.append(np.column_stack([xywh[order, 0], xywh[order, 1], conf[order]]).astype(np.float64))
        return out


def _with_detector(params: VisionParams, kind: str) -> VisionParams:
    import dataclasses

    return dataclasses.replace(params, detector=kind)


def make_detector(params: VisionParams) -> BallDetector:
    """The detector named by ``params.detector``."""
    kind = params.detector.lower()
    if kind == "yolo":
        return UltralyticsYoloDetector(params)
    if kind == "hough":
        return HoughCircleDetector(params)
    if kind == "motion":
        return MultiFrameMotionBallDetector(params)
    if kind == "dryrun":
        return DryRunDetector()
    raise ValueError(f"unknown detector '{params.detector}' (yolo | hough | motion | dryrun)")
