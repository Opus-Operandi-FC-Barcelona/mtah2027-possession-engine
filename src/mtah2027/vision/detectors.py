"""Ball detectors for the attention crops (Issue #11).

A :class:`BallDetector` takes a batch of BGR ``uint8`` crops and returns, for each, an array ``[K, 3]`` of
``(cx, cy, confidence)`` in the pixel coordinates of that crop (best first).

* :class:`DryRunDetector` -- finds nothing: the whole planning / reading / caching chain runs ("tracking-only"
  mode) without any model.
* :class:`HoughCircleDetector` -- deterministic, dependency-light fallback (OpenCV only): bright circular
  blobs.  It is meant for tests and sanity checks, not as a competition-grade detector.
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
    if kind == "dryrun":
        return "dryrun"
    raise ValueError(f"unknown detector '{params.detector}' (yolo | hough | dryrun)")


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
    if kind == "dryrun":
        return DryRunDetector()
    raise ValueError(f"unknown detector '{params.detector}' (yolo | hough | dryrun)")
