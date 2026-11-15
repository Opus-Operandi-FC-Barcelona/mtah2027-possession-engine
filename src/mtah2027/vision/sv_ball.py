"""Ball detector built on roboflow/supervision + a local football-ball YOLO model (Issue #63).

``supervision`` (MIT) is a toolkit, not a detector: here it provides ``InferenceSlicer`` (the 640x640 attention
crop is cut in overlapping tiles so that the ~4.5 px ball is seen at a larger relative size by the network).
The detector itself is the single-class ("ball") YOLOv8 model of roboflow/sports, loaded ONLY from a local
``.pt`` file through ``ultralytics.YOLO(path)`` (no hosted inference, no frame ever leaves the machine).

Two fixed configurations (chosen a priori, never tuned on the real match):

* ``sv_ball``      -- slices 320x320 with 20 % overlap (``overlap_wh=64``)
* ``sv_ball640``   -- the whole crop at ``imgsz=640``, no slicing

Optional temporal filter (``clip=5``): the reader supplies the 5 consecutive frames around the sampled one, the
model runs on each, candidates are reprojected to the pitch with ``H^-1`` and the confidence of every
candidate of the centre frame is scaled by the support it finds in the neighbour frames (a candidate within
the speed gate ``max_speed_mps`` of a candidate in a neighbour frame; extra weight when the two frames
around ``t`` agree with a constant-velocity motion).  ByteTrack is not used: supervision >= 0.31 moved it to
a separate package.

torch / ultralytics (AGPL-3.0) / supervision are imported lazily and only when this detector is built, so the
default path and CI never need them.  CPU, deterministic (``torch.manual_seed(0)``, one thread worker).
"""
# ruff: noqa: E501
from __future__ import annotations

from typing import Sequence

import numpy as np

from mtah2027.vision.detectors import EMPTY

CONFIGS = {
    "sv_ball": {"slice": 320, "overlap": 64},       # sliced
    "sv_ball640": {"slice": 0, "overlap": 0},        # whole crop, no slicing
}
CONF = 0.05                  # low on purpose: the ball is ~4.5 px; ranking is done by the confidence
MAX_CANDIDATES = 5
NEIGHBOUR_FLOOR = 0.5        # confidence factor of a candidate with no support in any neighbour frame


def sv_available() -> bool:
    """True when torch, ultralytics and supervision can all be imported."""
    try:
        import supervision  # noqa: F401
        import torch  # noqa: F401
        import ultralytics  # noqa: F401
    except Exception:
        return False
    return True


class SupervisionBallDetector:
    """See the module docstring.  ``clip`` = 0/1: single frame; odd ``clip`` >= 3: temporal filter."""

    def __init__(self, weights: str, config: str = "sv_ball", *, clip: int = 0, conf: float = CONF,
                 max_speed_mps: float = 38.0, speed_slack_m: float = 1.0, device: str = "cpu"):
        if config not in CONFIGS:
            raise ValueError(f"unknown config '{config}' ({' | '.join(CONFIGS)})")
        if clip not in (0, 1) and (clip < 3 or clip % 2 == 0):
            raise ValueError("clip must be 0, 1 or an odd number >= 3")
        try:
            import supervision as sv
            import torch
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError("sv_ball needs torch, ultralytics and supervision (separate venv, see the "
                              "'Issue #63' section of the technical report)") from exc
        torch.manual_seed(0)
        torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
        self._sv = sv
        self.weights = str(weights)
        self.model = YOLO(self.weights)
        self.config = config
        self.conf = float(conf)
        self.device = device
        self.max_speed_mps = float(max_speed_mps)
        self.speed_slack_m = float(speed_slack_m)
        self.clip = 0 if clip <= 1 else int(clip)
        self.name = config + (f"_clip{self.clip}" if self.clip else "")
        self.temporal = self.clip > 0
        self.context = self.clip // 2
        cfg = CONFIGS[config]
        self._slicer = None
        if cfg["slice"]:
            self._slicer = sv.InferenceSlicer(callback=self._callback, slice_wh=cfg["slice"],
                                              overlap_wh=cfg["overlap"], thread_workers=1,
                                              iou_threshold=0.5)

    def fingerprint(self) -> str:
        return f"{self.name}|{self.weights}|conf={self.conf}|clip={self.clip}"

    # -- model ---------------------------------------------------------------------------
    def _callback(self, tile: np.ndarray):
        res = self.model.predict(tile, imgsz=int(max(tile.shape[:2])), conf=self.conf, device=self.device,
                                 verbose=False)[0]
        return self._sv.Detections.from_ultralytics(res)

    def _detect_image(self, img: np.ndarray) -> np.ndarray:
        """``[K, 3]`` (cx, cy, conf) of one BGR image, best first."""
        det = self._slicer(img) if self._slicer is not None else self._callback(img)
        if len(det) == 0:
            return EMPTY.copy()
        xyxy = np.asarray(det.xyxy, dtype=np.float64)
        conf = np.asarray(det.confidence, dtype=np.float64)
        cx, cy = 0.5 * (xyxy[:, 0] + xyxy[:, 2]), 0.5 * (xyxy[:, 1] + xyxy[:, 3])
        order = np.lexsort((cx, cy, -conf))
        return np.column_stack([cx[order], cy[order], conf[order]])[:MAX_CANDIDATES * 2]

    # -- temporal filter -----------------------------------------------------------------
    @staticmethod
    def _to_pitch(cand: np.ndarray, geo: dict, k: int) -> np.ndarray:
        from mtah2027.vision.attention_crops import crop_points_to_pitch

        if not len(cand):
            return np.zeros((0, 2))
        org = np.repeat(np.asarray(geo["origin"], dtype=np.float64)[None], len(cand), axis=0)
        Hi = np.repeat(np.asarray(geo["H_inv"][k], dtype=np.float64)[None], len(cand), axis=0)
        return crop_points_to_pitch(cand[:, :2], org, Hi, tuple(geo.get("pitch", (105.3, 68.0))))

    def filter_clip(self, per_frame: Sequence[np.ndarray], geo: dict | None) -> np.ndarray:
        """Rescore the candidates of the centre frame with the support of the neighbour frames."""
        c = len(per_frame) // 2
        cur = per_frame[c]
        if len(cur) == 0 or geo is None or len(per_frame) < 3:
            return cur[:MAX_CANDIDATES]
        fps = float(geo.get("fps", 25.0))
        pts = [self._to_pitch(p, geo, k) for k, p in enumerate(per_frame)]
        n = len(cur)
        support = np.zeros(n)
        cv_bonus = np.zeros(n)
        n_nb = len(per_frame) - 1
        for i in range(n):
            matched: dict[int, np.ndarray] = {}
            for k in range(len(per_frame)):
                if k == c or not len(pts[k]):
                    continue
                gate = self.max_speed_mps * abs(k - c) / fps + self.speed_slack_m
                d = np.hypot(*(pts[k] - pts[c][i]).T)
                ok = np.flatnonzero(d <= gate)
                if len(ok):
                    j = ok[np.argmax(per_frame[k][ok, 2])]
                    matched[k] = pts[k][j]
                    support[i] += 1.0
            if (c - 1) in matched and (c + 1) in matched:
                mid = 0.5 * (matched[c - 1] + matched[c + 1])
                if float(np.hypot(*(pts[c][i] - mid))) <= self.speed_slack_m:
                    cv_bonus[i] = 1.0
        factor = NEIGHBOUR_FLOOR + (1.0 - NEIGHBOUR_FLOOR) * (0.8 * support / n_nb + 0.2 * cv_bonus)
        out = np.column_stack([cur[:, :2], cur[:, 2] * factor])
        out = out[np.lexsort((out[:, 0], out[:, 1], -out[:, 2]))]
        return out[:MAX_CANDIDATES]

    # -- BallDetector interface ----------------------------------------------------------
    def detect(self, crops: Sequence[np.ndarray], geometry: Sequence[dict | None] | None = None
               ) -> list[np.ndarray]:
        out = []
        for i, crop in enumerate(crops):
            arr = np.asarray(crop)
            if self.temporal and arr.ndim == 4:
                geo = geometry[i] if geometry is not None and i < len(geometry) else None
                per = [self._detect_image(np.ascontiguousarray(f)) for f in arr]
                out.append(self.filter_clip(per, geo))
            else:
                img = arr[arr.shape[0] // 2] if arr.ndim == 4 else arr
                out.append(self._detect_image(np.ascontiguousarray(img))[:MAX_CANDIDATES])
        return out
