"""Fine-tuned single-class ball detector on the GPU (Issue #68).

Same interface as the other detectors (``detect(crops)`` -> one ``[K, 3]`` (cx, cy, conf) array per crop, best
first) so that it plugs into :func:`mtah2027.vision.recovery.run_detection` and its cache.  The weights are
produced by ``scripts/train_ball_detector.py`` from the self-supervised ATD-ball labels of the match video and
are never committed.  torch / ultralytics (AGPL-3.0) are imported lazily, only when this detector is built.
Batched inference, deterministic (``torch.manual_seed(0)``).
"""
# ruff: noqa: E501
from __future__ import annotations

from typing import Sequence

import numpy as np

from mtah2027.vision.detectors import EMPTY


class FineTunedBallDetector:
    def __init__(self, weights: str, *, imgsz: int = 960, conf: float = 0.02, max_candidates: int = 5,
                 device: str | int = 0, half: bool = True, batch: int = 16):
        try:
            import torch
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError("the fine-tuned ball detector needs torch and ultralytics (vision environment, "
                              "see the 'Issue #68' section of the technical report)") from exc
        torch.manual_seed(0)
        self.weights = str(weights)
        self.model = YOLO(self.weights)
        self.imgsz, self.conf, self.k = int(imgsz), float(conf), int(max_candidates)
        self.device, self.half, self.batch = device, bool(half), int(batch)
        self.name = "ft_ball"

    def fingerprint(self) -> str:
        return f"ft_ball|{self.weights}|imgsz={self.imgsz}|conf={self.conf}|k={self.k}"

    def detect(self, crops: Sequence[np.ndarray]) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        imgs = []
        for c in crops:
            a = np.asarray(c)
            imgs.append(np.ascontiguousarray(a if a.ndim == 3 else a[a.shape[0] // 2]))
        for i in range(0, len(imgs), self.batch):
            res = self.model.predict(imgs[i:i + self.batch], imgsz=self.imgsz, conf=self.conf,
                                     device=self.device, half=self.half, verbose=False, max_det=self.k * 2)
            for r in res:
                b = r.boxes
                if b is None or len(b) == 0:
                    out.append(EMPTY.copy())
                    continue
                xyxy = b.xyxy.cpu().numpy().astype(np.float64)
                cf = b.conf.cpu().numpy().astype(np.float64)
                cx, cy = 0.5 * (xyxy[:, 0] + xyxy[:, 2]), 0.5 * (xyxy[:, 1] + xyxy[:, 3])
                order = np.lexsort((cx, cy, -cf))
                out.append(np.column_stack([cx[order], cy[order], cf[order]])[:self.k])
        return out
