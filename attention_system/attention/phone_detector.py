"""attention/phone_detector.py - OPTIONAL phone detection (faculty ranked 'phone or object use' 4th in the survey).

Off by default. To switch it on:   pip install ultralytics     then run the demo with  --phone
The first run downloads a small pretrained YOLOv8n model (COCO class 'cell phone').
NOTE: this module could not be tested inside the development sandbox (no model download there),
so test it on your own laptop before relying on it.  Licence note: Ultralytics YOLO is AGPL-3.0.
"""
from __future__ import annotations
from typing import List, Tuple
import numpy as np

COCO_CELL_PHONE = 67


class PhoneDetector:
    def __init__(self, weights: str = "yolov8n.pt", conf: float = 0.35):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise RuntimeError("Phone detection needs:  pip install ultralytics") from e
        self.model, self.conf = YOLO(weights), conf

    def detect(self, frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
        r = self.model.predict(frame_bgr, conf=self.conf, classes=[COCO_CELL_PHONE], verbose=False)[0]
        return [tuple(int(v) for v in b.xyxy[0].tolist()) for b in r.boxes]


def phone_near_face(phone_boxes, face_box) -> bool:
    """A phone counts for a student if its centre is within 1.5 face-widths of that face."""
    if not phone_boxes:
        return False
    fx0, fy0, fx1, fy1 = face_box
    fc = ((fx0 + fx1) / 2, (fy0 + fy1) / 2); reach = 1.5 * (fx1 - fx0)
    for x0, y0, x1, y1 in phone_boxes:
        if np.hypot((x0 + x1) / 2 - fc[0], (y0 + y1) / 2 - fc[1]) <= reach:
            return True
    return False
