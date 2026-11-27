"""attention/face_tracker.py  -  finds faces and their 468(+10 iris) landmarks with MediaPipe Face Mesh.

IMPORTANT: this project is pinned to mediapipe==0.10.14, the last family that ships the classic
`mediapipe.solutions` API with its models INSIDE the pip package (so it works offline).
"""
from __future__ import annotations
from typing import List, Tuple
import numpy as np


class FaceLandmarker:
    def __init__(self, max_faces: int = 5, refine: bool = True, det_conf: float = 0.5, track_conf: float = 0.5):
        import mediapipe as mp
        if not hasattr(mp, "solutions"):
            raise RuntimeError(
                "Your mediapipe version has no 'solutions' API. Install the pinned version:\n"
                "    pip install mediapipe==0.10.14")
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=max_faces, refine_landmarks=refine,
            min_detection_confidence=det_conf, min_tracking_confidence=track_conf)

    def process(self, frame_bgr: np.ndarray) -> List[List[Tuple[float, float]]]:
        """Returns one list of (x, y) pixel points per detected face."""
        h, w = frame_bgr.shape[:2]
        res = self._mesh.process(np.ascontiguousarray(frame_bgr[:, :, ::-1]))     # BGR -> RGB
        if not res.multi_face_landmarks:
            return []
        return [[(lm.x * w, lm.y * h) for lm in f.landmark] for f in res.multi_face_landmarks]

    def close(self):
        self._mesh.close()


def bbox_of(pts, w: int, h: int, pad: float = 0.1):
    xs, ys = [p[0] for p in pts[:468]], [p[1] for p in pts[:468]]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    px, py = (x1 - x0) * pad, (y1 - y0) * pad
    return (int(max(0, x0 - px)), int(max(0, y0 - py)), int(min(w, x1 + px)), int(min(h, y1 + py)))
