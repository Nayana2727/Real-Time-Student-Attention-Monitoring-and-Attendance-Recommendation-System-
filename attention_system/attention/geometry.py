"""
attention/geometry.py
Pure maths on face landmarks (no camera needed, so it is easy to test).

Landmarks are (x, y) pixel points from MediaPipe Face Mesh (468 points + 10 iris points).
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Sequence, Tuple
import numpy as np

# ---- MediaPipe Face Mesh landmark numbers ----------------------------------------------
# Eye corners/lids for the Eye Aspect Ratio (Soukupova & Cech, 2016): p1..p6 around each eye.
RIGHT_EYE = (33, 160, 158, 133, 153, 144)
LEFT_EYE = (362, 385, 387, 263, 373, 380)
# Mouth: outer corners 61/291, inner lips top 13 / bottom 14 plus two extra pairs for stability.
MOUTH_CORNERS = (61, 291)
MOUTH_PAIRS = ((13, 14), (81, 178), (311, 402))
# Six points for head pose with a generic 3-D face model.
POSE_IDS = {"nose": 1, "chin": 152, "left_eye": 263, "right_eye": 33, "left_mouth": 291, "right_mouth": 61}
MODEL_3D = np.array([(0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0),
                     (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)], dtype=np.float64)
# Iris centres (need refine_landmarks=True) and the eye corners used as a ruler.
LEFT_IRIS_CENTER, RIGHT_IRIS_CENTER = 473, 468
LEFT_EYE_CORNERS = (362, 263)    # inner, outer
RIGHT_EYE_CORNERS = (133, 33)    # inner, outer


def _d(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def eye_aspect_ratio(pts: Sequence[Tuple[float, float]], ids: Sequence[int]) -> float:
    """EAR = (|p2-p6| + |p3-p5|) / (2*|p1-p4|).  About 0.3 when open, near 0 when closed."""
    p1, p2, p3, p4, p5, p6 = (pts[i] for i in ids)
    horizontal = _d(p1, p4)
    return 0.0 if horizontal < 1e-6 else (_d(p2, p6) + _d(p3, p5)) / (2.0 * horizontal)


def mean_ear(pts) -> float:
    return 0.5 * (eye_aspect_ratio(pts, LEFT_EYE) + eye_aspect_ratio(pts, RIGHT_EYE))


def mouth_aspect_ratio(pts) -> float:
    """MAR = average lip opening / mouth width.  Small when closed, large when yawning."""
    width = _d(pts[MOUTH_CORNERS[0]], pts[MOUTH_CORNERS[1]])
    if width < 1e-6:
        return 0.0
    return float(np.mean([_d(pts[a], pts[b]) for a, b in MOUTH_PAIRS]) / width)


@dataclass
class HeadPose:
    yaw: float      # + = turned to the person's left... sign is consistent, magnitude is what we use
    pitch: float    # + = looking up
    roll: float


def head_pose(pts, frame_w: int, frame_h: int) -> Optional[HeadPose]:
    """Yaw/pitch/roll in degrees using OpenCV solvePnP on six landmarks."""
    import cv2
    img = np.array([pts[POSE_IDS[k]] for k in ("nose", "chin", "left_eye", "right_eye", "left_mouth", "right_mouth")],
                   dtype=np.float64)
    cam = np.array([[frame_w, 0, frame_w / 2.0], [0, frame_w, frame_h / 2.0], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(MODEL_3D, img, cam, np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    # Decompose with RQ-style angles that stay in [-90, 90] for a face looking at the camera.
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        x = np.arctan2(R[2, 1], R[2, 2]); y = np.arctan2(-R[2, 0], sy); z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1]); y = np.arctan2(-R[2, 0], sy); z = 0.0
    pitch, yaw, roll = (float(np.degrees(v)) for v in (x, y, z))
    # solvePnP with this model returns pitch near +/-180 for a frontal face; fold it into [-90, 90].
    if pitch > 90: pitch -= 180
    elif pitch < -90: pitch += 180
    if roll > 90: roll -= 180
    elif roll < -90: roll += 180
    return HeadPose(yaw=yaw, pitch=pitch, roll=roll)


def gaze_ratio(pts) -> Optional[float]:
    """Where the iris sits between the eye corners: 0 = far one side, 0.5 = centre, 1 = far other side."""
    if len(pts) < 478:
        return None   # iris points missing (refine_landmarks was off)
    vals = []
    for iris, (inner, outer) in ((LEFT_IRIS_CENTER, LEFT_EYE_CORNERS), (RIGHT_IRIS_CENTER, RIGHT_EYE_CORNERS)):
        a, b = pts[inner][0], pts[outer][0]
        span = b - a
        if abs(span) < 1e-6:
            continue
        vals.append(float(np.clip((pts[iris][0] - a) / span, 0.0, 1.0)))
    return float(np.mean(vals)) if vals else None


class TimedBuffer:
    """Keeps (time, value) pairs for the last `seconds` seconds."""
    def __init__(self, seconds: float):
        self.seconds = seconds
        self.items: Deque[Tuple[float, float]] = deque()

    def add(self, t: float, v: float):
        self.items.append((t, v))
        while self.items and self.items[0][0] < t - self.seconds:
            self.items.popleft()

    def values(self):
        return [v for _, v in self.items]

    def fraction(self, predicate) -> float:
        vs = self.values()
        return 0.0 if not vs else sum(1 for v in vs if predicate(v)) / len(vs)


def count_nods(pitch_series: Sequence[float], amplitude_deg: float) -> int:
    """Count head-pitch direction reversals bigger than `amplitude_deg` (a 'swing')."""
    if len(pitch_series) < 3:
        return 0
    swings, direction, extreme = 0, 0, pitch_series[0]
    for p in pitch_series[1:]:
        if direction >= 0 and p < extreme - amplitude_deg:
            if direction != 0: swings += 1
            direction, extreme = -1, p
        elif direction <= 0 and p > extreme + amplitude_deg:
            if direction != 0: swings += 1
            direction, extreme = 1, p
        else:
            extreme = max(extreme, p) if direction >= 0 else min(extreme, p)
    return swings
