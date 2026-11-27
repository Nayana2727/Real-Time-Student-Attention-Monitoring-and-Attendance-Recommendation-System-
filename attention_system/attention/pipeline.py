"""
attention/pipeline.py
One object that does everything for one video stream:

    frame -> face landmarks -> measurements (head pose, gaze, eyes, mouth, expression, phone)
          -> survey-weighted attention score per student -> SessionRecorder

Use it from the demo script, the dashboard, or your own code:

    pipe = AttentionPipeline(settings, thresholds)
    results = pipe.process_frame(frame, t)      # t = seconds (time.time() or video time)
    decisions = pipe.finish()                   # attendance RECOMMENDATIONS
"""
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from .attendance import AttendanceEngine, Decision
from .config import Settings, Thresholds
from .expression import ExpressionClassifier
from .face_tracker import FaceLandmarker, bbox_of
from .geometry import gaze_ratio, head_pose, mean_ear, mouth_aspect_ratio
from .scoring import CueScorer, FrameMeasurements
from .session import CentroidTracker, SessionRecorder


@dataclass
class FaceResult:
    student_id: str
    bbox: Tuple[int, int, int, int]
    score: Optional[float]
    cues: Dict[str, Optional[float]]
    meas: FrameMeasurements
    calibrating: bool


class AttentionPipeline:
    def __init__(self, settings: Settings, thresholds: Thresholds, phone_detector=None,
                 expression: Optional[ExpressionClassifier] = None, landmarker: Optional[FaceLandmarker] = None,
                 expression_every_n_frames: int = 3):
        self.cfg, self.th = settings, thresholds
        self.landmarker = landmarker or FaceLandmarker(settings.max_faces)
        self.expression = expression if expression is not None else ExpressionClassifier(settings.expression)
        self.phone = phone_detector
        self.tracker = CentroidTracker()
        self.recorder = SessionRecorder(thresholds, settings)
        self.scorers: Dict[str, CueScorer] = {}
        self._expr_cache: Dict[str, Optional[Dict[str, float]]] = {}
        self._every = max(1, expression_every_n_frames)
        self._frame_no = 0
        self.t0: Optional[float] = None

    def _scorer(self, sid: str) -> CueScorer:
        if sid not in self.scorers:
            self.scorers[sid] = CueScorer(self.cfg.cues, self.cfg.expression, self.cfg.weights)
        return self.scorers[sid]

    def process_frame(self, frame_bgr: np.ndarray, t: Optional[float] = None) -> List[FaceResult]:
        t = time.time() if t is None else t
        if self.t0 is None:
            self.t0 = t
        self._frame_no += 1
        h, w = frame_bgr.shape[:2]
        faces = self.landmarker.process(frame_bgr)
        boxes = [bbox_of(p, w, h) for p in faces]
        ids = self.tracker.assign([((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes], t)
        phone_boxes = self.phone.detect(frame_bgr) if self.phone is not None else None
        calibrating = (t - self.t0) < self.cfg.cues.calibration_seconds
        out: List[FaceResult] = []
        for pts, box, tid in zip(faces, boxes, ids):
            sid = f"Student {tid}"
            hp = head_pose(pts, w, h)
            m = FrameMeasurements(t=t, yaw=None if hp is None else hp.yaw, pitch=None if hp is None else hp.pitch,
                                  gaze=gaze_ratio(pts), ear=mean_ear(pts), mar=mouth_aspect_ratio(pts))
            if self.expression.is_ready():
                if self._frame_no % self._every == 0 or sid not in self._expr_cache:
                    x0, y0, x1, y1 = box
                    self._expr_cache[sid] = self.expression.predict(frame_bgr[y0:y1, x0:x1])
                m.expression_probs = self._expr_cache.get(sid)
            if phone_boxes is not None:
                from .phone_detector import phone_near_face
                m.phone = phone_near_face(phone_boxes, box)
            sc = self._scorer(sid)
            if calibrating:
                sc.add_calibration_sample(m)
                out.append(FaceResult(sid, box, None, {}, m, True))
                continue
            if not sc.calibrated:
                sc.finish_calibration()
            score, cues = sc.score(m)
            self.recorder.record(sid, t, score)
            out.append(FaceResult(sid, box, score, cues, m, False))
        return out

    def live_scores(self) -> Dict[str, Optional[float]]:
        return {sid: rec.live_score() for sid, rec in self.recorder.students.items()}

    def finish(self) -> List[Decision]:
        engine = AttendanceEngine(self.th)
        return [engine.evaluate(s) for s in self.recorder.summaries()]

    def close(self):
        self.landmarker.close()
