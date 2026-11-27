"""
attention/scoring.py
Turns face measurements into ONE attention score (0-100) for each student.

Idea for a beginner: every cue (head turn, eye closing, yawning...) gives a number between
0 (bad) and 1 (good). We then take a weighted average. The weights come from the faculty
survey (how often faculty picked each cue as one of their top signals).
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional
import numpy as np
from .config import CueSettings, ExpressionSettings, WeightSettings
from .geometry import TimedBuffer, count_nods


@dataclass
class FrameMeasurements:
    """Everything measured from one face in one frame. None = could not measure."""
    t: float
    yaw: Optional[float] = None
    pitch: Optional[float] = None
    gaze: Optional[float] = None
    ear: Optional[float] = None
    mar: Optional[float] = None
    expression_probs: Optional[Dict[str, float]] = None
    phone: Optional[bool] = None


def _ramp(value: float, ok_until: float, zero_at: float) -> float:
    """1.0 while value <= ok_until, falling in a straight line to 0.0 at zero_at."""
    if value <= ok_until:
        return 1.0
    if value >= zero_at:
        return 0.0
    return 1.0 - (value - ok_until) / (zero_at - ok_until)


class CueScorer:
    """Keeps a little memory per student (for blinks, yawns, nods) and scores each frame."""

    def __init__(self, cues: CueSettings, expr: ExpressionSettings, weights: WeightSettings):
        self.c, self.e, self.w = cues, expr, weights
        self.yaw0 = 0.0
        self.pitch0 = 0.0
        self._ear = TimedBuffer(cues.rolling_seconds)
        self._pitch = TimedBuffer(cues.nod_window_seconds)
        self._mouth_open_since: Optional[float] = None
        self._calib: list = []
        self.calibrated = False

    # ---- calibration: learn what "looking at the screen" looks like for THIS camera angle ----
    def add_calibration_sample(self, m: FrameMeasurements):
        if m.yaw is not None and m.pitch is not None:
            self._calib.append((m.yaw, m.pitch))

    def finish_calibration(self):
        if len(self._calib) >= 5:
            self.yaw0 = float(np.median([a for a, _ in self._calib]))
            self.pitch0 = float(np.median([b for _, b in self._calib]))
        self.calibrated = True

    # ---- per-cue scores (each returns 0..1, or None if that cue cannot be measured) ----
    def cue_scores(self, m: FrameMeasurements) -> Dict[str, Optional[float]]:
        c = self.c
        out: Dict[str, Optional[float]] = {}
        if m.yaw is not None and m.pitch is not None:
            dy, dp = abs(m.yaw - self.yaw0), abs(m.pitch - self.pitch0)
            worst = max(dy / c.yaw_limit_deg, dp / c.pitch_limit_deg)       # 1.0 = exactly at the limit
            out["head_pose"] = _ramp(worst, 1.0, c.head_pose_zero_factor)
            self._pitch.add(m.t, m.pitch)
            swings = count_nods(self._pitch.values(), c.nod_amplitude_deg)
            out["nodding"] = 0.0 if swings >= c.nod_min_swings else 1.0
        else:
            out["head_pose"] = out["nodding"] = None
        if m.gaze is not None:
            off = abs(m.gaze - 0.5)
            out["gaze"] = _ramp(off, 0.5 - c.gaze_ok_low, 0.5 - c.gaze_ok_low + 0.2)
        else:
            out["gaze"] = None
        if m.ear is not None:
            self._ear.add(m.t, m.ear)
            perclos = self._ear.fraction(lambda v: v < c.ear_closed)
            out["blink"] = 1.0 - min(1.0, perclos / c.perclos_bad)          # PERCLOS drowsiness measure
        else:
            out["blink"] = None
        if m.mar is not None:
            if m.mar > c.mar_yawn:
                if self._mouth_open_since is None:
                    self._mouth_open_since = m.t
                yawning = (m.t - self._mouth_open_since) >= c.yawn_min_seconds
            else:
                self._mouth_open_since, yawning = None, False
            out["mouth"] = 0.0 if yawning else 1.0
        else:
            out["mouth"] = None
        if m.expression_probs:
            out["expression"] = float(sum(m.expression_probs.get(k, 0.0) * self.e.class_attentiveness.get(k, 0.5)
                                          for k in m.expression_probs))
        else:
            out["expression"] = None
        out["phone"] = None if m.phone is None else (0.0 if m.phone else 1.0)
        return out

    def score(self, m: FrameMeasurements):
        """Returns (score 0-100 or None, per-cue dict)."""
        cues = self.cue_scores(m)
        avail = [k for k, v in cues.items() if v is not None]
        if not avail:
            return None, cues
        w = self.w.weights(available=avail)
        total = sum(w[k] * cues[k] for k in avail)
        if self.c.fusion == "gated":
            for g in self.c.gate_cues:
                if cues.get(g) is not None:
                    total *= cues[g]
        return 100.0 * total, cues
