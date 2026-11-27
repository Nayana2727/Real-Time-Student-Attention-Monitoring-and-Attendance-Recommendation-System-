"""
attention/session.py
Keeps the running record for each student during a class and produces SessionSummary objects.
No video is stored - only numbers (time, score).
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple
import numpy as np
from .attendance import SessionSummary
from .config import Settings, Thresholds


class StudentRecord:
    """Tracks one student. add(t, score) is called many times per second."""

    MAX_GAP = 2.0   # seconds; a longer silence is treated as 'not visible', not as attentive/inattentive

    def __init__(self, student_id: str, thresholds: Thresholds, settings: Settings):
        self.id, self.th, self.cfg = student_id, thresholds, settings
        self.first_t: Optional[float] = None
        self.last_t: Optional[float] = None
        self.visible = self.attentive = self.inattentive = 0.0
        self._pending = 0.0           # low-score time not yet long enough to count
        self._in_spell = False        # True once a low spell has lasted min_inattentive_seconds
        self._score_time = 0.0        # time-weighted sum for the average
        self._weighted = 0.0
        self._buckets: Dict[int, List[float]] = {}
        self.rolling: Deque[Tuple[float, float]] = deque()
        self.last_score: Optional[float] = None

    def add(self, t: float, score: Optional[float]):
        if self.first_t is None:
            self.first_t = t
        dt = 0.0 if self.last_t is None else min(max(t - self.last_t, 0.0), self.MAX_GAP)
        self.last_t = t
        if score is None:
            return
        self.last_score = score
        self.visible += dt
        self._score_time += dt
        self._weighted += dt * score
        self._buckets.setdefault(int((t - self.first_t) // self.cfg.timeline_bucket_seconds), []).append(score)
        self.rolling.append((t, score))
        while self.rolling and self.rolling[0][0] < t - self.cfg.cues.rolling_seconds:
            self.rolling.popleft()
        cutoff = self.th.cue_attentive_cutoff
        if cutoff is None:            # thresholds not chosen yet: just record, do not classify
            return
        if score >= cutoff:
            self.attentive += dt + self._pending
            self._pending = 0.0
            self._in_spell = False
        else:
            if self._in_spell:
                self.inattentive += dt
            else:
                self._pending += dt
                if self._pending >= (self.th.min_inattentive_seconds or 0.0):
                    self.inattentive += self._pending     # the whole spell counts, from its start
                    self._pending = 0.0
                    self._in_spell = True

    def live_score(self) -> Optional[float]:
        return None if not self.rolling else float(np.mean([s for _, s in self.rolling]))

    def summary(self) -> SessionSummary:
        total = 0.0 if self.first_t is None else (self.last_t - self.first_t)
        # leftover pending time was a short dip -> counts as attentive (it never reached the minimum length)
        attentive = self.attentive + self._pending
        avg = 0.0 if self._score_time <= 0 else self._weighted / self._score_time
        n = (max(self._buckets) + 1) if self._buckets else 0
        timeline = [float(np.mean(self._buckets[i])) if i in self._buckets else float("nan") for i in range(n)]
        dip = False
        cutoff = self.th.cue_attentive_cutoff
        if cutoff is not None and timeline:
            need = max(1, int(round(self.cfg.long_dip_minutes * 60 / self.cfg.timeline_bucket_seconds)))
            run = 0
            for v in timeline:
                run = run + 1 if (v == v and v < cutoff) else 0
                dip = dip or run >= need
        return SessionSummary(self.id, total, self.visible, attentive, self.inattentive, avg, timeline, dip)


class CentroidTracker:
    """Gives each face a stable number (1, 2, 3...) by matching it to the nearest face from the last frame."""
    def __init__(self, max_distance: float = 120.0, forget_after: float = 15.0):
        self.max_d, self.forget = max_distance, forget_after
        self.tracks: Dict[int, Tuple[float, float, float]] = {}
        self._next = 1

    def assign(self, centroids: List[Tuple[float, float]], t: float) -> List[int]:
        ids, used = [], set()
        for cx, cy in centroids:
            best, bd = None, self.max_d
            for tid, (x, y, _) in self.tracks.items():
                d = float(np.hypot(cx - x, cy - y))
                if tid not in used and d < bd:
                    best, bd = tid, d
            if best is None:
                best, self._next = self._next, self._next + 1
            used.add(best)
            self.tracks[best] = (cx, cy, t)
            ids.append(best)
        for tid in [k for k, (_, _, ts) in self.tracks.items() if t - ts > self.forget]:
            del self.tracks[tid]
        return ids


class SessionRecorder:
    def __init__(self, thresholds: Thresholds, settings: Settings):
        self.th, self.cfg = thresholds, settings
        self.students: Dict[str, StudentRecord] = {}

    def record(self, student_id: str, t: float, score: Optional[float]):
        if student_id not in self.students:
            self.students[student_id] = StudentRecord(student_id, self.th, self.cfg)
        self.students[student_id].add(t, score)

    def summaries(self) -> List[SessionSummary]:
        return [r.summary() for r in self.students.values()]
