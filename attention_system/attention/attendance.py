"""
attention/attendance.py
Turns a finished session into an attendance RECOMMENDATION.

Rules (all numbers come from Thresholds - nothing is hidden here):
  * EAP (Effective Attention Percentage) = time the student was attentive / time the face was visible x 100
  * A spell of low score only counts as "inattentive" if it lasts at least min_inattentive_seconds (survey Q8)
  * "Low score" means below cue_attentive_cutoff (survey Q7a)
  * EAP >= min_attentive_pct                       -> PRESENT                      (survey Q6)
  * else session average < absent_score_cutoff     -> ABSENT_RECOMMENDED           (survey Q7b)
  * else                                           -> PRESENT_LOW_ATTENTION
  * face seen for less than min_visible_pct        -> INSUFFICIENT_DATA (camera/seat problem, not inattention)
The result is ALWAYS a recommendation. The instructor confirms or overrides (survey Q10/Q11).
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional
from .config import Thresholds, Settings


class ThresholdsNotConfiguredError(Exception):
    pass


class Recommendation(str, Enum):
    PRESENT = "PRESENT"
    PRESENT_LOW_ATTENTION = "PRESENT - LOW ATTENTION"
    ABSENT_RECOMMENDED = "ABSENT - RECOMMENDED"
    INSUFFICIENT_DATA = "INSUFFICIENT DATA"


@dataclass
class SessionSummary:
    student_id: str
    total_seconds: float
    visible_seconds: float
    attentive_seconds: float
    inattentive_seconds: float
    average_score: float
    timeline: List[float] = field(default_factory=list)     # average score per bucket (default 1 minute)
    long_dip_flag: bool = False                             # score stayed low for >= long_dip_minutes (survey Q9b)

    @property
    def visible_pct(self) -> float:
        return 0.0 if self.total_seconds <= 0 else 100.0 * self.visible_seconds / self.total_seconds

    @property
    def eap(self) -> float:
        return 0.0 if self.visible_seconds <= 0 else 100.0 * self.attentive_seconds / self.visible_seconds


@dataclass
class Decision:
    student_id: str
    recommendation: Recommendation
    eap: float
    average_score: float
    visible_pct: float
    explanation: str
    needs_confirmation: bool = True
    instructor_decision: Optional[str] = None     # filled in when the instructor confirms/overrides
    instructor_note: str = ""


class AttendanceEngine:
    def __init__(self, thresholds: Thresholds):
        self.th = thresholds

    def evaluate(self, s: SessionSummary) -> Decision:
        th = self.th
        if not th.is_complete():
            raise ThresholdsNotConfiguredError(
                "Attendance thresholds are not set. Fill config/thresholds.json, or call "
                "Thresholds.from_survey() on purpose, then confirm them for this session.")
        eap, avg, vis = s.eap, s.average_score, s.visible_pct
        if vis < th.min_visible_pct:
            rec, why = Recommendation.INSUFFICIENT_DATA, (
                f"Face visible only {vis:.0f}% of the session (needs {th.min_visible_pct:.0f}%). "
                "Check camera angle/seating before judging attention.")
        elif eap >= th.min_attentive_pct:
            rec, why = Recommendation.PRESENT, f"EAP {eap:.1f}% meets the {th.min_attentive_pct:.0f}% requirement."
        elif avg < th.absent_score_cutoff:
            rec, why = Recommendation.ABSENT_RECOMMENDED, (
                f"EAP {eap:.1f}% is below {th.min_attentive_pct:.0f}% and the average score {avg:.1f}/100 is below "
                f"{th.absent_score_cutoff:.0f}/100. Recommendation only - the instructor decides.")
        else:
            rec, why = Recommendation.PRESENT_LOW_ATTENTION, (
                f"EAP {eap:.1f}% is below {th.min_attentive_pct:.0f}% but the average score {avg:.1f}/100 is not below "
                f"{th.absent_score_cutoff:.0f}/100. Flagged for the instructor's awareness.")
        if s.long_dip_flag:
            why += " Attention stayed low for a long stretch (see timeline)."
        return Decision(s.student_id, rec, round(eap, 1), round(avg, 1), round(vis, 1), why,
                        needs_confirmation=th.require_instructor_confirmation)
