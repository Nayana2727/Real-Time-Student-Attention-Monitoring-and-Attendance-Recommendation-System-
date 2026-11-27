"""
attention/config.py
All settings live here. Two ideas to remember:

1. NOTHING about attendance is decided silently. The attendance cut-offs start as None.
   You must either type your own numbers, or explicitly ask for the numbers that the
   faculty survey suggested (Thresholds.from_survey()).
2. Survey numbers are READ from survey/survey_findings.json (made by survey/analyze_survey.py),
   never typed into the code.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

ROOT = Path(__file__).resolve().parent.parent
SURVEY_JSON = ROOT / "survey" / "survey_findings.json"
MODEL_DIR = ROOT / "models"


def load_survey() -> dict:
    if not SURVEY_JSON.exists():
        raise FileNotFoundError(
            f"{SURVEY_JSON} not found. Run:  python survey/analyze_survey.py <your survey .xlsx>")
    return json.loads(SURVEY_JSON.read_text())


@dataclass
class Thresholds:
    """Attendance rules. All start empty on purpose."""
    cue_attentive_cutoff: Optional[float] = None       # 0-100: score below this = inattentive      (survey Q7a)
    min_attentive_pct: Optional[float] = None          # % of session attentive needed for PRESENT   (survey Q6)
    absent_score_cutoff: Optional[float] = None        # session average below this -> ABSENT rec.   (survey Q7b)
    min_inattentive_seconds: Optional[float] = None    # an inattentive spell must last this long    (survey Q8)
    min_visible_pct: float = 50.0                      # face must be seen this % of the time, else "not enough data"
    require_instructor_confirmation: bool = True       # always True in this project (survey Q10/Q11)

    def is_complete(self) -> bool:
        return None not in (self.cue_attentive_cutoff, self.min_attentive_pct,
                            self.absent_score_cutoff, self.min_inattentive_seconds)

    @classmethod
    def from_survey(cls) -> "Thresholds":
        """OPT-IN: medians/modes of the 106-faculty survey. Call this on purpose; nothing calls it for you."""
        s = load_survey()
        return cls(cue_attentive_cutoff=float(s["q7a"]["median"]),
                   min_attentive_pct=float(s["q6"]["median"]),
                   absent_score_cutoff=float(s["q7b"]["median"]),
                   min_inattentive_seconds=float(s["q8"]["mode_seconds"]))


@dataclass
class CueSettings:
    """Numbers that turn raw face measurements into 'good/bad' for each cue.
    These are engineering choices from the literature, NOT survey results. Tune them on your own camera."""
    yaw_limit_deg: float = 30.0          # head turned more than this = looking away (RT-GENE, Fischer 2018)
    pitch_limit_deg: float = 25.0        # head tilted up/down more than this = looking away
    head_pose_zero_factor: float = 1.5   # head cue falls from 1 (at the limit) to 0 at limit x this factor (30 deg -> 45 deg)
    gaze_ok_low: float = 0.30            # iris position (0..1) inside this band = looking forward
    gaze_ok_high: float = 0.70
    ear_closed: float = 0.20             # eye-aspect-ratio below this = eye closed
    perclos_bad: float = 0.30            # fraction of recent time eyes closed that counts as fully drowsy
    mar_yawn: float = 0.60               # mouth-aspect-ratio above this = mouth wide open
    yawn_min_seconds: float = 1.2        # ...for at least this long = a yawn
    nod_amplitude_deg: float = 10.0      # head pitch swing that counts as a nod
    nod_window_seconds: float = 3.0
    nod_min_swings: int = 2
    rolling_seconds: float = 5.0         # smoothing window for the live score
    calibration_seconds: float = 3.0     # look at the screen normally for this long when the session starts
    # How the cues are combined:
    #   "gated"    (default) survey-weighted average, MULTIPLIED by the head-pose and phone cues. Facing the screen is a
    #              necessary condition: a student turned away or on a phone cannot score high with wide-open eyes.
    #   "weighted" plain survey-weighted average only. (Kept for the comparative study: a student turned fully away
    #              can still score ~88 because only one cue, worth ~19%, drops.)
    fusion: str = "gated"
    gate_cues: tuple = ("head_pose", "phone")


@dataclass
class ExpressionSettings:
    model_path: Path = MODEL_DIR / "expression_model.tflite"
    classes: tuple = ("engaged", "neutral", "negative")
    input_size: int = 48
    # How attentive each expression class suggests a student is (0..1). A design choice - tune it.
    class_attentiveness: Dict[str, float] = field(default_factory=lambda: {
        "engaged": 1.0, "neutral": 0.8, "negative": 0.5})


@dataclass
class WeightSettings:
    mode: str = "survey"                  # "survey" | "equal" | "custom"
    custom: Optional[Dict[str, float]] = None

    def weights(self, available: Optional[list] = None) -> Dict[str, float]:
        cues = ["head_pose", "gaze", "expression", "blink", "mouth", "nodding", "phone"]
        if self.mode == "equal":
            w = {c: 1.0 for c in cues}
        elif self.mode == "custom" and self.custom:
            w = {c: float(self.custom.get(c, 0.0)) for c in cues}
        else:
            sw = load_survey()["derived_weights"]
            w = {c: float(sw.get(c, 0.0)) for c in cues}
        if available is not None:
            w = {c: v for c, v in w.items() if c in available}
        tot = sum(w.values())
        return {c: v / tot for c, v in w.items()} if tot > 0 else w


@dataclass
class Settings:
    cues: CueSettings = field(default_factory=CueSettings)
    expression: ExpressionSettings = field(default_factory=ExpressionSettings)
    weights: WeightSettings = field(default_factory=WeightSettings)
    max_faces: int = 5
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    timeline_bucket_seconds: float = 60.0   # one point per minute on the report graph
    long_dip_minutes: float = 10.0          # survey Q9b: flag when attention stays low this long
