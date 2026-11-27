import pytest
from attention.config import CueSettings, ExpressionSettings, WeightSettings
from attention.scoring import CueScorer, FrameMeasurements

def scorer(mode="survey"): return CueScorer(CueSettings(), ExpressionSettings(), WeightSettings(mode=mode))

def test_weights_sum_to_one_for_every_mode_and_subset():
    for mode in ("survey", "equal"):
        w = WeightSettings(mode=mode)
        assert sum(w.weights().values()) == pytest.approx(1.0)
        assert sum(w.weights(available=["head_pose", "gaze"]).values()) == pytest.approx(1.0)

def test_survey_weights_follow_the_survey():
    w = WeightSettings(mode="survey").weights()
    assert max(w, key=w.get) == "head_pose"                                         # faculty ranked head pose first (Q4b)

def test_attentive_frame_scores_high_and_turned_away_scores_low():
    s = scorer(); s.finish_calibration()
    good = s.score(FrameMeasurements(t=1, yaw=2, pitch=3, gaze=0.5, ear=0.3, mar=0.1))[0]
    s2 = scorer(); s2.finish_calibration()
    bad = s2.score(FrameMeasurements(t=1, yaw=70, pitch=3, gaze=0.5, ear=0.3, mar=0.1))[0]
    assert good > 95 and bad < good - 15

def test_missing_cues_are_ignored_not_counted_as_zero():
    s = scorer(); s.finish_calibration()
    score, cues = s.score(FrameMeasurements(t=1, yaw=0, pitch=0))                      # only head pose + nodding measurable
    assert cues["gaze"] is None and score > 95
    assert scorer().score(FrameMeasurements(t=1))[0] is None                          # nothing measurable -> None

def test_calibration_removes_camera_angle_offset():
    s = scorer()
    for i in range(10): s.add_calibration_sample(FrameMeasurements(t=i * .1, yaw=5, pitch=20))
    s.finish_calibration()
    score = s.score(FrameMeasurements(t=2, yaw=5, pitch=20, gaze=.5, ear=.3, mar=.1))[0]
    assert score > 95                                                                  # a laptop camera below eye level is fine

def test_closed_eyes_lower_score_over_time():
    s = scorer(); s.finish_calibration(); last = None
    for i in range(60): last = s.score(FrameMeasurements(t=i * .1, yaw=0, pitch=0, gaze=.5, ear=0.05, mar=.1))
    assert last[1]["blink"] < 0.05

def test_yawn_needs_to_last():
    s = scorer(); s.finish_calibration()
    brief = s.score(FrameMeasurements(t=0.0, yaw=0, pitch=0, mar=0.9))[1]["mouth"]
    long_ = None
    for i in range(1, 20): long_ = s.score(FrameMeasurements(t=i * .1, yaw=0, pitch=0, mar=0.9))[1]["mouth"]
    assert brief == 1.0 and long_ == 0.0

def test_expression_cue_uses_class_attentiveness():
    s = scorer(); s.finish_calibration()
    e = s.score(FrameMeasurements(t=1, expression_probs={"engaged": 1.0, "neutral": 0.0, "negative": 0.0}))[1]["expression"]
    n = s.score(FrameMeasurements(t=2, expression_probs={"engaged": 0.0, "neutral": 0.0, "negative": 1.0}))[1]["expression"]
    assert e > n

def test_phone_cue():
    s = scorer(); s.finish_calibration()
    assert s.score(FrameMeasurements(t=1, phone=True))[1]["phone"] == 0.0


def test_gate_catches_a_student_who_is_fully_turned_away():
    """Regression test for a real design flaw: with a plain weighted average, a student turned 55 degrees away
    still scored ~88 because the eyes/mouth/expression cues looked fine."""
    m = FrameMeasurements(t=1, yaw=55, pitch=0, gaze=0.5, ear=0.3, mar=0.1,
                          expression_probs={"engaged": 1.0, "neutral": 0.0, "negative": 0.0})
    gated = CueScorer(CueSettings(fusion="gated"), ExpressionSettings(), WeightSettings()); gated.finish_calibration()
    plain = CueScorer(CueSettings(fusion="weighted"), ExpressionSettings(), WeightSettings()); plain.finish_calibration()
    assert plain.score(m)[0] > 75            # the flaw (kept as an option for the comparative study)
    assert gated.score(m)[0] < 40            # the fix: below the survey's low-score cutoff

def test_gate_does_not_hurt_a_student_facing_the_screen():
    m = FrameMeasurements(t=1, yaw=3, pitch=2, gaze=0.5, ear=0.3, mar=0.1)
    g = CueScorer(CueSettings(fusion="gated"), ExpressionSettings(), WeightSettings()); g.finish_calibration()
    assert g.score(m)[0] > 95


def test_head_turned_forty_degrees_for_a_while_is_flagged():
    """Regression: with a 2x fall-off a student at 40 degrees still scored ~50 (above the survey's cutoff of 40)."""
    g = CueScorer(CueSettings(fusion="gated"), ExpressionSettings(), WeightSettings()); g.finish_calibration()
    m = FrameMeasurements(t=1, yaw=40, pitch=0, gaze=0.5, ear=0.3, mar=0.1)
    assert g.score(m)[0] < 40
    ok = FrameMeasurements(t=1, yaw=25, pitch=0, gaze=0.5, ear=0.3, mar=0.1)      # inside the 30 degree limit
    assert g.score(ok)[0] > 95
