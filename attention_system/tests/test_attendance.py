import pytest
from attention.attendance import AttendanceEngine, Recommendation, SessionSummary, ThresholdsNotConfiguredError
from attention.config import Settings, Thresholds
from attention.session import CentroidTracker, StudentRecord

TH = Thresholds(cue_attentive_cutoff=40, min_attentive_pct=80, absent_score_cutoff=50, min_inattentive_seconds=10)

def summ(att, vis=100, avg=70, tot=100, dip=False):
    return SessionSummary("S1", tot, vis, att, vis - att, avg, [], dip)

def test_engine_refuses_to_run_without_thresholds():
    with pytest.raises(ThresholdsNotConfiguredError): AttendanceEngine(Thresholds()).evaluate(summ(90))

def test_four_outcomes():
    e = AttendanceEngine(TH)
    assert e.evaluate(summ(90)).recommendation == Recommendation.PRESENT
    assert e.evaluate(summ(60, avg=55)).recommendation == Recommendation.PRESENT_LOW_ATTENTION
    assert e.evaluate(summ(40, avg=30)).recommendation == Recommendation.ABSENT_RECOMMENDED
    assert e.evaluate(summ(10, vis=20, tot=100)).recommendation == Recommendation.INSUFFICIENT_DATA

def test_exactly_at_the_limit_counts_as_present():
    assert AttendanceEngine(TH).evaluate(summ(80)).recommendation == Recommendation.PRESENT

def test_every_decision_needs_instructor_confirmation():
    assert AttendanceEngine(TH).evaluate(summ(90)).needs_confirmation is True

def test_from_survey_is_opt_in_and_complete():
    assert not Thresholds().is_complete()
    t = Thresholds.from_survey(); assert t.is_complete() and 0 < t.min_attentive_pct <= 100

def _feed(rec, start, seconds, score):
    for i in range(seconds): rec.add(start + i + 1, score)
    return start + seconds

def test_short_dips_do_not_count_but_long_ones_do():
    r = StudentRecord("S1", TH, Settings()); r.add(0, 90)
    t = _feed(r, 0, 60, 90); t = _feed(r, t, 5, 10); t = _feed(r, t, 30, 90)          # a 5 s dip (< 10 s) is forgiven
    s = r.summary(); assert s.inattentive_seconds == 0 and s.eap == pytest.approx(100)
    r2 = StudentRecord("S1", TH, Settings()); r2.add(0, 90)
    t = _feed(r2, 0, 60, 90); t = _feed(r2, t, 15, 10); t = _feed(r2, t, 30, 90)       # a 15 s dip counts, from its first second
    s2 = r2.summary(); assert s2.inattentive_seconds == pytest.approx(15) and s2.eap < 100

def test_not_visible_time_is_not_judged():
    r = StudentRecord("S1", TH, Settings()); r.add(0, 90); t = _feed(r, 0, 30, 90)
    for i in range(20): r.add(t + i + 1, None)                                          # student left the picture
    s = r.summary(); assert s.visible_pct < 70 and s.eap == pytest.approx(100)

def test_long_dip_flag_uses_survey_q9b_idea():
    cfg = Settings(); cfg.timeline_bucket_seconds = 60.0; cfg.long_dip_minutes = 3
    r = StudentRecord("S1", TH, cfg); r.add(0, 90); t = _feed(r, 0, 60, 90)
    t = _feed(r, t, 240, 10)
    assert r.summary().long_dip_flag is True

def test_tracker_keeps_ids_stable_and_adds_new_people():
    tr = CentroidTracker(max_distance=50)
    assert tr.assign([(100, 100), (300, 100)], 0) == [1, 2]
    assert tr.assign([(105, 102), (298, 99)], 1) == [1, 2]
    assert tr.assign([(105, 102), (298, 99), (500, 100)], 2) == [1, 2, 3]
