import json, pytest
from pathlib import Path
J = Path(__file__).resolve().parent.parent / "survey" / "survey_findings.json"

def test_findings_file_is_consistent():
    r = json.loads(J.read_text())
    assert r["n_respondents"] > 0
    assert sum(r["derived_weights"].values()) == pytest.approx(1.0, abs=0.01)
    assert set(r["derived_weights"]) == {"head_pose", "gaze", "expression", "blink", "mouth", "nodding", "phone"}
    assert 0 < r["q6"]["median"] <= 100 and 0 < r["q7b"]["median"] <= 100
    assert r["q4b_meta"]["exactly_3"] + r["q4b_meta"]["more_than_3"] + r["q4b_meta"]["fewer_than_3"] <= r["n_respondents"]
