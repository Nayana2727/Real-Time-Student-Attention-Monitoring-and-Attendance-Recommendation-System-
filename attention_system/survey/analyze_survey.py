"""
survey/analyze_survey.py
Reads the faculty-survey spreadsheet and writes survey/survey_findings.json.
Every number used by the system (feature weights, default thresholds) comes from here,
so nothing is typed in by hand.

Usage (from the project folder):
    python survey/analyze_survey.py path/to/VTU_Faculty_Survey_Responses.xlsx
"""
import sys, re, json
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd

OUT = Path(__file__).with_name("survey_findings.json")

# keyword -> canonical feature key (works for BOTH the Google-Form export "A; B; C"
# and the printed-form format "1. A, 3. B, 5. C")
FEATURES = {
    "head_pose": "head pose", "gaze": "gaze direction", "expression": "facial expression",
    "blink": "eye blink", "mouth": "mouth movements", "posture": "body posture",
    "nodding": "head nodding", "phone": "phone or object",
    "duration": "duration of inattention", "frequency": "frequency of inattention",
}
Q4A_KEYS = list(FEATURES)  # the 10 Q4a columns appear in this same order in the sheet

def leading_int(v):
    m = re.match(r"\s*(\d+)", str(v)); return int(m.group(1)) if m else np.nan

def below_value(v):
    m = re.search(r"Below (\d+)", str(v)); return int(m.group(1)) if m else None

def pct_value(v):
    for n in (90, 85, 80, 75, 70, 60, 50):
        if f"{n}%" in str(v): return n
    return None

def main(xlsx):
    df = pd.read_excel(xlsx); N = len(df); cols = list(df.columns)
    def first(prefix):
        hits = [c for c in cols if str(c).startswith(prefix)]
        if not hits: raise KeyError(prefix)
        return hits[0]
    res = {"n_respondents": N}

    q4a_cols = [c for c in cols if str(c).startswith("Q4a.")]
    assert len(q4a_cols) == 10, "expected 10 Q4a columns"
    res["q4a"] = {}
    for key, c in zip(Q4A_KEYS, q4a_cols):
        s = df[c].apply(leading_int).dropna()
        res["q4a"][key] = {"mean": round(float(s.mean()), 2), "pct_rated_4_or_5": round(100*float((s >= 4).mean()), 1)}

    cnt = Counter(); sizes = []
    for v in df[first("Q4b.")].dropna():
        t = str(v).lower(); hit = [k for k, w in FEATURES.items() if w in t]
        sizes.append(len(hit)); cnt.update(hit)
    res["q4b"] = {k: {"count": cnt[k], "pct": round(100*cnt[k]/N, 1)} for k in FEATURES}
    res["q4b_meta"] = {"exactly_3": sum(s == 3 for s in sizes), "more_than_3": sum(s > 3 for s in sizes),
                       "fewer_than_3": sum(s < 3 for s in sizes)}

    q6 = df[first("Q6.")].apply(pct_value).dropna()
    res["q6"] = {"median": float(q6.median()), "mean": round(float(q6.mean()), 1), "n": int(len(q6)),
                 "counts": {int(k): int(v) for k, v in q6.value_counts().sort_index().items()}}
    q7a = df[first("Q7a.")].apply(below_value).dropna()
    res["q7a"] = {"median": float(q7a.median()), "mode": int(q7a.mode().iloc[0]), "n": int(len(q7a)),
                  "counts": {int(k): int(v) for k, v in q7a.value_counts().sort_index().items()}}
    raw7b = df[first("Q7b. Overall")]
    q7b = raw7b.apply(below_value).dropna()
    non_num = int(raw7b.notna().sum() - len(q7b))
    res["q7b"] = {"median": float(q7b.median()), "mean": round(float(q7b.mean()), 1), "n_numeric": int(len(q7b)),
                  "n_non_numeric": non_num, "pct_non_numeric": round(100*non_num/N, 1),
                  "counts": {int(k): int(v) for k, v in q7b.value_counts().sort_index().items()}}

    q8 = df[first("Q8.")].astype(str)
    def secs(t):
        if "1 full minute" in t: return 60
        m = re.search(r"At least (\d+) consecutive", t); return int(m.group(1)) if m else None
    q8n = q8.apply(secs).dropna()
    res["q8"] = {"mode_seconds": int(q8n.mode().iloc[0]), "median_seconds": float(q8n.median()),
                 "counts": {int(k): int(v) for k, v in q8n.value_counts().sort_index().items()},
                 "pct_frequency_matters_more": round(100*float(q8.str.contains("frequency").mean()), 1)}

    q10 = [c for c in cols if str(c).startswith("Q10.")]
    res["q10"] = {re.sub(r"Q10\. | \(1-5\)", "", c): round(float(df[c].apply(leading_int).mean()), 2) for c in q10}
    q11 = df[first("Q11.")].astype(str)
    res["q11"] = {
        "recommend_then_instructor_confirms_pct": round(100*float(q11.str.contains("confirm or override").mean()), 1),
        "fully_automated_pct": round(100*float(q11.str.contains("automatically marks").mean()), 1),
        "soft_alert_pct": round(100*float(q11.str.contains("soft alert").mean()), 1)}
    res["q4c"] = {k: int(v) for k, v in df[first("Q4c.")].value_counts().items()}
    res["q9b"] = {k: int(v) for k, v in df[first("Q9b.")].value_counts().items()}

    implemented = ["head_pose", "gaze", "expression", "blink", "mouth", "nodding", "phone"]
    raw_w = {k: res["q4b"][k]["pct"] for k in implemented}; tot = sum(raw_w.values())
    res["derived_weights"] = {k: round(v/tot, 4) for k, v in raw_w.items()}
    res["derived_weights_note"] = ("Share of respondents choosing each cue in their top picks (Q4b), normalised over the "
        "cues this system implements. Body posture, duration and frequency are not separate cues here "
        "(duration/frequency are handled by the windowing logic).")
    OUT.write_text(json.dumps(res, indent=2)); print("wrote", OUT)
    return res

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/VTU_Faculty_Survey_Responses.xlsx")
