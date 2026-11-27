# Real-Time Student Attention Monitoring and Attendance Recommendation System
Computer vision + deep learning. Runs on a normal laptop CPU. Stores **no video**. Only **recommends** attendance - the instructor decides.

Department of AI & ML, AIEMS, Bidadi (VTU). Design is grounded in a survey of 106 engineering faculty (`survey/`).

## What is in here
| Folder | What it is |
|---|---|
| `attention/` | The system: face landmarks, head pose, gaze, eyes, mouth, expression, scoring, attendance rules, reports |
| `scripts/` | Programs you run: prepare data, train the model, live demo, collect data, compare methods |
| `dashboard/` | Streamlit instructor dashboard (`streamlit run dashboard/app.py`) |
| `survey/` | `analyze_survey.py` + `survey_findings.json` (every survey number used by the system) |
| `models/` | Trained expression model (`.tflite` used by the system, `.keras` for retraining) |
| `reports/` | Measured results of the trained model (metrics, confusion matrix, curves) |
| `tests/` | 34 automatic checks (`python -m pytest tests -q`) |
| `docs/` | Model card and limitations |

## Quick start (details: the Word guide "STEP_BY_STEP_GUIDE.docx")
```
python -m venv venv                 # Windows: venv\Scripts\activate    Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
python -m pytest tests -q           # health check - should say 34 passed
python scripts/run_demo.py --use-survey-thresholds
```

## How the score works (one paragraph)
Each cue gives 0 (bad) to 1 (good): head pose, gaze, eyes-closed (PERCLOS), yawning, nodding, expression, phone (optional).
They are averaged with weights taken from how often faculty picked each cue in the survey (`survey/survey_findings.json`).
Because facing the screen is *necessary* for attention, the average is then multiplied by the head-pose and phone cues ("gated").
EAP = share of visible time with an attentive score. Attendance rules (all four numbers are set by you, or loaded **on purpose** with
`--use-survey-thresholds`): see `attention/attendance.py`.

## Honest limits - read `docs/LIMITATIONS.md` before you present or publish.
