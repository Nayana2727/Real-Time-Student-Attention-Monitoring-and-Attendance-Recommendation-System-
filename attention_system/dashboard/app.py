"""
dashboard/app.py  -  the instructor dashboard.   Start it with:

    streamlit run dashboard/app.py

It must run on the SAME computer as the webcam (a website on the internet cannot reach your camera).
Steps in the page: 1) set/confirm thresholds  2) tick 'Start camera'  3) untick to finish  4) confirm or override
each recommendation  5) download CSV / PDF.   Nothing but numbers is saved.
"""
import sys, time, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import pandas as pd
import streamlit as st
from attention.config import Settings, Thresholds, WeightSettings, load_survey
from attention.pipeline import AttentionPipeline
from attention.report import write_reports
from attention.ui import draw_results

st.set_page_config(page_title="Attention Monitor", layout="wide")
st.title("Real-Time Student Attention Monitoring & Attendance Recommendation")
st.caption("The system only RECOMMENDS. You, the instructor, decide. No video is stored.")

# ------------------------------------------------------------------ sidebar: rules
sv = load_survey()
with st.sidebar:
    st.header("1. Rules for this class")
    st.write(f"Faculty survey: **{sv['n_respondents']}** respondents.")
    if st.button("Fill in the survey values"):
        t = Thresholds.from_survey()
        st.session_state.update(c=t.cue_attentive_cutoff, m=t.min_attentive_pct, a=t.absent_score_cutoff, s=t.min_inattentive_seconds)
    cutoff = st.number_input("Low-score cutoff (0-100)  [survey Q7a]", 0.0, 100.0, key="c", value=st.session_state.get("c"), placeholder="type a number")
    minpct = st.number_input("Minimum % attentive time for PRESENT  [Q6]", 0.0, 100.0, key="m", value=st.session_state.get("m"), placeholder="type a number")
    absent = st.number_input("Average score below this -> ABSENT recommended  [Q7b]", 0.0, 100.0, key="a", value=st.session_state.get("a"), placeholder="type a number")
    secs = st.number_input("Seconds a low spell must last to count  [Q8]", 0.0, 120.0, key="s", value=st.session_state.get("s"), placeholder="type a number")
    weights_mode = st.selectbox("How cues are weighted", ["survey", "equal"], help="'survey' uses how often faculty picked each cue.")
    fusion_mode = st.selectbox("How cues are combined", ["gated", "weighted"], help="'gated': facing the screen is required to score high.")
    confirmed = st.checkbox("I have checked these rules and confirm them for this class")
    st.caption("Survey Q10: faculty gave 'the system must let me override' "
               f"{sv['q10'].get('The system should always allow me to override its recommendation', '')}/5 on average.")

th = Thresholds(cutoff, minpct, absent, secs)
ready = confirmed and th.is_complete()

# ------------------------------------------------------------------ main area
left, right = st.columns([3, 2])
with left:
    run = st.checkbox("2. Start camera", disabled=not ready, key="run")
    if not ready:
        st.info("Fill in the four rules in the sidebar and tick the confirmation box to enable the camera.")
    frame_slot = st.empty()
with right:
    st.subheader("Live scores")
    live_slot = st.empty()
    status_slot = st.empty()

if run and ready:
    cfg = Settings(weights=WeightSettings(mode=weights_mode))
    cfg.cues.fusion = fusion_mode
    pipe = AttentionPipeline(cfg, th)
    status_slot.info("Expression model: " + ("loaded" if pipe.expression.is_ready() else "not found - running without it"))
    cap = cv2.VideoCapture(cfg.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.frame_width); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.frame_height)
    if not cap.isOpened():
        st.error("Could not open the camera. Close other apps that use it.")
    else:
        last_table = 0.0
        while st.session_state.get("run", False):
            ok, frame = cap.read()
            if not ok:
                break
            res = pipe.process_frame(frame, time.time())
            frame_slot.image(cv2.cvtColor(draw_results(frame, res, th.cue_attentive_cutoff), cv2.COLOR_BGR2RGB), channels="RGB")
            if time.time() - last_table > 1:
                live_slot.dataframe(pd.DataFrame([{"Student": k, "Live score": None if v is None else round(v)} for k, v in pipe.live_scores().items()]),
                                    use_container_width=True)
                last_table = time.time()
        cap.release()
    st.session_state["decisions"] = pipe.finish()
    st.session_state["summaries"] = pipe.recorder.summaries()
    st.session_state["th"] = th
    pipe.close()

# ------------------------------------------------------------------ end of class
if st.session_state.get("decisions"):
    st.header("3. Confirm or override")
    decisions, summaries = st.session_state["decisions"], st.session_state["summaries"]
    for d in decisions:
        c1, c2, c3 = st.columns([2, 3, 3])
        c1.markdown(f"**{d.student_id}**  \n{d.recommendation.value}  \nEAP {d.eap}% | avg {d.average_score}")
        c2.caption(d.explanation)
        choice = c3.selectbox("Your decision", ["(accept recommendation)", "PRESENT", "ABSENT"], key=f"dec_{d.student_id}")
        note = c3.text_input("Reason (optional)", key=f"note_{d.student_id}")
        d.instructor_decision = d.recommendation.value if choice.startswith("(") else choice
        d.instructor_note = note
    if st.button("4. Save reports"):
        out = ROOT / "reports" / time.strftime("session_%Y%m%d_%H%M%S")
        files = write_reports(out, decisions, summaries, st.session_state["th"])
        st.success(f"Saved to {out}")
        for k, p in files.items():
            st.download_button(f"Download {k.upper()}", Path(p).read_bytes(), file_name=Path(p).name)
