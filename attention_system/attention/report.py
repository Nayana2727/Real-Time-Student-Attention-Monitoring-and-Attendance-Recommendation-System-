"""attention/report.py - writes the end-of-class files: CSV, JSON decision log and a PDF with a timeline graph.
Only numbers are saved. No images or video are ever written."""
from __future__ import annotations
import csv, json, math
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List
from .attendance import Decision, SessionSummary
from .config import Thresholds


def write_reports(out_dir, decisions: List[Decision], summaries: List[SessionSummary], th: Thresholds,
                  title: str = "Attention session report") -> dict:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    by_id = {s.student_id: s for s in summaries}

    csv_path = out / "attendance_recommendations.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["student", "recommendation", "EAP_percent", "average_score", "face_visible_percent",
                    "minutes", "instructor_decision", "instructor_note", "explanation"])
        for d in decisions:
            s = by_id[d.student_id]
            w.writerow([d.student_id, d.recommendation.value, d.eap, d.average_score, d.visible_pct,
                        round(s.total_seconds / 60, 1), d.instructor_decision or "", d.instructor_note, d.explanation])

    json_path = out / "decision_log.json"
    json_path.write_text(json.dumps({
        "created": stamp, "thresholds_used": {k: v for k, v in th.__dict__.items()},
        "decisions": [{**{k: (v.value if hasattr(v, "value") else v) for k, v in asdict(d).items()}} for d in decisions],
    }, indent=2, default=str))

    pdf_path = out / "session_report.pdf"
    _pdf(pdf_path, title, stamp, decisions, by_id, th)
    return {"csv": csv_path, "json": json_path, "pdf": pdf_path}


def _pdf(path, title, stamp, decisions, by_id, th):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    st = getSampleStyleSheet(); doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=1.8*cm, rightMargin=1.8*cm,
                                                        topMargin=1.5*cm, bottomMargin=1.5*cm)
    els = [Paragraph(title, st["Title"]), Paragraph(f"Generated {stamp}. This is a RECOMMENDATION report; "
           "the instructor confirms or overrides every line. No video was stored.", st["Normal"]), Spacer(1, 8)]
    els.append(Paragraph("Thresholds used in this session", st["Heading3"]))
    els.append(Paragraph(
        f"Low-score cutoff (per moment): {th.cue_attentive_cutoff} | Minimum attentive time for PRESENT: {th.min_attentive_pct}% | "
        f"Session score below which ABSENT is recommended: {th.absent_score_cutoff} | Minimum length of an inattentive spell: "
        f"{th.min_inattentive_seconds} s", st["Normal"]))
    els.append(Spacer(1, 8))
    rows = [["Student", "Recommendation", "EAP %", "Avg score", "Visible %", "Instructor"]]
    for d in decisions:
        rows.append([d.student_id, d.recommendation.value, f"{d.eap:.1f}", f"{d.average_score:.1f}",
                     f"{d.visible_pct:.0f}", d.instructor_decision or "(pending)"])
    t = Table(rows, repeatRows=1, colWidths=[2.6*cm, 5.0*cm, 1.8*cm, 2.2*cm, 2.2*cm, 3.2*cm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E79")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                           ("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                           ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F7FC")])]))
    els += [t, Spacer(1, 10)]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    for d in decisions:
        tl = by_id[d.student_id].timeline
        if tl: ax.plot([i + 1 for i in range(len(tl))], [v if v == v else math.nan for v in tl], marker="o", ms=3, label=d.student_id)
    if th.cue_attentive_cutoff is not None:
        ax.axhline(th.cue_attentive_cutoff, ls="--", c="red", lw=1, label="low-score cutoff")
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(0, 100); ax.set_xlabel("minute of class"); ax.set_ylabel("attention score"); ax.legend(fontsize=7)
    ax.set_title("Attention score through the class (one point per minute)")
    fig.tight_layout(); png = Path(path).with_suffix(".timeline.png"); fig.savefig(png, dpi=130); plt.close(fig)
    els.append(Image(str(png), width=16.5*cm, height=7.3*cm))
    for d in decisions:
        els.append(Paragraph(f"<b>{d.student_id}</b>: {d.explanation}", st["Normal"]))
    doc.build(els)
