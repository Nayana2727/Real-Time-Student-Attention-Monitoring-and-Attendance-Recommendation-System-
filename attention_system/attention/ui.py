"""attention/ui.py - drawing the overlay on the camera picture (shared by the demo and the dashboard)."""
from __future__ import annotations
import cv2


def draw_results(frame, results, th_cutoff=None):
    for r in results:
        x0, y0, x1, y1 = r.bbox
        if r.calibrating:
            color, label = (200, 200, 0), f"{r.student_id}: calibrating - look at the screen"
        elif r.score is None:
            color, label = (128, 128, 128), f"{r.student_id}: no data"
        else:
            ok = th_cutoff is None or r.score >= th_cutoff
            color = (0, 180, 0) if ok else (0, 0, 220)
            label = f"{r.student_id}: {r.score:.0f}"
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
        cv2.putText(frame, label, (x0, max(15, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return frame
