"""
scripts/collect_attention_data.py  -  collect YOUR OWN labelled attention data (features only, no pictures).

Why: public datasets with attention labels (like DAiSEE) need a request form. This tool lets your team create a small
dataset in a few minutes. Ask a volunteer to sit in front of the camera and press keys while a second person watches:

    1 = attentive   2 = distracted (looking away / phone)   3 = drowsy (eyes closing, yawning, nodding)   0 = pause
    Q = quit

What is saved: numbers only (head angles, eye/mouth ratios, expression probabilities, the label). NO images or video.
The volunteer must agree first - the script asks them to type YES.
"""
import argparse, csv, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import cv2
from attention.config import Settings
from attention.expression import ExpressionClassifier
from attention.face_tracker import FaceLandmarker, bbox_of
from attention.geometry import gaze_ratio, head_pose, mean_ear, mouth_aspect_ratio

COLS = ["participant", "t", "label", "yaw", "pitch", "gaze", "ear", "mar", "p_engaged", "p_neutral", "p_negative"]
LABELS = {ord("1"): "attentive", ord("2"): "distracted", ord("3"): "drowsy", ord("0"): ""}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--participant", required=True, help="a code like P01 (NOT a real name)")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--hz", type=float, default=10.0, help="rows saved per second")
    a = ap.parse_args()
    print("This records numbers measured from the camera picture (no images are saved).")
    if input("The volunteer agrees to take part. Type YES to continue: ").strip() != "YES":
        sys.exit("Stopped: consent not given.")
    out = ROOT / "data" / "collected" / f"{a.participant}.csv"; out.parent.mkdir(parents=True, exist_ok=True)
    cfg = Settings(); fl = FaceLandmarker(1); expr = ExpressionClassifier(cfg.expression)
    cap = cv2.VideoCapture(a.camera); label, last, n = "", 0.0, 0
    t0 = time.time()
    with open(out, "a", newline="") as f:
        w = csv.writer(f)
        if out.stat().st_size == 0:
            w.writerow(COLS)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            now = time.time(); faces = fl.process(frame); h, wd = frame.shape[:2]
            if faces and label and now - last >= 1.0 / a.hz:
                p = faces[0]; hp = head_pose(p, wd, h); probs = None
                if expr.is_ready():
                    x0, y0, x1, y1 = bbox_of(p, wd, h); probs = expr.predict(frame[y0:y1, x0:x1])
                w.writerow([a.participant, round(now - t0, 3), label, None if hp is None else round(hp.yaw, 2),
                            None if hp is None else round(hp.pitch, 2), gaze_ratio(p), round(mean_ear(p), 4), round(mouth_aspect_ratio(p), 4),
                            *(round(probs[c], 4) if probs else "" for c in ("engaged", "neutral", "negative"))])
                last, n = now, n + 1
            cv2.putText(frame, f"label: {label or 'PAUSED'} | rows: {n} | 1/2/3 label, 0 pause, Q quit", (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            cv2.imshow("Collect data", frame)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), ord("Q")):
                break
            if k in LABELS:
                label = LABELS[k]
    cap.release(); cv2.destroyAllWindows(); fl.close(); print("saved", n, "rows to", out)


if __name__ == "__main__":
    main()
