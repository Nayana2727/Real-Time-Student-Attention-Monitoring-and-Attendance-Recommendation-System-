"""
scripts/run_demo.py  -  the live demo.

    python scripts/run_demo.py --use-survey-thresholds            # webcam, survey values as the rules
    python scripts/run_demo.py --video my_class.mp4 --use-survey-thresholds
    python scripts/run_demo.py --thresholds config/thresholds.json  # your own numbers

Press  Q  in the video window to finish. Then you confirm or override each recommendation in the terminal.
Nothing is recorded: no frames or video are saved - only the numbers in the report.
"""
import argparse, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
from attention.config import Settings, Thresholds, WeightSettings
from attention.pipeline import AttentionPipeline
from attention.report import write_reports
from attention.ui import draw_results


def load_thresholds(args) -> Thresholds:
    if args.use_survey_thresholds:
        th = Thresholds.from_survey()
        print("Using thresholds from the faculty survey:", {k: v for k, v in th.__dict__.items()})
        return th
    data = json.loads(Path(args.thresholds).read_text())
    data.pop("_help", None)
    return Thresholds(**data)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", help="video file instead of the webcam")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--thresholds", default=str(ROOT / "config" / "thresholds.json"))
    ap.add_argument("--use-survey-thresholds", action="store_true")
    ap.add_argument("--weights", choices=["survey", "equal"], default="survey")
    ap.add_argument("--fusion", choices=["gated", "weighted"], default="gated", help="how cues are combined (see config.py)")
    ap.add_argument("--phone", action="store_true", help="turn on phone detection (needs: pip install ultralytics)")
    ap.add_argument("--seconds", type=float, default=None, help="stop automatically after this many seconds")
    ap.add_argument("--no-window", action="store_true", help="do not open a video window (for servers/tests)")
    ap.add_argument("--no-prompt", action="store_true", help="skip the confirm/override questions")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    th = load_thresholds(a)
    if not th.is_complete():
        sys.exit("Thresholds are empty. Edit config/thresholds.json or add --use-survey-thresholds.")
    cfg = Settings(weights=WeightSettings(mode=a.weights), camera_index=a.camera)
    cfg.cues.fusion = a.fusion
    phone = None
    if a.phone:
        from attention.phone_detector import PhoneDetector
        phone = PhoneDetector()
    pipe = AttentionPipeline(cfg, th, phone_detector=phone)
    print("Expression model:", "LOADED" if pipe.expression.is_ready() else "not found (running without the expression cue)")
    cap = cv2.VideoCapture(a.video if a.video else a.camera)
    if not cap.isOpened():
        sys.exit("Could not open the camera/video. Close other apps using the camera, or try --camera 1.")
    if not a.video:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.frame_width); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.frame_height)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    start, n, last_fps_t, shown_fps = time.time(), 0, time.time(), 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # for video files use the video's own clock so results do not depend on how fast your computer is
            t = (n / fps) if (a.video and fps > 0) else time.time()
            results = pipe.process_frame(frame, t); n += 1
            if time.time() - last_fps_t >= 1.0:
                shown_fps, last_fps_t = n / max(1e-6, time.time() - start), time.time()
            if not a.no_window:
                draw_results(frame, results, th.cue_attentive_cutoff)
                cv2.putText(frame, f"{shown_fps:.1f} FPS  |  Q = finish", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                cv2.imshow("Attention monitor", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                    break
            if a.seconds and (time.time() - start) >= a.seconds:
                break
    finally:
        cap.release(); cv2.destroyAllWindows()
    print(f"\nProcessed {n} frames in {time.time() - start:.1f}s  (~{n / max(1e-6, time.time() - start):.1f} frames/second on this computer)")
    decisions, summaries = pipe.finish(), pipe.recorder.summaries()
    pipe.close()
    if not decisions:
        sys.exit("No faces were scored (the calibration period alone may be longer than the video).")
    for d in decisions:
        print(f"\n{d.student_id}: {d.recommendation.value}   EAP={d.eap}%  average={d.average_score}  visible={d.visible_pct}%\n  {d.explanation}")
        if not a.no_prompt and d.needs_confirmation:
            ans = input("  Instructor: [Enter]=accept, P=Present, A=Absent  > ").strip().upper()
            if ans in ("P", "A"):
                d.instructor_decision = "PRESENT" if ans == "P" else "ABSENT"
                d.instructor_note = input("  Reason (optional) > ").strip()
            else:
                d.instructor_decision = "ACCEPTED: " + d.recommendation.value
    out = a.out or str(ROOT / "reports" / time.strftime("session_%Y%m%d_%H%M%S"))
    files = write_reports(out, decisions, summaries, th)
    print("\nReports written:", {k: str(v) for k, v in files.items()})


if __name__ == "__main__":
    main()
