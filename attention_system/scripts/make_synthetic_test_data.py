"""
scripts/make_synthetic_test_data.py - makes FAKE feature files so you can check that the training script runs.
The files are named SYNTHETIC_*.csv. NEVER report anything trained on them. Real data comes from collect_attention_data.py.
"""
import csv
from pathlib import Path
import numpy as np
OUT = Path(__file__).resolve().parent.parent / "data" / "collected"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(7)
for p in range(4):
    rows, t = [], 0.0
    for block in range(18):
        lab = ["attentive", "distracted", "drowsy"][int(rng.choice(3, p=[0.55, 0.25, 0.20]))] if block > 0 else "attentive"
        for _ in range(100):                                             # 10 s at 10 Hz
            t += 0.1; off = rng.normal(2 * p, 1)
            if lab == "attentive":
                yaw, pit, gaze, ear, mar = rng.normal(off, 5), rng.normal(0, 5), rng.normal(.5, .04), rng.normal(.31, .03), abs(rng.normal(.1, .05))
            elif lab == "distracted":
                yaw, pit, gaze, ear, mar = rng.normal(off + rng.choice([-1, 1]) * 38, 8), rng.normal(0, 9), rng.normal(.5, .18), rng.normal(.3, .03), abs(rng.normal(.1, .05))
            else:
                yaw, pit, gaze, ear, mar = rng.normal(off, 6), rng.normal(-12, 12), rng.normal(.5, .08), rng.normal(.17, .07), abs(rng.normal(.45, .3))
            shown = lab if rng.random() > 0.05 else "attentive"          # 5% label noise
            rows.append([f"S{p+1}", round(t, 3), shown, round(yaw, 2), round(pit, 2), round(gaze, 4), round(ear, 4), round(mar, 4), "", "", ""])
    with open(OUT / f"SYNTHETIC_P{p+1}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["participant", "t", "label", "yaw", "pitch", "gaze", "ear", "mar", "p_engaged", "p_neutral", "p_negative"]); w.writerows(rows)
print("wrote 4 SYNTHETIC files to", OUT, "- for code testing only")
