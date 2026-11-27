"""
scripts/prepare_expression_data.py  -  STEP 1 of training the expression model.

What it does (in plain words):
  1. Gets a folder of face photos that people have labelled with an emotion.
  2. Finds the face in each photo and cuts it out as a tiny 48x48 grey picture.
  3. Joins the many emotion names into 3 groups that matter for a classroom:
         engaged  <- happiness
         neutral  <- neutral
         negative <- anger, contempt, disgust, fear, sadness     (looks confused / unhappy / tired)
     'surprise' is left out because it could mean either good or bad attention.
  4. Splits people into train / validation / test groups so that the SAME PERSON never appears in both
     training and testing (otherwise the score would look better than it really is).
  5. Saves everything in data/expression_faces.npz

Two data sources:
  --source github    (default) downloads the public GitHub set 'muxspace/facial_expressions' (about 250 MB).
                     Labels were made by students and are noisy; see docs. Images are NOT redistributed in this project.
  --source fer2013   uses FER-2013 that YOU downloaded from Kaggle:  --fer-dir path/to/fer2013   (folders train/ and test/)
"""
import argparse, csv, hashlib, io, re, sys, urllib.request, zipfile
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "expression_faces.npz"
CLASSES = ["engaged", "neutral", "negative"]
GROUP = {"happiness": "engaged", "happy": "engaged", "neutral": "neutral",
         "anger": "negative", "angry": "negative", "contempt": "negative", "disgust": "negative", "disgusted": "negative",
         "fear": "negative", "sadness": "negative", "sad": "negative"}
URL = "https://codeload.github.com/muxspace/facial_expressions/zip/refs/heads/master"
SIZE = 48


def bucket(key: str) -> str:
    """Deterministic 70/15/15 split by person (or by file when the person is unknown)."""
    v = int(hashlib.md5(key.encode()).hexdigest(), 16) % 100
    return "train" if v < 70 else ("val" if v < 85 else "test")


def person_of(filename: str) -> str:
    m = re.match(r"^(.+)_\d{4}\.(jpg|jpeg|png)$", filename, re.I)       # LFW style: Firstname_Lastname_0001.jpg
    return m.group(1) if m else filename


class FaceCropper:
    def __init__(self):
        import mediapipe as mp
        self.det = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.4)

    def crop(self, bgr):
        import cv2
        h, w = bgr.shape[:2]
        r = self.det.process(np.ascontiguousarray(bgr[:, :, ::-1]))
        if not r.detections:
            return None
        box = max(r.detections, key=lambda d: d.score[0]).location_data.relative_bounding_box
        x0, y0 = int((box.xmin - 0.05) * w), int((box.ymin - 0.10) * h)
        x1, y1 = int((box.xmin + box.width + 0.05) * w), int((box.ymin + box.height + 0.05) * h)
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        if x1 - x0 < 24 or y1 - y0 < 24:
            return None
        g = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        return cv2.resize(g, (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def from_github(limit=None):
    import cv2
    RAW.mkdir(parents=True, exist_ok=True)
    zpath = RAW / "facial_expressions.zip"
    if not zpath.exists():
        print("Downloading public image set (about 250 MB) ... this can take a few minutes")
        urllib.request.urlretrieve(URL, zpath)
    zf = zipfile.ZipFile(zpath)
    prefix = zf.namelist()[0]                                   # 'facial_expressions-master/'
    rows = list(csv.DictReader(io.TextIOWrapper(zf.open(prefix + "data/legend.csv"), encoding="utf-8", errors="ignore")))
    print("labelled rows in legend.csv:", len(rows))
    labels = {}
    for r in rows:
        name, emo = r["image"].strip(), r["emotion"].strip().lower()
        if emo in ("surprise",) or emo not in GROUP:
            labels.setdefault(name, set()).add(None); continue
        labels.setdefault(name, set()).add(GROUP[emo])
    cropper, X, y, split, who = FaceCropper(), [], [], [], []
    skipped = Counter()
    items = list(labels.items())[: limit or None]
    for i, (name, labs) in enumerate(items):
        if len(labs) != 1 or None in labs:
            skipped["conflicting or excluded label"] += 1; continue
        try:
            data = zf.read(prefix + "images/" + name)
        except KeyError:
            skipped["image missing"] += 1; continue
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            skipped["unreadable"] += 1; continue
        face = cropper.crop(img)
        if face is None:
            skipped["no face found"] += 1; continue
        p = person_of(name)
        X.append(face); y.append(CLASSES.index(next(iter(labs)))); split.append(bucket(p)); who.append(p)
        if (i + 1) % 1000 == 0:
            print(f"  processed {i+1}/{len(items)}")
    return np.array(X, np.uint8), np.array(y, np.int64), np.array(split), np.array(who), skipped


def from_fer2013(fer_dir: Path):
    import cv2
    X, y, split, who, skipped = [], [], [], [], Counter()
    for sub in ("train", "test"):
        for cls_dir in sorted((fer_dir / sub).iterdir()):
            emo = cls_dir.name.lower()
            if emo not in GROUP:
                skipped[f"excluded class {emo}"] += len(list(cls_dir.glob("*"))); continue
            for f in cls_dir.glob("*"):
                g = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
                if g is None: skipped["unreadable"] += 1; continue
                X.append(cv2.resize(g, (SIZE, SIZE))); y.append(CLASSES.index(GROUP[emo]))
                key = f.name  # FER-2013 has no person ids; official 'test' folder stays the test set
                split.append("test" if sub == "test" else ("val" if bucket(key) == "test" else "train")); who.append(key)
    return np.array(X, np.uint8), np.array(y, np.int64), np.array(split), np.array(who), skipped


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["github", "fer2013"], default="github")
    ap.add_argument("--fer-dir", type=Path)
    ap.add_argument("--limit", type=int, default=None, help="only use the first N images (quick test)")
    a = ap.parse_args()
    if a.source == "fer2013":
        if not a.fer_dir: sys.exit("Give --fer-dir path/to/fer2013 (it must contain train/ and test/)")
        X, y, split, who, skipped = from_fer2013(a.fer_dir)
    else:
        X, y, split, who, skipped = from_github(a.limit)
    OUT.parent.mkdir(exist_ok=True)
    np.savez_compressed(OUT, X=X, y=y, split=split, who=who, classes=np.array(CLASSES), source=a.source)
    print("\nSaved", OUT, "with", len(X), "faces")
    for s in ("train", "val", "test"):
        m = split == s
        print(f"  {s:5s}: {int(m.sum()):5d}  ", {CLASSES[k]: int((y[m] == k).sum()) for k in range(3)})
    print("  skipped:", dict(skipped))
