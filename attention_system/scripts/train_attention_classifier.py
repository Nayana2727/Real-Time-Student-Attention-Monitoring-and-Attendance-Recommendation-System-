"""
scripts/train_attention_classifier.py  -  the 'rule-based vs machine-learning' comparison (project objective O7).

Input : CSV files made by collect_attention_data.py  (data/collected/*.csv)
Output: reports/attention_comparison.json  and  models/attention_classifier.joblib

It compares three ways of deciding 'attentive' vs 'inattentive' for every 3-second window:
   A) RULE_GATED / RULE_WEIGHTED - the survey-weighted score used by the live system (with and without the 'must face the screen' gate), using the survey's low-score cutoff
   B) LOGREG - logistic regression on window features
   C) GBM    - gradient boosting on window features
People in the test set are NEVER in the training set (GroupKFold by participant).
If you have fewer than 3 participants the numbers are only a rough guide - the script warns you.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from attention.config import CueSettings, ExpressionSettings, Settings, Thresholds, WeightSettings
from attention.scoring import CueScorer, FrameMeasurements

WINDOW, STEP = 3.0, 1.0


def load(paths):
    dfs = [pd.read_csv(p) for p in paths]
    df = pd.concat(dfs, ignore_index=True)
    df = df[df["label"].astype(str).isin(["attentive", "distracted", "drowsy"])].copy()
    return df


def frame_scores(g: pd.DataFrame, cfg: Settings, calib_seconds: float):
    sc = CueScorer(cfg.cues, cfg.expression, cfg.weights)
    t0 = g["t"].iloc[0]
    for _, r in g[g["t"] - t0 < calib_seconds].iterrows():
        sc.add_calibration_sample(FrameMeasurements(t=r["t"], yaw=r["yaw"], pitch=r["pitch"]))
    sc.finish_calibration()
    out = []
    for _, r in g.iterrows():
        probs = None
        if not pd.isna(r.get("p_engaged", np.nan)):
            probs = {"engaged": r["p_engaged"], "neutral": r["p_neutral"], "negative": r["p_negative"]}
        m = FrameMeasurements(t=r["t"], yaw=None if pd.isna(r["yaw"]) else r["yaw"], pitch=None if pd.isna(r["pitch"]) else r["pitch"],
                              gaze=None if pd.isna(r["gaze"]) else r["gaze"], ear=r["ear"], mar=r["mar"], expression_probs=probs)
        s, _ = sc.score(m); out.append(np.nan if s is None else s)
    return np.array(out), sc


def windows(df, cfg, calib_seconds):
    rows = []
    for pid, g in df.groupby("participant"):
        g = g.sort_values("t").reset_index(drop=True)
        scores, sc = frame_scores(g, cfg, calib_seconds)
        cfg_w = Settings(); cfg_w.cues.fusion = "weighted"
        scores_w, _ = frame_scores(g, cfg_w, calib_seconds)
        yaw = (g["yaw"] - sc.yaw0).abs().values; pit = (g["pitch"] - sc.pitch0).abs().values
        gaze = (g["gaze"] - 0.5).abs().values
        t = g["t"].values; start = t[0] + calib_seconds
        while start + WINDOW <= t[-1]:
            m = (t >= start) & (t < start + WINDOW)
            if m.sum() >= 5:
                lab = g.loc[m, "label"].mode().iloc[0]
                rows.append(dict(participant=pid, label=lab, rule_score=np.nanmean(scores[m]), rule_weighted_score=np.nanmean(scores_w[m]),
                                 yaw_mean=np.nanmean(yaw[m]), yaw_max=np.nanmax(yaw[m]), pitch_mean=np.nanmean(pit[m]), pitch_max=np.nanmax(pit[m]),
                                 gaze_dev=np.nanmean(gaze[m]), perclos=float((g.loc[m, "ear"] < cfg.cues.ear_closed).mean()),
                                 mar_open=float((g.loc[m, "mar"] > cfg.cues.mar_yawn).mean()),
                                 ear_mean=g.loc[m, "ear"].mean(), p_engaged=g.loc[m, "p_engaged"].mean() if "p_engaged" in g else np.nan))
            start += STEP
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="*", help="CSV files (default: everything in data/collected/)")
    ap.add_argument("--calib-seconds", type=float, default=3.0)
    ap.add_argument("--rule-cutoff", type=float, default=None, help="default: the survey's Q7a value")
    a = ap.parse_args()
    paths = [Path(p) for p in a.csv] if a.csv else sorted((ROOT / "data" / "collected").glob("*.csv"))
    if not paths:
        sys.exit("No CSV files found. Collect data first (scripts/collect_attention_data.py).")
    synthetic = any(p.name.upper().startswith("SYNTHETIC") for p in paths)
    if synthetic:
        print("\n" + "!" * 78 + "\n!! SYNTHETIC (fake) data detected: this only tests that the code runs.\n"
              "!! Do NOT report these numbers anywhere.\n" + "!" * 78 + "\n")
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    import joblib

    cfg = Settings(); cutoff = a.rule_cutoff if a.rule_cutoff is not None else Thresholds.from_survey().cue_attentive_cutoff
    W = windows(load(paths), cfg, a.calib_seconds)
    W["y"] = (W["label"] != "attentive").astype(int)                  # 1 = inattentive (distracted or drowsy)
    feats = [c for c in W.columns if c not in ("participant", "label", "rule_score", "rule_weighted_score", "y") and not W[c].isna().all()]
    X = W[feats].fillna(0.0).values; y = W["y"].values; groups = W["participant"].values
    n_people = len(set(groups))
    print(f"{len(W)} windows from {n_people} participant(s); inattentive share = {y.mean():.2f}")
    warn = None
    if n_people < 3:
        warn = f"Only {n_people} participant(s): results are a rough guide only (people cannot be fully separated)."
        print("WARNING:", warn)

    def score(pred):
        return {"accuracy": round(float(accuracy_score(y_te, pred)), 4), "balanced_accuracy": round(float(balanced_accuracy_score(y_te, pred)), 4),
                "macro_f1": round(float(f1_score(y_te, pred, average="macro")), 4),
                "recall_inattentive": round(float(recall_score(y_te, pred, pos_label=1, zero_division=0)), 4)}

    models = {"LOGREG": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
              "GBM": GradientBoostingClassifier(random_state=0)}
    pooled = {"RULE_GATED": ([], []), "RULE_WEIGHTED": ([], []), "LOGREG": ([], []), "GBM": ([], [])}
    splits = list(GroupKFold(n_splits=min(5, max(2, n_people))).split(X, y, groups)) if n_people >= 2 else None
    if splits is None:
        cut = int(0.7 * len(W)); splits = [(np.arange(cut), np.arange(cut, len(W)))]
    for tr, te in splits:
        yt = y[te]
        pooled["RULE_GATED"][0].extend(yt); pooled["RULE_GATED"][1].extend((W["rule_score"].values[te] < cutoff).astype(int))
        pooled["RULE_WEIGHTED"][0].extend(yt); pooled["RULE_WEIGHTED"][1].extend((W["rule_weighted_score"].values[te] < cutoff).astype(int))
        for name, mdl in models.items():
            if len(set(y[tr])) < 2: continue
            mdl.fit(X[tr], y[tr]); pooled[name][0].extend(yt); pooled[name][1].extend(mdl.predict(X[te]))
    results = {}
    for name, (yt, yp) in pooled.items():
        if not yt: continue
        y_te = np.array(yt); results[name] = score(np.array(yp))
    print("\nmethod   accuracy  bal.acc  macroF1  recall(inattentive)")
    for k, v in results.items():
        print(f"{k:7s}  {v['accuracy']:.3f}    {v['balanced_accuracy']:.3f}    {v['macro_f1']:.3f}    {v['recall_inattentive']:.3f}")
    best = max((k for k in results if not k.startswith("RULE")), key=lambda k: results[k]["macro_f1"], default=None)
    if best:
        models[best].fit(X, y); (ROOT / "models").mkdir(exist_ok=True)
        joblib.dump({"model": models[best], "features": feats, "name": best}, ROOT / "models" / ("SYNTHETIC_attention_classifier.joblib" if synthetic else "attention_classifier.joblib"))
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / ("SYNTHETIC_attention_comparison.json" if synthetic else "attention_comparison.json")).write_text(json.dumps({
        "synthetic_data_not_a_result": synthetic, "participants": n_people, "windows": int(len(W)), "rule_cutoff": cutoff,
        "warning": warn, "results": results, "best_ml": best}, indent=2))


if __name__ == "__main__":
    main()
