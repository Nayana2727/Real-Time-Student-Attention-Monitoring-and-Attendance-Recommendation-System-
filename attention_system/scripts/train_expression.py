"""
scripts/train_expression.py  -  STEP 2: teach a small neural network (CNN) to read expressions.

Input : data/expression_faces.npz         (made by prepare_expression_data.py)
Output: models/expression_model.keras  and  models/expression_model.tflite
        reports/expression_metrics.json, reports/expression_confusion_matrix.png, reports/expression_training_curves.png

Honest scoring: the test faces belong to DIFFERENT PEOPLE than the training faces, and we always print the
'always guess the biggest class' baseline next to our accuracy, because when one class is much bigger,
a high accuracy alone can be misleading. We also report balanced accuracy and macro-F1.
"""
import argparse, json, os
from pathlib import Path
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
ROOT = Path(__file__).resolve().parent.parent
DATA, MODELS, REPORTS = ROOT / "data" / "expression_faces.npz", ROOT / "models", ROOT / "reports"


def build_model(n_classes=3, size=48):
    import tensorflow as tf
    from tensorflow.keras import layers as L
    aug = tf.keras.Sequential([L.RandomFlip("horizontal"), L.RandomRotation(0.08), L.RandomZoom(0.1),
                               L.RandomContrast(0.2)], name="augment")
    inp = L.Input((size, size, 1))
    x = aug(inp)
    for filters in (32, 64, 96):                      # small on purpose: runs fast on a laptop CPU
        x = L.Conv2D(filters, 3, padding="same", use_bias=False)(x); x = L.BatchNormalization()(x); x = L.ReLU()(x)
        x = L.MaxPooling2D()(x); x = L.Dropout(0.25)(x)
    x = L.Conv2D(128, 3, padding="same", use_bias=False)(x); x = L.BatchNormalization()(x); x = L.ReLU()(x)
    x = L.GlobalAveragePooling2D()(x)
    x = L.Dense(64, activation="relu")(x); x = L.Dropout(0.4)(x)
    out = L.Dense(n_classes, activation="softmax")(x)
    m = tf.keras.Model(inp, out, name="expression_cnn")
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true", help="continue from the last saved epoch")
    ap.add_argument("--time-budget-min", type=float, default=None, help="pause after this many minutes (use with --resume)")
    a = ap.parse_args()
    import tensorflow as tf
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                                 classification_report)
    from sklearn.utils.class_weight import compute_class_weight
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    np.random.seed(a.seed); tf.random.set_seed(a.seed)
    d = np.load(DATA, allow_pickle=True)
    X = d["X"].astype("float32")[..., None] / 255.0; y = d["y"]; split = d["split"]; classes = list(d["classes"])
    tr, va, te = (split == "train"), (split == "val"), (split == "test")
    print("train/val/test:", tr.sum(), va.sum(), te.sum(), "| source:", d["source"])
    cw = compute_class_weight("balanced", classes=np.arange(len(classes)), y=y[tr])
    cw = {i: float(w) for i, w in enumerate(cw)}
    print("class weights:", {classes[i]: round(w, 2) for i, w in cw.items()})

    # ---------- training loop that can be PAUSED and RESUMED (laptops sleep, sessions end) ----------
    MODELS.mkdir(exist_ok=True); REPORTS.mkdir(exist_ok=True)
    last_p, best_p, state_p = MODELS / "_checkpoint_last.keras", MODELS / "_checkpoint_best.keras", REPORTS / "expression_train_state.json"
    state = {"epoch": 0, "best_val_loss": 1e9, "wait": 0, "history": {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}}
    if a.resume and last_p.exists() and state_p.exists():
        model = tf.keras.models.load_model(last_p); state = json.loads(state_p.read_text())
        print(f"Resuming from epoch {state['epoch']} (best val_loss so far {state['best_val_loss']:.4f})")
    else:
        model = build_model(len(classes))
    import time as _t
    t_start, PATIENCE = _t.time(), 8
    while state["epoch"] < a.epochs and state["wait"] < PATIENCE:
        if a.time_budget_min and (_t.time() - t_start) / 60 >= a.time_budget_min:
            print(f"\nPAUSED after epoch {state['epoch']}. Run the same command again with --resume to continue.")
            return
        h = model.fit(X[tr], y[tr], validation_data=(X[va], y[va]), epochs=1, batch_size=a.batch, class_weight=cw, verbose=2)
        state["epoch"] += 1
        for k in state["history"]: state["history"][k].append(float(h.history[k][0]))
        vl = state["history"]["val_loss"][-1]
        if vl < state["best_val_loss"] - 1e-4:
            state["best_val_loss"], state["wait"] = vl, 0; model.save(best_p)
        else:
            state["wait"] += 1
            if state["wait"] % 4 == 0:                                    # same idea as ReduceLROnPlateau
                lr = float(model.optimizer.learning_rate.numpy()); model.optimizer.learning_rate.assign(max(lr * 0.5, 1e-5))
        model.save(last_p); state_p.write_text(json.dumps(state))
    print(f"Training finished after {state['epoch']} epochs (best val_loss {state['best_val_loss']:.4f}).")
    model = tf.keras.models.load_model(best_p)                           # go back to the best epoch
    hist = type("H", (), {"history": state["history"]})()

    # ---------- honest evaluation on the held-out people ----------
    proba = model.predict(X[te], verbose=0); pred = proba.argmax(1); yt = y[te]
    majority = int(np.bincount(y[tr]).argmax())
    metrics = {
        "source": str(d["source"]), "n_train": int(tr.sum()), "n_val": int(va.sum()), "n_test": int(te.sum()),
        "classes": classes, "test_class_counts": {classes[k]: int((yt == k).sum()) for k in range(len(classes))},
        "accuracy": round(float(accuracy_score(yt, pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(yt, pred)), 4),
        "macro_f1": round(float(f1_score(yt, pred, average="macro")), 4),
        "majority_class_baseline_accuracy": round(float((yt == majority).mean()), 4),
        "chance_balanced_accuracy": round(1.0 / len(classes), 4),
        "per_class": classification_report(yt, pred, target_names=classes, output_dict=True, zero_division=0),
        "epochs_run": len(hist.history["loss"]),
    }
    cm = confusion_matrix(yt, pred, labels=range(len(classes)))
    metrics["confusion_matrix"] = cm.tolist()
    model.save(MODELS / "expression_model.keras")

    (REPORTS / "expression_metrics.json").write_text(json.dumps(metrics, indent=2))      # saved FIRST, so it can never be lost

    # ---------- export to TFLite (small & fast on a laptop CPU) and CHECK it matches ----------
    # Keras 3 + TF 2.16: from_keras_model() fails, so export a SavedModel first and convert that.
    import shutil, tempfile
    tmp = Path(tempfile.mkdtemp()) / "saved"
    # the random-augmentation layer is only for training; export a copy that starts AFTER it
    infer = tf.keras.Model(inputs=model.get_layer("augment").output, outputs=model.output, name="expression_cnn_inference")
    infer.export(str(tmp), verbose=False)
    tfl = tf.lite.TFLiteConverter.from_saved_model(str(tmp)).convert()
    shutil.rmtree(tmp.parent, ignore_errors=True)
    (MODELS / "expression_model.tflite").write_bytes(tfl)
    interp = tf.lite.Interpreter(model_content=tfl); interp.allocate_tensors()
    i_d, o_d = interp.get_input_details()[0], interp.get_output_details()[0]
    k = min(300, len(X[te])); agree, maxdiff = 0, 0.0
    for j in range(k):
        interp.set_tensor(i_d["index"], X[te][j:j + 1]); interp.invoke()
        p = interp.get_tensor(o_d["index"])[0]
        agree += int(p.argmax() == proba[j].argmax()); maxdiff = max(maxdiff, float(np.abs(p - proba[j]).max()))
    metrics["tflite_check"] = {"samples": k, "same_prediction_pct": round(100 * agree / k, 1),
                               "max_probability_difference": round(maxdiff, 6), "size_kb": round(len(tfl) / 1024, 1)}
    (REPORTS / "expression_metrics.json").write_text(json.dumps(metrics, indent=2))

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes); ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Expression model - test people only")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout(); fig.savefig(REPORTS / "expression_confusion_matrix.png", dpi=150); plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
    axs[0].plot(hist.history["accuracy"], label="train"); axs[0].plot(hist.history["val_accuracy"], label="validation"); axs[0].set_title("Accuracy"); axs[0].legend()
    axs[1].plot(hist.history["loss"], label="train"); axs[1].plot(hist.history["val_loss"], label="validation"); axs[1].set_title("Loss"); axs[1].legend()
    for ax in axs: ax.set_xlabel("epoch")
    fig.tight_layout(); fig.savefig(REPORTS / "expression_training_curves.png", dpi=150); plt.close(fig)

    print("\n===== RESULTS ON PEOPLE THE MODEL NEVER SAW =====")
    print(f"accuracy                  : {metrics['accuracy']:.3f}")
    print(f"majority-class baseline   : {metrics['majority_class_baseline_accuracy']:.3f}   (accuracy of always guessing '{classes[majority]}')")
    print(f"balanced accuracy         : {metrics['balanced_accuracy']:.3f}   (chance = {metrics['chance_balanced_accuracy']:.3f})")
    print(f"macro F1                  : {metrics['macro_f1']:.3f}")
    print("confusion matrix (rows=true):", cm.tolist())
    print("tflite check:", metrics["tflite_check"])


if __name__ == "__main__":
    main()
