"""attention/expression.py - runs the trained expression model (3 classes: engaged / neutral / negative).

If no model file exists yet, is_ready() is False and the rest of the system simply runs without
the expression cue (its weight is shared among the other cues). Nothing crashes.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional
import numpy as np
from .config import ExpressionSettings


class ExpressionClassifier:
    def __init__(self, settings: Optional[ExpressionSettings] = None):
        self.s = settings or ExpressionSettings()
        self._interp = None; self._keras = None
        path = Path(self.s.model_path)
        try:
            if path.exists():
                import tensorflow as tf
                self._interp = tf.lite.Interpreter(model_path=str(path))
                self._interp.allocate_tensors()
                self._in = self._interp.get_input_details()[0]
                self._out = self._interp.get_output_details()[0]
            elif path.with_suffix(".keras").exists():
                import tensorflow as tf
                self._keras = tf.keras.models.load_model(path.with_suffix(".keras"))
        except Exception as e:                       # a broken model must not break the camera demo
            print(f"[expression] could not load model ({e}); running without the expression cue")
            self._interp = self._keras = None

    def is_ready(self) -> bool:
        return self._interp is not None or self._keras is not None

    def predict(self, face_bgr: np.ndarray) -> Optional[Dict[str, float]]:
        if not self.is_ready() or face_bgr is None or face_bgr.size == 0:
            return None
        import cv2
        g = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (self.s.input_size, self.s.input_size)).astype(np.float32) / 255.0
        x = g.reshape(1, self.s.input_size, self.s.input_size, 1)
        if self._interp is not None:
            self._interp.set_tensor(self._in["index"], x.astype(self._in["dtype"]))
            self._interp.invoke()
            p = self._interp.get_tensor(self._out["index"])[0]
        else:
            p = self._keras(x, training=False).numpy()[0]
        return {c: float(v) for c, v in zip(self.s.classes, p)}
