# Expression model - model card
**Task:** classify a 48x48 grey face crop into `engaged` / `neutral` / `negative`. It is ONE cue (weight about 16% under survey weighting), not the whole system.

**Data:** public GitHub image set `muxspace/facial_expressions` (13,185 usable faces after face detection). Labels were made by students and are noisy.
Mapping: happiness -> engaged; neutral -> neutral; anger, contempt, disgust, fear, sadness -> negative; surprise excluded.
The images are NOT included in this repository (licence unclear); `scripts/prepare_expression_data.py` downloads them to your own computer.
Split by person (train 9,633 / validation 1,743 / test 1,809) so nobody appears in two splits.

**Model:** small CNN (4 conv blocks, BatchNorm, dropout, augmentation), class-weighted loss, early stopping on validation loss (best epoch kept). 20 epochs run. TFLite export: 760 KB, identical predictions to the Keras model on 300 test faces.

**Results on the held-out test people (measured by `scripts/train_expression.py`, see `reports/expression_metrics.json`):**
| | Result |
|---|---|
| Accuracy | 78.6% (always guessing "neutral" gives 50.8%) |
| Balanced accuracy | 64.2% (chance 33.3%) |
| Macro-F1 | 0.650 |
| Recall engaged / neutral / negative | 76.0% / 87.3% / **29.3%** |
| Precision negative | 35.0% |

**Weak spot:** the `negative` class (only 501 training faces) is recognised poorly. Do not treat a "negative" reading as reliable.
**Not comparable** to published FER-2013 numbers (different data, 3 classes, different labels). For a paper-grade number retrain on FER-2013:
`python scripts/prepare_expression_data.py --source fer2013 --fer-dir path/to/fer2013` then `python scripts/train_expression.py`.
**Bias warning:** photos are mostly celebrity news photos (LFW). Performance on other ages, skin tones, lighting, glasses or masks is unknown and probably worse.
