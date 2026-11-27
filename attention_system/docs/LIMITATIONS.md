# Limitations - say these out loud in your viva and put them in your paper
1. **Not tested on a live webcam by the authors of this code package.** All testing used still images/video files (34 automatic tests, simulated class videos). Test on your own laptop first.
2. **No ground-truth attention data exists yet.** The rule weights come from a faculty survey (opinion), not from measured accuracy against labelled classroom video. Use `collect_attention_data.py` + `train_attention_classifier.py` to measure it, and DAiSEE (request form) for a public benchmark.
3. **Head pose** uses a generic 3-D face model. The angle maths was verified exactly on synthetic rotations, but on real faces accuracy drops at large angles (a 51-degree head read as 39-degree once tracking took over). Calibration (first 3 s) removes the camera-height offset only.
4. **Gaze** here is eye-position-in-head only (iris between eye corners). It cannot tell where on the screen someone looks and misses eyes-down-at-a-phone.
5. **Expression model** is weak on the negative class and trained on noisy, mostly celebrity photos (see MODEL_CARD.md).
6. **Cue thresholds** (30 degree head limit, EAR 0.20, MAR 0.60, ...) are engineering choices, not survey results. Tune them on your own camera. `head_pose_zero_factor` and the 'gated' fusion were set after testing on ONE example - re-check them.
7. **Phone detection** is optional, uses a third-party model (AGPL licence) and was **not tested** by the authors.
8. **Multi-person** tracking is simple nearest-face matching. Two students swapping seats or leaving the frame can swap IDs.
9. **Survey:** self-selected respondents from VTU colleges in Karnataka; Q4b asked for a top-3 but 18 of 106 answered otherwise (all picks counted); 16 of 106 (15.1%) gave no numeric absent-threshold.
10. **Ethics:** attendance affects students. This is a recommendation tool; the instructor must confirm. Always get consent, tell students the system runs, and never store video. Not suitable for automatic or high-stakes use without institutional approval and a bias audit.
