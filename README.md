# Pork Rasher Packaging Inspection

Automated detection of packaging and product-placement defects in trays of pork rashers, using a YOLO object detector. The project audits the dataset, builds a leakage-free validation split, compares two model sizes, and turns detections into a **pass/fail decision** for each tray.

*MMA3001 Numerical Methods and Machine Learning, Individual Project 2026 (Dataset 3).*

---

## The engineering problem

Packaged meat trays can leave the line with defects: an unsealed film, meat hanging loose or twisted, or wrinkled packaging. Defective trays can spoil or be rejected by retailers, and inspecting them by eye is slow and inconsistent. **The question:** can a camera and an object detector flag defective trays accurately, and fast enough, for a production line?

**Scope:** single top-down RGB frames from one camera on one line. The method is not tested on other lines, lighting conditions or tray types.

### Inputs and outputs

| | Description | Format |
|---|---|---|
| **Input** | One RGB image of a tray on the conveyor | JPEG, 720 × 540 px |
| **Output 1** | Detected defects | List of boxes: class (one of 5), confidence 0–1, box centre/width/height as fractions of the image |
| **Output 2** | Tray decision | `FAIL` if any detection has confidence ≥ threshold *t*, otherwise `PASS` |
| **Parameter** | Confidence threshold *t* | 0–1 (chosen on validation data; see results) |

Defect classes: `loose-meat`, `packaging-error`, `twisted-meat`, `unsealed`, `wrinkle`.

---

## Results at a glance

**1. The provided split leaks, but the leak did not inflate the score.** The images are video frames. In the provided split, 67.5 % of test images are within 5 frames of a training image, and 75 % are near-identical by perceptual hash. A leakage-free *blocked* split gave a similar score (YOLO11n, test mAP50):

| Split | Test images | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| Original (Roboflow) | 75 | 0.725 | 0.371 | 0.427 | 0.237 |
| Blocked (leakage-free) | 197 | 0.564 | 0.367 | **0.459** | 0.230 |

All later results use the blocked split.

**2. The smaller model is better on this dataset.** Both models were trained with identical settings (50 epochs, 640 px, seed 0):

| Model | Parameters | GFLOPs | File | Test mAP50 | GPU ms/image | CPU ms/image |
|---|---|---|---|---|---|---|
| **YOLO11n** | 2.6 M | 6.6 | 5.5 MB | **0.459** | 13.3 | **86** |
| YOLO11s | 9.4 M | 21.7 | 19.2 MB | 0.336 | 14.8 | 210 |

Per-class test mAP50 (YOLO11n / YOLO11s): unsealed 0.92 / 0.91, packaging-error 0.63 / 0.58, twisted-meat 0.39 / 0.16, loose-meat 0.36 / 0.03, wrinkle 0.00 / 0.00. The larger model loses most on the rare classes, which is consistent with overfitting on few examples.

**3. Pass/fail trade-off (YOLO11n, blocked test set: 168 bad trays, 29 good trays).** 95 % bootstrap intervals are in brackets.

| Threshold | Bad trays missed | Good trays rejected |
|---|---|---|
| 0.25 (YOLO default) | 20 of 168: 11.9 % [7.1–17.1 %] | 9 of 29: 31.0 % [14.3–48.6 %] |
| 0.04 (chosen on validation for ≤ 5 % misses) | 3 of 168: 1.8 % [0.0–4.2 %] | 22 of 29: 75.9 % [59.4–90.5 %] |

Meeting a 5 % miss-rate target requires rejecting about three-quarters of good trays, so the model is not yet practical for unattended inspection. See `results/figures/threshold_sweep.png`.

---

## Repository structure

```
pork-rasher-qc/
├── README.md                  ← this file
├── LICENSE                    ← MIT licence (code)
├── AI_USE_LOG.md              ← record of AI use during the project
├── requirements.txt           ← Python packages
├── pyproject.toml             ← package + pytest settings
├── notebooks/
│   ├── 01_data_exploration.ipynb              ← dataset audit and leakage check
│   ├── 02_split_and_train.ipynb               ← blocked split + YOLO11n on both splits
│   └── 03_model_comparison_and_passfail.ipynb ← YOLO11s, pass/fail, threshold, timing
├── src/porkqc/                ← tested core functions
│   ├── data.py                ← label parsing, file names
│   ├── leakage.py             ← frame gaps, perceptual hashing
│   ├── splitting.py           ← leakage-free blocked split
│   ├── passfail.py            ← pass/fail rule, threshold choice, bootstrap intervals
│   └── metrics.py             ← average precision (interpolation + trapezoidal rule)
├── tests/                     ← pytest tests
├── docs/                      ← HTML documentation generated from docstrings (pdoc)
├── reports/test_report.html   ← latest test report
└── results/                   ← tables (CSV) and figures from the notebooks
```

---

## How to run

### A. The notebooks (Google Colab)

1. **Get the data.** Download *Pork Rasher Error(Packaging)* **version 4** in **YOLOv11** format from [Roboflow Universe](https://universe.roboflow.com/hello-2aqe0/pork-rasher-error-packaging) as a zip. The data is not included in this repository.
2. Rename the zip `pork_rasher_v4.zip` and put it in the top level of your Google Drive ("My Drive").
3. Open each notebook in [Google Colab](https://colab.research.google.com) (*File → Upload notebook*), in order **01 → 02 → 03**.
4. For notebooks 02 and 03, choose *Runtime → Change runtime type → T4 GPU*.
5. Choose *Runtime → Run all* and allow Google Drive access. Results are saved to `pork_runs/` in your Drive. Finished training runs are detected and skipped if you re-run.

Approximate run times on a Colab T4 GPU: notebook 01 takes about 2 min, notebook 02 about 30 min, notebook 03 about 20–40 min.

### B. Tests and documentation (any computer with Python 3.9+)

```bash
pip install -r requirements.txt
pytest                                                  # run all tests
pytest --html=reports/test_report.html --self-contained-html   # regenerate the test report
pdoc src/porkqc --docformat numpy -o docs               # regenerate the HTML docs
```

Open `docs/index.html` in a browser to read the documentation.

---

## Method summary

1. **Dataset audit (notebook 01).** Class counts, co-occurrence and box sizes. Leakage check using frame-number gaps and 64-bit difference hashes (Hamming distance).
2. **Blocked split (notebook 02).** Frames are grouped into blocks of 200; whole blocks are assigned to train/valid/test (target 70/15/15). Of 500 random assignments, the one with the best class balance in valid and test is kept (stratified group split, chosen on labels only). Training frames within 15 frames of any valid/test frame are removed. One copy of each original image is used, and Roboflow's augmented copies are excluded, so both splits use the same image pool.
3. **Training.** Ultralytics YOLO11n and YOLO11s, pre-trained on COCO (transfer learning), fine-tuned for 50 epochs at 640 px, batch 16, seed 0. The best epoch by validation mAP is kept.
4. **Pass/fail decision (notebook 03).** A tray fails if its highest detection confidence is ≥ *t*. The threshold is swept from 0.01 to 0.95 on the **validation** set. The chosen *t* is the one with the fewest false rejects among those with a miss rate ≤ 5 %. It is then applied once to the test set.
5. **Uncertainty.** 95 % percentile bootstrap intervals (2000 resamples) for miss and false-reject rates.
6. **Performance.** Median end-to-end time per image (batch size 1, after warm-up) on GPU and CPU, plus parameter count and GFLOPs.

---

## Verification and testing

`pytest` runs 58 tests (see `reports/test_report.html`), including:

- **Known-answer tests:** pass/fail counts on a hand-calculated example; pixel conversion of a known box; average precision of shapes with an exact area (perfect detector = 1, rectangle = 0.4, constant precision = 0.5625).
- **Numerical convergence:** the trapezoidal-integration error for AP shrinks as the interpolation grid is refined (11 → 101 → 1001 points).
- **Comparison with a reference implementation:** `average_precision` matches Ultralytics' `compute_ap` to 1e-12 on random curves.
- **Property tests:** raising the threshold never decreases the miss rate or increases the false-reject rate; blocks are never split across sets; no training frame lies within the buffer of a test frame; bootstrap intervals contain the point estimate and narrow with more data.
- **Regression tests on real results:** re-applying the threshold rule to the saved validation sweep reproduces the thresholds chosen in notebook 03 (0.04 and 0.03).
- **Input validation:** malformed, polygon and non-normalised label lines are rejected.

The `porkqc.splitting` functions were also checked to reproduce notebook 02's split exactly (same seed, same assignment) on a synthetic dataset.

---

## Limitations

- **Rare classes:** `wrinkle` (18 images including augmented copies) is never detected. Per-class scores for loose and twisted meat rest on a handful of test examples.
- **Few good trays:** the blocked test set has only 29 defect-free trays, so the false-reject rate is very uncertain (interval width about 30 percentage points).
- **Class balance unlike a real line:** about 85 % of test trays are defective, the reverse of real production. Plain accuracy is therefore misleading and is not used for decisions.
- **Partly visible trays:** the missed defects are at the image edge, on trays entering the camera view.
- **Single training run per model**, so small differences between models may be due to randomness.
- **One camera, one line, one lighting setup:** the blocked split removes near-duplicate frames but not this wider domain shift.

**Possible improvements:** combine several frames of the same tray (fail if any frame flags it); trigger capture when the tray is fully in view; collect more wrinkle and good-tray examples; repeat training with several seeds.

---

## Data and citation

Dataset: *Pork Rasher Error(Packaging)*, Roboflow Universe, licensed **CC BY 4.0**.

```bibtex
@misc{pork-rasher-error-packaging_dataset,
  title        = {Pork Rasher Error(Packaging) Dataset},
  type         = {Open Source Dataset},
  author       = {Hello},
  howpublished = {\url{https://universe.roboflow.com/hello-2aqe0/pork-rasher-error-packaging}},
  publisher    = {Roboflow},
  year         = {2025},
  month        = {may},
}
```

Object detection: [Ultralytics YOLO11](https://docs.ultralytics.com) (AGPL-3.0).

## AI use

AI assistance was used in this project and is recorded in [`AI_USE_LOG.md`](AI_USE_LOG.md).

## Licence

Code in this repository is released under the MIT licence (see `LICENSE`). The dataset is not included and remains under its own CC BY 4.0 licence.
