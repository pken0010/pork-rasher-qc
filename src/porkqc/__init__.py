"""
porkqc: automated quality inspection of packaged pork rashers.

This package holds the tested core functions used by the project notebooks:

- `porkqc.data`: reading YOLO label files and Roboflow file names.
- `porkqc.leakage`: detecting near-duplicate video frames between splits.
- `porkqc.splitting`: building a leakage-free *blocked* train/valid/test split.
- `porkqc.passfail`: turning detections into a pass/fail decision per tray,
  choosing the confidence threshold, and bootstrap confidence intervals.
- `porkqc.metrics`: average precision (the area under the precision-recall
  curve), computed by interpolation and trapezoidal integration.

Model training and evaluation use the Ultralytics YOLO library and are run in
the notebooks in `notebooks/`.
"""

__version__ = "1.0.0"
