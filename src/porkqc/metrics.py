"""
Average precision (AP): the area under the precision-recall curve.

AP is the building block of **mAP50**, the main detection score in this
project (mAP = AP averaged over the defect classes; "50" means a predicted box
counts as correct if it overlaps the true box by at least 50 %).

Computing AP is a small **numerical integration** problem:

1. Add end points so the curve runs from recall 0 to recall 1.
2. Replace precision by its *envelope* (the highest precision at that recall or
   any higher recall), so the curve never goes up as recall increases.
3. **Interpolate** the envelope at 101 evenly spaced recall values (0, 0.01, ..., 1),
   the COCO convention.
4. **Integrate** with the trapezoidal rule.

This mirrors the method used inside Ultralytics YOLO; the tests check that both
give the same answer.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

_trapezoid = getattr(np, "trapezoid", None) or np.trapz  # np.trapz was renamed in NumPy 2.0


def precision_envelope(precision: Sequence[float]) -> np.ndarray:
    """
    Running maximum of precision taken from the right.

    >>> precision_envelope([0.5, 1.0, 0.6, 0.8]).tolist()
    [1.0, 1.0, 0.8, 0.8]
    """
    p = np.asarray(precision, dtype=float)
    return np.flip(np.maximum.accumulate(np.flip(p)))


def average_precision(recall: Sequence[float], precision: Sequence[float],
                      n_points: int = 101) -> float:
    """
    Area under the precision-recall curve (interpolated, trapezoidal rule).

    Parameters
    ----------
    recall : sequence of float
        Recall values, increasing (as the confidence threshold decreases).
    precision : sequence of float
        Precision at each recall value.
    n_points : int, default 101
        Number of evenly spaced recall points used for interpolation.

    Returns
    -------
    float
        AP between 0 and 1. A perfect detector (precision 1 at every recall
        up to 1) gives 1.
    """
    recall = np.asarray(recall, dtype=float)
    precision = np.asarray(precision, dtype=float)
    last = recall[-1] if len(recall) else 1.0
    mrec = np.concatenate(([0.0], recall, [last], [1.0]))
    mpre = precision_envelope(np.concatenate(([1.0], precision, [0.0], [0.0])))
    x = np.linspace(0, 1, n_points)
    return float(_trapezoid(np.interp(x, mrec, mpre), x))
