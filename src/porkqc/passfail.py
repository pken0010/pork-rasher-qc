"""
Turning detections into a pass/fail decision per tray.

**Rule.** A tray *fails* if the detector finds any defect with confidence at or
above a threshold `t`; otherwise it *passes*. Each tray is summarised by
`max_conf`, the highest confidence of any detection on it (0 if nothing was
detected), so the rule is simply `fail = max_conf >= t`.

**Truth.** A tray is *bad* if its label file has at least one box.

| | Model says FAIL | Model says PASS |
|---|---|---|
| **Tray is bad** | caught | **missed** (bad tray reaches customers) |
| **Tray is good** | **false reject** (good tray wasted) | correct pass |

- miss rate = missed / all bad trays
- false-reject rate = false rejects / all good trays

Plain accuracy is deliberately *not* used to choose the threshold. In this
dataset most trays are bad, so a rule that fails every tray would already score
about 85 % accuracy.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def passfail_counts(max_conf: Sequence[float], is_bad: Sequence[bool], threshold: float) -> dict:
    """
    Pass/fail counts and rates at one threshold.

    Parameters
    ----------
    max_conf : sequence of float
        Highest detection confidence per tray (0 if no detections).
    is_bad : sequence of bool
        True if the tray has at least one labelled defect.
    threshold : float
        Confidence threshold `t`; a tray fails if `max_conf >= t`.

    Returns
    -------
    dict
        `threshold, caught, missed, false_rejects, correct_passes, miss_rate,
        false_reject_rate, accuracy`. A rate is 0 if its denominator is 0.
    """
    fail = np.asarray(max_conf, dtype=float) >= threshold
    bad = np.asarray(is_bad, dtype=bool)
    if fail.shape != bad.shape:
        raise ValueError("max_conf and is_bad must have the same length")
    tp, fn = int((fail & bad).sum()), int((~fail & bad).sum())
    fp, tn = int((fail & ~bad).sum()), int((~fail & ~bad).sum())
    return dict(threshold=threshold, caught=tp, missed=fn, false_rejects=fp, correct_passes=tn,
                miss_rate=fn / max(tp + fn, 1), false_reject_rate=fp / max(fp + tn, 1),
                accuracy=(tp + tn) / max(len(bad), 1))


def threshold_sweep(max_conf: Sequence[float], is_bad: Sequence[bool],
                    thresholds: Sequence[float] | None = None) -> pd.DataFrame:
    """
    `passfail_counts` for every threshold; one row per threshold.

    Defaults to thresholds 0.01, 0.02, ..., 0.95 (as in notebook 03).
    """
    if thresholds is None:
        thresholds = np.round(np.arange(0.01, 0.96, 0.01), 2)
    return pd.DataFrame([passfail_counts(max_conf, is_bad, t) for t in thresholds])


def choose_threshold(sweep: pd.DataFrame, max_miss_rate: float = 0.05) -> tuple[float, bool]:
    """
    Pick a threshold from a sweep (run this on the **validation** set).

    Among thresholds with `miss_rate <= max_miss_rate`, choose the one with the
    lowest false-reject rate (ties: the higher threshold). If no threshold meets
    the target, choose the lowest miss rate (ties: the higher threshold).

    Returns
    -------
    threshold : float
        The chosen threshold.
    target_reached : bool
        Whether `max_miss_rate` was achievable.
    """
    ok = sweep[sweep.miss_rate <= max_miss_rate]
    if len(ok):
        best = ok.sort_values(["false_reject_rate", "threshold"], ascending=[True, False]).iloc[0]
    else:
        best = sweep.sort_values(["miss_rate", "threshold"], ascending=[True, False]).iloc[0]
    return float(best.threshold), bool(len(ok))


def bootstrap_ci(max_conf: Sequence[float], is_bad: Sequence[bool], threshold: float,
                 n_boot: int = 2000, seed: int = 0, level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    """
    Bootstrap confidence intervals for the miss rate and false-reject rate.

    The test set is resampled with replacement `n_boot` times; the rates are
    recomputed for each resample, and the central `level` share of the results
    gives the interval (percentile bootstrap).

    Returns
    -------
    miss_ci, false_reject_ci : numpy.ndarray
        `[low, high]` for each rate.
    """
    rng = np.random.default_rng(seed)
    fail = np.asarray(max_conf, dtype=float) >= threshold
    bad = np.asarray(is_bad, dtype=bool)
    idx = rng.integers(0, len(bad), size=(n_boot, len(bad)))
    f, b = fail[idx], bad[idx]
    miss = (~f & b).sum(1) / np.maximum(b.sum(1), 1)
    frej = (f & ~b).sum(1) / np.maximum((~b).sum(1), 1)
    tail = 100 * (1 - level) / 2
    return np.percentile(miss, [tail, 100 - tail]), np.percentile(frej, [tail, 100 - tail])
