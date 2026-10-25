"""Tests for porkqc.metrics: average precision by interpolation + trapezoidal integration."""

import numpy as np
import pytest

from porkqc.metrics import average_precision, precision_envelope

GRID_ERROR = 0.01   # 101-point grid has spacing 0.01, so a step edge costs at most ~1 grid step


def test_envelope_never_increases():
    env = precision_envelope([0.3, 0.9, 0.5, 0.7, 0.2])
    assert env.tolist() == [0.9, 0.9, 0.7, 0.7, 0.2]
    assert (np.diff(env) <= 0).all()


def test_perfect_detector_has_ap_one():
    assert average_precision([0.5, 1.0], [1.0, 1.0]) == pytest.approx(1.0, abs=GRID_ERROR)


def test_known_area_rectangle():
    # Precision 1 up to recall 0.4, then nothing: the exact area is 0.4
    assert average_precision([0.2, 0.4], [1.0, 1.0]) == pytest.approx(0.4, abs=GRID_ERROR)


def test_known_area_constant_precision():
    # Precision 0.5 from recall 0.25 to 1, plus the sentinel ramp from (0, 1) to (0.25, 0.5):
    # exact area = 0.25 * (1 + 0.5) / 2 + 0.75 * 0.5 = 0.5625
    assert average_precision([0.25, 0.5, 1.0], [0.5, 0.5, 0.5]) == pytest.approx(0.5625, abs=GRID_ERROR)


def test_finer_grid_converges_to_exact_area():
    # Numerical convergence: more interpolation points -> smaller integration error
    errors = [abs(average_precision([0.2, 0.4], [1.0, 1.0], n_points=n) - 0.4) for n in (11, 101, 1001)]
    assert errors[0] > errors[1] > errors[2]


def test_matches_ultralytics_implementation():
    metrics = pytest.importorskip("ultralytics.utils.metrics")
    rng = np.random.default_rng(0)
    for _ in range(20):
        recall = np.sort(rng.random(30))
        precision = rng.random(30)
        expected, _, _ = metrics.compute_ap(recall, precision)
        assert average_precision(recall, precision) == pytest.approx(expected, abs=1e-12)
