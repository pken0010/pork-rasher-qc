"""Tests for porkqc.passfail: pass/fail rule, threshold choice, bootstrap intervals."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from porkqc.passfail import bootstrap_ci, choose_threshold, passfail_counts, threshold_sweep

RESULTS = Path(__file__).resolve().parents[1] / "results"


def test_passfail_hand_calculated_example():
    # Trays:     bad   bad   good  good  bad
    max_conf = [0.9, 0.2, 0.6, 0.1, 0.0]
    is_bad = [True, True, False, False, True]
    r = passfail_counts(max_conf, is_bad, 0.5)
    assert (r["caught"], r["missed"], r["false_rejects"], r["correct_passes"]) == (1, 2, 1, 1)
    assert r["miss_rate"] == pytest.approx(2 / 3)
    assert r["false_reject_rate"] == pytest.approx(1 / 2)
    assert r["accuracy"] == pytest.approx(2 / 5)


def test_confidence_equal_to_threshold_fails():
    assert passfail_counts([0.5], [True], 0.5)["caught"] == 1


def test_no_good_trays_gives_zero_false_reject_rate():
    assert passfail_counts([0.9, 0.1], [True, True], 0.5)["false_reject_rate"] == 0


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        passfail_counts([0.1, 0.2], [True], 0.5)


def test_sweep_trade_off_is_monotonic():
    # Raising the threshold can only miss more bad trays and reject fewer good ones
    rng = np.random.default_rng(0)
    conf, bad = rng.random(300), rng.random(300) < 0.8
    s = threshold_sweep(conf, bad)
    assert (np.diff(s.miss_rate) >= 0).all()
    assert (np.diff(s.false_reject_rate) <= 0).all()


def test_choose_threshold_meets_target_with_fewest_rejects():
    sweep = pd.DataFrame({"threshold": [0.1, 0.2, 0.3],
                          "miss_rate": [0.01, 0.04, 0.10],
                          "false_reject_rate": [0.8, 0.5, 0.2]})
    assert choose_threshold(sweep, 0.05) == (0.2, True)


def test_choose_threshold_falls_back_when_target_unreachable():
    sweep = pd.DataFrame({"threshold": [0.1, 0.2],
                          "miss_rate": [0.2, 0.3],
                          "false_reject_rate": [0.5, 0.1]})
    assert choose_threshold(sweep, 0.05) == (0.1, False)


@pytest.mark.parametrize("model, expected", [("YOLO11n", 0.04), ("YOLO11s", 0.03)])
def test_reproduces_thresholds_chosen_in_notebook_03(model, expected):
    """Regression test: re-running the rule on the saved validation sweep gives the same choice."""
    sweep = pd.read_csv(RESULTS / "threshold_sweep.csv", index_col=[0, 1])
    t, reached = choose_threshold(sweep.loc[f"{model}_valid"], 0.05)
    assert t == pytest.approx(expected)
    assert reached


def test_saved_test_results_are_consistent():
    """The saved test sweep at the chosen threshold matches the reported pass/fail table."""
    sweep = pd.read_csv(RESULTS / "threshold_sweep.csv", index_col=[0, 1])
    row = sweep.loc["YOLO11n_test"].set_index("threshold").loc[0.04]
    assert (row.missed, row.caught + row.missed) == (3, 168)
    assert (row.false_rejects, row.false_rejects + row.correct_passes) == (22, 29)
    assert row.miss_rate == pytest.approx(row.missed / (row.missed + row.caught))


def test_bootstrap_interval_contains_point_estimate():
    rng = np.random.default_rng(1)
    conf, bad = rng.random(200), rng.random(200) < 0.7
    r = passfail_counts(conf, bad, 0.5)
    miss_ci, frej_ci = bootstrap_ci(conf, bad, 0.5, n_boot=1000)
    assert miss_ci[0] <= r["miss_rate"] <= miss_ci[1]
    assert frej_ci[0] <= r["false_reject_rate"] <= frej_ci[1]


def test_bootstrap_is_reproducible():
    conf, bad = [0.9, 0.1, 0.6, 0.3], [True, True, False, False]
    a = bootstrap_ci(conf, bad, 0.5, n_boot=200, seed=3)
    b = bootstrap_ci(conf, bad, 0.5, n_boot=200, seed=3)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_perfect_classifier_has_zero_width_interval():
    conf, bad = [0.9, 0.8, 0.1, 0.2], [True, True, False, False]
    miss_ci, frej_ci = bootstrap_ci(conf, bad, 0.5, n_boot=200)
    assert miss_ci.tolist() == [0, 0] and frej_ci.tolist() == [0, 0]


def test_bootstrap_interval_narrows_with_more_data():
    rng = np.random.default_rng(2)
    small = rng.random(50), rng.random(50) < 0.7
    large = rng.random(5000), rng.random(5000) < 0.7
    width_small = np.diff(bootstrap_ci(*small, 0.5, n_boot=500)[0])[0]
    width_large = np.diff(bootstrap_ci(*large, 0.5, n_boot=500)[0])[0]
    assert width_large < width_small
