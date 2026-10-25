"""Tests for porkqc.splitting: the leakage-free blocked split."""

import numpy as np
import pandas as pd

from porkqc.leakage import nearest_gap
from porkqc.splitting import (apply_buffer, assign_blocks, best_blocked_split, frame_blocks,
                              split_score)

CLASSES = ["common", "rare"]


def synthetic_video(n_runs=60, seed=0):
    """Frames in short consecutive runs (like a tray moving past the camera)."""
    rng = np.random.default_rng(seed)
    frames, classes = [], []
    for start in rng.choice(20000, n_runs, replace=False):
        for f in range(start, start + 10, 2):
            frames.append(int(f))
            r = rng.random()
            classes.append(set() if r < 0.15 else {"rare"} if r < 0.20 else {"common"})
    return frames, classes


def test_frame_blocks():
    assert frame_blocks([0, 199, 200, 401], 200).tolist() == [0, 0, 1, 2]
    assert frame_blocks([None], 200).isna().all()


def test_whole_blocks_stay_together():
    frames, _ = synthetic_video()
    blocks = frame_blocks(frames, 200)
    assignment = assign_blocks(blocks, seed=3)
    splits_per_block = pd.DataFrame({"b": blocks, "s": assignment}).groupby("b").s.nunique()
    assert (splits_per_block == 1).all()


def test_same_seed_same_split():
    frames, _ = synthetic_video()
    blocks = frame_blocks(frames, 200)
    assert assign_blocks(blocks, 7).equals(assign_blocks(blocks, 7))


def test_images_without_frame_go_to_train():
    blocks = frame_blocks([0, 500, None], 200)
    assert assign_blocks(blocks, 0).iloc[2] == "train"


def test_split_score_penalises_missing_class():
    classes = [{"rare"}] * 3 + [{"common"}] * 7
    all_train = pd.Series(["train"] * 10)
    balanced = pd.Series(["valid", "test", "train"] + ["train"] * 5 + ["valid", "test"])
    assert split_score(classes, all_train, CLASSES) > split_score(classes, balanced, CLASSES)


def test_buffer_removes_close_training_frames():
    frames = [0, 10, 20, 100, 200]
    assignment = pd.Series(["train", "train", "test", "train", "valid"])
    out = apply_buffer(frames, assignment, buffer=15)
    assert out.tolist() == ["train", "dropped (buffer)", "test", "train", "valid"]


def test_blocked_split_has_no_close_neighbours():
    frames, classes = synthetic_video()
    assignment, _ = best_blocked_split(frames, classes, CLASSES, block_size=200, buffer=15,
                                       n_seeds=50)
    frames = np.array(frames)
    train = frames[(assignment == "train").to_numpy()]
    held_out = frames[assignment.isin(["valid", "test"]).to_numpy()]
    assert nearest_gap(held_out, train).min() > 15


def test_blocked_split_puts_rare_class_in_test_and_valid():
    frames, classes = synthetic_video()
    assignment, _ = best_blocked_split(frames, classes, CLASSES, n_seeds=100)
    rare = pd.Series(["rare" in c for c in classes])
    for split in ["valid", "test"]:
        assert (rare & (assignment == split)).sum() > 0


def test_blocked_split_sizes_near_target():
    frames, classes = synthetic_video()
    assignment, _ = best_blocked_split(frames, classes, CLASSES, n_seeds=50)
    kept = assignment[assignment != "dropped (buffer)"]
    share_test = (kept == "test").mean()
    assert 0.08 < share_test < 0.25
