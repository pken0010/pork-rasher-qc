"""
Building a leakage-free "blocked" train/valid/test split.

Frame numbers are cut into blocks of `block_size` consecutive frames, and each
*whole block* is sent to one split. Neighbouring frames (which show the same
tray) therefore can't end up on both sides of the train/test boundary.

Because some defect classes are rare, a single random assignment might put all
examples of a class in one split. `best_blocked_split` therefore tries many
random assignments and keeps the one whose class mix in valid and test is
closest to the target fractions (a *stratified group split*). Only the labels
are used to choose, never any model results, so the test set stays unbiased.

Finally, `apply_buffer` removes training images that are within `buffer`
frames of any valid/test image, as a safety margin at block edges.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

SPLITS = ("train", "valid", "test")
DEFAULT_TARGET = {"train": 0.70, "valid": 0.15, "test": 0.15}
NO_DEFECT = "(no defect)"


def frame_blocks(frames: Sequence[Optional[float]], block_size: int) -> pd.Series:
    """
    Block index (`frame // block_size`) for each frame; missing frames give <NA>.

    >>> frame_blocks([0, 199, 200, None], 200).tolist()
    [0, 0, 1, <NA>]
    """
    return (pd.Series(frames, dtype="Float64") // block_size).astype("Int64")


def assign_blocks(blocks: pd.Series, seed: int, target: dict = DEFAULT_TARGET) -> pd.Series:
    """
    Randomly send each whole block to train, valid or test.

    Blocks are shuffled, then filled into **test** until it holds about
    `target["test"]` of the images, then **valid**, then everything else goes to
    **train**. Images without a block (no frame number) go to train.

    Parameters
    ----------
    blocks : pandas.Series of Int64
        Block index per image, from `frame_blocks`.
    seed : int
        Random seed (the same seed always gives the same split).
    target : dict
        Target share of images per split.

    Returns
    -------
    pandas.Series of str
        Split name per image, aligned with `blocks`.
    """
    rng = np.random.default_rng(seed)
    unique_blocks = blocks.dropna().unique().to_numpy(dtype="int64")
    rng.shuffle(unique_blocks)
    sizes = blocks.groupby(blocks).size()
    total, running, block_split = sizes.sum(), 0, {}
    for b in unique_blocks:
        frac = running / total
        block_split[b] = ("test" if frac < target["test"] else
                          "valid" if frac < target["test"] + target["valid"] else "train")
        running += sizes[b]
    return blocks.map(block_split).fillna("train")


def split_score(classes: Sequence[set], assignment: pd.Series, class_names: Sequence[str],
                target: dict = DEFAULT_TARGET) -> float:
    """
    How far a split's class mix is from the target (lower is better).

    For each class (plus "no defect"), the share of that class's images in
    valid and in test is compared with the target share. A penalty of 1 is
    added if a class with at least 3 images is missing entirely from valid or
    test.

    Parameters
    ----------
    classes : sequence of set of str
        Classes present in each image (empty set = good tray).
    assignment : pandas.Series of str
        Split name per image.
    class_names : sequence of str
        All class names.
    """
    classes = pd.Series(list(classes), index=assignment.index)
    score = 0.0
    for c in list(class_names) + [NO_DEFECT]:
        has_c = (classes.apply(len) == 0) if c == NO_DEFECT else classes.apply(lambda s: c in s)
        n = has_c.sum()
        if n == 0:
            continue
        for s in ("valid", "test"):
            share = (has_c & (assignment == s)).sum() / n
            score += abs(share - target[s])
            if n >= 3 and share == 0:
                score += 1.0
    return score


def apply_buffer(frames: Sequence[Optional[float]], assignment: pd.Series, buffer: int) -> pd.Series:
    """
    Mark training images within `buffer` frames of a valid/test image as dropped.

    Returns a new assignment where those images are labelled
    `"dropped (buffer)"`; other images are unchanged.
    """
    frames = pd.Series(frames, index=assignment.index, dtype="Float64")
    held_out = np.sort(frames[assignment != "train"].dropna().to_numpy(dtype=float))
    out = assignment.copy()
    if len(held_out) == 0:
        return out
    for i in out.index[(out == "train") & frames.notna()]:
        f = float(frames[i])
        j = np.searchsorted(held_out, f)
        near = held_out[max(j - 1, 0): j + 1]
        if np.min(np.abs(near - f)) <= buffer:
            out[i] = "dropped (buffer)"
    return out


def best_blocked_split(frames: Sequence[Optional[float]], classes: Sequence[set],
                       class_names: Sequence[str], block_size: int = 200, buffer: int = 15,
                       n_seeds: int = 500, target: dict = DEFAULT_TARGET) -> tuple[pd.Series, int]:
    """
    Build the blocked split: try `n_seeds` assignments, keep the best, apply the buffer.

    Parameters
    ----------
    frames : sequence of int or None
        Frame number per image (one copy per original image).
    classes : sequence of set of str
        Classes present in each image.
    class_names : sequence of str
        All class names.
    block_size : int, default 200
        Frames per block.
    buffer : int, default 15
        Safety gap in frames between train and valid/test.
    n_seeds : int, default 500
        Number of random assignments to try.
    target : dict
        Target share of images per split.

    Returns
    -------
    assignment : pandas.Series of str
        Split name per image (`"train"`, `"valid"`, `"test"` or `"dropped (buffer)"`).
    best_seed : int
        The seed that gave the best class balance.
    """
    blocks = frame_blocks(frames, block_size)
    scores = {s: split_score(classes, assign_blocks(blocks, s, target), class_names, target)
              for s in range(n_seeds)}
    best_seed = min(scores, key=scores.get)
    assignment = assign_blocks(blocks, best_seed, target)
    return apply_buffer(frames, assignment, buffer), best_seed
