"""
Detecting data leakage between dataset splits.

The images in this dataset are frames from a video of a conveyor belt.
Neighbouring frames show the same tray a fraction of a second apart, so if one
frame is in the training set and the next one is in the test set, the test is
not a fair measure of performance on unseen trays.

Two independent checks are provided:

1. **Frame-number gap** (`nearest_gap`): how many frames separate each test
   image from the closest training image.
2. **Perceptual hash** (`dhash`, `hamming_distance`, `nearest_hamming`): a
   64-bit "fingerprint" of what an image looks like. Similar images have
   fingerprints that differ in only a few bits.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def dhash(image: str | Path | Image.Image, size: int = 8) -> int:
    """
    Compute a difference hash ("dHash") of an image.

    The image is converted to greyscale and shrunk to `(size + 1) x size`
    pixels. Each bit of the hash records whether a pixel is brighter than its
    right-hand neighbour, which captures the image's overall structure while
    ignoring small changes in brightness or colour.

    Parameters
    ----------
    image : path or PIL.Image
        The image to hash.
    size : int, default 8
        Hash grid size. `size = 8` gives a 64-bit hash.

    Returns
    -------
    int
        The hash as a non-negative integer with `size * size` bits.
    """
    img = image if isinstance(image, Image.Image) else Image.open(image)
    g = np.asarray(img.convert("L").resize((size + 1, size)), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming_distance(a: int, b: int) -> int:
    """
    Number of bits that differ between two hashes.

    0 means the images look the same; for 64-bit dHashes, about 10 or fewer
    means very similar.
    """
    return bin(a ^ b).count("1")


def nearest_hamming(query: list[int], reference: list[int]) -> np.ndarray:
    """
    For each hash in `query`, the smallest Hamming distance to any hash in `reference`.

    Parameters
    ----------
    query, reference : list of int
        64-bit hashes from `dhash`.

    Returns
    -------
    numpy.ndarray of int
        One distance per query hash.
    """
    q = np.array(query, dtype=np.uint64)
    r = np.array(reference, dtype=np.uint64)
    x = np.bitwise_xor(q[:, None], r[None, :])
    bits = np.unpackbits(x.view(np.uint8).reshape(*x.shape, 8), axis=-1)
    return bits.sum(axis=-1).min(axis=1)


def nearest_gap(frames: np.ndarray, reference_frames: np.ndarray) -> np.ndarray:
    """
    Frame-number distance from each frame to the closest reference frame.

    Parameters
    ----------
    frames : array of int
        Frame numbers to check (e.g. the test set).
    reference_frames : array of int
        Frame numbers to compare against (e.g. the training set). Must not be empty.

    Returns
    -------
    numpy.ndarray of int
        `min |f - r|` over the reference frames, for each frame `f`.
    """
    ref = np.sort(np.asarray(reference_frames))
    if len(ref) == 0:
        raise ValueError("reference_frames must not be empty")
    frames = np.asarray(frames)
    i = np.searchsorted(ref, frames)
    left = ref[np.clip(i - 1, 0, len(ref) - 1)]
    right = ref[np.clip(i, 0, len(ref) - 1)]
    return np.minimum(np.abs(frames - left), np.abs(frames - right))
