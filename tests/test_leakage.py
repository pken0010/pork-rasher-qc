"""Tests for porkqc.leakage: perceptual hashes and frame gaps."""

import numpy as np
import pytest
from PIL import Image, ImageEnhance

from porkqc.leakage import dhash, hamming_distance, nearest_gap, nearest_hamming


def random_image(seed, size=(720, 540)):
    """A reproducible random test image (smooth, so it has structure)."""
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, (6, 8, 3), dtype=np.uint8)
    return Image.fromarray(small).resize(size, Image.BILINEAR)


def test_identical_images_have_distance_zero():
    img = random_image(0)
    assert hamming_distance(dhash(img), dhash(img.copy())) == 0


def test_brightness_change_keeps_hash_close():
    # Roboflow's augmentations change exposure; the hash should barely move
    img = random_image(1)
    brighter = ImageEnhance.Brightness(img).enhance(1.2)
    assert hamming_distance(dhash(img), dhash(brighter)) <= 5


def test_different_images_are_far_apart():
    assert hamming_distance(dhash(random_image(2)), dhash(random_image(3))) > 15


def test_hash_has_64_bits_or_fewer():
    assert 0 <= dhash(random_image(4)) < 2 ** 64


def test_dhash_accepts_a_file_path(tmp_path):
    p = tmp_path / "x.png"
    random_image(5).save(p)
    assert dhash(p) == dhash(random_image(5))


def test_hamming_distance_known_answer():
    assert hamming_distance(0b1011, 0b0001) == 2


def test_nearest_hamming_matches_brute_force():
    rng = np.random.default_rng(0)
    query = [int(x) for x in rng.integers(0, 2 ** 63, 20, dtype=np.uint64)]
    ref = [int(x) for x in rng.integers(0, 2 ** 63, 30, dtype=np.uint64)] + [2 ** 64 - 1]
    expected = [min(hamming_distance(q, r) for r in ref) for q in query]
    assert nearest_hamming(query, ref).tolist() == expected


def test_nearest_gap_known_answer():
    assert nearest_gap([5, 100, 10], [1, 10, 98]).tolist() == [4, 2, 0]


def test_nearest_gap_outside_range():
    assert nearest_gap([0, 500], [10, 20]).tolist() == [10, 480]


def test_nearest_gap_empty_reference_raises():
    with pytest.raises(ValueError):
        nearest_gap([1, 2], [])
