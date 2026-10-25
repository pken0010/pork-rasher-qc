"""Tests for porkqc.data: label parsing and file-name handling."""

import pytest

from porkqc.data import (Box, area_fraction, frame_number, label_classes, load_class_names,
                         parse_label_line, read_label_file, source_stem, to_pixels)

CLASSES = ["loose-meat", "packaging-error", "twisted-meat", "unsealed", "wrinkle"]


def test_source_stem_removes_roboflow_suffix():
    assert source_stem("frame_with_red_8669_jpg.rf.ab12cd.jpg") == "frame_with_red_8669_jpg"


def test_augmented_copies_share_a_stem():
    a = source_stem("frame_with_red_10_jpg.rf.111.jpg")
    b = source_stem("frame_with_red_10_jpg.rf.222.jpg")
    assert a == b


def test_frame_number_found_and_missing():
    assert frame_number("frame_with_red_8669_jpg.rf.ab12cd.jpg") == 8669
    assert frame_number("/some/folder/frame_with_red_7_jpg.rf.x.jpg") == 7
    assert frame_number("IMG_0001.jpg") is None


def test_parse_valid_line():
    assert parse_label_line("3 0.71 0.62 0.20 0.15") == Box(3, 0.71, 0.62, 0.20, 0.15)


def test_parse_blank_line_returns_none():
    assert parse_label_line("   ") is None


@pytest.mark.parametrize("bad_line", [
    "3 0.5 0.5 0.2",                      # too few values
    "3 0.1 0.1 0.2 0.1 0.3 0.3 0.1 0.4",  # polygon label, not a box
    "3 0.5 0.5 1.2 0.1",                  # width > 1: not normalised
    "-1 0.5 0.5 0.2 0.1",                 # negative class id
])
def test_parse_rejects_invalid_lines(bad_line):
    with pytest.raises(ValueError):
        parse_label_line(bad_line)


def test_missing_and_empty_label_files_mean_no_defects(tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text("")
    assert read_label_file(tmp_path / "does_not_exist.txt") == []
    assert read_label_file(empty) == []
    assert label_classes(empty, CLASSES) == set()


def test_read_label_file_and_classes(tmp_path):
    f = tmp_path / "img.txt"
    f.write_text("3 0.5 0.5 0.2 0.1\n\n1 0.2 0.3 0.1 0.1\n")
    assert len(read_label_file(f)) == 2
    assert label_classes(f, CLASSES) == {"unsealed", "packaging-error"}


def test_to_pixels_known_answer():
    # A box centred in a 720 x 540 image, 20 % wide and 40 % tall
    x0, y0, x1, y1 = to_pixels(Box(0, 0.5, 0.5, 0.2, 0.4), 720, 540)
    assert (x0, y0, x1, y1) == pytest.approx((288, 162, 432, 378))


def test_area_fraction():
    assert area_fraction(Box(0, 0.5, 0.5, 0.2, 0.15)) == pytest.approx(0.03)


@pytest.mark.parametrize("names_yaml", [
    "names: ['a', 'b', 'c']\n",
    "names:\n  0: a\n  1: b\n  2: c\n",
])
def test_load_class_names_list_and_dict_forms(tmp_path, names_yaml):
    y = tmp_path / "data.yaml"
    y.write_text(names_yaml)
    assert load_class_names(y) == ["a", "b", "c"]
