"""
Reading the dataset: YOLO label files and Roboflow file names.

**YOLO label format.** Each image `name.jpg` has a text file `name.txt` with
one line per bounding box:

    class_id  x_centre  y_centre  width  height

`class_id` is an integer index into the class list in `data.yaml`. The four box
values are *normalised*: fractions (0 to 1) of the image width and height, not
pixels. An empty (or missing) label file means the image has no defects.

**Roboflow file names.** Roboflow renames each exported image to
`<original name>.rf.<hash>.jpg`. Augmented copies of the same original image
share the part before `.rf.`. In this dataset the original names look like
`frame_with_red_8669_jpg`, where the number is the frame index in the source
video.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple, Optional

import yaml


class Box(NamedTuple):
    """One bounding box from a YOLO label file (normalised coordinates)."""

    class_id: int
    x_centre: float
    y_centre: float
    width: float
    height: float


def load_class_names(yaml_path: str | Path) -> list[str]:
    """
    Read the list of class names from a YOLO `data.yaml` file.

    Parameters
    ----------
    yaml_path : str or Path
        Path to `data.yaml`.

    Returns
    -------
    list of str
        Class names, ordered by class index. Both the list form
        (`names: [a, b]`) and the dictionary form (`names: {0: a, 1: b}`)
        are accepted.
    """
    with open(yaml_path) as f:
        names = yaml.safe_load(f)["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    return list(names)


def source_stem(filename: str) -> str:
    """
    Return the original image name, before Roboflow's `.rf.<hash>` suffix.

    Augmented copies of one original image all return the same stem.

    >>> source_stem("frame_with_red_8669_jpg.rf.ab12cd.jpg")
    'frame_with_red_8669_jpg'
    """
    return Path(filename).name.split(".rf.")[0]


def frame_number(filename: str) -> Optional[int]:
    """
    Return the video frame number in a file name, or None if there isn't one.

    >>> frame_number("frame_with_red_8669_jpg.rf.ab12cd.jpg")
    8669
    >>> frame_number("some_other_image.jpg") is None
    True
    """
    m = re.search(r"frame_with_red_(\d+)", Path(filename).name)
    return int(m.group(1)) if m else None


def parse_label_line(line: str) -> Optional[Box]:
    """
    Parse one line of a YOLO label file.

    Parameters
    ----------
    line : str
        A line such as `"3 0.71 0.62 0.20 0.15"`.

    Returns
    -------
    Box or None
        The parsed box, or None for a blank line.

    Raises
    ------
    ValueError
        If the line does not have exactly 5 values, the class id is negative,
        or a coordinate is outside the range 0 to 1. Polygon (segmentation)
        labels have more than 5 values and are rejected on purpose, since this
        project uses boxes only.
    """
    parts = line.split()
    if not parts:
        return None
    if len(parts) != 5:
        raise ValueError(f"expected 5 values per label line, got {len(parts)}: {line!r}")
    class_id = int(parts[0])
    coords = [float(p) for p in parts[1:]]
    if class_id < 0:
        raise ValueError(f"negative class id: {line!r}")
    if any(not 0.0 <= c <= 1.0 for c in coords):
        raise ValueError(f"box values must be normalised to 0-1: {line!r}")
    return Box(class_id, *coords)


def read_label_file(path: str | Path) -> list[Box]:
    """
    Read all boxes from a YOLO label file.

    A missing or empty file returns an empty list, which means the image has
    no defects (a "good" tray).
    """
    path = Path(path)
    if not path.exists():
        return []
    boxes = [parse_label_line(line) for line in path.read_text().splitlines()]
    return [b for b in boxes if b is not None]


def label_classes(path: str | Path, class_names: list[str]) -> set[str]:
    """Return the set of class names present in a label file (empty set = good tray)."""
    return {class_names[b.class_id] for b in read_label_file(path)}


def to_pixels(box: Box, image_width: int, image_height: int) -> tuple[float, float, float, float]:
    """
    Convert a normalised box to pixel corner coordinates.

    Returns
    -------
    tuple of float
        `(x_min, y_min, x_max, y_max)` in pixels.
    """
    x0 = (box.x_centre - box.width / 2) * image_width
    y0 = (box.y_centre - box.height / 2) * image_height
    return x0, y0, x0 + box.width * image_width, y0 + box.height * image_height


def area_fraction(box: Box) -> float:
    """Box area as a fraction of the image area (0.01 = 1 % of the image)."""
    return box.width * box.height
