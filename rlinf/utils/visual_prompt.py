# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Rasterize source and target markers consistently for datasets and evaluation."""

import numpy as np
from PIL import Image, ImageDraw

G1_PROMPT = (
    "pick the object in the cyan solid box and place it in the magenta dashed box"
)
G1_STYLE = {
    "source_color": [0, 255, 255],
    "target_color": [255, 0, 255],
    "line_width": 2,
    "target_dash_length": 6,
    "target_dash_gap": 4,
}


def validate_box(box: list[int], width: int, height: int) -> None:
    """Require inclusive integer xyxy coordinates inside the reference image."""
    if len(box) != 4 or any(type(v) is not int for v in box):
        raise ValueError("Boxes must contain four integer xyxy coordinates.")
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 < width and 0 <= y0 < y1 < height):
        raise ValueError(f"Box {box} is outside the {width}x{height} image.")


def render_g1_prompt(
    image: np.ndarray, source_box: list[int], target_box: list[int]
) -> np.ndarray:
    """Draw G1 outlines on a copy of an RGB reference, preserving other pixels."""
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("G1 reference must be an HWC uint8 RGB image.")
    height, width = image.shape[:2]
    for box in (source_box, target_box):
        validate_box(box, width, height)
    result = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(result)
    draw.rectangle(source_box, outline=tuple(G1_STYLE["source_color"]), width=2)
    x0, y0, x1, y1 = target_box
    color = tuple(G1_STYLE["target_color"])
    for x in range(x0, x1 + 1, 10):
        for y in (y0, y1 - 1):
            draw.rectangle([x, y, min(x + 5, x1), y + 1], fill=color)
    for y in range(y0, y1 + 1, 10):
        for x in (x0, x1 - 1):
            draw.rectangle([x, y, x + 1, min(y + 5, y1)], fill=color)
    return np.asarray(result)
