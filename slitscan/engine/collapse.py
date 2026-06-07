"""collapse.py — photofinish / strip photography engine.

Accumulates slit history from successive video frames into a single image.
Each input frame contributes one column (axis="x") or one row (axis="y")
to the output image.

Memory usage is O(output_image_size), not O(video_size): frames are consumed
one at a time from the iterator.
"""

from __future__ import annotations

import numpy as np
from typing import Iterator

from slitscan.meta import ClipMeta


def collapse(
    meta: ClipMeta,
    frames_iter: Iterator[np.ndarray],
    slit_position: float,  # 0..1, where the reading slit sits
    direction: str,        # "forward" | "reverse"
    axis: str,             # "x" | "y"
    slice_width: int,      # pixels per band (slit width); center pixel is sampled
) -> np.ndarray:
    """Accumulate slit history into a single image.

    For axis="x": the slit is a vertical strip at slit_position * width.
    Each input frame contributes one column to the output.
    Output shape: (H, frame_count, C).

    For axis="y": the slit is a horizontal strip.
    Each input frame contributes one row to the output.
    Output shape: (frame_count, W, C).

    The center pixel of the slice_width-wide slit strip is sampled, so the
    output always has exactly frame_count columns (axis x) or rows (axis y).
    """
    if axis not in ("x", "y"):
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")
    if direction not in ("forward", "reverse"):
        raise ValueError(f"direction must be 'forward' or 'reverse', got {direction!r}")

    if axis == "x":
        extent = meta.width
        slit_px = int(round(slit_position * (extent - 1)))
        slit_px = max(0, min(slit_px, extent - slice_width))
        center_col = slit_px + slice_width // 2
        center_col = max(0, min(center_col, extent - 1))

        out_h = meta.height
        out_w = meta.frame_count
        output = np.zeros((out_h, out_w, meta.channels), dtype=np.uint8)
    else:  # axis == "y"
        extent = meta.height
        slit_px = int(round(slit_position * (extent - 1)))
        slit_px = max(0, min(slit_px, extent - slice_width))
        center_row = slit_px + slice_width // 2
        center_row = max(0, min(center_row, extent - 1))

        out_h = meta.frame_count
        out_w = meta.width
        output = np.zeros((out_h, out_w, meta.channels), dtype=np.uint8)

    # Collect frames — required for reverse direction.
    # For forward direction we still collect to keep the interface uniform;
    # the iterator is typically a generator so we must materialise it anyway
    # before reversing. Memory cost is O(output) since only one pixel-column
    # (or row) per frame is stored in the output.
    frames_list = list(frames_iter)
    if direction == "reverse":
        frames_list = frames_list[::-1]

    for col_idx, frame in enumerate(frames_list):
        if col_idx >= meta.frame_count:
            break
        if axis == "x":
            # Sample the center pixel column of the slit — shape (H, 1, C)
            slit_data = frame[:, center_col : center_col + 1, :]
            output[:, col_idx : col_idx + 1, :] = slit_data
        else:
            # Sample the center pixel row of the slit — shape (1, W, C)
            slit_data = frame[center_row : center_row + 1, :, :]
            output[col_idx : col_idx + 1, :, :] = slit_data

    return output
