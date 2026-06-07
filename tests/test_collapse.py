"""Tests for slitscan.engine.collapse (Phase 7)."""

import numpy as np
import pytest

from slitscan.engine.collapse import collapse
from slitscan.meta import ClipMeta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_meta(n_frames, h=10, w=20, c=3):
    return ClipMeta(fps=24.0, frame_count=n_frames, width=w, height=h, channels=c)


def make_frames_iter(n_frames, h=10, w=20, c=3):
    """Each frame: pixels in column x have value x (easy to identify slit)."""
    for _ in range(n_frames):
        frame = np.zeros((h, w, c), dtype=np.uint8)
        for x in range(w):
            frame[:, x, :] = x
        yield frame


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

def test_collapse_output_shape_axis_x():
    meta = make_meta(5, h=10, w=20)
    result = collapse(
        meta, make_frames_iter(5),
        slit_position=0.5, direction="forward", axis="x", slice_width=1,
    )
    assert result.shape == (10, 5, 3)


def test_collapse_output_shape_axis_y():
    meta = make_meta(5, h=10, w=20)
    frames = (np.zeros((10, 20, 3), dtype=np.uint8) for _ in range(5))
    result = collapse(
        meta, frames,
        slit_position=0.5, direction="forward", axis="y", slice_width=1,
    )
    assert result.shape == (5, 20, 3)


# ---------------------------------------------------------------------------
# Slit-position sampling tests
# ---------------------------------------------------------------------------

def test_collapse_slit_samples_correct_column():
    """With slit_position=0.0, samples column 0 (value 0 in our frames)."""
    meta = make_meta(5, h=10, w=20)
    result = collapse(
        meta, make_frames_iter(5),
        slit_position=0.0, direction="forward", axis="x", slice_width=1,
    )
    assert result[0, 0, 0] == 0  # column 0 → value 0


def test_collapse_slit_position_1_samples_last_column():
    """With slit_position=1.0, samples last column (value w-1 = 19)."""
    meta = make_meta(5, h=10, w=20)
    result = collapse(
        meta, make_frames_iter(5),
        slit_position=1.0, direction="forward", axis="x", slice_width=1,
    )
    assert result[0, 0, 0] == 19  # column 19 → value 19


def test_collapse_slit_mid_position():
    """slit_position=0.5 on a 20-wide frame → column 10 (rounded)."""
    meta = make_meta(3, h=4, w=20)
    result = collapse(
        meta, make_frames_iter(3, h=4, w=20),
        slit_position=0.5, direction="forward", axis="x", slice_width=1,
    )
    # slit_px = round(0.5 * 19) = round(9.5) = 10 (Python banker's rounding → 10)
    # center_col = 10 + 0 = 10
    assert result[0, 0, 0] == 10


# ---------------------------------------------------------------------------
# Direction test
# ---------------------------------------------------------------------------

def test_collapse_direction_reverse():
    """Reverse direction reverses time accumulation in the output."""
    meta = make_meta(5, h=4, w=10)

    def id_frames():
        for i in range(5):
            yield np.full((4, 10, 3), i, dtype=np.uint8)

    fwd = collapse(
        meta, id_frames(),
        slit_position=0.5, direction="forward", axis="x", slice_width=1,
    )

    def id_frames2():
        for i in range(5):
            yield np.full((4, 10, 3), i, dtype=np.uint8)

    rev = collapse(
        meta, id_frames2(),
        slit_position=0.5, direction="reverse", axis="x", slice_width=1,
    )

    # Forward: output col 0 = frame 0, output col 4 = frame 4
    assert fwd[0, 0, 0] == 0
    assert fwd[0, 4, 0] == 4

    # Reverse: output col 0 = frame 4, output col 4 = frame 0
    assert rev[0, 0, 0] == 4
    assert rev[0, 4, 0] == 0


# ---------------------------------------------------------------------------
# axis="y" sampling
# ---------------------------------------------------------------------------

def test_collapse_axis_y_samples_row():
    """Each frame has row r = value r; slit_position=0.0 samples row 0."""
    meta = make_meta(3, h=10, w=8)

    def row_frames():
        for _ in range(3):
            frame = np.zeros((10, 8, 3), dtype=np.uint8)
            for r in range(10):
                frame[r, :, :] = r
            yield frame

    result = collapse(
        meta, row_frames(),
        slit_position=0.0, direction="forward", axis="y", slice_width=1,
    )
    assert result.shape == (3, 8, 3)
    # slit_position=0.0 → row 0 → value 0
    assert result[0, 0, 0] == 0


# ---------------------------------------------------------------------------
# slice_width > 1
# ---------------------------------------------------------------------------

def test_collapse_slice_width_samples_center():
    """slice_width=3 on a 20-wide frame with slit_position=0.0.
    slit_px=0, center_col = 0 + 3//2 = 1 → value 1."""
    meta = make_meta(3, h=4, w=20)
    result = collapse(
        meta, make_frames_iter(3, h=4, w=20),
        slit_position=0.0, direction="forward", axis="x", slice_width=3,
    )
    assert result.shape == (4, 3, 3)
    assert result[0, 0, 0] == 1  # center of [0,1,2] → col 1 → value 1


# ---------------------------------------------------------------------------
# Invalid parameter guards
# ---------------------------------------------------------------------------

def test_collapse_invalid_axis():
    meta = make_meta(2)
    with pytest.raises(ValueError, match="axis"):
        collapse(meta, iter([]), slit_position=0.5, direction="forward", axis="z", slice_width=1)


def test_collapse_invalid_direction():
    meta = make_meta(2)
    with pytest.raises(ValueError, match="direction"):
        collapse(meta, iter([]), slit_position=0.5, direction="backward", axis="x", slice_width=1)
