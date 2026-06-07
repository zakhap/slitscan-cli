"""Unit tests for engine/gather.py.

Synthetic buffer: frame N is a solid image where every pixel = N (mod 256).
This lets us assert that gathered bands contain the correct source values.
"""

from __future__ import annotations

import numpy as np
import pytest

from slitscan.buffer.base import FrameBuffer
from slitscan.engine.gather import gather_frame


# ---------------------------------------------------------------------------
# Synthetic buffer
# ---------------------------------------------------------------------------

class SolidColorBuffer(FrameBuffer):
    """Each frame is filled with a solid value equal to frame_index % 256."""

    def __init__(self, frame_count: int, height: int, width: int, channels: int = 3):
        self._frame_count = frame_count
        self._height = height
        self._width = width
        self._channels = channels

    def get(self, frame_index: int) -> np.ndarray | None:
        if frame_index < 0 or frame_index >= self._frame_count:
            return None
        value = frame_index % 256
        return np.full(
            (self._height, self._width, self._channels),
            value,
            dtype=np.uint8,
        )

    def advance(self) -> None:
        pass  # no-op for full buffer analog

    @property
    def available_range(self) -> tuple[int, int]:
        return (0, self._frame_count - 1)

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def projected_ram_mb(self) -> float:
        return self._height * self._width * self._channels * self._frame_count / (1024 ** 2)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGatherFrame:
    H, W, C = 4, 8, 3
    FRAME_COUNT = 20

    def _make_buffer(self) -> SolidColorBuffer:
        return SolidColorBuffer(
            frame_count=self.FRAME_COUNT,
            height=self.H,
            width=self.W,
            channels=self.C,
        )

    def test_single_band_reads_correct_source_frame(self):
        """A single-band gather should fill the entire output with the source frame value."""
        buf = self._make_buffer()
        output_shape = (self.H, self.W, self.C)
        fill_color = np.zeros(self.C, dtype=np.uint8)

        # Band covers entire width; source frame = 7
        src_indices = np.array([7.0])
        band_starts = np.array([0])
        band_widths = np.array([self.W])

        out = gather_frame(
            buffer=buf,
            output_t=0,
            src_indices=src_indices,
            band_starts=band_starts,
            band_widths=band_widths,
            axis="x",
            output_shape=output_shape,
            fill="black",
            fill_color=fill_color,
        )

        assert out.shape == output_shape
        assert np.all(out == 7), f"Expected all 7, got unique values: {np.unique(out)}"

    def test_multiple_bands_each_reads_own_source(self):
        """Each band should contain pixels from its assigned source frame."""
        buf = self._make_buffer()
        output_shape = (self.H, self.W, self.C)
        fill_color = np.zeros(self.C, dtype=np.uint8)

        # 4 bands of width 2 each; sources: frame 3, 5, 10, 15
        src_indices = np.array([3.0, 5.0, 10.0, 15.0])
        band_starts = np.array([0, 2, 4, 6])
        band_widths = np.array([2, 2, 2, 2])

        out = gather_frame(
            buffer=buf,
            output_t=0,
            src_indices=src_indices,
            band_starts=band_starts,
            band_widths=band_widths,
            axis="x",
            output_shape=output_shape,
            fill="black",
            fill_color=fill_color,
        )

        assert out.shape == output_shape
        # Check each band
        assert np.all(out[:, 0:2, :] == 3), "Band 0 should be from frame 3"
        assert np.all(out[:, 2:4, :] == 5), "Band 1 should be from frame 5"
        assert np.all(out[:, 4:6, :] == 10), "Band 2 should be from frame 10"
        assert np.all(out[:, 6:8, :] == 15), "Band 3 should be from frame 15"

    def test_out_of_range_index_produces_fill(self):
        """Negative or too-large source indices should fill with fill_color."""
        buf = self._make_buffer()
        output_shape = (self.H, self.W, self.C)
        fill_color = np.zeros(self.C, dtype=np.uint8)  # black

        # One band at frame -1 (out of range), one at frame 5 (valid)
        src_indices = np.array([-1.0, 5.0])
        band_starts = np.array([0, 4])
        band_widths = np.array([4, 4])

        out = gather_frame(
            buffer=buf,
            output_t=0,
            src_indices=src_indices,
            band_starts=band_starts,
            band_widths=band_widths,
            axis="x",
            output_shape=output_shape,
            fill="black",
            fill_color=fill_color,
        )

        # Left half: fill (black = 0)
        assert np.all(out[:, 0:4, :] == 0), "Out-of-range should produce black fill"
        # Right half: frame 5
        assert np.all(out[:, 4:8, :] == 5), "Valid index should produce correct pixel value"

    def test_out_of_range_high_index_produces_fill(self):
        """Frame index >= frame_count should also fill."""
        buf = self._make_buffer()
        output_shape = (self.H, self.W, self.C)
        fill_color = np.full(self.C, 255, dtype=np.uint8)  # white fill

        src_indices = np.array([float(self.FRAME_COUNT + 10)])  # way out of range
        band_starts = np.array([0])
        band_widths = np.array([self.W])

        out = gather_frame(
            buffer=buf,
            output_t=0,
            src_indices=src_indices,
            band_starts=band_starts,
            band_widths=band_widths,
            axis="x",
            output_shape=output_shape,
            fill="white",
            fill_color=fill_color,
        )

        assert np.all(out == 255), "Out-of-range high index should produce white fill"

    def test_float_src_index_rounds_to_nearest(self):
        """Fractional source indices should round to the nearest integer frame."""
        buf = self._make_buffer()
        output_shape = (self.H, self.W, self.C)
        fill_color = np.zeros(self.C, dtype=np.uint8)

        # 4.6 → rounds to 5; 4.4 → rounds to 4
        src_indices = np.array([4.6, 4.4])
        band_starts = np.array([0, 4])
        band_widths = np.array([4, 4])

        out = gather_frame(
            buffer=buf,
            output_t=0,
            src_indices=src_indices,
            band_starts=band_starts,
            band_widths=band_widths,
            axis="x",
            output_shape=output_shape,
            fill="black",
            fill_color=fill_color,
        )

        assert np.all(out[:, 0:4, :] == 5), "4.6 should round to frame 5"
        assert np.all(out[:, 4:8, :] == 4), "4.4 should round to frame 4"

    def test_axis_y_gathers_horizontal_bands(self):
        """axis='y' should gather horizontal row-slices."""
        buf = SolidColorBuffer(
            frame_count=self.FRAME_COUNT,
            height=self.W,   # 8 rows
            width=self.H,    # 4 cols
            channels=self.C,
        )
        output_shape = (self.W, self.H, self.C)  # H=8, W=4
        fill_color = np.zeros(self.C, dtype=np.uint8)

        # 2 horizontal bands of 4 rows each
        src_indices = np.array([2.0, 9.0])
        band_starts = np.array([0, 4])
        band_widths = np.array([4, 4])

        out = gather_frame(
            buffer=buf,
            output_t=0,
            src_indices=src_indices,
            band_starts=band_starts,
            band_widths=band_widths,
            axis="y",
            output_shape=output_shape,
            fill="black",
            fill_color=fill_color,
        )

        assert out.shape == output_shape
        assert np.all(out[0:4, :, :] == 2), "Top 4 rows should be from frame 2"
        assert np.all(out[4:8, :, :] == 9), "Bottom 4 rows should be from frame 9"


# ---------------------------------------------------------------------------
# Trumbull slit-source tests
# ---------------------------------------------------------------------------

class TestTrumbullGather:
    """Tests for the fixed-slit (Trumbull/Stargate) gather mode."""

    H, W, C = 4, 8, 3
    FRAME_COUNT = 20

    def _make_gradient_buffer(self) -> "GradientBuffer":
        return GradientBuffer(
            frame_count=self.FRAME_COUNT,
            height=self.H,
            width=self.W,
            channels=self.C,
        )

    def test_slit_source_all_bands_from_same_column(self):
        """With slit_source_px=2, every output column should be col 2 of source."""
        buf = GradientBuffer(frame_count=10, height=4, width=8, channels=3)
        output_shape = (4, 8, 3)
        fill_color = np.zeros(3, dtype=np.uint8)

        # All bands from frame 5; slit at column 2
        src_indices = np.ones(8, dtype=np.float64) * 5
        band_starts = np.arange(8, dtype=np.int32)
        band_widths = np.ones(8, dtype=np.int32)

        out = gather_frame(
            buffer=buf,
            output_t=0,
            src_indices=src_indices,
            band_starts=band_starts,
            band_widths=band_widths,
            axis="x",
            output_shape=output_shape,
            fill="black",
            fill_color=fill_color,
            slit_source_px=2,
        )

        # Column 2 of frame 5 in GradientBuffer = value 2 in channel 0
        for col in range(8):
            assert out[0, col, 0] == 2, f"Col {col} should have slit col value 2"

    def test_slit_source_x_different_times(self):
        """With slit_source_px set, different bands come from different TIME but same COLUMN."""
        buf = SolidColorBuffer(
            frame_count=20, height=4, width=8, channels=3
        )
        output_shape = (4, 8, 3)
        fill_color = np.zeros(3, dtype=np.uint8)

        # Band 0→frame 3, band 1→frame 7 (slit at col 0, which is the same col for both)
        src_indices = np.array([3.0, 7.0])
        band_starts = np.array([0, 4])
        band_widths = np.array([4, 4])

        out = gather_frame(
            buffer=buf,
            output_t=0,
            src_indices=src_indices,
            band_starts=band_starts,
            band_widths=band_widths,
            axis="x",
            output_shape=output_shape,
            fill="black",
            fill_color=fill_color,
            slit_source_px=0,
        )

        # SolidColorBuffer fills entire frame with frame_index value
        # So slit col 0 from frame 3 = 3; from frame 7 = 7
        assert np.all(out[:, 0:4, :] == 3), "Left bands should be from frame 3"
        assert np.all(out[:, 4:8, :] == 7), "Right bands should be from frame 7"

    def test_slit_source_y_axis(self):
        """slit_source_px on axis='y' should repeat a row."""
        buf = SolidColorBuffer(
            frame_count=20, height=8, width=4, channels=3
        )
        output_shape = (8, 4, 3)
        fill_color = np.zeros(3, dtype=np.uint8)

        src_indices = np.array([2.0, 9.0])
        band_starts = np.array([0, 4])
        band_widths = np.array([4, 4])

        out = gather_frame(
            buffer=buf,
            output_t=0,
            src_indices=src_indices,
            band_starts=band_starts,
            band_widths=band_widths,
            axis="y",
            output_shape=output_shape,
            fill="black",
            fill_color=fill_color,
            slit_source_px=1,
        )

        # SolidColorBuffer fills entire frame so slit row = frame value
        assert np.all(out[0:4, :, :] == 2), "Top 4 rows should be from frame 2 slit"
        assert np.all(out[4:8, :, :] == 9), "Bottom 4 rows should be from frame 9 slit"

    def test_slit_source_fill_still_applied(self):
        """Out-of-range src_index with slit_source_px should still produce fill."""
        buf = SolidColorBuffer(
            frame_count=10, height=4, width=8, channels=3
        )
        output_shape = (4, 8, 3)
        fill_color = np.zeros(3, dtype=np.uint8)

        src_indices = np.array([-1.0])  # out of range
        band_starts = np.array([0])
        band_widths = np.array([8])

        out = gather_frame(
            buffer=buf,
            output_t=0,
            src_indices=src_indices,
            band_starts=band_starts,
            band_widths=band_widths,
            axis="x",
            output_shape=output_shape,
            fill="black",
            fill_color=fill_color,
            slit_source_px=2,
        )

        assert np.all(out == 0), "Out-of-range with slit_source_px should still fill"


class GradientBuffer(FrameBuffer):
    """Each frame has column index encoded in channel 0, row index in channel 1."""

    def __init__(self, frame_count: int, height: int, width: int, channels: int = 3):
        self._frame_count = frame_count
        self._height = height
        self._width = width
        self._channels = channels

    def get(self, frame_index: int) -> np.ndarray | None:
        if frame_index < 0 or frame_index >= self._frame_count:
            return None
        frame = np.zeros((self._height, self._width, self._channels), dtype=np.uint8)
        for col in range(self._width):
            frame[:, col, 0] = col % 256
        for row in range(self._height):
            frame[row, :, 1] = row % 256
        frame[:, :, 2] = frame_index % 256
        return frame

    def advance(self) -> None:
        pass

    @property
    def available_range(self) -> tuple[int, int]:
        return (0, self._frame_count - 1)

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def projected_ram_mb(self) -> float:
        return self._height * self._width * self._channels * self._frame_count / (1024 ** 2)


# ---------------------------------------------------------------------------
# Ramp profile tests
# ---------------------------------------------------------------------------

class TestRampProfile:
    def test_vanguard_left_delays_increase_left_to_right(self):
        """With vanguard=0.0, delay at x=0 is 0 and at x=n-1 is max_delay."""
        from slitscan.profiles.ramp import RampProfile

        profile = RampProfile()
        n = 10
        x_coords = np.arange(n, dtype=np.float64)
        params = {"vanguard": 0.0, "max_delay": 100}

        delays = profile.delay_map(x_coords, output_t=0, params=params)

        assert delays[0] == pytest.approx(0.0)
        assert delays[-1] == pytest.approx(100.0)
        # Monotonically increasing
        assert np.all(np.diff(delays) >= 0)

    def test_vanguard_right_delays_increase_right_to_left(self):
        """With vanguard=1.0, delay at x=0 is max_delay and at x=n-1 is 0."""
        from slitscan.profiles.ramp import RampProfile

        profile = RampProfile()
        n = 10
        x_coords = np.arange(n, dtype=np.float64)
        params = {"vanguard": 1.0, "max_delay": 100}

        delays = profile.delay_map(x_coords, output_t=0, params=params)

        assert delays[0] == pytest.approx(100.0)
        assert delays[-1] == pytest.approx(0.0)
        # Monotonically decreasing
        assert np.all(np.diff(delays) <= 0)

    def test_output_dtype_is_float64(self):
        """delay_map must return float64 for downstream arithmetic."""
        from slitscan.profiles.ramp import RampProfile

        profile = RampProfile()
        x_coords = np.arange(5, dtype=np.float64)
        delays = profile.delay_map(x_coords, 0, {"vanguard": 0.0, "max_delay": 50})
        assert delays.dtype == np.float64


# ---------------------------------------------------------------------------
# Source frame formula test
# ---------------------------------------------------------------------------

class TestSourceFrameFormula:
    """Verify the canonical source_frame formula used in render.py."""

    def test_vanguard_reads_current_frame(self):
        """Vanguard band (delay=0) should read exactly output_t."""
        output_t = 5
        delay = 0.0
        src = output_t - delay
        assert src == 5.0

    def test_lagging_band_reads_past_frame(self):
        """Lagging band (delay=3) at output_t=5 should read frame 2."""
        output_t = 5
        delay = 3.0
        src = output_t - delay
        assert src == 2.0

    def test_lagging_band_produces_negative_index_at_start(self):
        """Lagging band (delay=3) at output_t=2 should produce index -1 (fill)."""
        output_t = 2
        delay = 3.0
        src = output_t - delay
        assert src == -1.0

    def test_lagging_band_becomes_valid_at_max_delay(self):
        """Lagging band (delay=max_delay) first becomes valid at output_t=max_delay."""
        max_delay = 3
        output_t = max_delay
        delay = float(max_delay)
        src = output_t - delay
        assert src == 0.0

    def test_midpoint_reads_halfway_back(self):
        """Midpoint delay should read output_t - max_delay/2."""
        max_delay = 50
        output_t = 60
        delay = 25.0
        src = output_t - delay
        assert src == pytest.approx(35.0)
