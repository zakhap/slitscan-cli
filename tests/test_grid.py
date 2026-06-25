"""Unit tests for grid-based slit-scan (engine/grid.py).

Grid mode generalizes the 1D band gather to a 2D mosaic: the frame is split
into cols×rows cells, each reading from its own source frame
    source(cell, output_t) = output_t - delay[row, col]

Synthetic buffer: frame N is a solid image where every pixel = N (mod 256),
so a cell that reads source frame K must be entirely filled with value K.
"""

from __future__ import annotations

import numpy as np
import pytest

from slitscan.buffer.base import FrameBuffer
from slitscan.engine.grid import (
    grid_geometry,
    profile_delay_grid,
    lfo_delay_grid,
    GridLFO,
    gather_grid_frame,
    render_grid,
)
from slitscan.meta import ClipMeta


class SolidColorBuffer(FrameBuffer):
    """Each frame is a solid image whose value equals frame_index % 256."""

    def __init__(self, frame_count: int, height: int, width: int, channels: int = 3):
        self._frame_count = frame_count
        self._height = height
        self._width = width
        self._channels = channels

    def get(self, frame_index: int) -> np.ndarray | None:
        if frame_index < 0 or frame_index >= self._frame_count:
            return None
        return np.full((self._height, self._width, self._channels),
                       frame_index % 256, dtype=np.uint8)

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
        return 0.0


# ---------------------------------------------------------------------------
# grid_geometry
# ---------------------------------------------------------------------------

class TestGridGeometry:
    def test_partitions_tile_the_full_frame_exactly(self):
        col_starts, col_widths, row_starts, row_heights = grid_geometry(
            width=100, height=60, cols=8, rows=6)
        # Exactly the requested number of cells on each axis.
        assert len(col_starts) == 8 and len(col_widths) == 8
        assert len(row_starts) == 6 and len(row_heights) == 6
        # Cells abut with no gaps or overlaps and cover [0, extent].
        assert col_starts[0] == 0 and row_starts[0] == 0
        assert int(col_widths.sum()) == 100
        assert int(row_heights.sum()) == 60
        np.testing.assert_array_equal(col_starts[1:], np.cumsum(col_widths)[:-1])
        np.testing.assert_array_equal(row_starts[1:], np.cumsum(row_heights)[:-1])

    def test_handles_non_divisible_sizes_without_dropping_pixels(self):
        _, col_widths, _, row_heights = grid_geometry(
            width=10, height=7, cols=3, rows=3)
        assert int(col_widths.sum()) == 10   # 10 px into 3 cells, none lost
        assert int(row_heights.sum()) == 7

    def test_one_cell_covers_everything(self):
        col_starts, col_widths, row_starts, row_heights = grid_geometry(
            width=64, height=48, cols=1, rows=1)
        assert list(col_starts) == [0] and list(col_widths) == [64]
        assert list(row_starts) == [0] and list(row_heights) == [48]


# ---------------------------------------------------------------------------
# profile_delay_grid  (delay source A: reuse 1D profiles in 2D)
# ---------------------------------------------------------------------------

class TestProfileDelayGrid:
    def test_shape_and_range(self):
        d = profile_delay_grid("ramp", cols=5, rows=4, max_delay=30, vanguard=0.0)
        assert d.shape == (4, 5)
        assert d.min() >= 0.0 and d.max() <= 30.0

    def test_ramp_avg_increases_toward_far_corner(self):
        # vanguard at origin corner: delay 0 top-left, grows toward bottom-right.
        d = profile_delay_grid("ramp", cols=4, rows=4, max_delay=12,
                               vanguard=0.0, combine="avg")
        assert d[0, 0] == pytest.approx(0.0)
        assert d[3, 3] == pytest.approx(12.0)
        assert d[3, 3] > d[0, 0]

    def test_max_combine_with_single_row_degenerates_to_1d_ramp(self):
        # rows=1 + combine=max collapses to the pure 1D column ramp.
        d = profile_delay_grid("ramp", cols=5, rows=1, max_delay=8,
                               vanguard=0.0, combine="max")
        expected = np.array([0, 2, 4, 6, 8], dtype=float)  # t*max_delay, t=i/4
        np.testing.assert_allclose(d[0], expected)

    def test_tent_avg_is_zero_at_center(self):
        d = profile_delay_grid("tent", cols=5, rows=5, max_delay=10,
                               vanguard=0.5, combine="avg")
        assert d[2, 2] == pytest.approx(0.0)        # center cell = "now"
        assert d[0, 0] > d[2, 2]                      # corners are older


# ---------------------------------------------------------------------------
# lfo_delay_grid  (delay source B: spatial oscillator field)
# ---------------------------------------------------------------------------

class TestLfoDelayGrid:
    def test_single_column_sine_matches_manual(self):
        lfo = GridLFO(axis="col", osc="sine", rate=1.0, depth=1.0, phase=0.0)
        d = lfo_delay_grid([lfo], cols=4, rows=2, max_delay=10)
        assert d.shape == (2, 4)
        # delay01 = 0.5 + 0.5*sin(2π · rate · col_norm + phase); col_norm=c/3.
        col_norm = np.arange(4) / 3.0
        expected01 = 0.5 + 0.5 * np.sin(2 * np.pi * col_norm)
        np.testing.assert_allclose(d[0], expected01 * 10, atol=1e-9)

    def test_col_lfo_is_constant_down_each_column(self):
        lfo = GridLFO(axis="col", osc="sine", rate=2.0, depth=1.0, phase=0.0)
        d = lfo_delay_grid([lfo], cols=6, rows=3, max_delay=20)
        for c in range(6):
            assert d[0, c] == pytest.approx(d[1, c]) == pytest.approx(d[2, c])

    def test_two_axes_sum_into_a_2d_field(self):
        col = GridLFO(axis="col", osc="sine", rate=1.0, depth=0.5, phase=0.0)
        row = GridLFO(axis="row", osc="sine", rate=1.0, depth=0.5, phase=0.0)
        d = lfo_delay_grid([col, row], cols=5, rows=5, max_delay=1.0)
        # Genuine 2D field: it varies both down columns and across rows.
        assert not np.allclose(d[0, :], d[1, :])   # varies down a column (row LFO)
        assert not np.allclose(d[:, 0], d[:, 1])   # varies across a row (col LFO)

    def test_clamped_to_valid_delay_range(self):
        # Huge depth must not push delay outside [0, max_delay].
        lfo = GridLFO(axis="col", osc="sine", rate=1.0, depth=5.0, phase=0.0)
        d = lfo_delay_grid([lfo], cols=8, rows=2, max_delay=15)
        assert d.min() >= 0.0 and d.max() <= 15.0


# ---------------------------------------------------------------------------
# gather_grid_frame
# ---------------------------------------------------------------------------

class TestGatherGridFrame:
    H, W, C = 6, 8, 3

    def _geom(self, cols, rows):
        return grid_geometry(self.W, self.H, cols, rows)

    def test_each_cell_reads_its_own_source_frame(self):
        buf = SolidColorBuffer(frame_count=50, height=self.H, width=self.W)
        cs, cw, rs, rh = self._geom(cols=2, rows=2)
        # delay[r,c] chosen so each cell maps to a distinct, known source.
        delay = np.array([[0, 5], [10, 15]], dtype=float)  # output_t=20
        out = gather_grid_frame(
            buffer=buf, output_t=20, delay_grid=delay,
            col_starts=cs, col_widths=cw, row_starts=rs, row_heights=rh,
            output_shape=(self.H, self.W, self.C), fill="black",
            fill_color=np.zeros(3, np.uint8), frame_count=50)
        # source = 20 - delay → 20, 15, 10, 5
        assert out[rs[0], cs[0], 0] == 20
        assert out[rs[0], cs[1], 0] == 15
        assert out[rs[1], cs[0], 0] == 10
        assert out[rs[1], cs[1], 0] == 5

    def test_out_of_range_cell_gets_fill_color(self):
        buf = SolidColorBuffer(frame_count=10, height=self.H, width=self.W)
        cs, cw, rs, rh = self._geom(cols=2, rows=1)
        delay = np.array([[0, 99]], dtype=float)   # cell 1 → source -99 (OOB)
        white = np.full(3, 255, np.uint8)
        out = gather_grid_frame(
            buffer=buf, output_t=5, delay_grid=delay,
            col_starts=cs, col_widths=cw, row_starts=rs, row_heights=rh,
            output_shape=(self.H, self.W, self.C), fill="white",
            fill_color=white, frame_count=10)
        assert out[0, cs[0], 0] == 5         # in-range cell reads source 5
        assert out[0, cs[1], 0] == 255       # OOB cell filled white

    def test_wrap_fill_cycles_out_of_range_into_the_clip(self):
        buf = SolidColorBuffer(frame_count=10, height=self.H, width=self.W)
        cs, cw, rs, rh = self._geom(cols=2, rows=1)
        delay = np.array([[0, 12]], dtype=float)   # source 3-12 = -9 → wrap → 1
        out = gather_grid_frame(
            buffer=buf, output_t=3, delay_grid=delay,
            col_starts=cs, col_widths=cw, row_starts=rs, row_heights=rh,
            output_shape=(self.H, self.W, self.C), fill="wrap",
            fill_color=np.zeros(3, np.uint8), frame_count=10)
        assert out[0, cs[0], 0] == 3
        assert out[0, cs[1], 0] == 1         # (-9) % 10 == 1

    def test_interpolate_blends_adjacent_frames(self):
        buf = SolidColorBuffer(frame_count=50, height=self.H, width=self.W)
        cs, cw, rs, rh = self._geom(cols=1, rows=1)
        delay = np.array([[10.5]], dtype=float)    # source 30 - 10.5 = 19.5
        out = gather_grid_frame(
            buffer=buf, output_t=30, delay_grid=delay,
            col_starts=cs, col_widths=cw, row_starts=rs, row_heights=rh,
            output_shape=(self.H, self.W, self.C), fill="black",
            fill_color=np.zeros(3, np.uint8), frame_count=50, interpolate=True)
        # 0.5 blend of frame 19 and 20 → 19.5 → rounds to 19 or 20 as uint8.
        assert out[0, 0, 0] in (19, 20)


# ---------------------------------------------------------------------------
# render_grid (the per-frame loop)
# ---------------------------------------------------------------------------

class _CaptureEncoder:
    def __init__(self):
        self.frames = []
        self.closed = False

    def write_frame(self, frame):
        self.frames.append(frame.copy())

    def close(self):
        self.closed = True


class TestRenderGrid:
    H, W, C = 6, 8, 3

    def test_emits_one_frame_per_source_frame_and_closes(self):
        buf = SolidColorBuffer(frame_count=12, height=self.H, width=self.W)
        meta = ClipMeta(fps=24.0, frame_count=12, width=self.W,
                        height=self.H, channels=self.C)
        delay = profile_delay_grid("ramp", cols=4, rows=3, max_delay=5,
                                   vanguard=0.0)
        enc = _CaptureEncoder()
        render_grid(meta=meta, buffer=buf, delay_grid=delay, fill="hold",
                    interpolate=False, encoder=enc)
        assert len(enc.frames) == 12
        assert enc.closed is True
        assert enc.frames[0].shape == (self.H, self.W, self.C)

    def test_vanguard_cell_reads_current_output_frame(self):
        # The delay-0 cell (top-left for ramp vanguard=0) must read output_t.
        buf = SolidColorBuffer(frame_count=12, height=self.H, width=self.W)
        meta = ClipMeta(fps=24.0, frame_count=12, width=self.W,
                        height=self.H, channels=self.C)
        delay = profile_delay_grid("ramp", cols=4, rows=3, max_delay=5,
                                   vanguard=0.0, combine="avg")
        cs, _, rs, _ = grid_geometry(self.W, self.H, 4, 3)
        enc = _CaptureEncoder()
        render_grid(meta=meta, buffer=buf, delay_grid=delay, fill="hold",
                    interpolate=False, encoder=enc)
        assert enc.frames[8][rs[0], cs[0], 0] == 8   # top-left = "now"


# ---------------------------------------------------------------------------
# parse_grid_lfo (CLI --grid-mod string)
# ---------------------------------------------------------------------------

from slitscan.engine.grid import parse_grid_lfo  # noqa: E402


class TestParseGridLfo:
    def test_full_string(self):
        lfo = parse_grid_lfo("col=sine:rate=2,depth=0.8,phase=0.25,offset=0.1")
        assert lfo.axis == "col" and lfo.osc == "sine"
        assert lfo.rate == 2.0 and lfo.depth == 0.8
        assert lfo.phase == 0.25 and lfo.offset == pytest.approx(0.1)

    def test_defaults_when_omitted(self):
        lfo = parse_grid_lfo("row=triangle:rate=3")
        assert lfo.axis == "row" and lfo.osc == "triangle"
        assert lfo.rate == 3.0 and lfo.depth == 1.0
        assert lfo.phase == 0.0 and lfo.offset == 0.0

    def test_bad_axis_rejected(self):
        with pytest.raises(ValueError):
            parse_grid_lfo("diag=sine:rate=1")

    def test_malformed_rejected(self):
        with pytest.raises(ValueError):
            parse_grid_lfo("garbage")
