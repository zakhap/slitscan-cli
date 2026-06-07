"""Unit tests for slitscan/buffer/ring.py.

Synthetic frame pattern: frame N is a solid image where every pixel == N % 256.
This makes it trivial to verify which source frame was read.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pytest

from slitscan.buffer.ring import RingBuffer, validate_ring_compatible, parse_memory_budget
from slitscan.buffer.full import FullBuffer
from slitscan.engine.gather import gather_frame
from slitscan.engine.fill import make_fill_color
from slitscan.meta import ClipMeta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_fake_iter(n_frames: int, h: int = 4, w: int = 4, c: int = 3) -> Iterator[np.ndarray]:
    """Yield *n_frames* arrays where frame N is filled with value N % 256."""
    for i in range(n_frames):
        yield np.full((h, w, c), i % 256, dtype=np.uint8)


def make_fake_meta(n_frames: int, h: int = 4, w: int = 4, c: int = 3) -> ClipMeta:
    return ClipMeta(fps=24.0, frame_count=n_frames, width=w, height=h, channels=c)


# ---------------------------------------------------------------------------
# RingBuffer tests
# ---------------------------------------------------------------------------

class TestRingBuffer:
    """Tests for the RingBuffer implementation."""

    N = 10
    MAX_DELAY = 3  # capacity = 4 frames (max_delay + 1)
    _capacity = MAX_DELAY + 1  # 4

    def _make_ring(self, n: int = N, max_delay: int = MAX_DELAY) -> RingBuffer:
        meta = make_fake_meta(n)
        return RingBuffer(meta=meta, frames_iter=make_fake_iter(n), max_delay=max_delay)

    # ------------------------------------------------------------------
    # Initial window
    # ------------------------------------------------------------------

    def test_ring_get_in_window(self):
        """After init only frame 0 is resident; get(0) returns frame with value 0."""
        ring = self._make_ring()
        # Only frame 0 is loaded at init time.
        frame = ring.get(0)
        assert frame is not None, "Frame 0 should be resident after init"
        assert np.all(frame == 0), f"Frame 0 should have pixel value 0, got {np.unique(frame)}"
        # Frame 1 onwards are not yet loaded.
        for i in range(1, self.MAX_DELAY + 1):
            assert ring.get(i) is None, (
                f"Frame {i} should not be resident at init (loaded lazily via advance())"
            )

    def test_ring_get_out_of_range_negative(self):
        """get(-1) must return None."""
        ring = self._make_ring()
        assert ring.get(-1) is None

    def test_ring_get_out_of_range_high(self):
        """get(frame_count) must return None."""
        ring = self._make_ring()
        assert ring.get(self.N) is None

    # ------------------------------------------------------------------
    # After advance()
    # ------------------------------------------------------------------

    def test_ring_advance_evicts_oldest(self):
        """Frame 0 is evicted once the ring fills and advances past capacity."""
        ring = self._make_ring()
        # After init: only frame 0 is loaded (_loaded=1, capacity=4).
        # We need to advance capacity times to fill the ring and then evict frame 0.
        for _ in range(self._capacity):  # advance capacity (=MAX_DELAY+1=4) times
            ring.advance()
        # Now _loaded = 1 + 4 = 5; oldest = max(0, 5-4) = 1 → frame 0 is evicted
        assert ring.get(0) is None, "Frame 0 should be evicted after ring fills"
        next_frame = ring.get(self._capacity)  # frame index 4
        assert next_frame is not None, f"Frame {self._capacity} should be loaded"
        assert np.all(next_frame == self._capacity)

    def test_ring_window_tracks(self):
        """available_range should expand then slide as advance() is called."""
        ring = self._make_ring()
        # Initial window after 1-frame pre-load: [0, 0]
        assert ring.available_range == (0, 0)

        # First capacity-1 advances fill the ring without eviction
        for step in range(1, self._capacity):
            ring.advance()
            oldest, newest = ring.available_range
            assert newest == step, f"After {step} advance(s), newest should be {step}"
            assert oldest == 0, f"Before ring fills, oldest should remain 0"

        # Once we've done `capacity` total advances (including pre-load as step 0),
        # the ring starts evicting the oldest.
        ring.advance()
        oldest, newest = ring.available_range
        assert newest == self._capacity, f"newest should be {self._capacity}"
        assert oldest == 1, "oldest should advance to 1 once ring is full"

    def test_ring_frame_count(self):
        """frame_count should reflect the clip's total frames, not the window."""
        ring = self._make_ring()
        assert ring.frame_count == self.N

    def test_ring_get_evicted_is_none(self):
        """Frames that have been evicted should return None even if valid indices."""
        ring = self._make_ring()
        # Pre-load gives frame 0.  capacity=4.  We need capacity+3 advances to
        # evict frames 0, 1, 2 (window becomes [3, 3+capacity-1] = [3, 6]).
        # After advance x times from _loaded=1: _loaded=1+x; oldest=max(0,1+x-4).
        # oldest becomes >=3 when 1+x-4>=3 → x>=6.  Do 6 advances.
        for _ in range(6):
            ring.advance()
        # _loaded = 7; oldest = max(0, 7-4) = 3
        for i in range(3):
            assert ring.get(i) is None, f"Frame {i} should be evicted"
        # Frames 3..6 should be accessible
        for i in range(3, 7):
            frame = ring.get(i)
            assert frame is not None, f"Frame {i} should be resident"
            assert np.all(frame == i % 256)

    def test_ring_exhaustion(self):
        """Ring should handle clips shorter than capacity gracefully."""
        n = 3
        max_delay = 5  # capacity = 6, but only 3 frames exist
        meta = make_fake_meta(n)
        ring = RingBuffer(meta=meta, frames_iter=make_fake_iter(n), max_delay=max_delay)

        # After init only frame 0 is loaded; advance to load all n frames.
        for _ in range(n - 1):
            ring.advance()

        # All existing frames should be accessible
        for i in range(n):
            frame = ring.get(i)
            assert frame is not None, f"Frame {i} should be resident"
            assert np.all(frame == i)
        # Beyond clip is always None
        assert ring.get(n) is None

    def test_ring_projected_ram_mb(self):
        """projected_ram_mb should be positive for a loaded ring."""
        ring = self._make_ring()
        assert ring.projected_ram_mb > 0.0


# ---------------------------------------------------------------------------
# Full vs Ring equivalence test
# ---------------------------------------------------------------------------

class TestFullVsRingEquivalence:
    """Ensure RingBuffer produces pixel-identical gather results to FullBuffer."""

    N_FRAMES = 20
    H = 8
    W = 8
    C = 3
    MAX_DELAY = 4

    def _gather_all(self, buffer, meta: ClipMeta) -> list[np.ndarray]:
        """Simulate the render loop: gather each frame and call advance()."""
        output_shape = (meta.height, meta.width, meta.channels)
        fill_color = make_fill_color("black", meta.channels)

        # Ramp delay surface: delay[x] = round(x / (W - 1) * MAX_DELAY)
        n_bands = meta.width  # slice_width=1
        x_coords = np.arange(n_bands, dtype=np.float64)
        band_starts = x_coords.astype(np.int32)
        band_widths = np.ones(n_bands, dtype=np.int32)
        # delays: 0 .. MAX_DELAY linearly
        delays_static = x_coords / (n_bands - 1) * self.MAX_DELAY

        results = []
        for output_t in range(meta.frame_count):
            src_indices = output_t - delays_static
            frame = gather_frame(
                buffer=buffer,
                output_t=output_t,
                src_indices=src_indices,
                band_starts=band_starts,
                band_widths=band_widths,
                axis="x",
                output_shape=output_shape,
                fill="black",
                fill_color=fill_color,
                frame_count=meta.frame_count,
                interpolate=False,
            )
            results.append(frame)
            buffer.advance()

        return results

    def test_full_vs_ring_equivalence(self):
        """FullBuffer and RingBuffer must produce pixel-identical output."""
        meta = make_fake_meta(self.N_FRAMES, h=self.H, w=self.W, c=self.C)

        full_buf = FullBuffer(meta=meta, frames_iter=make_fake_iter(self.N_FRAMES, self.H, self.W, self.C))
        ring_buf = RingBuffer(
            meta=meta,
            frames_iter=make_fake_iter(self.N_FRAMES, self.H, self.W, self.C),
            max_delay=self.MAX_DELAY,
        )

        full_frames = self._gather_all(full_buf, meta)
        ring_frames = self._gather_all(ring_buf, meta)

        assert len(full_frames) == len(ring_frames) == self.N_FRAMES

        for t, (f_frame, r_frame) in enumerate(zip(full_frames, ring_frames)):
            assert f_frame.shape == r_frame.shape, f"Shape mismatch at output_t={t}"
            np.testing.assert_array_equal(
                f_frame, r_frame,
                err_msg=f"Pixel mismatch at output_t={t}: "
                        f"full unique={np.unique(f_frame)} ring unique={np.unique(r_frame)}",
            )


# ---------------------------------------------------------------------------
# validate_ring_compatible tests
# ---------------------------------------------------------------------------

class TestValidateRingCompatible:
    def test_wrap_raises(self):
        with pytest.raises(ValueError, match="wrap"):
            validate_ring_compatible(max_delay=10, fill="wrap")

    def test_hold_ok(self):
        validate_ring_compatible(max_delay=10, fill="hold")  # no exception

    def test_black_ok(self):
        validate_ring_compatible(max_delay=10, fill="black")

    def test_white_ok(self):
        validate_ring_compatible(max_delay=10, fill="white")

    def test_transparent_ok(self):
        validate_ring_compatible(max_delay=10, fill="transparent")


# ---------------------------------------------------------------------------
# parse_memory_budget tests
# ---------------------------------------------------------------------------

class TestParseMemoryBudget:
    def test_gigabytes(self):
        assert parse_memory_budget("8G") == 8 * 1024 ** 3

    def test_megabytes(self):
        assert parse_memory_budget("512M") == 512 * 1024 ** 2

    def test_kilobytes(self):
        assert parse_memory_budget("1024K") == 1024 * 1024

    def test_bytes(self):
        assert parse_memory_budget("500000") == 500000

    def test_lowercase(self):
        assert parse_memory_budget("4g") == 4 * 1024 ** 3

    def test_fractional_gigabytes(self):
        assert parse_memory_budget("1.5G") == int(1.5 * 1024 ** 3)

    def test_whitespace(self):
        assert parse_memory_budget("  256M  ") == 256 * 1024 ** 2
