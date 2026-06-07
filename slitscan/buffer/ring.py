"""Circular frame buffer for memory-bounded rendering.

Holds exactly ``max_delay + 1`` frames in RAM at a time. As the engine
advances frame-by-frame, the oldest slot is overwritten with the next
decoded frame, keeping the window [output_t - max_delay, output_t] resident.

Ring constraint
---------------
The delay surface may only reference frames within the resident window.
For output frame *t* the window is [t - max_delay, t], which is exactly
``max_delay + 1`` frames — the ring capacity.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np

from slitscan.buffer.base import FrameBuffer
from slitscan.meta import ClipMeta


class RingBuffer(FrameBuffer):
    """Circular buffer holding exactly ``max_delay + 1`` frames.

    Memory usage is constant regardless of clip length: only the sliding
    window ``[oldest_resident, newest_resident]`` is kept in RAM.
    """

    def __init__(
        self,
        meta: ClipMeta,
        frames_iter: Iterator[np.ndarray],
        max_delay: int,
    ) -> None:
        self._meta = meta
        self._iter = frames_iter
        self._capacity = max_delay + 1
        self._ring: list[np.ndarray | None] = [None] * self._capacity
        self._loaded = 0        # total frames loaded so far
        self._exhausted = False

        # Pre-load only the first frame.
        # The render loop calls advance() after each gathered frame, which
        # progressively fills the window.  Pre-loading all capacity frames
        # would cause the ring to evict frames that are still needed during
        # the first max_delay output frames.
        self._load_next()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_next(self) -> bool:
        """Load the next frame from the iterator into the ring.

        Returns False if the iterator is exhausted.
        """
        if self._exhausted:
            return False
        try:
            frame = next(self._iter)
        except StopIteration:
            self._exhausted = True
            return False
        slot = self._loaded % self._capacity
        self._ring[slot] = frame
        self._loaded += 1
        return True

    # ------------------------------------------------------------------
    # FrameBuffer interface
    # ------------------------------------------------------------------

    def get(self, frame_index: int) -> np.ndarray | None:
        """Return the frame at *frame_index*, or None if out of range or evicted."""
        if frame_index < 0 or frame_index >= self._meta.frame_count:
            return None
        newest = self._loaded - 1
        oldest = max(0, self._loaded - self._capacity)
        if frame_index < oldest or frame_index > newest:
            return None  # outside the current resident window
        slot = frame_index % self._capacity
        return self._ring[slot]

    def advance(self) -> None:
        """Slide the window forward by loading the next frame and evicting the oldest."""
        self._load_next()

    @property
    def available_range(self) -> tuple[int, int]:
        """(oldest, newest) resident frame indices (inclusive)."""
        newest = self._loaded - 1
        oldest = max(0, self._loaded - self._capacity)
        return (oldest, newest)

    @property
    def frame_count(self) -> int:
        return self._meta.frame_count

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def projected_ram_mb(self) -> float:
        """Estimated RAM usage in megabytes."""
        for slot in self._ring:
            if slot is not None:
                return (slot.nbytes * self._capacity) / (1024 ** 2)
        return 0.0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_ring_compatible(max_delay: int, fill: str) -> None:
    """Raise ValueError if *fill* mode is incompatible with ring buffer.

    ``--fill=wrap`` requires random access to any frame in the clip, which
    the ring buffer cannot provide.  All other fill modes are safe because
    they either stay within the window (hold) or use a synthetic color.
    """
    if fill == "wrap":
        raise ValueError(
            "--fill=wrap requires access to any frame in the clip and is "
            "incompatible with --buffer=ring. Use --buffer=full instead."
        )
    # hold, black, white, transparent are all compatible with ring


# ---------------------------------------------------------------------------
# Memory budget helper
# ---------------------------------------------------------------------------

def parse_memory_budget(s: str) -> int:
    """Parse a human-readable memory string to bytes.

    Supported suffixes (case-insensitive): G, M, K.
    No suffix is treated as raw bytes.

    Examples
    --------
    >>> parse_memory_budget("8G")
    8589934592
    >>> parse_memory_budget("512M")
    536870912
    >>> parse_memory_budget("1024K")
    1048576
    >>> parse_memory_budget("500000")
    500000
    """
    try:
        s_stripped = s.strip().upper()
        if s_stripped.endswith("G"):
            return int(float(s_stripped[:-1]) * 1024 ** 3)
        elif s_stripped.endswith("M"):
            return int(float(s_stripped[:-1]) * 1024 ** 2)
        elif s_stripped.endswith("K"):
            return int(float(s_stripped[:-1]) * 1024)
        return int(s_stripped)
    except (ValueError, AttributeError) as exc:
        raise ValueError(
            f"Invalid memory budget {s!r}. Expected a number with optional suffix: 8G, 512M, 1024K, or raw bytes."
        ) from exc
