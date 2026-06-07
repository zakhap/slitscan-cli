"""Whole-clip-in-RAM frame buffer.

Decodes all frames eagerly at construction. Suitable for clips that
fit comfortably in memory. Phase 1 only uses this backing.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np

from slitscan.buffer.base import FrameBuffer
from slitscan.meta import ClipMeta


class FullBuffer(FrameBuffer):
    """Load every frame into a list of numpy arrays at construction time."""

    def __init__(self, meta: ClipMeta, frames_iter: Iterator[np.ndarray]) -> None:
        self._frames: list[np.ndarray] = list(frames_iter)
        # Update frame count in case the stream's metadata was inaccurate
        self._frame_count: int = len(self._frames)

    # ------------------------------------------------------------------
    # FrameBuffer interface
    # ------------------------------------------------------------------

    def get(self, frame_index: int) -> np.ndarray | None:
        """Return frame at *frame_index*, or None if out of range."""
        if frame_index < 0 or frame_index >= self._frame_count:
            return None
        return self._frames[frame_index]

    def advance(self) -> None:
        """No-op: all frames are already loaded."""

    @property
    def available_range(self) -> tuple[int, int]:
        if self._frame_count == 0:
            return (0, -1)
        return (0, self._frame_count - 1)

    @property
    def frame_count(self) -> int:
        return self._frame_count

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def projected_ram_mb(self) -> float:
        """Estimated RAM usage in megabytes."""
        if not self._frames:
            return 0.0
        return self._frames[0].nbytes * self._frame_count / (1024 ** 2)
