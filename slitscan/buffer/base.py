"""Abstract base class for frame buffers.

The engine always talks to the buffer — never directly to PyAV.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class FrameBuffer(ABC):
    """Interface for frame-indexed video access."""

    @abstractmethod
    def get(self, frame_index: int) -> np.ndarray | None:
        """Return the frame at *frame_index*, or ``None`` if out of range.

        Out-of-range means frame_index < 0 or frame_index >= frame_count.
        Never raises; the engine interprets None as a fill region.
        """

    @abstractmethod
    def advance(self) -> None:
        """Pull the next decoded frame from the source (used by ring buffer).

        For FullBuffer this is a no-op because all frames are pre-loaded.
        """

    @property
    @abstractmethod
    def available_range(self) -> tuple[int, int]:
        """(oldest, newest) resident frame indices (inclusive)."""

    @property
    @abstractmethod
    def frame_count(self) -> int:
        """Total number of frames in the clip."""

    @property
    @abstractmethod
    def projected_ram_mb(self) -> float:
        """Estimated RAM usage in megabytes."""
