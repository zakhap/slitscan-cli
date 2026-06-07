"""Buffer subsystem: abstract frame access interface and concrete implementations."""

from slitscan.buffer.base import FrameBuffer
from slitscan.buffer.full import FullBuffer

__all__ = ["FrameBuffer", "FullBuffer"]
