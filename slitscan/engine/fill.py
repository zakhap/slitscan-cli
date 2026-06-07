"""Fill policies for out-of-range frame references.

Phase 1: black fill only.
Phase 3+: white, transparent, hold, wrap.
"""

from __future__ import annotations

import numpy as np


def make_fill_color(fill: str, channels: int) -> np.ndarray:
    """Return a 1-D array of *channels* values for the named fill color.

    Parameters
    ----------
    fill:
        Fill mode name. ``"hold"`` and ``"wrap"`` never produce a fill color
        (they resolve to a valid frame), so their color arrays default to black.
    channels:
        Number of color channels (3 for RGB, 4 for RGBA).

    Returns
    -------
    np.ndarray
        Shape ``(channels,)``, dtype uint8.
    """
    if fill == "black":
        return np.zeros(channels, dtype=np.uint8)
    elif fill == "white":
        return np.full(channels, 255, dtype=np.uint8)
    elif fill == "transparent":
        # Transparent = zero-alpha RGBA; all channels are 0
        return np.zeros(channels, dtype=np.uint8)
    elif fill in ("hold", "wrap"):
        # hold/wrap never reach the fill-color path; return black as a safe default
        return np.zeros(channels, dtype=np.uint8)
    else:
        raise ValueError(
            f"Unknown fill mode: {fill!r}. "
            f"Supported: black, white, transparent, hold, wrap"
        )


def _resolve_src_index(src_idx: float, fill: str, frame_count: int) -> int:
    """Resolve a (possibly out-of-range) source frame index according to *fill*.

    Parameters
    ----------
    src_idx:
        Raw (possibly fractional) source frame index.
    fill:
        Fill mode. ``"hold"`` clamps; ``"wrap"`` wraps; others pass through
        so that ``buffer.get()`` can return None and trigger a color fill.
    frame_count:
        Total number of frames in the source clip.

    Returns
    -------
    int
        Resolved frame index.
    """
    idx = int(round(src_idx))
    if fill == "hold":
        return max(0, min(idx, frame_count - 1))
    elif fill == "wrap":
        return idx % frame_count
    # black / white / transparent: let buffer.get() return None naturally
    return idx


def make_fill_band(
    fill_color: np.ndarray,
    band_h: int,
    band_w: int,
) -> np.ndarray:
    """Create a solid fill band array.

    Parameters
    ----------
    fill_color:
        Per-channel fill value, shape ``(C,)``.
    band_h:
        Height of the band.
    band_w:
        Width of the band.

    Returns
    -------
    np.ndarray
        Shape ``(band_h, band_w, C)``. The caller handles axis orientation.
    """
    C = len(fill_color)
    band = np.zeros((band_h, band_w, C), dtype=np.uint8)
    band[:] = fill_color
    return band
