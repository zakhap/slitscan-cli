"""Vectorized column (or row) gather for one output frame.

The gather function is the inner loop of the render engine.
It reads bands from the buffer using precomputed source frame indices
and assembles the output frame.

Phase 1: per-band Python loop (no interpolation).
Phase 5: add sub-frame interpolation via np fancy indexing.
"""

from __future__ import annotations

import numpy as np

from slitscan.buffer.base import FrameBuffer
from slitscan.engine.fill import make_fill_color, make_fill_band, _resolve_src_index


def _resolve_src_index_floor_ceil(
    floor_raw: int,
    ceil_raw: int,
    fill: str,
    frame_count: int,
) -> tuple[int, int]:
    """Resolve floor and ceil raw indices according to fill mode.

    For hold/wrap, the resolved index is always valid (buffer.get returns a frame).
    For color fills, out-of-range indices are passed through so buffer.get returns None.

    Unlike _resolve_src_index, this does NOT round — floor/ceil are already integers.
    """
    if fill == "hold":
        floor_resolved = max(0, min(floor_raw, frame_count - 1))
        ceil_resolved = max(0, min(ceil_raw, frame_count - 1))
    elif fill == "wrap":
        floor_resolved = floor_raw % frame_count
        ceil_resolved = ceil_raw % frame_count
    else:
        # black / white / transparent: pass through; buffer.get returns None for OOB
        floor_resolved = floor_raw
        ceil_resolved = ceil_raw
    return floor_resolved, ceil_resolved


def _blend_frames(
    floor_frame: np.ndarray | None,
    ceil_frame: np.ndarray | None,
    alpha: float,
    fill_band: np.ndarray,
    axis: str,
    start: int,
    width: int,
    slit_source_px: int | None = None,
) -> np.ndarray:
    """Blend two adjacent frames with weight *alpha* (0.0 = all floor, 1.0 = all ceil).

    If either frame is None (fill zone), uses *fill_band* for that frame.
    When *slit_source_px* is set, each frame contributes a single slit column/row
    repeated across the band (Trumbull mode).
    """
    def get_band(frame: np.ndarray | None, fill: np.ndarray) -> np.ndarray:
        if frame is None:
            return fill
        if slit_source_px is not None:
            if axis == "x":
                slit = frame[:, slit_source_px:slit_source_px + 1, :]
                return np.repeat(slit, width, axis=1)
            else:
                slit = frame[slit_source_px:slit_source_px + 1, :, :]
                return np.repeat(slit, width, axis=0)
        if axis == "x":
            return frame[:, start:start + width, :]
        return frame[start:start + width, :, :]

    f0 = get_band(floor_frame, fill_band)
    f1 = get_band(ceil_frame, fill_band)
    return np.clip(
        (1.0 - alpha) * f0.astype(np.float32) + alpha * f1.astype(np.float32),
        0, 255,
    ).astype(np.uint8)


def gather_frame(
    buffer: FrameBuffer,
    output_t: int,
    src_indices: np.ndarray,      # shape (n_bands,), float — source frame indices
    band_starts: np.ndarray,      # shape (n_bands,), int  — pixel start per band
    band_widths: np.ndarray,      # shape (n_bands,), int  — pixel width per band
    axis: str,                    # "x" or "y"
    output_shape: tuple,          # (H, W, C)
    fill: str,
    fill_color: np.ndarray,       # shape (C,), fill pixel value
    frame_count: int = 0,         # total number of source frames (required for hold/wrap)
    interpolate: bool = False,
    slit_source_px: int | None = None,  # Trumbull: fixed pixel coord to gather from
) -> np.ndarray:
    """Assemble one output frame by gathering bands from the buffer.

    For each band *i*:
    - Round ``src_indices[i]`` to the nearest integer frame index.
    - Call ``buffer.get(src_int)`` → frame or None.
    - Copy the band columns/rows from that frame into *out*.
    - If None, write the fill color instead.

    When *interpolate* is True and the source index is fractional, blend
    the floor and ceil frames with linear weighting.

    Parameters
    ----------
    buffer:
        The frame source. ``get()`` returns None for out-of-range indices.
    output_t:
        Current output frame index (informational; not used in Phase 1).
    src_indices:
        Per-band source frame index (float). Values outside [0, frame_count)
        will map to None from the buffer.
    band_starts:
        Pixel start coordinate for each band along *axis*.
    band_widths:
        Pixel width of each band along *axis*.
    axis:
        ``"x"`` — bands are vertical slices (columns).
        ``"y"`` — bands are horizontal slices (rows).
    output_shape:
        ``(H, W, C)`` of the output frame.
    fill:
        Fill mode name. ``"hold"`` clamps the index; ``"wrap"`` wraps it;
        ``"black"``, ``"white"``, and ``"transparent"`` let the buffer return
        None and use *fill_color* instead.
    fill_color:
        Pixel value for fill regions, shape ``(C,)``.
    frame_count:
        Total number of frames in the source clip. Required for ``"hold"``
        and ``"wrap"`` fills; ignored for color fills.
    interpolate:
        If True, apply sub-frame linear interpolation between adjacent frames
        when the source index is fractional.
    slit_source_px:
        If set, use this fixed pixel coordinate (along *axis*) as the slit
        position instead of each band's own position. This is the Trumbull /
        Stargate slit-scan effect: all output bands are gathered from the same
        column (axis="x") or row (axis="y") of their respective source frames.

    Returns
    -------
    np.ndarray
        Shape ``output_shape``, dtype uint8.
    """
    if fill in ("hold", "wrap") and frame_count == 0:
        raise ValueError(f"frame_count must be provided when fill='{fill}'")

    H, W, C = output_shape
    out = np.zeros(output_shape, dtype=np.uint8)

    for i in range(len(src_indices)):
        src_idx_f = float(src_indices[i])
        start = int(band_starts[i])
        width = int(band_widths[i])

        if interpolate and not np.isclose(src_idx_f, round(src_idx_f)):
            floor_raw = int(np.floor(src_idx_f))
            ceil_raw = floor_raw + 1
            alpha = float(src_idx_f - floor_raw)

            floor_resolved, ceil_resolved = _resolve_src_index_floor_ceil(
                floor_raw, ceil_raw, fill, frame_count
            )
            floor_frame = buffer.get(floor_resolved)
            ceil_frame = buffer.get(ceil_resolved)

            if axis == "x":
                fill_band = make_fill_band(fill_color, H, width)
            else:
                fill_band = make_fill_band(fill_color, width, W)

            band = _blend_frames(
                floor_frame, ceil_frame, alpha, fill_band, axis, start, width,
                slit_source_px=slit_source_px,
            )

            if axis == "x":
                out[:, start:start + width, :] = band
            else:
                out[start:start + width, :, :] = band
        else:
            src_int = _resolve_src_index(src_idx_f, fill, frame_count)
            frame = buffer.get(src_int)

            if axis == "x":
                # Band is a vertical slice: out[:, start:start+width, :]
                if frame is None:
                    fill_band = make_fill_band(fill_color, H, width)
                    out[:, start: start + width, :] = fill_band
                elif slit_source_px is not None:
                    slit_col = frame[:, slit_source_px:slit_source_px + 1, :]
                    out[:, start: start + width, :] = np.repeat(slit_col, width, axis=1)
                else:
                    out[:, start: start + width, :] = frame[:, start: start + width, :]
            else:
                # Band is a horizontal slice: out[start:start+width, :, :]
                if frame is None:
                    fill_band = make_fill_band(fill_color, width, W)
                    out[start: start + width, :, :] = fill_band
                elif slit_source_px is not None:
                    slit_row = frame[slit_source_px:slit_source_px + 1, :, :]
                    out[start: start + width, :, :] = np.repeat(slit_row, width, axis=0)
                else:
                    out[start: start + width, :, :] = frame[start: start + width, :, :]

    return out
