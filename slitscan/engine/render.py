"""Sweep-mode render engine.

One output frame is produced per input frame. Each output frame is
assembled by gathering bands from potentially different source frames
as determined by the active profile's delay_map.

Source frame formula (canonical, never change):
    source_frame(band_x, output_t) = output_t - delay[band_x]

Interpretation:
- vanguard band (delay=0): source = output_t          → reads current frame
- lagging band (delay=max_delay): source = output_t - max_delay → oldest side
- For the first max_delay output frames, lagging bands reference negative
  indices which buffer.get() returns None for → fill applied on lagging side
- buffer.get() returns None for indices < 0 or >= frame_count → fill applied
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from slitscan.buffer.base import FrameBuffer
from slitscan.engine.fill import make_fill_color
from slitscan.engine.gather import gather_frame
from slitscan.meta import ClipMeta, RenderParams
from slitscan.profiles.base import get_profile


def render(
    meta: ClipMeta,
    buffer: FrameBuffer,
    params: RenderParams,
    encoder,
    resolved_params_fn: Callable[[int], dict] | None = None,
) -> None:
    """Render all frames and write them via *encoder*.

    Parameters
    ----------
    meta:
        Clip metadata (fps, frame_count, width, height, channels).
    buffer:
        Loaded frame buffer. The engine never calls PyAV directly.
    params:
        Render parameters (profile, axis, vanguard, max_delay, etc.).
    encoder:
        An open ``Encoder`` instance. ``close()`` is called at the end.
    resolved_params_fn:
        Optional callable ``(output_t: int) -> dict`` for modulation.
        If None, static params are used for every frame.
        Phase 4+ populates this.
    """
    profile = get_profile(params.profile)

    # Axis extent determines n_bands and max_delay default
    if params.axis == "x":
        extent = meta.width
    elif params.axis == "y":
        extent = meta.height
    else:
        raise ValueError(f"Unknown axis: {params.axis!r}. Choose 'x' or 'y'.")

    max_delay = params.max_delay
    vanguard = params.vanguard if params.vanguard is not None else 0.0
    slice_width = params.slice_width

    # Precompute band geometry (static for Phase 1)
    n_bands = math.ceil(extent / slice_width)
    band_starts = np.arange(n_bands, dtype=np.int32) * slice_width
    band_widths = np.minimum(slice_width, extent - band_starts).astype(np.int32)

    # x_coords: raw band indices 0..n_bands-1 (profiles normalize internally)
    x_coords = np.arange(n_bands, dtype=np.float64)

    output_shape = (meta.height, meta.width, meta.channels)
    fill_color = make_fill_color(params.fill, meta.channels)

    # Trumbull slit-source pixel position (None = normal gather)
    if params.slit_source is not None:
        if params.axis == "x":
            slit_source_px: int | None = int(round(params.slit_source * (meta.width - 1)))
        else:
            slit_source_px = int(round(params.slit_source * (meta.height - 1)))
    else:
        slit_source_px = None

    # Static params dict (used when no modulation)
    static_params = {
        "vanguard": vanguard,
        "max_delay": max_delay,
        "slice_width": slice_width,
    }

    try:
        for output_t in range(meta.frame_count):
            # Resolve per-frame params (modulation or static)
            if resolved_params_fn is not None:
                p = resolved_params_fn(output_t)
            else:
                p = static_params

            # Compute per-band delays via profile
            delays = profile.delay_map(x_coords, output_t, p)

            # Source frame indices: canonical formula
            # source(x, t) = t - delay[x]
            src_indices = output_t - delays

            frame = gather_frame(
                buffer=buffer,
                output_t=output_t,
                src_indices=src_indices,
                band_starts=band_starts,
                band_widths=band_widths,
                axis=params.axis,
                output_shape=output_shape,
                fill=params.fill,
                fill_color=fill_color,
                frame_count=meta.frame_count,
                interpolate=params.interpolate,
                slit_source_px=slit_source_px,
            )

            encoder.write_frame(frame)

            # Slide the ring buffer window forward.
            # For FullBuffer this is a no-op; for RingBuffer it evicts the
            # oldest frame and loads the next one from the decoder iterator.
            buffer.advance()
    finally:
        encoder.close()
