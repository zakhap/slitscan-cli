"""Grid-based slit-scan: the 2D generalization of the 1D band gather.

The 1D engine (engine/render.py + engine/gather.py) slices a frame into bands
along ONE axis; each band reads from one source frame via

    source(band, output_t) = output_t - delay[band]

Grid mode splits the frame into a cols×rows mosaic and gives every *cell* its
own source frame:

    source(cell, output_t) = output_t - delay[row, col]

At cols×1 this degenerates to vertical bands, at 1×rows to horizontal bands;
in between it is a mosaic where each tile is frozen at a different moment.

Two delay sources are provided (the "where in time is this cell" question):

* ``profile_delay_grid`` — reuse the existing 1D profiles (ramp/reverse/tent)
  evaluated across the grid and combined between the two axes. No new math.
* ``lfo_delay_grid`` — a spatial oscillator field: the CLI's modulation
  oscillators (modulation/oscillators.py) with their phase driven by the
  cell's grid position instead of output time. One LFO per axis → a plaid /
  interference field of times.

The gather (``gather_grid_frame``) reuses the fill / wrap / interpolation
primitives from engine/gather.py so ``--fill`` (including ``wrap``) and
``--interpolate`` behave identically to 1D mode.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from slitscan.engine.fill import _resolve_src_index, make_fill_band, make_fill_color
from slitscan.engine.gather import _resolve_src_index_floor_ceil
from slitscan.modulation.oscillators import get_oscillator
from slitscan.profiles.base import get_profile


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def grid_geometry(
    width: int,
    height: int,
    cols: int,
    rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Partition a ``width``×``height`` frame into ``cols``×``rows`` cells.

    Returns ``(col_starts, col_widths, row_starts, row_heights)`` as int32
    arrays. Cells tile the frame exactly (no gaps, no overlap, no dropped
    pixels), with near-equal sizes even when the extent is not divisible.
    """
    col_edges = np.linspace(0, width, cols + 1).round().astype(np.int32)
    row_edges = np.linspace(0, height, rows + 1).round().astype(np.int32)
    col_starts = col_edges[:-1]
    col_widths = np.diff(col_edges).astype(np.int32)
    row_starts = row_edges[:-1]
    row_heights = np.diff(row_edges).astype(np.int32)
    return col_starts, col_widths, row_starts, row_heights


# ---------------------------------------------------------------------------
# Delay source A: 1D profiles, reused in 2D
# ---------------------------------------------------------------------------

_COMBINERS = {
    "avg": lambda a, b: (a + b) / 2.0,
    "max": np.maximum,
    "min": np.minimum,
    "multiply": lambda a, b: a * b,
}


def profile_delay_grid(
    profile: str,
    cols: int,
    rows: int,
    max_delay: float,
    vanguard: float,
    combine: str = "avg",
) -> np.ndarray:
    """Per-cell delay (shape ``(rows, cols)``) from a 1D profile in 2D.

    The profile's normalized shape is sampled along the columns and along the
    rows independently, then the two are combined per ``combine`` and scaled by
    ``max_delay``. ``combine='max'`` with a single row/col reproduces the pure
    1D ramp/tent exactly.
    """
    if combine not in _COMBINERS:
        raise ValueError(
            f"Unknown combine mode: {combine!r}. "
            f"Choose from {sorted(_COMBINERS)}."
        )
    prof = get_profile(profile)
    unit = {"max_delay": 1.0, "vanguard": vanguard, "slice_width": 1}
    col_d = prof.delay_map(np.arange(cols, dtype=np.float64), 0, unit)  # (cols,)
    row_d = prof.delay_map(np.arange(rows, dtype=np.float64), 0, unit)  # (rows,)
    combined = _COMBINERS[combine](row_d[:, None], col_d[None, :])
    return (np.clip(combined, 0.0, 1.0) * max_delay).astype(np.float64)


# ---------------------------------------------------------------------------
# Delay source B: spatial oscillator field
# ---------------------------------------------------------------------------

@dataclass
class GridLFO:
    """One spatial oscillator over the grid.

    ``axis`` selects the coordinate the oscillator runs over (``"col"`` or
    ``"row"``); ``rate`` is in cycles across that axis; ``depth``/``phase``/
    ``offset`` match the CLI's modulation semantics. Oscillator output is
    mapped to [0,1] (``0.5 + 0.5·osc``) before contributing.
    """
    axis: str = "col"
    osc: str = "sine"
    rate: float = 1.0
    depth: float = 1.0
    phase: float = 0.0
    offset: float = 0.0


_GRID_AXES = ("col", "row")


def parse_grid_lfo(s: str) -> GridLFO:
    """Parse a ``--grid-mod`` string into a :class:`GridLFO`.

    Format mirrors the CLI's ``--mod`` but the destination is a grid axis::

        axis=osc:rate=<r>[,depth=<d>][,phase=<p>][,offset=<o>]

    e.g. ``col=sine:rate=2,depth=0.8,phase=0.25``.
    """
    try:
        axis_part, rest = s.split("=", 1)
        axis = axis_part.strip()
        osc_part, params_str = rest.split(":", 1)
        osc = osc_part.strip()
        params: dict[str, str] = {}
        for item in params_str.split(","):
            item = item.strip()
            if "=" in item:
                k, v = item.split("=", 1)
                params[k.strip()] = v.strip()
    except ValueError as exc:
        raise ValueError(
            f"Invalid --grid-mod string {s!r}. Expected format: "
            "axis=osc:rate=<r>[,depth=<d>][,phase=<p>][,offset=<o>]"
        ) from exc

    if axis not in _GRID_AXES:
        raise ValueError(
            f"Unknown grid LFO axis: {axis!r}. Choose 'col' or 'row'."
        )
    return GridLFO(
        axis=axis,
        osc=osc,
        rate=float(params.get("rate", 1.0)),
        depth=float(params.get("depth", 1.0)),
        phase=float(params.get("phase", 0.0)),
        offset=float(params.get("offset", 0.0)),
    )


def lfo_delay_grid(
    lfos: list[GridLFO],
    cols: int,
    rows: int,
    max_delay: float,
) -> np.ndarray:
    """Per-cell delay (shape ``(rows, cols)``) from a stack of spatial LFOs.

    For each cell, every LFO contributes ``offset + depth·osc01(rate·coord +
    phase)`` where ``coord`` is the cell's normalized position along the LFO's
    axis and ``osc01`` maps the oscillator to [0,1]. Contributions sum (as the
    CLI sums multiple ``--mod`` entries), clamp to [0,1], and scale by
    ``max_delay``.
    """
    if not lfos:
        raise ValueError("lfo_delay_grid requires at least one GridLFO.")
    col_norm = np.arange(cols, dtype=np.float64) / max(cols - 1, 1)
    row_norm = np.arange(rows, dtype=np.float64) / max(rows - 1, 1)

    delay01 = np.zeros((rows, cols), dtype=np.float64)
    for lfo in lfos:
        osc_fn = get_oscillator(lfo.osc)
        if lfo.axis == "col":
            coord = col_norm[None, :]
        elif lfo.axis == "row":
            coord = row_norm[:, None]
        else:
            raise ValueError(
                f"Unknown grid LFO axis: {lfo.axis!r}. Choose 'col' or 'row'."
            )
        # osc_fn is scalar; vectorize over the coordinate grid.
        phase_field = lfo.rate * coord + lfo.phase
        osc_vals = np.vectorize(osc_fn)(phase_field)
        osc01 = 0.5 + 0.5 * osc_vals
        delay01 = delay01 + lfo.offset + lfo.depth * osc01

    return (np.clip(delay01, 0.0, 1.0) * max_delay).astype(np.float64)


# ---------------------------------------------------------------------------
# Gather
# ---------------------------------------------------------------------------

def gather_grid_frame(
    buffer,
    output_t: int,
    delay_grid: np.ndarray,          # (rows, cols) float — per-cell delay
    col_starts: np.ndarray,
    col_widths: np.ndarray,
    row_starts: np.ndarray,
    row_heights: np.ndarray,
    output_shape: tuple,             # (H, W, C)
    fill: str,
    fill_color: np.ndarray,          # (C,)
    frame_count: int = 0,
    interpolate: bool = False,
) -> np.ndarray:
    """Assemble one output frame as a 2D mosaic of per-cell time reads.

    For each cell, the source frame index is ``output_t - delay[row, col]``;
    the cell rectangle is copied from that frame. Out-of-range indices follow
    ``fill`` (``hold``/``wrap`` resolve to a valid frame; ``black``/``white``/
    ``transparent`` paint ``fill_color``). With ``interpolate`` and a fractional
    delay, the two straddling frames are linearly blended.
    """
    if fill in ("hold", "wrap") and frame_count == 0:
        raise ValueError(f"frame_count must be provided when fill='{fill}'")

    rows, cols = delay_grid.shape
    out = np.zeros(output_shape, dtype=np.uint8)

    for r in range(rows):
        r0 = int(row_starts[r])
        r1 = r0 + int(row_heights[r])
        for c in range(cols):
            c0 = int(col_starts[c])
            c1 = c0 + int(col_widths[c])
            src_f = float(output_t) - float(delay_grid[r, c])

            if interpolate and not np.isclose(src_f, round(src_f)):
                floor_raw = int(np.floor(src_f))
                ceil_raw = floor_raw + 1
                alpha = src_f - floor_raw
                f0i, f1i = _resolve_src_index_floor_ceil(
                    floor_raw, ceil_raw, fill, frame_count)
                f0 = buffer.get(f0i)
                f1 = buffer.get(f1i)
                fb = make_fill_band(fill_color, r1 - r0, c1 - c0)
                b0 = fb if f0 is None else f0[r0:r1, c0:c1, :]
                b1 = fb if f1 is None else f1[r0:r1, c0:c1, :]
                cell = np.clip(
                    (1.0 - alpha) * b0.astype(np.float32)
                    + alpha * b1.astype(np.float32), 0, 255).astype(np.uint8)
                out[r0:r1, c0:c1, :] = cell
            else:
                src_int = _resolve_src_index(src_f, fill, frame_count)
                frame = buffer.get(src_int)
                if frame is None:
                    out[r0:r1, c0:c1, :] = make_fill_band(
                        fill_color, r1 - r0, c1 - c0)
                else:
                    out[r0:r1, c0:c1, :] = frame[r0:r1, c0:c1, :]

    return out


# ---------------------------------------------------------------------------
# Render loop
# ---------------------------------------------------------------------------

def render_grid(
    meta,
    buffer,
    delay_grid: np.ndarray,          # (rows, cols) static per-cell delay
    fill: str,
    interpolate: bool,
    encoder,
) -> None:
    """Render every frame of a grid-mode slit-scan and write via ``encoder``.

    The delay grid is static (computed once); motion comes from ``output_t``
    advancing through the clip, exactly as in 1D sweep mode. Mirrors
    engine/render.render but with the 2D gather.
    """
    rows, cols = delay_grid.shape
    col_starts, col_widths, row_starts, row_heights = grid_geometry(
        meta.width, meta.height, cols, rows)
    output_shape = (meta.height, meta.width, meta.channels)
    fill_color = make_fill_color(fill, meta.channels)

    try:
        for output_t in range(meta.frame_count):
            frame = gather_grid_frame(
                buffer=buffer,
                output_t=output_t,
                delay_grid=delay_grid,
                col_starts=col_starts,
                col_widths=col_widths,
                row_starts=row_starts,
                row_heights=row_heights,
                output_shape=output_shape,
                fill=fill,
                fill_color=fill_color,
                frame_count=meta.frame_count,
                interpolate=interpolate,
            )
            encoder.write_frame(frame)
            buffer.advance()
    finally:
        encoder.close()
