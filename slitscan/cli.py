"""slitscan CLI — all commands and all flags pre-wired.

Later phases add implementation files but do NOT touch this file.
Flags for future phases raise NotImplementedError until their phase lands.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import typer

app = typer.Typer(
    name="slitscan",
    help="Slit-scan video rendering toolkit.",
    add_completion=False,
)


# ---------------------------------------------------------------------------
# render command
# ---------------------------------------------------------------------------

@app.command()
def render(
    # Positional
    input: Path = typer.Argument(..., help="Source video file."),
    output: Path = typer.Argument(..., help="Output path; file extension selects codec."),

    # Profile & geometry
    profile: str = typer.Option(
        "ramp",
        "--profile",
        help="Delay profile: ramp | reverse | tent | (Phase 2+: wave, custom).",
    ),
    axis: str = typer.Option(
        "x",
        "--axis",
        help="Scan axis: x (vertical bands) | y (horizontal bands).",
    ),
    vanguard: float | None = typer.Option(
        None,
        "--vanguard",
        help="Vanguard position 0.0–1.0 (zero-delay edge). Default: profile-specific.",
        min=0.0,
        max=1.0,
    ),
    max_delay: int | None = typer.Option(
        None,
        "--max-delay",
        help="Maximum delay in frames. Default: extent-1 (full width or height).",
        min=0,
    ),
    slice_width: int = typer.Option(
        1,
        "--slice-width",
        help="Pixels per band (column/row slice width). Default: 1.",
        min=1,
    ),

    # Fill
    fill: str = typer.Option(
        "black",
        "--fill",
        help="Fill mode for out-of-range frames: black | white | transparent | hold | wrap.",
    ),

    # Interpolation (Phase 5)
    interpolate: bool = typer.Option(
        False,
        "--interpolate/--no-interpolate",
        help="Sub-frame linear interpolation (Phase 5).",
    ),

    # Resize / fit (Phase 1: crop only)
    resize: str | None = typer.Option(
        None,
        "--resize",
        help="Target dimensions as WxH, e.g. 1920x1080. Default: use source dimensions.",
    ),
    fit: str = typer.Option(
        "crop",
        "--fit",
        help="Scaling strategy: crop | letterbox (Phase 2) | stretch (Phase 2).",
    ),

    # Buffer policy (Phase 6: ring buffer)
    buffer: str = typer.Option(
        "auto",
        "--buffer",
        help="Buffer policy: auto | full | ring. Default: auto (full in Phase 1).",
    ),
    memory_budget: str | None = typer.Option(
        None,
        "--memory-budget",
        help="Max RAM for frame buffer, e.g. 8G, 512M. Used by 'auto' and 'ring'.",
    ),

    # Trumbull / fixed-slit mode
    slit_source: float | None = typer.Option(
        None,
        "--slit-source",
        help=(
            "Trumbull slit-scan: fixed slit position 0.0–1.0 along the axis. "
            "All output bands are gathered from this single column/row in each "
            "source frame (Stargate/2001 effect). Default: None (normal gather)."
        ),
        min=0.0,
        max=1.0,
    ),

    # Modulation (Phase 4)
    mod: list[str] = typer.Option(
        [],
        "--mod",
        help=(
            "Modulation patch (repeatable). Format: dest=osc:rate=<r>,depth=<d>. "
            "Example: --mod vanguard=osc:rate=0.5Hz,depth=0.3"
        ),
    ),
    mod_file: Path | None = typer.Option(
        None,
        "--mod-file",
        help="YAML file containing modulation patch definitions.",
    ),

    # Output options
    fps: float | None = typer.Option(
        None,
        "--fps",
        help="Override output frame rate. Default: match source.",
        min=1.0,
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet/--no-quiet",
        help="Suppress progress output.",
    ),
) -> None:
    """Render a slit-scan video (sweep mode).

    Each output frame is assembled by gathering bands from different
    source frames as determined by the delay profile.

    Source frame formula:
        source(band_x, output_t) = output_t - delay[band_x]
    """
    # --- Early parameter validation (before any I/O) ---
    if axis not in ("x", "y"):
        raise typer.BadParameter(f"axis must be 'x' or 'y', got {axis!r}")
    from slitscan.profiles.base import get_profile
    try:
        get_profile(profile)
    except ValueError as e:
        raise typer.BadParameter(str(e))

    # --- Validate flags that are not yet implemented ---
    if fit not in ("crop",):
        _not_implemented(f"--fit={fit}", phase=2)
    if memory_budget is not None and buffer not in ("auto", "ring"):
        typer.echo(
            "Warning: --memory-budget is only relevant with --buffer=auto or --buffer=ring.",
            err=True,
        )

    # --- Validate inputs ---
    if not input.exists():
        typer.echo(f"Error: input file not found: {input}", err=True)
        raise typer.Exit(code=1)

    # --- Imports (deferred to keep --help fast) ---
    from slitscan.io.decode import open_video
    from slitscan.io.encode import open_encoder
    from slitscan.buffer.full import FullBuffer
    from slitscan.buffer.ring import RingBuffer, validate_ring_compatible, parse_memory_budget
    from slitscan.meta import RenderParams, ClipMeta
    from slitscan.engine.render import render as _render

    # --- Probe / open input ---
    try:
        meta, frames_iter = open_video(str(input), rgba=(fill == "transparent"))
    except Exception as exc:
        typer.echo(f"Error opening input: {exc}", err=True)
        raise typer.Exit(code=1)

    # --- Handle resize ---
    target_w, target_h = meta.width, meta.height
    if resize is not None:
        try:
            w_str, h_str = resize.lower().split("x")
            target_w, target_h = int(w_str), int(h_str)
        except ValueError:
            typer.echo(
                f"Error: --resize must be WxH format (e.g. 1920x1080), got: {resize!r}",
                err=True,
            )
            raise typer.Exit(code=1)

    output_fps = fps if fps is not None else meta.fps

    # --- Resolve max_delay default (uses target dims, before buffer is built) ---
    axis_extent = target_w if axis == "x" else target_h
    resolved_max_delay = max_delay if max_delay is not None else axis_extent - 1

    # --- Determine buffer type ---
    frame_bytes = target_w * target_h * meta.channels
    full_buffer_bytes = frame_bytes * meta.frame_count

    if memory_budget is not None:
        budget_bytes = parse_memory_budget(memory_budget)
    else:
        budget_bytes = 8 * 1024 ** 3  # 8 GB default

    if buffer == "full":
        use_ring = False
    elif buffer == "ring":
        use_ring = True
    else:  # "auto"
        use_ring = full_buffer_bytes > budget_bytes

    # --- Ring buffer reach validation ---
    if use_ring and resolved_max_delay >= meta.frame_count:
        typer.echo(
            f"Error: --max-delay={resolved_max_delay} >= frame_count={meta.frame_count}; "
            "ring buffer window cannot span the full clip. Reduce --max-delay or use --buffer=full.",
            err=True,
        )
        raise typer.Exit(code=1)

    # --- Validate ring/fill compatibility ---
    if use_ring:
        try:
            validate_ring_compatible(resolved_max_delay, fill)
        except ValueError as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1)

    # --- Build buffer ---
    if not quiet:
        if use_ring:
            typer.echo("Using ring buffer (streaming)…")
        else:
            typer.echo("Loading frames into buffer…")

    if use_ring:
        # Ring buffer: apply resize normalization lazily if needed
        if resize is not None:
            from slitscan.io.normalize import normalize_frame
            frames_iter = (normalize_frame(f, target_w, target_h, fit) for f in frames_iter)
        buf = RingBuffer(meta, frames_iter, max_delay=resolved_max_delay)
        # actual frame count comes from meta (ring doesn't eagerly load)
        actual_frame_count = meta.frame_count
    else:
        # Full buffer: eagerly load all frames (with optional resize)
        if resize is not None:
            from slitscan.io.normalize import normalize_frame

            def _norm_iter():
                for f in frames_iter:
                    yield normalize_frame(f, target_w, target_h, fit)

            buf = FullBuffer(meta, _norm_iter())
        else:
            buf = FullBuffer(meta, frames_iter)
        actual_frame_count = buf.frame_count

    actual_meta = ClipMeta(
        fps=output_fps,
        frame_count=actual_frame_count,
        width=target_w,
        height=target_h,
        channels=meta.channels,
    )

    # --- Print plan ---
    if not quiet:
        _print_plan(actual_meta, buf, resolved_max_delay, axis, profile, slice_width, output, use_ring=use_ring)

    # --- Build params ---
    render_params = RenderParams(
        profile=profile,
        axis=axis,
        vanguard=vanguard,
        max_delay=resolved_max_delay,
        slice_width=slice_width,
        fill=fill,
        interpolate=interpolate,
        slit_source=slit_source,
    )

    # --- Open encoder ---
    try:
        encoder = open_encoder(
            path=str(output),
            width=actual_meta.width,
            height=actual_meta.height,
            fps=actual_meta.fps,
            fill=fill,
        )
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)
    except Exception as exc:
        typer.echo(f"Error opening encoder: {exc}", err=True)
        raise typer.Exit(code=1)

    # --- Build modulation resolver ---
    resolved_params_fn = None
    if mod or mod_file is not None:
        from slitscan.modulation.patch import parse_mod_string, load_mod_file, ModEntry
        from slitscan.modulation.resolve import make_resolved_params_fn

        mod_entries: list[ModEntry] = []
        base_patch: dict = {}

        if mod_file is not None:
            base_patch, mod_entries = load_mod_file(str(mod_file))

        for m in mod:
            mod_entries.append(parse_mod_string(m))

        base_params = {
            "vanguard": vanguard if vanguard is not None else 0.0,
            "max_delay": resolved_max_delay,
            "slice_width": slice_width,
        }
        base_params.update(base_patch)

        resolved_params_fn = make_resolved_params_fn(
            base_params, mod_entries,
            fps=actual_meta.fps,
            frame_count=actual_meta.frame_count,
        )

    # --- Render ---
    if not quiet:
        typer.echo(f"Rendering {actual_meta.frame_count} frames…")

    try:
        _render(
            meta=actual_meta,
            buffer=buf,
            params=render_params,
            encoder=encoder,
            resolved_params_fn=resolved_params_fn,
        )
    except Exception as exc:
        typer.echo(f"Render error: {exc}", err=True)
        raise typer.Exit(code=1)

    if not quiet:
        typer.echo(f"Done → {output}")


# ---------------------------------------------------------------------------
# collapse command (Phase 7)
# ---------------------------------------------------------------------------

@app.command()
def collapse(
    # Positional
    input: Path = typer.Argument(..., help="Source video file."),
    output: Path = typer.Argument(..., help="Output image path."),

    # Slit options
    slit_position: float = typer.Option(
        0.5,
        "--slit-position",
        help="Normalized slit position 0.0–1.0 across the frame. Default: 0.5 (center).",
        min=0.0,
        max=1.0,
    ),
    direction: str = typer.Option(
        "forward",
        "--direction",
        help="Time direction: forward | reverse.",
    ),
    axis: str = typer.Option(
        "x",
        "--axis",
        help="Slit axis: x (vertical slit) | y (horizontal slit).",
    ),
    slice_width: int = typer.Option(
        1,
        "--slice-width",
        help="Pixels per temporal slice.",
        min=1,
    ),

    # Resize / fit
    resize: str | None = typer.Option(
        None,
        "--resize",
        help="Target dimensions as WxH.",
    ),
    fit: str = typer.Option(
        "crop",
        "--fit",
        help="Scaling strategy: crop | letterbox | stretch.",
    ),
) -> None:
    """Photofinish: accumulate slit history into a single image.

    Each frame contributes one vertical (or horizontal) slice at the
    slit position, producing a time-space composite image.
    """
    # --- Validate parameters ---
    if direction not in ("forward", "reverse"):
        typer.echo(f"Error: --direction must be 'forward' or 'reverse', got {direction!r}", err=True)
        raise typer.Exit(code=1)
    if axis not in ("x", "y"):
        typer.echo(f"Error: --axis must be 'x' or 'y', got {axis!r}", err=True)
        raise typer.Exit(code=1)
    if not input.exists():
        typer.echo(f"Error: input file not found: {input}", err=True)
        raise typer.Exit(code=1)

    # --- Deferred imports ---
    from slitscan.io.decode import open_video
    from slitscan.io.normalize import normalize_frame
    from slitscan.meta import ClipMeta
    from slitscan.engine.collapse import collapse as _collapse
    from PIL import Image

    # --- Open input ---
    try:
        meta, frames_iter = open_video(str(input))
    except Exception as exc:
        typer.echo(f"Error opening input: {exc}", err=True)
        raise typer.Exit(code=1)

    # --- Handle resize ---
    target_w, target_h = meta.width, meta.height
    if resize is not None:
        try:
            w_str, h_str = resize.lower().split("x")
            target_w, target_h = int(w_str), int(h_str)
        except ValueError:
            typer.echo(
                f"Error: --resize must be WxH format (e.g. 1920x1080), got: {resize!r}",
                err=True,
            )
            raise typer.Exit(code=1)
        frames_iter = (normalize_frame(f, target_w, target_h, fit) for f in frames_iter)
        meta = ClipMeta(
            fps=meta.fps,
            frame_count=meta.frame_count,
            width=target_w,
            height=target_h,
            channels=meta.channels,
        )

    # --- Run collapse ---
    try:
        result = _collapse(
            meta=meta,
            frames_iter=frames_iter,
            slit_position=slit_position,
            direction=direction,
            axis=axis,
            slice_width=slice_width,
        )
    except Exception as exc:
        typer.echo(f"Collapse error: {exc}", err=True)
        raise typer.Exit(code=1)

    # --- Save output ---
    try:
        img = Image.fromarray(result)
        img.save(str(output))
    except Exception as exc:
        typer.echo(f"Error saving output: {exc}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Saved {output} ({result.shape[1]}x{result.shape[0]})")


# ---------------------------------------------------------------------------
# info command — print clip metadata
# ---------------------------------------------------------------------------

@app.command()
def info(
    input: Path = typer.Argument(..., help="Video file to inspect."),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output metadata as JSON.",
    ),
) -> None:
    """Print metadata for a video file."""
    if not input.exists():
        typer.echo(f"Error: file not found: {input}", err=True)
        raise typer.Exit(code=1)

    from slitscan.io.decode import open_video

    try:
        meta, frames_iter = open_video(str(input))
    except Exception as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    if json_output:
        import json
        data = {
            "fps": meta.fps,
            "frame_count": meta.frame_count,
            "width": meta.width,
            "height": meta.height,
            "channels": meta.channels,
        }
        typer.echo(json.dumps(data, indent=2))
    else:
        typer.echo(f"File:        {input}")
        typer.echo(f"FPS:         {meta.fps:.3f}")
        typer.echo(f"Frame count: {meta.frame_count}")
        typer.echo(f"Dimensions:  {meta.width} × {meta.height}")
        typer.echo(f"Channels:    {meta.channels} ({'RGB' if meta.channels == 3 else 'RGBA'})")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_plan(
    meta,
    buf,
    max_delay: int,
    axis: str,
    profile: str,
    slice_width: int,
    output: Path,
    use_ring: bool = False,
) -> None:
    """Print a human-readable render plan at startup."""
    from slitscan.io.encode import codec_name_for_path
    codec = codec_name_for_path(output)
    extent = meta.width if axis == "x" else meta.height
    n_bands = math.ceil(extent / slice_width)

    if use_ring:
        # Ring buffer: projected RAM is the window size, not the full clip
        ram_mb = buf.projected_ram_mb if hasattr(buf, "projected_ram_mb") else float("nan")
        buf_label = f"ring  ({ram_mb:.1f} MB window, {max_delay + 1} frames)"
    else:
        ram_mb = buf.projected_ram_mb if hasattr(buf, "projected_ram_mb") else float("nan")
        buf_label = f"full  ({ram_mb:.1f} MB loaded)"

    typer.echo("─" * 50)
    typer.echo("Render plan")
    typer.echo("─" * 50)
    typer.echo(f"  Input:        {meta.width}×{meta.height}  {meta.fps:.3f} fps  {meta.frame_count} frames")
    typer.echo(f"  Profile:      {profile}  axis={axis}  slice_width={slice_width}")
    typer.echo(f"  Max delay:    {max_delay} frames")
    typer.echo(f"  Bands:        {n_bands}")
    typer.echo(f"  Buffer:       {buf_label}")
    typer.echo(f"  Codec:        {codec}  →  {output.name}")
    typer.echo("─" * 50)


def _not_implemented(flag: str, phase: int) -> None:
    """Emit a clear error and exit if a not-yet-implemented flag is used."""
    typer.echo(
        f"Error: {flag} is not yet implemented (scheduled for Phase {phase}).",
        err=True,
    )
    raise typer.Exit(code=2)


if __name__ == "__main__":
    app()
