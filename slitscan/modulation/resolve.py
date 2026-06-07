"""Per-frame resolution: takes base params + active mods + output_t -> resolved params dict."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from slitscan.modulation.oscillators import get_oscillator
from slitscan.modulation.rates import parse_rate
from slitscan.modulation.patch import ModEntry

DEST_CLAMPS = {
    "vanguard": (0.0, 1.0),
    "max_delay": (0.0, 1e6),
    "slice_width": (1.0, 1e6),
    "fill_alpha": (0.0, 1.0),
}


@dataclass
class ResolvedMod:
    dest: str
    osc_fn: Callable[[float], float]   # callable(phase) -> [-1, 1]
    cycles_per_frame: float
    depth: float
    phase: float
    offset: float


VALID_DESTS = set(DEST_CLAMPS.keys())


def prepare_mods(mods: list[ModEntry], fps: float, frame_count: int) -> list[ResolvedMod]:
    """Convert ModEntry list to ResolvedMod list (resolve rate strings to cycles/frame)."""
    resolved = []
    for m in mods:
        if m.dest not in VALID_DESTS:
            raise ValueError(
                f"Unknown modulation destination: {m.dest!r}. "
                f"Valid destinations: {sorted(VALID_DESTS)}"
            )
        resolved.append(ResolvedMod(
            dest=m.dest,
            osc_fn=get_oscillator(m.osc),
            cycles_per_frame=parse_rate(m.rate_str, fps, frame_count),
            depth=m.depth,
            phase=m.phase,
            offset=m.offset,
        ))
    return resolved


def resolve_params(base: dict, mods: list[ResolvedMod], output_t: int) -> dict:
    """Compute per-frame resolved params: base + modulation applied + clamped."""
    params = dict(base)

    # Group mods by destination, sum contributions
    for m in mods:
        osc_val = m.osc_fn(m.cycles_per_frame * output_t + m.phase)
        contribution = m.offset + m.depth * osc_val
        params[m.dest] = params.get(m.dest, 0.0) + contribution

    # Clamp all modulated destinations
    for dest, (lo, hi) in DEST_CLAMPS.items():
        if dest in params:
            params[dest] = max(lo, min(hi, float(params[dest])))

    # max_delay and slice_width must be integers
    if "max_delay" in params:
        params["max_delay"] = int(round(params["max_delay"]))
    if "slice_width" in params:
        params["slice_width"] = max(1, int(round(params["slice_width"])))

    return params


def make_resolved_params_fn(
    base_params: dict,
    mods: list[ModEntry],
    fps: float,
    frame_count: int,
) -> Callable[[int], dict]:
    """Build a per-frame params resolver from base params + mod list."""
    resolved_mods = prepare_mods(mods, fps, frame_count)

    def _resolve(output_t: int) -> dict:
        return resolve_params(base_params, resolved_mods, output_t)

    return _resolve
