from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClipMeta:
    fps: float
    frame_count: int
    width: int
    height: int
    channels: int  # 3=RGB, 4=RGBA


@dataclass
class RenderParams:
    profile: str = "ramp"
    axis: str = "x"
    vanguard: float | None = None   # None = profile default (0.0)
    max_delay: int | None = None    # None = extent - 1
    slice_width: int = 1
    fill: str = "black"
    interpolate: bool = False
    slit_source: float | None = None  # None = normal gather; 0-1 = Trumbull fixed-slit
