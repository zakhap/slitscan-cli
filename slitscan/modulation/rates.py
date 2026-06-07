"""Rate string parsing for modulation.

Converts rate strings with units to cycles-per-frame.
"""


def parse_rate(rate_str: str, fps: float, frame_count: int) -> float:
    """Parse a rate string like '0.5cyc', '2hz', '4frames' -> cycles per frame."""
    rate_str = rate_str.strip().lower()
    if rate_str.endswith("hz"):
        hz = float(rate_str[:-2])
        if fps <= 0:
            raise ValueError(f"fps must be > 0 to convert Hz rate, got fps={fps}")
        return hz / fps
    elif rate_str.endswith("cyc"):
        cyc = float(rate_str[:-3])
        if frame_count <= 0:
            raise ValueError(f"frame_count must be > 0 to convert cyc rate, got frame_count={frame_count}")
        return cyc / frame_count
    elif rate_str.endswith("frames"):
        n_frames = float(rate_str[:-6])
        if n_frames == 0:
            raise ValueError("rate in 'frames' units cannot be 0 (division by zero)")
        return 1.0 / n_frames
    else:
        raise ValueError(
            f"Unknown rate unit in {rate_str!r}. Use 'hz', 'cyc', or 'frames'."
        )
