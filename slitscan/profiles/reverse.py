"""Reverse ramp profile.

Mirror of the ramp: vanguard at the right (or bottom), delay=0 at the right
edge, increasing toward the left. With vanguard >= 0.5, band N-1 (rightmost)
has delay=0 and band 0 (leftmost) has delay=max_delay.
"""

from __future__ import annotations

import numpy as np

from slitscan.profiles.base import register


@register("reverse")
class ReverseProfile:
    """Reverse ramp: delay 0 at right edge, max_delay at left edge."""

    def delay_map(
        self,
        x_coords: np.ndarray,
        output_t: int,
        params: dict,
    ) -> np.ndarray:
        n = len(x_coords)
        vanguard = params.get("vanguard", 1.0)  # default: right edge
        max_delay = params["max_delay"]
        t = x_coords / max(n - 1, 1)
        # vanguard at right: delay = (1-t) * max_delay at left
        if vanguard >= 0.5:
            delay = (1.0 - t) * max_delay
        else:
            delay = t * max_delay
        return delay.astype(float)
