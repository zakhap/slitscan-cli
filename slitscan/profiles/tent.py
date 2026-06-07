"""Tent / vee profile.

Vanguard at ``vanguard`` position (default center 0.5); delay flares outward
from the vanguard toward both edges.  The longer arm (farther edge) always
reaches ``max_delay``; the shorter arm (nearer edge) reaches a proportionally
smaller value.  When ``vanguard=0.5`` the profile is symmetric and both edges
reach ``max_delay``.
"""

from __future__ import annotations

import numpy as np

from slitscan.profiles.base import register


@register("tent")
class TentProfile:
    """Tent: delay=0 at vanguard, linearly increasing outward.

    The longer arm always reaches ``max_delay``; the shorter arm (if the
    vanguard is off-center) reaches a proportionally smaller value.
    """

    def delay_map(
        self,
        x_coords: np.ndarray,
        output_t: int,
        params: dict,
    ) -> np.ndarray:
        n = len(x_coords)
        vanguard = params.get("vanguard", 0.5)  # default: center
        max_delay = params["max_delay"]
        t = x_coords / max(n - 1, 1)  # normalized 0..1
        # distance from vanguard, normalized to [0,1] on each side
        dist_normalized = np.abs(t - vanguard) / max(max(vanguard, 1.0 - vanguard), 1e-9)
        # clamp to [0,1] and scale by max_delay
        return (np.clip(dist_normalized, 0.0, 1.0) * max_delay).astype(float)
