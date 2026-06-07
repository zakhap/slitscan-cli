"""Linear delay ramp profile.

With vanguard=0.0 (default): delay increases left→right.
  - Band 0 (leftmost): delay = 0  → reads the most recent frame
  - Band N-1 (rightmost): delay = max_delay → reads the oldest frame

With vanguard=1.0: delay increases right→left (reversed ramp).
  - Band N-1 (rightmost): delay = 0
  - Band 0 (leftmost): delay = max_delay

This is the canonical ramp; other profiles (tent, etc.) extend the registry.
"""

from __future__ import annotations

import numpy as np

from slitscan.profiles.base import register


@register("ramp")
class RampProfile:
    """Linear ramp: delay 0 at the vanguard edge, max_delay at the far edge."""

    def delay_map(
        self,
        x_coords: np.ndarray,
        output_t: int,
        params: dict,
    ) -> np.ndarray:
        """Compute per-band delays for a linear ramp.

        Parameters
        ----------
        x_coords:
            Band indices ``[0, 1, ..., n_bands-1]``.
        output_t:
            Current output frame (unused by static ramp, present for protocol).
        params:
            Must contain ``"max_delay"`` (int) and ``"vanguard"`` (float 0..1).

        Returns
        -------
        np.ndarray
            Float64 delays of shape ``(n_bands,)``.
        """
        n = len(x_coords)
        vanguard: float = float(params.get("vanguard", 0.0))
        max_delay: float = float(params["max_delay"])

        # Normalize band positions to [0, 1]
        t = x_coords / max(n - 1, 1)  # shape (n,), dtype float64

        if vanguard <= 0.5:
            # Vanguard at left edge: delay increases left → right
            delay = t * max_delay
        else:
            # Vanguard at right edge: delay increases right → left
            delay = (1.0 - t) * max_delay

        return delay.astype(np.float64)
