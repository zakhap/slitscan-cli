"""Profile protocol and registration registry.

A Profile is a pure function object — no I/O, no state, no caching.
It maps band positions and time to per-band delays.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Profile(Protocol):
    """Protocol that all slit-scan profiles must satisfy."""

    def delay_map(
        self,
        x_coords: np.ndarray,
        output_t: int,
        params: dict,
    ) -> np.ndarray:
        """Return per-band delay values.

        Parameters
        ----------
        x_coords:
            Band indices ``[0, 1, ..., n_bands-1]`` as a float64 array.
        output_t:
            Current output frame index (0-based). Used by modulated profiles.
        params:
            Runtime parameters dict. Must contain at minimum:
            - ``"max_delay"`` (int): maximum delay in frames
            - ``"vanguard"`` (float, 0..1): position of the zero-delay band
            - ``"slice_width"`` (int): pixels per band (informational)

        Returns
        -------
        np.ndarray
            Float64 array of shape ``(n_bands,)`` with delay values in
            ``[0, max_delay]``.
        """
        ...


# Module-level registry: name → Profile instance
_REGISTRY: dict[str, "Profile"] = {}


def register(name: str):
    """Class decorator that registers a profile under *name*."""
    def decorator(cls):
        _REGISTRY[name] = cls()
        return cls
    return decorator


def _load_all() -> None:
    """Import all profile modules so their @register decorators run."""
    from . import ramp, reverse, tent  # noqa: F401


def get_profile(name: str) -> "Profile":
    """Look up a registered profile by name.

    Raises
    ------
    ValueError
        If *name* is not in the registry.
    """
    _load_all()
    if name not in _REGISTRY:
        available = sorted(_REGISTRY.keys())
        raise ValueError(
            f"Unknown profile: {name!r}. Available profiles: {available}"
        )
    return _REGISTRY[name]
