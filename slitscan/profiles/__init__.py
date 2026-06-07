"""Profile registry and built-in profiles."""

from slitscan.profiles.base import get_profile, register, _REGISTRY
from slitscan.profiles import ramp  # noqa: F401 — triggers @register side-effect

__all__ = ["get_profile", "register", "_REGISTRY"]
