"""Modulation subsystem (Phase 4+).

LFO definitions and the modulation resolver.
"""

from .patch import parse_mod_string, load_mod_file, ModEntry
from .resolve import make_resolved_params_fn, resolve_params, prepare_mods
from .oscillators import get_oscillator, OSCILLATORS
from .rates import parse_rate

__all__ = [
    "parse_mod_string",
    "load_mod_file",
    "ModEntry",
    "make_resolved_params_fn",
    "resolve_params",
    "prepare_mods",
    "get_oscillator",
    "OSCILLATORS",
    "parse_rate",
]
