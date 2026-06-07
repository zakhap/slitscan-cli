"""Oscillator functions for modulation.

Each oscillator is a pure function: osc(phase) -> float in [-1, 1].
phase is a float (total accumulated phase, in cycles).
"""

import math


def sine(phase: float) -> float:
    return math.sin(2 * math.pi * phase)


def triangle(phase: float) -> float:
    # Period 1, goes -1->1->-1
    p = phase % 1.0
    if p < 0.5:
        return 4.0 * p - 1.0
    return 3.0 - 4.0 * p


OSCILLATORS = {"sine": sine, "triangle": triangle}


def get_oscillator(name: str):
    if name not in OSCILLATORS:
        raise ValueError(f"Unknown oscillator: {name!r}. Available: {list(OSCILLATORS)}")
    return OSCILLATORS[name]
