"""Tests for the modulation subsystem (Phase 4)."""

import pytest

from slitscan.modulation.oscillators import sine, triangle
from slitscan.modulation.rates import parse_rate
from slitscan.modulation.patch import ModEntry, parse_mod_string
from slitscan.modulation.resolve import make_resolved_params_fn


# ---------------------------------------------------------------------------
# Oscillator tests
# ---------------------------------------------------------------------------

def test_sine_zero_at_zero():
    assert sine(0) == pytest.approx(0.0)


def test_sine_one_at_quarter():
    assert sine(0.25) == pytest.approx(1.0)


def test_triangle_midpoint():
    assert triangle(0.5) == pytest.approx(1.0)


def test_triangle_at_zero():
    assert triangle(0.0) == pytest.approx(-1.0)


def test_triangle_at_three_quarters():
    assert triangle(0.75) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Rate parsing tests
# ---------------------------------------------------------------------------

def test_hz_to_cpf():
    assert parse_rate("2hz", fps=24.0, frame_count=240) == pytest.approx(2 / 24)


def test_cyc_to_cpf():
    assert parse_rate("1cyc", fps=24.0, frame_count=240) == pytest.approx(1 / 240)


def test_frames_to_cpf():
    assert parse_rate("4frames", fps=24.0, frame_count=240) == pytest.approx(0.25)


def test_unknown_rate_unit_raises():
    with pytest.raises(ValueError, match="Unknown rate unit"):
        parse_rate("2bpm", fps=24.0, frame_count=240)


def test_fractional_cyc():
    assert parse_rate("0.5cyc", fps=24.0, frame_count=240) == pytest.approx(0.5 / 240)


# ---------------------------------------------------------------------------
# Mod string parsing tests
# ---------------------------------------------------------------------------

def test_parse_mod_string():
    m = parse_mod_string("vanguard=sine:rate=0.5cyc,depth=0.5,phase=0.0")
    assert m.dest == "vanguard"
    assert m.osc == "sine"
    assert m.rate_str == "0.5cyc"
    assert m.depth == 0.5
    assert m.phase == 0.0


def test_parse_mod_string_defaults():
    m = parse_mod_string("max_delay=triangle:rate=2hz")
    assert m.dest == "max_delay"
    assert m.osc == "triangle"
    assert m.rate_str == "2hz"
    assert m.depth == 1.0
    assert m.phase == 0.0
    assert m.offset == 0.0


def test_parse_mod_string_with_offset():
    m = parse_mod_string("fill_alpha=sine:rate=1cyc,depth=0.3,offset=0.7")
    assert m.offset == pytest.approx(0.7)
    assert m.depth == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Resolve tests
# ---------------------------------------------------------------------------

def test_one_cyc_returns_to_phase():
    """1cyc sine should return to same value at t=0 and t=frame_count (exactly 1 cycle later)."""
    mods = [ModEntry(dest="vanguard", osc="sine", rate_str="1cyc", depth=0.5, phase=0.0, offset=0.0)]
    fn = make_resolved_params_fn({"vanguard": 0.5}, mods, fps=24.0, frame_count=100)
    p_start = fn(0)["vanguard"]
    p_end = fn(100)["vanguard"]  # exactly 1 cycle later
    assert p_start == pytest.approx(p_end, abs=1e-9)


def test_vanguard_clamped_to_0_1():
    """With depth=2.0, raw value could exceed 1.0 — must be clamped to [0, 1]."""
    mods = [ModEntry(dest="vanguard", osc="sine", rate_str="1cyc", depth=2.0, phase=0.0, offset=0.0)]
    fn = make_resolved_params_fn({"vanguard": 0.5}, mods, fps=24.0, frame_count=100)
    for t in range(100):
        v = fn(t)["vanguard"]
        assert 0.0 <= v <= 1.0, f"vanguard out of range at t={t}: {v}"


def test_max_delay_returns_int():
    """max_delay should always be an integer after resolution."""
    mods = [ModEntry(dest="max_delay", osc="sine", rate_str="1cyc", depth=10.0, phase=0.0, offset=0.0)]
    fn = make_resolved_params_fn({"max_delay": 50}, mods, fps=24.0, frame_count=100)
    for t in range(10):
        v = fn(t)["max_delay"]
        assert isinstance(v, int), f"max_delay should be int, got {type(v)} at t={t}"


def test_slice_width_returns_int_at_least_1():
    """slice_width should always be an integer >= 1 after resolution."""
    mods = [ModEntry(dest="slice_width", osc="sine", rate_str="1cyc", depth=100.0, phase=0.0, offset=0.0)]
    fn = make_resolved_params_fn({"slice_width": 10}, mods, fps=24.0, frame_count=100)
    for t in range(100):
        v = fn(t)["slice_width"]
        assert isinstance(v, int)
        assert v >= 1, f"slice_width must be >= 1, got {v} at t={t}"


def test_fill_alpha_clamped():
    """fill_alpha must stay in [0, 1] even with large depth."""
    mods = [ModEntry(dest="fill_alpha", osc="triangle", rate_str="1cyc", depth=5.0, phase=0.0, offset=0.0)]
    fn = make_resolved_params_fn({"fill_alpha": 0.5}, mods, fps=24.0, frame_count=100)
    for t in range(100):
        v = fn(t)["fill_alpha"]
        assert 0.0 <= v <= 1.0, f"fill_alpha out of range at t={t}: {v}"


def test_no_mods_returns_base_params():
    """With no mods, resolved params should equal base params."""
    fn = make_resolved_params_fn({"vanguard": 0.3, "max_delay": 42}, [], fps=24.0, frame_count=100)
    p = fn(0)
    assert p["vanguard"] == pytest.approx(0.3)
    assert p["max_delay"] == 42
