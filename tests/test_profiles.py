"""Tests for slitscan profile implementations."""

from __future__ import annotations

import numpy as np
import pytest

from slitscan.profiles.base import get_profile


N = 100  # number of bands used in tests


def make_x(n: int = N) -> np.ndarray:
    """Raw band indices as float64, as the render engine produces them."""
    return np.arange(n, dtype=np.float64)


def make_params(max_delay: int = 50, **kwargs) -> dict:
    base = {"max_delay": max_delay, "slice_width": 1}
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# Ramp
# ---------------------------------------------------------------------------

class TestRamp:
    def test_ramp_endpoints(self):
        """With vanguard=0, delay[0]=0 and delay[-1]=max_delay."""
        profile = get_profile("ramp")
        x = make_x()
        params = make_params(max_delay=50, vanguard=0.0)
        delays = profile.delay_map(x, 0, params)

        assert delays[0] == pytest.approx(0.0), "Left edge should have delay=0"
        assert delays[-1] == pytest.approx(50.0), "Right edge should have delay=max_delay"

    def test_ramp_monotone_increasing(self):
        """Ramp with vanguard=0 should be monotonically non-decreasing."""
        profile = get_profile("ramp")
        x = make_x()
        delays = profile.delay_map(x, 0, make_params(max_delay=100, vanguard=0.0))
        assert np.all(np.diff(delays) >= 0)

    def test_ramp_vanguard_right(self):
        """With vanguard=1.0, delay[0]=max_delay and delay[-1]=0."""
        profile = get_profile("ramp")
        x = make_x()
        delays = profile.delay_map(x, 0, make_params(max_delay=30, vanguard=1.0))
        assert delays[0] == pytest.approx(30.0)
        assert delays[-1] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Reverse
# ---------------------------------------------------------------------------

class TestReverse:
    def test_reverse_endpoints(self):
        """With vanguard=1.0, delay[-1]=0 and delay[0]=max_delay."""
        profile = get_profile("reverse")
        x = make_x()
        params = make_params(max_delay=50, vanguard=1.0)
        delays = profile.delay_map(x, 0, params)

        assert delays[-1] == pytest.approx(0.0), "Right edge should have delay=0"
        assert delays[0] == pytest.approx(50.0), "Left edge should have delay=max_delay"

    def test_reverse_monotone_decreasing(self):
        """Reverse with default vanguard (right) should be monotonically non-increasing."""
        profile = get_profile("reverse")
        x = make_x()
        delays = profile.delay_map(x, 0, make_params(max_delay=80))
        assert np.all(np.diff(delays) <= 0)


# ---------------------------------------------------------------------------
# Tent
# ---------------------------------------------------------------------------

class TestTent:
    def test_tent_center_symmetry(self):
        """With vanguard=0.5, delay is symmetric; delay[0] ≈ delay[-1] ≈ max_delay."""
        profile = get_profile("tent")
        # Use even number so edges are equidistant from center
        x = make_x(101)
        params = make_params(max_delay=40, vanguard=0.5)
        delays = profile.delay_map(x, 0, params)

        assert delays[0] == pytest.approx(delays[-1], abs=1e-6), (
            "Tent should be symmetric: delay[0] ≈ delay[-1]"
        )
        assert delays[0] == pytest.approx(40.0, abs=1e-6), (
            "Tent edge delay should equal max_delay"
        )

    def test_tent_vanguard_zero_delay(self):
        """At the vanguard band, delay should be 0 (or near 0)."""
        profile = get_profile("tent")
        n = 101
        x = make_x(n)
        vanguard = 0.5
        params = make_params(max_delay=60, vanguard=vanguard)
        delays = profile.delay_map(x, 0, params)

        # Find band closest to vanguard position
        vanguard_band = int(round(vanguard * (n - 1)))
        assert delays[vanguard_band] == pytest.approx(0.0, abs=1e-6), (
            f"Delay at vanguard band {vanguard_band} should be 0"
        )

    def test_tent_non_negative(self):
        """All tent delays should be non-negative."""
        profile = get_profile("tent")
        x = make_x()
        delays = profile.delay_map(x, 0, make_params(max_delay=50, vanguard=0.5))
        assert np.all(delays >= 0.0)

    def test_tent_max_delay_not_exceeded(self):
        """Tent delays should never exceed max_delay."""
        profile = get_profile("tent")
        x = make_x()
        max_delay = 75
        delays = profile.delay_map(x, 0, make_params(max_delay=max_delay, vanguard=0.5))
        assert np.all(delays <= max_delay + 1e-9)

    def test_tent_offcenter_long_arm_reaches_max_delay(self):
        """With off-center vanguard, only the long arm reaches max_delay."""
        profile = get_profile("tent")
        n = 21
        x = np.arange(n, dtype=float)
        p = {"max_delay": 100, "vanguard": 0.2}  # vanguard near left
        delays = profile.delay_map(x, 0, p)
        # Long arm (right side) should reach max_delay
        assert delays[-1] == pytest.approx(100.0, abs=1e-6)
        # Short arm (left edge) should NOT reach max_delay
        assert delays[0] < 100.0
        # Delay at vanguard position should be 0
        vanguard_band = int(round(0.2 * (n - 1)))
        assert delays[vanguard_band] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Ramp / Reverse mirror relationship
# ---------------------------------------------------------------------------

class TestRampReverseMirror:
    def test_ramp_reverse_mirror(self):
        """Ramp with vanguard=0 reversed should match reverse with vanguard=1.0."""
        ramp = get_profile("ramp")
        reverse = get_profile("reverse")
        x = make_x()
        max_delay = 60
        params_ramp = make_params(max_delay=max_delay, vanguard=0.0)
        params_rev = make_params(max_delay=max_delay, vanguard=1.0)

        ramp_delays = ramp.delay_map(x, 0, params_ramp)
        rev_delays = reverse.delay_map(x, 0, params_rev)

        np.testing.assert_allclose(
            ramp_delays,
            rev_delays[::-1],
            atol=1e-9,
            err_msg="Reversed ramp should equal mirror of reverse profile",
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestProfileRegistry:
    def test_get_ramp(self):
        p = get_profile("ramp")
        assert hasattr(p, "delay_map"), "ramp profile must have delay_map method"

    def test_get_reverse(self):
        p = get_profile("reverse")
        assert hasattr(p, "delay_map"), "reverse profile must have delay_map method"

    def test_get_tent(self):
        p = get_profile("tent")
        assert hasattr(p, "delay_map"), "tent profile must have delay_map method"

    def test_get_nonexistent_raises(self):
        with pytest.raises(ValueError, match="nonexistent"):
            get_profile("nonexistent")

    def test_delay_map_returns_ndarray(self):
        """All profiles should return a numpy array of correct shape."""
        for name in ("ramp", "reverse", "tent"):
            p = get_profile(name)
            x = make_x(20)
            delays = p.delay_map(x, 0, make_params(max_delay=10, vanguard=0.5))
            assert isinstance(delays, np.ndarray), f"{name}: delay_map must return ndarray"
            assert delays.shape == (20,), f"{name}: wrong shape {delays.shape}"
