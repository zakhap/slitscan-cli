"""Unit tests for fill policies (engine/fill.py)."""

from __future__ import annotations

import numpy as np
import pytest

from slitscan.engine.fill import make_fill_color, _resolve_src_index


class TestMakeFillColor:
    def test_black_fill_is_zeros(self):
        color = make_fill_color("black", 3)
        assert color.shape == (3,)
        assert color.dtype == np.uint8
        assert np.all(color == 0)

    def test_white_fill_is_255(self):
        color = make_fill_color("white", 3)
        assert color.shape == (3,)
        assert color.dtype == np.uint8
        assert np.all(color == 255)

    def test_transparent_fill_alpha_zero(self):
        color = make_fill_color("transparent", 4)
        assert color.shape == (4,)
        assert color.dtype == np.uint8
        assert np.all(color == 0)

    def test_unknown_fill_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown fill mode"):
            make_fill_color("bogus", 3)


class TestResolveSrcIndex:
    def test_resolve_hold_clamps_low(self):
        assert _resolve_src_index(-5, "hold", 100) == 0

    def test_resolve_hold_clamps_high(self):
        assert _resolve_src_index(105, "hold", 100) == 99

    def test_resolve_hold_valid_index_unchanged(self):
        assert _resolve_src_index(50, "hold", 100) == 50

    def test_resolve_wrap(self):
        assert _resolve_src_index(105, "wrap", 100) == 5

    def test_resolve_wrap_zero(self):
        assert _resolve_src_index(0, "wrap", 100) == 0

    def test_resolve_wrap_exact_boundary(self):
        assert _resolve_src_index(100, "wrap", 100) == 0

    def test_resolve_black_passes_through(self):
        # Negative index passes through for black so buffer.get returns None
        assert _resolve_src_index(-3, "black", 100) == -3

    def test_resolve_white_passes_through(self):
        assert _resolve_src_index(-1, "white", 100) == -1

    def test_resolve_transparent_passes_through(self):
        assert _resolve_src_index(200, "transparent", 100) == 200

    def test_resolve_rounds_float(self):
        # 4.6 → 5 for hold within range
        assert _resolve_src_index(4.6, "hold", 100) == 5
        # -0.6 → -1 → clamped to 0
        assert _resolve_src_index(-0.6, "hold", 100) == 0
