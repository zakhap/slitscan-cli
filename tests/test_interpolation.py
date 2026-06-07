import numpy as np
import pytest
from slitscan.engine.gather import gather_frame
from slitscan.buffer.full import FullBuffer
from slitscan.meta import ClipMeta

# Helper: frame N is solid value N
def make_buffer(n_frames, h=8, w=8, c=3):
    meta = ClipMeta(fps=24.0, frame_count=n_frames, width=w, height=h, channels=c)
    frames = [np.full((h, w, c), i, dtype=np.uint8) for i in range(n_frames)]
    buf = FullBuffer(meta, iter(frames))
    return buf, meta

def test_interpolation_midpoint_blends_50_50():
    """With src=1.5, output should be 50% frame 1 + 50% frame 2 = 1.5 → rounds to 1 or 2."""
    buf, meta = make_buffer(10)
    H, W, C = meta.height, meta.width, meta.channels
    band_starts = np.array([0])
    band_widths = np.array([W])
    src_indices = np.array([1.5])
    out = gather_frame(
        buffer=buf, output_t=0, src_indices=src_indices,
        band_starts=band_starts, band_widths=band_widths,
        axis="x", output_shape=(H, W, C),
        fill="black", fill_color=np.zeros(C, dtype=np.uint8),
        frame_count=meta.frame_count, interpolate=True,
    )
    # Expected: 0.5*1 + 0.5*2 = 1.5, clipped to uint8 → 1 or 2
    center_val = out[H//2, W//2, 0]
    assert center_val in (1, 2), f"Expected 1 or 2, got {center_val}"

def test_no_interpolation_rounds_to_nearest():
    """Without interpolation, src=1.5 rounds to 2."""
    buf, meta = make_buffer(10)
    H, W, C = meta.height, meta.width, meta.channels
    out = gather_frame(
        buffer=buf, output_t=0, src_indices=np.array([1.5]),
        band_starts=np.array([0]), band_widths=np.array([W]),
        axis="x", output_shape=(H, W, C),
        fill="black", fill_color=np.zeros(C, dtype=np.uint8),
        frame_count=meta.frame_count, interpolate=False,
    )
    center_val = out[H//2, W//2, 0]
    assert center_val == 2, f"Expected 2 (round 1.5), got {center_val}"

def test_interpolation_at_integer_index_same_as_no_interpolation():
    """At integer src index, interpolation produces same result as no interpolation."""
    buf, meta = make_buffer(10)
    H, W, C = meta.height, meta.width, meta.channels
    kwargs = dict(
        buffer=buf, output_t=0, src_indices=np.array([3.0]),
        band_starts=np.array([0]), band_widths=np.array([W]),
        axis="x", output_shape=(H, W, C),
        fill="black", fill_color=np.zeros(C, dtype=np.uint8),
        frame_count=meta.frame_count,
    )
    out_no_interp = gather_frame(**kwargs, interpolate=False)
    out_interp = gather_frame(**kwargs, interpolate=True)
    np.testing.assert_array_equal(out_no_interp, out_interp)

def test_interpolation_fill_zone_blend():
    """Blending at the boundary: one frame is in-range, the other is None (fill zone)."""
    buf, meta = make_buffer(5)
    H, W, C = meta.height, meta.width, meta.channels
    # src=-0.5: floor=-1 (fill=black=0), ceil=0 (frame 0 = solid 0), alpha=0.5
    out = gather_frame(
        buffer=buf, output_t=0, src_indices=np.array([-0.5]),
        band_starts=np.array([0]), band_widths=np.array([W]),
        axis="x", output_shape=(H, W, C),
        fill="black", fill_color=np.zeros(C, dtype=np.uint8),
        frame_count=meta.frame_count, interpolate=True,
    )
    # 0.5*black(0) + 0.5*frame_0(0) = 0
    assert out[0, 0, 0] == 0
