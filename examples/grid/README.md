# Grid slit-scan — worked examples

Generated from `examples/source.mp4` (480×360, 96 frames). The source is built
so the effect reads at a glance: **whole-frame hue encodes time** (so each tile
becomes a solid color = the moment it froze), a bright disc gives spatial
motion, and the frame number is burned in large.

Regenerate everything:

```sh
python examples/make_source_clip.py          # writes examples/source.mp4
# then the render commands below
```

See `_contact_sheet_frame60.png` for a side-by-side of one frame from each.

| File | Command (args after `render source.mp4 OUT`) | What it shows |
|------|----------------------------------------------|---------------|
| `01_baseline_1d_ramp.mp4` | `--profile ramp --axis x --fill hold` | Classic 1D vertical-band slit-scan (reference) |
| `02_grid_2x2_ramp.mp4` | `--grid 2x2 --profile ramp --fill wrap` | Four quadrants, four times |
| `03_grid_4x4_ramp.mp4` | `--grid 4x4 --profile ramp --fill wrap` | 16-tile mosaic |
| `04_grid_8x8_ramp.mp4` | `--grid 8x8 --profile ramp --fill wrap` | Diagonal time field (ramp avg) |
| `05_grid_16x16_ramp.mp4` | `--grid 16x16 --profile ramp --fill wrap` | Finer diagonal gradient |
| `06_grid_32x24_ramp.mp4` | `--grid 32x24 --profile ramp --fill wrap` | Near-continuous gradient |
| `07_grid_8x8_tent_radial.mp4` | `--grid 8x8 --profile tent --grid-combine max --fill wrap` | Radial time rings (center = now) |
| `08_grid_12x12_lfo_stripes.mp4` | `--grid 12x12 --grid-mod "col=sine:rate=2,depth=1" --fill wrap` | Vertical stripes of oscillating time |
| `09_grid_12x12_lfo_plaid.mp4` | `--grid 12x12 --grid-mod "col=sine:rate=2,depth=0.5" --grid-mod "row=sine:rate=2,depth=0.5" --fill wrap` | 2D plaid interference field |
| `10_grid_16x12_lfo_tri.mp4` | `--grid 16x12 --grid-mod "col=triangle:rate=3,depth=0.6" --grid-mod "row=triangle:rate=2,depth=0.4" --fill wrap` | Triangle-osc plaid texture |

The grid-size sweep (02→06) is the "various sizes" comparison: same delay field,
increasing tile resolution.
