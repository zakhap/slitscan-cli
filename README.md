# slitscan

Time-displacement slit-scan rendering for video.

Treats a video clip as an X-Y-T spacetime volume. Each output column is sourced from a different moment in time according to a configurable delay surface.

## Installation

Requires Python 3.11+ and libav.

```sh
pip install -e .
```

## Quick Start

```sh
# Sweep mode — video to video
slitscan render input.mp4 output.mp4

# Seamless loop with wrap fill
slitscan render input.mp4 output.mp4 --fill wrap

# Photofinish — video to image
slitscan collapse input.mp4 photofinish.png --slit-position 0.5
```

## render

Assemble each output frame from bands drawn from different source frames.

```sh
slitscan render INPUT OUTPUT [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--profile` | `ramp` | Delay surface: `ramp`, `tent`, `reverse` |
| `--axis` | `x` | Scan axis: `x` (columns) or `y` (rows) |
| `--max-delay N` | width−1 | Frames between vanguard and lagging edge |
| `--slice-width N` | `1` | Pixels per temporal band |
| `--fill` | `black` | Out-of-range fill: `black`, `white`, `transparent`, `hold`, `wrap` |
| `--vanguard 0–1` | profile default | Position of the zero-delay edge |
| `--interpolate` | off | Sub-frame linear interpolation |
| `--slit-source 0–1` | — | Trumbull fixed-slit: all bands gathered from one position |
| `--mod PATCH` | — | LFO modulation string, repeatable |
| `--mod-file PATH` | — | Load modulation patch from YAML |
| `--resize WxH` | source dims | Output dimensions |
| `--fps N` | source fps | Output frame rate |
| `--buffer auto\|full\|ring` | `auto` | Buffer policy |
| `--memory-budget 8G` | `8G` | RAM cap for frame buffer |

Output format is selected by file extension:

| Extension | Codec |
|-----------|-------|
| `.mp4` | H.264 |
| `.mov` | ProRes (ProRes 4444 with `--fill transparent`) |
| `.webm` | VP9 |
| `.gif` | Animated GIF, 256-color palette, Floyd-Steinberg dither |
| `.png` / `.tiff` | Image sequence |

## collapse

Accumulate a single slit's history into one image. Time runs horizontally.

```sh
slitscan collapse INPUT OUTPUT [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--slit-position 0–1` | `0.5` | Normalized slit position |
| `--axis` | `x` | Slit orientation |
| `--direction` | `forward` | `forward` or `reverse` |
| `--slice-width N` | `1` | Pixels per temporal slice |

## info

Print clip metadata.

```sh
slitscan info input.mp4
slitscan info input.mp4 --json
```

## Profiles

**ramp** — delay increases linearly from the vanguard edge to the opposite side. The classic temporal rake.

**tent** — delay peaks at center and falls to zero at both edges. Creates a bilateral temporal fold.

**reverse** — mirror of ramp; the opposite edge leads in time.

## Modulation

Drive any render parameter with an LFO oscillator.

```sh
# Oscillate vanguard position at 0.1 Hz, ±0.4
slitscan render input.mp4 out.mp4 --mod "vanguard=sine:rate=0.1hz,depth=0.4"

# Breathe the temporal spread at 0.25 Hz
slitscan render input.mp4 out.mp4 --mod "max_delay=sine:rate=0.25hz,depth=400"

# Stack multiple modulators
slitscan render input.mp4 out.mp4 \
  --profile tent \
  --mod "vanguard=sine:rate=0.1hz,depth=0.4" \
  --mod "max_delay=triangle:rate=0.05hz,depth=300"
```

Destinations: `vanguard`, `max_delay`, `slice_width`. Oscillators: `sine`, `triangle`. Rate units: `hz`, `cyc` (cycles per clip), `frames` (period in frames).

Load a patch from YAML:

```yaml
# patch.yaml
base:
  vanguard: 0.0
  max_delay: 600
mods:
  - dest: vanguard
    osc: sine
    rate: "0.1hz"
    depth: 0.4
```

```sh
slitscan render input.mp4 out.mp4 --mod-file patch.yaml
```

## Tests

```sh
pytest tests/
```
