# Slitscan — Technical Specification & Documentation

**Status:** v1 design spec / living documentation
**Last updated:** 2026-06-06

A command-line toolkit for time-displacement slit-scan rendering of video. The tool treats a clip as an X-Y-T volume and renders surfaces through it, where each spatial slice of the output is sourced from a different moment in time according to a configurable, modulatable delay surface.

---

## Table of contents

1. [Concept & vocabulary](#1-concept--vocabulary)
2. [The core model](#2-the-core-model)
3. [Architecture](#3-architecture)
4. [The frame buffer](#4-the-frame-buffer)
5. [Delay profiles](#5-delay-profiles)
6. [Modulation (LFO system)](#6-modulation-lfo-system)
7. [Rate units](#7-rate-units)
8. [The normalization front-end](#8-the-normalization-front-end)
9. [Fill / boundary behavior](#9-fill--boundary-behavior)
10. [Interpolation](#10-interpolation)
11. [CLI reference](#11-cli-reference)
12. [Patch file format](#12-patch-file-format)
13. [Rendering pipeline (end to end)](#13-rendering-pipeline-end-to-end)
14. [Performance & memory](#14-performance--memory)
15. [Codec & format matrix](#15-codec--format-matrix)
16. [Technical background & references](#16-technical-background--references)
17. [v1 scope, deferred features, roadmap](#17-v1-scope-deferred-features-roadmap)
18. [Glossary](#18-glossary)

---

## 1. Concept & vocabulary

A normal video is a stack of frames played in lockstep: at output time *t*, every pixel comes from input frame *t*. **Slit-scan** breaks that lockstep. Different spatial regions of the output frame are pulled from *different* input frames.

The canonical example: in a width-W video, the leftmost column shows input that is W−1 frames *ahead* of the rightmost column. Reading across a single output frame from left to right is therefore reading *backward through time*. As the clip plays, this temporal rake sweeps through the footage.

Key terms used throughout this document:

- **Slice / band** — a vertical (or horizontal) strip of the frame, one or more pixels wide, that shares a single time offset.
- **Vanguard** — the slice that is furthest *ahead* in time (offset 0, the "leading edge" of the rake). In the classic ramp, the vanguard is the leftmost column.
- **Delay / offset** — for a given slice, how many frames *behind* the vanguard it is sourced from. Always ≥ 0.
- **Delay surface** — the full function mapping every column (and the current output time) to a delay. Conceptually a surface cutting through the X-Y-T volume.
- **Sweep mode** — output is a video; the delay surface is held (or modulated) while input plays past it. Output length ≈ input length.
- **Collapse mode** — output is a single image; one moving slit's history is accumulated across the whole clip (photofinish / strip photography).
- **Modulation / LFO** — any tool parameter driven by an oscillator that is a function of output time.

---

## 2. The core model

Everything in the tool reduces to one function, evaluated once per output frame:

```
delay_map(x_coords, output_t, params) -> ndarray[int|float]  # one delay per column
```

- `x_coords` — array of band indices `[0, 1, ... n_bands-1]`.
- `output_t` — the current output frame index.
- `params` — the resolved parameter set for this frame, *after* all modulation has been applied (see §6). Static renders are simply the case where no parameter changes between frames.

The returned array gives, per band, the number of frames to look *back* from a reference. The engine then gathers each output column from input frame:

```
source_frame(x, output_t) = output_t + (max_delay - delay_map[x])
```

so that the vanguard (delay 0) reads the most recent frame and lagging bands read older ones. (Sign conventions are internal; the user thinks only in "vanguard position" and "how far back the spread reaches.")

Because the signature takes `output_t` and time-varying `params`, **static surfaces, animated surfaces, and LFO-modulated surfaces are all the same code path.** This is the single most important design invariant; everything else is built to preserve it.

---

## 3. Architecture

```
┌─────────────┐   ┌──────────────┐   ┌───────────────┐   ┌──────────┐   ┌─────────┐
│  Decode     │──▶│  Normalize   │──▶│  Frame Buffer │──▶│  Engine  │──▶│ Encode  │
│  (PyAV)     │   │ resize/fit   │   │  (RAM | ring) │   │  gather  │   │ (PyAV)  │
└─────────────┘   └──────────────┘   └───────────────┘   └────▲─────┘   └─────────┘
                                                              │
                                          ┌───────────────────┴───────────────────┐
                                          │  delay_map(x, t, params)               │
                                          │    ├─ profile (ramp/reverse/tent)      │
                                          │    └─ params resolved by modulation    │
                                          │         └─ oscillators (sine/triangle) │
                                          └────────────────────────────────────────┘
```

**Layers, each independently testable:**

- **I/O layer** — PyAV decode/encode, container/codec selection, alpha handling.
- **Normalization** — resize, crop/letterbox to known dimensions (§8).
- **Buffer** — abstract interface with two backings: full in-RAM and ring buffer (§4).
- **Profile layer** — pure functions producing a base delay surface (§5).
- **Modulation layer** — resolves time-varying params before each frame's `delay_map` call (§6).
- **Engine** — the per-frame vectorized gather; backing-agnostic and profile-agnostic.

**Stack:** Python 3.11+, PyAV (libav bindings, decode/encode incl. alpha codecs), NumPy (vectorized gather), Typer (CLI), PyYAML (patch files). All Apple-silicon (M1) native.

---

## 4. The frame buffer

The buffer is the heart of the memory story. The engine never talks to PyAV directly for reads — it asks the buffer for "input frame *n*," and the buffer decides whether that frame is in RAM, must be decoded, or no longer exists.

**Interface (conceptual):**

```python
class FrameBuffer:
    def get(self, frame_index: int) -> np.ndarray | None: ...   # None => out of range
    def advance(self) -> None: ...                              # producer pulls next decoded frame
    @property
    def available_range(self) -> tuple[int, int]: ...           # [oldest, newest] resident
```

**Two backings:**

- **`full` (default)** — decode the entire clip into RAM. Simplest, supports arbitrary random access (needed for `collapse` and for backward-looking surfaces). Chosen automatically when `max_delay × frame_size ≤ memory_budget` and the clip fits.
- **`ring`** — a circular buffer holding exactly `max_delay + 1` frames. Memory is constant regardless of clip length; only the window `[output_t, output_t + max_delay]` is resident. Used when the full clip would exceed `--memory-budget`. Constraint: the delay surface may only reference frames within the resident window — enforced at validation time.

Selection is automatic from clip metadata and `--memory-budget`, overridable with `--buffer {full,ring,auto}`.

**Frame size math** (for budgeting): `W × H × channels × dtype_bytes`. Example: 1920×1080×3×1 = 6.22 MB/frame; a 1080-deep buffer ≈ 6.7 GB. RGBA (transparent fill) is 4 channels → ~8.3 GB. The tool prints the projected buffer size at startup and refuses (with a clear message) if it would exceed the budget without `ring`.

---

## 5. Delay profiles

A profile is a pure function `(x_coords, output_t, params) -> base_delay_per_band`. v1 ships three. All are parameterized by `vanguard` (normalized 0–1 position along the axis) and `max_delay` (frames), both of which may be modulated.

- **`ramp`** — linear. Delay 0 at the vanguard edge, rising linearly to `max_delay` at the far edge. The classic. With `vanguard=0.0`, leftmost is the lead; `vanguard=1.0` is equivalent to `reverse`.
- **`reverse`** — `ramp` mirrored; vanguard at the right (or bottom).
- **`tent` / `vee`** — vanguard at `vanguard` position (default center, 0.5); delay flares *outward* symmetrically toward both edges. Note the halving property: with the vanguard centered, each edge is reached over half the axis, so for a given slope the edge delay is half that of an edge-vanguard `ramp` of the same slope — or, to reach the same `max_delay` at the edges, the slope doubles. The tool parameterizes by edge `max_delay` so the user controls the endpoint directly.

**Axis:** profiles operate along X by default; `--axis y` runs the same math along rows (the rolling-shutter / scanline family).

**Band sampling:** with `--slice-width w`, the axis is divided into `ceil(extent / w)` bands; the profile is sampled once per band and the delay is constant across each band's pixels.

Deferred profiles (v2): `radial`, `sine`, arbitrary `--profile-image` heightmap, and `--expr` custom expressions. The signature already accommodates them.

---

## 6. Modulation (LFO system)

Modulation patches an **oscillator source** onto a **parameter destination**, in direct analogy to LFO routing on a synthesizer. Before each output frame, the modulation layer evaluates every active oscillator at `output_t`, applies it to its destination's base value, and produces the resolved `params` passed to `delay_map`.

**Destinations (v1):** `vanguard`, `max_delay`, `slice_width`, `fill_alpha`. (`profile_blend` deferred to v2.)

**Oscillator sources (v1):** `sine`, `triangle`. (Deferred v2: `saw`, `square`, `sample-hold`, `noise`, `envelope`.)

**Oscillator parameters:**

- `rate` — frequency, with a unit suffix (§7): `hz`, `cyc`, or `frames`.
- `depth` — modulation amount, in the destination's native units (pixels for `slice_width`, frames for `max_delay`, normalized 0–1 for `vanguard`, 0–1 for `fill_alpha`).
- `phase` — 0–1, fraction of a cycle offset at `output_t = 0`.
- `offset` — added to the destination's base value (DC offset of the modulation).

**Application:** `resolved = base + offset + depth × osc(2π · (cycles_per_frame · output_t + phase))`, then clamped to the destination's valid range. Multiple mods on one destination sum.

**Patch syntax (CLI):**

```
--mod DEST=OSC:rate=…,depth=…,phase=…,offset=…
```

Example:

```
slitscan render in.mp4 out.mov \
  --profile tent \
  --mod vanguard=sine:rate=0.5cyc,depth=0.5,phase=0 \
  --mod max_delay=triangle:rate=2cyc,depth=200,offset=400
```

Complex setups go in a `--mod-file patch.yaml` (§12) instead of long command lines.

---

## 7. Rate units

All three units collapse internally to **cycles-per-frame**; the unit is purely a parsing lens. This gives a low-level (frame/absolute-time) pole and a high-level (whole-content) pole, both first-class.

| Suffix     | Meaning                       | Conversion to cycles/frame      | Needs        | Pole        |
|------------|-------------------------------|----------------------------------|--------------|-------------|
| `hz`       | cycles per real second        | `rate / fps`                     | fps          | low-level   |
| `cyc`      | cycles per whole clip         | `rate / total_frames`            | frame count  | high-level  |
| `frames`   | one cycle per N frames        | `1 / rate`                       | none         | low-level   |

**Default:** `cyc` — resolution-independent, survives a re-render at a different fps.

**Resolution timing:** `hz` and `cyc` depend on metadata known only *after* decode (fps, frame count). For `render` this is automatic. For the v2 `mods` preview command, supply `--assume-fps` / `--assume-frames` to plot without a real clip.

---

## 8. The normalization front-end

Rather than a runtime constraint system, the tool normalizes input to known dimensions in a single front-end step, after which the engine works in fixed, predictable dimensions.

```
slitscan render input.mp4 out.mov --resize 1080x720 --fit crop
```

- `--resize WxH` — target dimensions. If omitted, native dimensions are used.
- `--fit {crop, letterbox, stretch}` — how to reconcile aspect ratio. `crop` (center-crop, default) preserves scale and fills the frame; `letterbox` preserves the whole image with bars; `stretch` ignores aspect ratio.

Normalization runs once, before buffering. The slit-scan engine never sees ragged input.

---

## 9. Fill / boundary behavior

For the first `max_delay` output frames, lagging bands reference input frames that don't exist yet (negative or pre-roll indices). `--fill` controls this:

- **`black`** (default) — missing bands are black.
- **`white`** — missing bands are white.
- **`transparent`** — missing bands have alpha 0. **Forces RGBA output and an alpha-capable codec** (ProRes 4444 or a PNG/TIFF sequence); H.264/H.265 cannot carry alpha and the tool will error with a suggested alternative.
- **`hold`** — clamp to the nearest existing frame (freeze the edge).
- **`wrap`** — wrap the index modulo clip length (cyclic).

`fill_alpha` is also a modulation destination, allowing the fill region's opacity to be animated when in `transparent` mode.

---

## 10. Interpolation

When `vanguard` or `max_delay` is modulated quickly relative to slice width, the source-frame index for a band can jump by more than one frame per output frame, producing temporal stutter or banding.

`--interpolate` (off by default) blends between the two buffered frames straddling a fractional source index, smoothing motion at the cost of one extra blend per band. Off is faithful/crisp; on is smooth/cinematic. Fractional delays (from continuous modulation) only have visible effect when interpolation is on; otherwise they are rounded to the nearest frame.

---

## 11. CLI reference

```
slitscan <command> [options]
```

### `render` — video → video (sweep mode)

| Option | Type | Default | Description |
|---|---|---|---|
| `INPUT` | path | — | Source video (positional). |
| `OUTPUT` | path | — | Output path; extension selects container/codec (positional). |
| `--profile` | `ramp\|reverse\|tent` | `ramp` | Base delay surface. |
| `--axis` | `x\|y` | `x` | Slice orientation. |
| `--vanguard` | float 0–1 | profile-dependent | Vanguard position along the axis. |
| `--max-delay` | int (frames) | `extent − 1` | Deepest delay in the spread. |
| `--slice-width` | int (px) | `1` | Pixels per band. |
| `--fill` | `black\|white\|transparent\|hold\|wrap` | `black` | Out-of-range behavior. |
| `--interpolate` | flag | off | Sub-frame blending. |
| `--resize` | `WxH` | native | Normalize dimensions. |
| `--fit` | `crop\|letterbox\|stretch` | `crop` | Aspect reconciliation. |
| `--buffer` | `auto\|full\|ring` | `auto` | Buffer backing. |
| `--memory-budget` | size (e.g. `8G`) | `auto` | Cap before switching to ring. |
| `--mod` | patch string | — | Repeatable modulation routing (§6). |
| `--mod-file` | path | — | YAML patch file (§12). |
| `--fps` | float | source | Override output fps. |

### `collapse` — video → image (photofinish)

Accumulates one moving slit's history across the clip into a single image. Shares `--axis`, `--slice-width`, `--resize`, `--fit`. Additional:

| Option | Type | Default | Description |
|---|---|---|---|
| `--slit-position` | float 0–1 | `0.5` | Where the reading slit sits in the frame. |
| `--direction` | `forward\|reverse` | `forward` | Time direction of accumulation. |
| `OUTPUT` | path | — | Image path; extension (`.png`/`.tiff`) selects format. |

### `profiles` *(v2)* — preview a delay surface as a heightmap PNG.
### `mods` *(v2)* — plot an LFO/patch; supports `--assume-fps` / `--assume-frames`.

---

## 12. Patch file format

For complex modulation, a YAML file is equivalent to a set of `--mod` flags and is easier to read and version-control:

```yaml
# patch.yaml
profile: tent
vanguard: 0.5
max_delay: 600
slice_width: 2
fill: black

mods:
  - dest: vanguard
    osc: sine
    rate: 0.5cyc
    depth: 0.5
    phase: 0.0
    offset: 0.0
  - dest: max_delay
    osc: triangle
    rate: 2cyc
    depth: 200
    offset: 400
```

`--mod-file patch.yaml`. Command-line `--mod` flags, if also present, append to (and override by destination order) the file's mods.

---

## 13. Rendering pipeline (end to end)

For `render`, per the architecture in §3:

1. **Decode metadata** — open input, read fps, frame count, dimensions.
2. **Resolve modulation timing** — convert all oscillator rates to cycles-per-frame using fps / frame count (§7).
3. **Plan buffer** — compute projected buffer size; pick `full` or `ring`; validate that the surface's reach fits a `ring` window if used (§4); print the plan.
4. **Validate fill/codec** — if `transparent`, confirm the output codec carries alpha or error early (§9, §15).
5. **Per output frame `t`:**
   a. Evaluate oscillators at `t` → resolved `params`.
   b. `delay_map(x_coords, t, params)` → per-band delay.
   c. Compute per-band source-frame indices.
   d. Gather columns from the buffer (one vectorized fancy-index; blend if `--interpolate`).
   e. Apply fill to out-of-range bands.
   f. Encode the assembled frame.
   g. Advance buffer (ring) / proceed (full).
6. **Finalize** — flush encoder, close container.

`collapse` is similar but accumulates into a single output image: each output column is the slit's content at a successive input time, written once.

---

## 14. Performance & memory

- **Inner loop is one NumPy gather per frame.** With per-band source indices `src[x]`, the assembled frame is `buffer_stack[src, :, x_range]` — fully vectorized, no Python per-column loop.
- **Memory** is dominated by the buffer (§4). Use `ring` for long clips; print-and-refuse guards against accidental multi-GB allocations.
- **M1 notes:** unified memory makes `full` viable up to a large fraction of system RAM, but leave headroom for the encoder. 1080-deep RGB ≈ 6.7 GB suits 16 GB machines; drop to `ring` or lower `--memory-budget` on 8 GB.
- **Interpolation** roughly doubles gather cost (two reads + blend per band); leave off unless motion artifacts appear.
- **Slice width > 1** reduces band count and speeds evaluation proportionally.

---

## 15. Codec & format matrix

| Output ext | Container | Typical codec | Alpha? | Notes |
|---|---|---|---|---|
| `.mp4` | MP4 | H.264 / H.265 | No | Smallest; cannot use `--fill transparent`. |
| `.mov` | QuickTime | ProRes 422 | No | High quality intermediate. |
| `.mov` | QuickTime | **ProRes 4444** | **Yes** | Use for `--fill transparent`. |
| `.png` (seq) | image sequence | PNG | Yes | Lossless frames; alpha-capable. |
| `.tiff` (seq) | image sequence | TIFF | Yes | Lossless; `collapse` default-friendly. |

If `--fill transparent` is set with a non-alpha codec, the tool errors and suggests ProRes 4444 or a PNG sequence.

---

## 16. Technical background & references

The tool sits in the **time-displacement** family of slit-scan: spatial position maps to a temporal offset. The unifying mental model is the **X-Y-T spacetime cube** — the video as a volume with two spatial axes and one time axis. Any surface cut through that cube is a slit-scan variant; the delay surface `t = f(x, output_t)` *is* that cutting surface. This framing subsumes the whole catalog:

- **Time-slice / time displacement** — the present tool's `render`. Position → time offset.
- **Photofinish / strip photography** — a single moving slit accumulated over time (`collapse`). Finish-line cameras and rotating panoramic cameras.
- **Optical slit-scan (Stargate)** — Trumbull's *2001* effect; imagery streaked through a physical slit on an animation stand.
- **Scanline / rolling-shutter** — per-row temporal offset; the `--axis y` case.

The canonical reference is **Golan Levin's "An Informal Catalogue of Slit-Scan Video Artworks and Research"**, which surveys the technique's history and practitioners. Recommended reading alongside this spec for the artistic lineage. (Worth pulling the current version directly, as the catalogue is periodically updated.)

The modulation system borrows its conceptual model from **subtractive-synthesis LFO routing**: parameters are destinations, oscillators are sources, and a patch connects them — making time-varying surfaces a natural extension rather than a special case.

---

## 17. v1 scope, deferred features, roadmap

**v1 (this spec, build target):**

- Engine with `delay_map(x, t, params)` invariant and buffer interface (full + ring).
- Normalization front-end (`--resize`, `--fit`).
- `render`: profiles `ramp`, `reverse`, `tent`; `--slice-width`, `--max-delay`, `--axis`, `--fill`, `--interpolate`.
- `collapse` (photofinish).
- Modulation: `--mod` + `--mod-file`; oscillators `sine`, `triangle`; units `hz`/`cyc`/`frames`.
- Stack: Python, PyAV, NumPy, Typer, PyYAML.

**Deferred to v2+:**

- `profiles` / `mods` preview commands.
- Oscillators: `saw`, `square`, `sample-hold`, `noise`, `envelope`.
- `profile_blend` modulation (crossfade two profiles over time).
- Profiles: `radial`, `sine`, `--profile-image` heightmap, `--expr`.
- `beats` / BPM rate unit (audio-synced scoring).

**Design invariants to preserve across versions:**

1. The `delay_map(x, output_t, params)` signature — never special-case static vs. animated.
2. Buffer is an interface; the engine is backing-agnostic.
3. Profiles are pure functions; no I/O, no global state.
4. Rate units are a parsing lens over cycles-per-frame.

---

## 18. Glossary

- **Band / slice** — a strip of one or more pixels sharing one time offset.
- **Collapse mode** — video → single image; photofinish accumulation.
- **Cycles-per-frame** — the internal canonical rate unit.
- **Delay / offset** — frames a band lags behind the vanguard (≥ 0).
- **Delay surface** — `t = f(x, output_t)`; the surface cut through the X-Y-T cube.
- **Fill** — what fills bands whose source frame doesn't exist.
- **LFO / oscillator** — a periodic function of output time driving a parameter.
- **Modulation destination** — a parameter an oscillator can drive.
- **Ring buffer** — constant-memory circular frame store of `max_delay + 1` frames.
- **Sweep mode** — video → video; surface held/modulated while input plays past.
- **Vanguard** — the leading slice, delay 0.
- **X-Y-T cube** — the video as a volume; two spatial axes plus time.
