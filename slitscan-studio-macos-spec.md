# Slitscan Studio (macOS) — Technical Specification

**Status:** v1 design spec
**Companion documents:** `slitscan-spec.md` (CLI tool — the reference implementation of the surface math), `slitscan-build-plan.md`, `slitscan-audio-spec.md` (speculative)
**Last updated:** 2026-06-06

A macOS desktop application for **time-collapse slit-scan image making**. It ingests a short video (10–15s), holds it in memory as an X-Y-T volume ("the cube"), and lets the user cut a single shaped surface through that volume in real time — producing a still image where every pixel is a different moment in time. The output is a "photograph of time": video in, image out, with the surface manipulated through direct, immediate controls.

This document specifies the v1 desktop app. It positions the app as a **second implementation of the surface math defined by the CLI tool**, not a frontend bolted onto it — see §3.

---

## Table of contents

1. [Product intent](#1-product-intent)
2. [Relationship to the CLI tool](#2-relationship-to-the-cli-tool)
3. [The shared-core question (what is and isn't reused)](#3-the-shared-core-question)
4. [Core model: the cube and the surface](#4-core-model-the-cube-and-the-surface)
5. [Architecture](#5-architecture)
6. [The cube: ingest, residency, budgets](#6-the-cube-ingest-residency-budgets)
7. [The gather (GPU)](#7-the-gather-gpu)
8. [Preview vs. final render fork](#8-preview-vs-final-render-fork)
9. [Control taxonomy](#9-control-taxonomy)
10. [Surface / profile system](#10-surface--profile-system)
11. [The two time controls](#11-the-two-time-controls)
12. [UI layout & interaction](#12-ui-layout--interaction)
13. [Output & export](#13-output--export)
14. [Project / session persistence](#14-project--session-persistence)
15. [Performance targets](#15-performance-targets)
16. [Tech stack](#16-tech-stack)
17. [v1 scope, deferred, roadmap](#17-v1-scope-deferred-roadmap)
18. [Open questions](#18-open-questions)
19. [Glossary](#19-glossary)

---

## 1. Product intent

Most slit-scan tools are effects: apply a transform, get a weird video. This app is an **instrument for making still images** — a camera whose shutter is a function the user draws across space. The core experience is direct manipulation with realtime feedback: turn a knob, see the image change *now*, not after a render.

The artifact is a single high-resolution still in which spatial position encodes time. This places the tool in the lineage of strip photography, chronophotography, Hockney's photo-joiners, and the slit-scan catalogue — a coherent arts/production practice, not a novelty filter.

**Non-goals for v1:** video output (deferred — see roadmap), audio, batch processing, plugin/host integration. v1 is one clip in, one still out, manipulated in realtime.

---

## 2. Relationship to the CLI tool

The CLI tool (`slitscan-spec.md`) renders surfaces *sweeping through* the cube over output time (video out). This app cuts **one fixed surface** through the cube and freezes it to a still — conceptually the CLI's `render` **gather** evaluated at a *single* output time, reimagined as a direct-manipulation GUI. We call this operation **the cut**.

> **Terminology note.** "The cut" is the CLI's `render`/gather path with output time held fixed: the output keeps the full W×H picture, and every pixel reads its own time-depth. This is **distinct from the CLI's `collapse` command**, which is photofinish — one fixed slit column read per frame, with time mapped onto an output axis (output width = frame count). The app does *not* do photofinish; the word "collapse" is avoided here to prevent collision with that command.

The two tools share a **conceptual core**: the `delay(x, y, params) -> per-element delay` contract, the profile catalog (`ramp`, `reverse`, `tent`), the parameter taxonomy, and the source-frame math. They do **not** share executable code (see §3).

---

## 3. The shared-core question

This is the central architectural decision and is stated explicitly so it isn't quietly violated.

**The CLI's pure functions are NumPy on CPU. This app's hot path is a Metal shader on GPU.** You cannot call NumPy from a fragment shader. Therefore:

- **The app is fully standalone.** It lives in its own subdirectory and has **zero dependency on the CLI package** — it does not import, call, or build against `slitscan/`. Any math it needs is *copied* into the app's own tree, not referenced across the package boundary.
- **The surface math is shared as a *specification*, not as imported code.** `ramp`, `reverse`, `tent`, band-sampling, the source-frame formula — each is a few lines of arithmetic, reimplemented in Metal Shading Language here.
- **A small standalone reference, vendored into the app's tree, is the executable spec.** A self-contained NumPy script (copied from / modeled on the CLI's profile math, but living inside the app subdir) is the canonical, debuggable definition of "what is a tent surface" and the generator for test vectors. It does not import the CLI.
- **Agreement is guaranteed by a test-vector suite:** `(profile, params) -> expected delay array` cases emitted by that vendored reference, which the Metal implementation must reproduce within tolerance. This is a CI gate.
- **Sign convention is pinned to the CLI's:** source depth is `src = buffer_origin − spread_scale · delay(x, y)`, where `delay ≥ 0` counts frames *back* from the reference. The vanguard (delay 0) reads `buffer_origin` ("now"); every other region reads older footage. Test vectors are generated to this convention so Metal and the reference agree by construction.

**Mental model:** this app is a second, standalone implementation of the same spec, kept honest by test vectors — not a skin over the CLI, and not a fork of it.

---

## 4. Core model: the cube and the surface

The ingested clip is an **X-Y-T volume** ("the cube"): width × height × frames, held resident in GPU memory. Normal playback reads flat XY sheets at successive depths T. **Slit-scan reads a non-flat surface** through the volume — different spatial positions sampled at different depths (times).

The surface is defined by:

```
delay(x, y, params) -> t_offset    # how deep into the cube to read, per spatial position
```

Cutting to a still: for every output pixel `(x, y)`, sample the cube at `(x, y, buffer_origin − spread_scale · delay(x, y, params))` — note the **minus**, matching the CLI's `src = output_t − delay`. `delay ≥ 0` counts frames back; the vanguard reads `buffer_origin`. One cut, one image. Every pixel is a different moment.

This is the same surface contract as the CLI. v1 profiles (`ramp`, `reverse`, `tent`) are 1D — delay is a function of position along one axis. Genuinely 2D surfaces (`diagonal`, `radial`) that would make delay a function of both x and y are **deferred to v2** (see §10, §17), because the CLI has no reference implementation to gate them against (§3).

---

## 5. Architecture

```
┌──────────────┐   ┌──────────────┐   ┌─────────────────────┐   ┌──────────────┐
│  Ingest      │──▶│  Cube        │──▶│  Gather (Metal)     │──▶│  Preview     │
│  AVFoundation│   │  texture     │   │  surface eval +     │   │  (Metal view)│
│  decode      │   │  array (GPU) │   │  sample per pixel   │   └──────────────┘
└──────────────┘   └──────────────┘   └─────────▲───────────┘
                                                 │ uniforms (params)
                          ┌──────────────────────┴───────────────────────┐
                          │  Parameter state (Swift, observable)          │
                          │    ├─ surface profile + params                │
                          │    ├─ buffer window (origin) + spread (range) │
                          │    ├─ grain / slice-width, axis, fill         │
                          │    └─ driven by UI controls (knobs/gestures)  │
                          └───────────────────────────────────────────────┘
                                                 │
                                   ┌─────────────┴──────────────┐
                                   │  Final render (Metal,       │
                                   │  full-res, 16-bit) → export │
                                   └─────────────────────────────┘
```

- **Ingest** (AVFoundation): decode the clip to a sequence of pixel buffers, normalized to a chosen working resolution.
- **Cube**: frames uploaded to GPU as a **texture array** (one array slice per frame) or a 3D texture.
- **Parameter state** (Swift, observable/Combine): the single source of truth for all controls; mutations push uniforms to the shader.
- **Gather (Metal)**: a fragment shader that, per output pixel, evaluates the surface and samples the cube. This is the realtime hot path.
- **Preview**: an `MTKView` showing the current gather, updated every interaction frame.
- **Final render**: the same shader at full resolution and higher bit depth, run once on export.

---

## 6. The cube: ingest, residency, budgets

The realtime feel depends entirely on the cube being resident so re-cuts are cheap. Budgeting:

- **Frame size** = W × H × 4 bytes (RGBA8) at preview, or × 8 (RGBA16F) at high quality.
- **A 15s clip at 30fps = 450 frames.** At 1280×720 RGBA8 ≈ 3.7 MB/frame → ~1.66 GB for the cube. At 1920×1080 ≈ 8.3 MB/frame → ~3.7 GB.
- **Strategy:** ingest at a **working resolution** for the resident preview cube (e.g. ≤720p), sized to fit comfortably in GPU memory. Keep the original decoded frames (or re-decode on demand) only for the **final render** at full resolution, which is not latency-bound and can stream rather than hold the whole high-res cube.
- **Controls:** `working_resolution`, `max_cube_memory` budget. The app prints/inspects projected cube size on ingest and downsamples to fit, warning the user.
- **Clip length cap (v1):** 10–15s target; hard cap (e.g. 20s) to keep the cube bounded. Longer clips prompt a trim step on ingest.
- **Metal texture-array limits:** array slice count is bounded by the device; 450 frames is well within limits, but the app validates `frame_count ≤ device max array length` and falls back to a 3D texture or tiling if exceeded. (Flagged as an open item to verify on target hardware — §18.)

---

## 7. The gather (GPU)

The fragment shader is the heart of realtime. Per output pixel `(x, y)`:

1. Evaluate the active surface `delay(x, y, params)` from uniforms (profile selector + params).
2. Apply band quantization if `slice_width > 1` (snap x and/or y to band centers).
3. Compute source depth `t = buffer_origin − spread_scale · delay(...)` (minus — delay rakes backward in time).
4. Sample the cube texture array at slice `floor(t)` (and `floor(t)+1` blended, if interpolation on).
5. Apply fill if `t` is out of the resident window (black/white/transparent/hold).

All inputs are uniforms; changing any control is a uniform update + redraw, **not** a recompute of the cube. This is what makes 60fps manipulation possible. The math mirrors the CLI's gather exactly (verified by §3 test vectors).

**Interpolation** (`--interpolate` analogue): blend the two straddling slices for smooth fractional-depth sampling; toggle, off by default (crisp).

---

## 8. Preview vs. final render fork

The one place the architecture intentionally forks:

| | Preview | Final render |
|---|---|---|
| Resolution | working (≤720p) | full (up to source / user-set) |
| Bit depth | RGBA8 | RGBA16F (16-bit) |
| Cube | fully resident | streamed / re-decoded at full res |
| Latency | realtime (≤16ms/frame) | one-shot, seconds OK |
| Trigger | every interaction | on Export |

Same shader, same surface, different quality knobs. The user manipulates on the fast preview, then commits to a high-quality gather on export. Preview must be *representative* (same surface math) so what you see is what you render, modulo resolution/precision.

---

## 9. Control taxonomy

Controls fall into clear families. Each maps to a shader uniform.

- **Surface controls** — choose and shape the cut:
  - `profile` — `ramp`, `reverse`, `tent` (see §10).
  - `vanguard` — position of the "now" point (the delay-0 locus), 0–1 along the axis.
  - `axis` — `x` or `y`.
  - `spread` / `range` — how far back in time the surface rakes (see §11).
- **Time / buffer controls** (§11):
  - `buffer_origin` — where the sampled window sits within the clip (scrub all 15s).
  - `spread` — duplicated above; it's a time control and a surface control at once.
- **Texture controls:**
  - `slice_width` / `grain` — band size in px; a first-class *aesthetic* slider (stratification / venetian-blind look), not just performance.
- **Boundary:**
  - `fill` — black / white / transparent / hold for out-of-window samples.
- **Quality:**
  - `interpolate` — sub-slice blending toggle.

All continuous controls available as both a **knob/slider** and **direct gesture on the image** where it makes sense (drag to move vanguard, pinch to set spread).

---

## 10. Surface / profile system

v1 profiles port directly from the CLI — all 1D (delay is a function of position along one axis):

- **`ramp`** — linear gradient of delay along one axis. Vanguard at one edge.
- **`reverse`** — mirrored ramp.
- **`tent` / `vee`** — vanguard at a position; delay flares outward to both edges. (Halving property as in CLI spec §5.)

Each is a pure function `delay(x, params)` evaluated per-column (y ignored). Implemented in MSL; verified against the vendored NumPy reference via test vectors (§3). The registry is extensible — new surfaces are a shader function + a param schema.

**Deferred to v2 — genuinely 2D surfaces** (delay a function of *both* x and y). These have **no CLI reference implementation**, so they cannot be gated by §3's test-vector CI as the 1D profiles are; adding them means first writing a 2D reference. Deferred deliberately for that reason:
- **`diagonal`** — ramp along an arbitrary angle, not axis-aligned.
- **`radial`** — vanguard at a 2D point; delay grows with distance from it (cone/paraboloid cut). Center-is-now, time spirals outward.
- Freeform surface (paint a heightmap), `--profile-image` import, expression-based surfaces.

---

## 11. The two time controls

A deliberate, easily-conflated distinction the UI must keep separate:

- **`buffer_origin` (scrub)** — *where* the sampled time window sits in the clip. Slides the whole window across the 10–15s. Changes *which moments* the image is built from.
- **`spread` / `range`** — *how much* time the surface rakes across. Changes *how far apart in time* adjacent regions of the image are — from "all nearly the same instant" (tiny spread) to "raking across seconds" (large spread).

Merging these into one control would destroy expressivity; they answer different questions ("when" vs. "how wide"). Both are first-class, separately labeled, separately gesturable.

---

## 12. UI layout & interaction

**Single-window, three zones:**

- **Canvas (center)** — the live preview, dominant. Direct manipulation happens here: drag to set vanguard, pinch/scroll to set spread, modifier-drag to rotate diagonal angle. The image *is* the primary control surface.
- **Inspector (right)** — knobs/sliders for every parameter (§9), grouped by family. Numeric entry for precision. This is the "precision" complement to canvas gesture, important on desktop.
- **Filmstrip / timeline (bottom)** — visualizes the clip and the current `buffer_origin` window + `spread` as a highlighted band, so the two time controls are *visible* and scrubbable. Thumbnails of the clip frames.

**Interaction principles:**
- Every change is realtime; no "apply" button for preview.
- Gesture and knob are two views of the same uniform — moving one updates the other.
- Reset-per-control and global reset.
- Non-destructive: the cube is never modified; only the surface params change.

**Precision affordances (desktop-specific):** numeric fields, arrow-key nudges, snapping (e.g. vanguard to center/edges), and a curve/angle readout. These are why this is a Mac app and not just the iOS app on a bigger screen.

---

## 13. Output & export

- **Formats:** PNG (lossless, alpha), TIFF (16-bit, print/production), JPEG (sharing). Alpha-bearing formats required when `fill = transparent`.
- **Resolution:** up to source resolution (or user-set), via the high-quality final render path (§8).
- **Bit depth:** 16-bit for TIFF/PNG where the precision matters (smooth gradients across time can band at 8-bit).
- **Metadata:** optionally embed the parameter set (the "recipe") in the file (e.g. PNG text chunk / EXIF user comment) so a still can be traced back to its surface — useful for an arts practice.
- **Export is the only non-realtime operation;** a few seconds is acceptable.

---

## 14. Project / session persistence

- **Session file** stores: source clip reference (or embedded), all surface/time/texture params, working resolution, and UI state. Reopening restores the exact editable state.
- Params are small (a handful of floats + enums) — the "recipe" is tiny and shareable independent of the clip.
- A saved recipe applied to a *different* clip is a meaningful operation (same cut, new footage) — supported.

---

## 15. Performance targets

- **Preview interaction:** ≤16 ms/frame (60fps) for any control change on the resident cube at working resolution. This is the headline requirement; the whole architecture exists to hit it.
- **Ingest:** decode + upload a 15s clip in a few seconds, with progress.
- **Final render/export:** seconds, not minutes, for a full-res 16-bit still.
- **Memory:** resident preview cube within `max_cube_memory` (default sized to leave headroom on 16GB machines); warn and downsample if over.

---

## 16. Tech stack

| Concern | Choice | Notes |
|---|---|---|
| Language | Swift | Native macOS. |
| UI | SwiftUI (+ AppKit where needed) | Inspector, filmstrip; AppKit for fine pointer handling if required. |
| Realtime view | Metal / `MTKView` | The gather shader + preview. |
| GPU | Metal Shading Language | Surface eval + cube sampling; the hot path. |
| Decode | AVFoundation | Clip ingest to pixel buffers. |
| Export | ImageIO / Core Image | PNG/TIFF/JPEG, 16-bit, metadata. |
| Min OS | macOS 14+ (suggested) | Confirm against Metal features used. |

Apple-silicon-first; unified memory helps the cube residency story.

---

## 17. v1 scope, deferred, roadmap

**v1 (this spec):**
- Single clip ingest (10–15s), resident GPU cube at working resolution.
- Realtime Metal gather; preview at 60fps.
- Profiles: `ramp`, `reverse`, `tent` (1D, CLI-referenced).
- Controls: vanguard, spread, buffer_origin, slice_width/grain, axis, fill, interpolate.
- Canvas direct manipulation + inspector knobs + filmstrip.
- Still export: PNG/TIFF/JPEG, up to 16-bit, recipe metadata.
- Session save/load.

**Deferred (v2+):**
- **Video / loop export** (the big one — convergence with the CLI's sweep mode; animate the controls over an output duration). See note below.
- **2D surfaces: `diagonal`, `radial`** (require a 2D reference implementation first — see §10).
- Freeform / painted surfaces; profile-image import; expression surfaces.
- Keyframe and/or LFO animation of controls (prerequisite for video).
- Batch / recipe-apply-to-many.
- iOS sibling app (shared Metal core, touch-first UI).

**Video note (forward design):** video export = the still renderer run per output frame while the surface animates. The v1 still renderer should therefore be written internally as "render surface at output-time T" with T pinned to a single value — so video is a loop around existing code, not a rewrite. Seamless loops require integer-cycle parameter animation (the CLI's `cyc` rate unit). This is a v2 feature but the v1 architecture must not preclude it.

---

## 18. Open questions

1. **Metal texture-array slice limit** on target hardware vs. max frame count (450+). Verify; fall back to 3D texture or tiling if needed (§6).
2. **Working resolution default** — what's the sweet spot between preview fidelity and 60fps headroom across the M-series range? Needs profiling.
3. **High-res final render**: stream from re-decode, or hold a second full-res cube transiently? Memory vs. speed tradeoff.
4. **Freeform surface in v1 or v2?** Parametric profiles cover a lot; painted heightmaps are powerful but a bigger UI lift. Currently deferred — confirm.
5. **Recipe portability to CLI**: should a session recipe be exportable as CLI flags / patch file, so a still designed in the GUI can be batch-rendered or video-swept via the CLI? Strong synergy; confirm priority.
6. **Color management**: working color space, display P3 vs. sRGB preview, and export profile handling — matters for a production/arts tool. Needs a decision before export ships.

---

## 19. Glossary

- **Cube / X-Y-T volume** — the decoded clip held whole in memory as width × height × frames; the thing the surface cuts through.
- **Surface / cut** — the shape `delay(x, y, params)` sampled through the cube; the artwork's defining function.
- **Vanguard** — the delay-0 locus (the "now" point) of the surface.
- **Spread / range** — how far back in time the surface rakes; one of the two time controls.
- **Buffer origin** — where the sampled time window sits within the clip; the other time control.
- **Grain / slice-width** — band size; an aesthetic stratification control.
- **Gather** — the per-pixel sampling of the cube along the surface; the GPU hot path.
- **Recipe** — the full parameter set defining a still; tiny, shareable, clip-independent.
- **Working resolution** — the downsampled resolution of the resident preview cube.
- **Final render** — the full-resolution, higher-bit-depth gather run on export.
