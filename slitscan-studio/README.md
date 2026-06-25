# Slitscan Studio (macOS)

A native macOS app for **time-collapse slit-scan image making**. Hold a short
clip in GPU memory as an X-Y-T "cube," cut one shaped surface through it, and get
a still where every pixel is a different moment in time — manipulated in realtime.

This is a **standalone** second implementation of the surface math defined by the
`slitscan` CLI (one directory up). It does not import or depend on the CLI; the
profile math is vendored and kept honest by a shared test-vector suite. See
`../slitscan-studio-macos-spec.md` for the full spec.

## Requirements

- Apple-silicon Mac, macOS 14+
- Xcode / Swift 6.2 toolchain

## Run

```bash
swift run SlitscanStudio          # launch the app
swift run SlitscanStudio --selftest   # headless engine check (renders + exports to /tmp)
swift test                        # profile parity gate (Swift vs NumPy reference)
```

In the app: **Demo clip** generates a synthetic cube so you can play immediately;
**Open video…** ingests a real 10–15s clip. Drag on the canvas to move the
vanguard (x) and spread (y); use the inspector for precision; scrub the filmstrip
to move the buffer origin. **Export still…** writes PNG/TIFF/JPEG plus a
`.recipe.json` sidecar.

## What's here (v1 prototype)

- Resident GPU cube (`texture2d_array`, one slice per frame), memory-budgeted at a
  working resolution (≤720p by default).
- Realtime Metal gather at 60fps. Sign convention pinned to the CLI's
  `src = origin − spread·delay`.
- Profiles `ramp`, `reverse`, `tent` (1D, gated against the NumPy reference).
- Controls: vanguard, spread, buffer-origin, grain/slice-width, axis, fill,
  interpolate.
- Canvas direct manipulation + inspector + filmstrip.
- Still export with recipe metadata; session recipe save/load.

## Architecture

```
reference/generate_vectors.py     vendored NumPy reference → test_vectors.json
Sources/SlitscanCore/             pure profile math + recipe model (testable)
Sources/SlitscanStudio/
  ShaderSource.swift              the gather MSL (compiled at runtime)
  GatherRenderer.swift            MTKView delegate + offscreen export render
  Cube.swift / VideoIngest.swift  residency + AVFoundation/MPS ingest + demo cube
  AppState.swift                  single source of truth (params + cube)
  Views/                          ContentView, Inspector, Filmstrip
  Exporter.swift / Session.swift  ImageIO export + recipe persistence
Tests/SlitscanCoreTests/          §3 parity gate
```

## Deferred (v2)

2D surfaces (`diagonal`, `radial` — need a 2D reference first), full-source-res
export via re-decode, 16-bit pipeline, color management, video/loop export,
keyframe/LFO animation. See spec §17.
