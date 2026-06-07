# Slitscan — Implementation Build Plan & Developer Handoff

**Status:** v1 build plan
**Companion document:** `slitscan-spec.md` (the technical spec / product docs — read it first)
**Last updated:** 2026-06-06

This document is a build-order plan for an engineering agent implementing the Slitscan tool described in `slitscan-spec.md`. It assumes the spec is the source of truth for *what* and *why*; this document covers *how* and *in what order*, plus the constraints, acceptance criteria, and pitfalls that aren't obvious from the spec alone.

Read the spec sections in parentheses as you reach each phase.

---

## 0. Ground rules for the implementing agent

These are non-negotiable and any deviation should be raised before coding, not after:

1. **The `delay_map(x_coords, output_t, params) -> ndarray` signature is sacred** (spec §2). Static, animated, and modulated renders are ONE code path. Do not add a fast-path that special-cases static surfaces — it will bit-rot and diverge. If a static render is slow, optimize inside this signature, not around it.
2. **The buffer is an interface** (spec §4). The engine asks the buffer for "input frame N" and never touches PyAV for reads. Both `full` and `ring` backings implement the same interface. The engine must contain zero `if buffer_type == ...` branches.
3. **Profiles are pure functions.** No I/O, no global state, no caching across frames. Input arrays in, delay array out. This is what makes them testable and composable.
4. **Rate units are a parsing lens** (spec §7). Convert to cycles-per-frame at the boundary; the oscillator core only ever sees cycles-per-frame.
5. **Fail loud and early.** Validate buffer plan, codec/alpha compatibility, and ring-window reach *before* decoding a single frame of payload. A user should never wait through a long decode to hit an error that was knowable at startup.
6. **Print the plan.** On every `render`, before processing, print: resolved dimensions, fps, frame count, buffer backing + projected size, codec, and a one-line summary of active modulation. This is the primary debugging affordance.

---

## 1. Tech stack & environment

| Concern | Choice | Notes |
|---|---|---|
| Language | Python 3.11+ | Match typing features used below. |
| Decode/encode | PyAV | libav bindings; handles alpha codecs (ProRes 4444). Pin a known-good version. |
| Numerics | NumPy | The gather is the hot path. |
| CLI | Typer | Subcommands map to functions cleanly. |
| Patch files | PyYAML | `--mod-file`. |
| Packaging | `pyproject.toml`, console entry point `slitscan` | — |
| Target platform | Apple Silicon (M1) native, but keep it portable | No platform-specific code; PyAV/NumPy are universal. |

**Environment caveat for whoever builds this:** PyAV wheels need a working ffmpeg/libav. On the build/test machine, confirm `import av; av.codecs_available` includes `prores_ks` (or equivalent) and an H.264 encoder before starting Phase 5, or the codec matrix can't be validated. If ffmpeg isn't present, document the install step (`brew install ffmpeg` on macOS) in the README.

Note: the sandbox network allowlist may not include PyPI mirrors for av/ffmpeg in every environment — verify package installation works in the target dev environment first and flag if blocked.

---

## 2. Module layout

```
slitscan/
├── __init__.py
├── cli.py            # Typer app; arg parsing; wires everything; owns "print the plan"
├── io/
│   ├── decode.py     # PyAV decode → normalized RGB(A) ndarrays + metadata
│   ├── encode.py     # PyAV encode; codec/container selection; alpha validation
│   └── normalize.py  # resize / fit (crop|letterbox|stretch)
├── buffer/
│   ├── base.py       # FrameBuffer ABC (the interface)
│   ├── full.py       # whole-clip-in-RAM backing
│   └── ring.py       # circular buffer backing
├── profiles/
│   ├── base.py       # Profile protocol + registry
│   ├── ramp.py
│   ├── reverse.py
│   └── tent.py
├── modulation/
│   ├── oscillators.py # sine, triangle; osc(phase)->[-1,1]
│   ├── rates.py       # unit parsing: hz|cyc|frames -> cycles/frame
│   ├── patch.py       # --mod string parser + YAML loader; merge order
│   └── resolve.py     # per-frame: base params + active mods -> resolved params
├── engine/
│   ├── render.py      # sweep mode: per-frame gather loop
│   ├── collapse.py    # photofinish accumulation
│   ├── gather.py      # the vectorized column gather + interpolation
│   └── fill.py        # black|white|transparent|hold|wrap
└── meta.py            # ClipMeta dataclass (fps, frame_count, W, H, channels)
```

Keep `engine/gather.py` free of any knowledge of profiles, modulation, or PyAV. It receives a buffer, an array of per-band source indices, and a fill policy. That isolation is what makes it unit-testable with synthetic buffers.

---

## 3. Core data types (define these first)

```python
@dataclass(frozen=True)
class ClipMeta:
    fps: float
    frame_count: int
    width: int
    height: int
    channels: int          # 3 (RGB) or 4 (RGBA)

@dataclass
class RenderParams:        # the "base" values, pre-modulation
    profile: str
    axis: str              # "x" | "y"
    vanguard: float        # 0..1
    max_delay: int         # frames
    slice_width: int       # px
    fill: str
    interpolate: bool

# resolved per-frame params are the same shape with modulation applied;
# represent as a mutable copy or a dict keyed by destination name.
```

`ClipMeta` is produced by `io/decode.py` and threaded everywhere timing or sizing is needed. Modulation rate resolution (spec §7) consumes `fps` and `frame_count` from it.

---

## 4. Build phases (strict order)

Each phase ends in something runnable and testable. Do not start a phase until the prior phase's acceptance criteria pass.

### Phase 1 — Vertical slice end-to-end (the skeleton)
**Goal:** `slitscan render in.mp4 out.mp4 --profile ramp` works with a static linear ramp, `full` buffer, `black` fill, no modulation, axis x, slice-width 1.

Build: `meta.py`, `io/decode.py` (decode + RGB only), `io/normalize.py` (resize/crop only), `io/encode.py` (H.264 only), `buffer/base.py` + `buffer/full.py`, `profiles/base.py` + `profiles/ramp.py`, `engine/gather.py` (no interpolation), `engine/fill.py` (black only), `engine/render.py`, minimal `cli.py`.

**Acceptance:**
- A short clip renders without error; output frame count and fps match input.
- Visual check: leftmost column leads, rightmost lags by `max_delay`; reading left→right is reading back in time.
- First `max_delay` frames show black fill on the lagging side.
- The "print the plan" line appears at startup.

**Pitfall:** get the source-frame sign convention right here (spec §2): `source_frame = output_t + (max_delay - delay_map[x])`. Verify with a clip that has a burnt-in frame counter or an obvious moving object; off-by-one and reversed-time bugs are the most common failure and are obvious with a counter.

### Phase 2 — Profiles & axis
Add `reverse.py`, `tent.py`, the profile registry, and `--axis y`. Add `--vanguard`, `--max-delay`, `--slice-width` plumbing.

**Acceptance:**
- `reverse` mirrors `ramp` exactly (vanguard at right).
- `tent` with centered vanguard reaches `max_delay` at both edges; confirm the halving relationship (spec §5) by comparing edge delays at matching slope.
- `--axis y` produces the row-wise (rolling-shutter) version.
- `--slice-width 4` yields visibly banded output with 1/4 the band count; performance improves.

**Pitfall:** band sampling math — `ceil(extent / slice_width)` bands, profile sampled per band, delay constant across each band's pixels. Off-by-one at the final partial band is the trap; test with widths that don't divide evenly.

### Phase 3 — Fill policies & codec/alpha
Complete `engine/fill.py` (white, hold, wrap, transparent) and `io/encode.py` codec/container selection per the matrix (spec §15). Implement the alpha validation gate.

**Acceptance:**
- Each fill mode behaves per spec §9.
- `--fill transparent` to an `.mp4` errors *before* decode with a message suggesting ProRes 4444 or PNG sequence.
- `--fill transparent` to a ProRes 4444 `.mov` or PNG sequence produces correct alpha (verify the fill region is actually transparent, not black).

**Pitfall:** RGBA changes channel count to 4, which changes buffer size math (spec §4) and must be reflected in the buffer plan. Decode must produce RGBA when transparent fill is requested.

### Phase 4 — Modulation
Build `modulation/` in this internal order: `oscillators.py` → `rates.py` → `resolve.py` → `patch.py`. Wire `--mod` and `--mod-file` into `cli.py`. Resolve params per-frame in the render loop.

**Acceptance:**
- `--mod vanguard=sine:rate=0.5cyc,depth=0.5` visibly sweeps the vanguard over the clip.
- All three rate units produce correct frequencies (test: a `1cyc` sine returns to phase 0 exactly at the last frame; a `2frames` oscillator has period 2).
- Multiple mods on one destination sum; resolved values clamp to valid ranges.
- `--mod-file` reproduces equivalent `--mod` flags; CLI flags append/override per spec §12.

**Pitfall:** rate resolution must happen AFTER metadata is known (spec §7). The patch parser produces unresolved oscillators (rate + unit); `resolve.py` converts to cycles-per-frame using `ClipMeta`. Don't resolve at parse time — `hz`/`cyc` need fps/frame_count that aren't available then.

### Phase 5 — Interpolation
Add sub-frame blending in `engine/gather.py` behind `--interpolate` (spec §10). Fractional source indices blend the two straddling buffered frames.

**Acceptance:**
- With a fast modulation, stutter/banding visible without `--interpolate` is smoothed with it.
- `--interpolate` off rounds fractional indices to nearest (crisp); on blends (smooth).
- Performance cost is roughly 2× gather, as expected.

### Phase 6 — Ring buffer
Implement `buffer/ring.py` against the same interface. Auto-selection logic in `cli.py`/buffer planner: choose `ring` when projected `full` size > `--memory-budget`. Validate the surface's reach fits the `max_delay + 1` window.

**Acceptance:**
- A long clip that would blow the budget renders in constant memory via `ring`.
- `full` and `ring` produce *identical* output for a clip that fits both (this is the key correctness test — diff the outputs).
- A surface whose reach exceeds the ring window errors at validation, before decode.

**Pitfall:** `ring` only holds `[output_t, output_t + max_delay]`. Any fill mode or surface that references outside that window (e.g. `wrap` needs clip-end frames; `collapse` needs full random access) is incompatible with `ring` — detect and either force `full` or error clearly.

### Phase 7 — Collapse (photofinish)
Build `engine/collapse.py`: video → single image, accumulating one moving slit's history (spec §1 "collapse mode", §11 `collapse` command). Add `--slit-position`, `--direction`. Output PNG/TIFF.

**Acceptance:**
- Output is one image whose successive columns are the slit's content at successive input times.
- A clip of something moving horizontally past the slit produces the expected smear/strip.
- `--direction reverse` reverses the time axis of accumulation.

**Note:** `collapse` needs full random access → forces `full` buffer (or streaming accumulation, which is simpler here since each input frame contributes exactly one output column — you can accumulate without holding all frames). Prefer the streaming accumulation; it sidesteps the buffer entirely.

---

## 5. Testing strategy (build alongside, not after)

- **Synthetic-buffer unit tests for `gather.py`:** construct a fake buffer where frame N is a solid image of value N. Then a gathered output column's value directly reveals which source frame it came from — making source-index correctness assertable numerically, no video needed. This is the single highest-value test; write it in Phase 1.
- **Profile unit tests:** assert the delay array shape and known values (ramp endpoints, tent symmetry, reverse mirror) for fixed params.
- **Rate unit tests:** assert cycles-per-frame conversions for each unit against hand-computed values.
- **`full` vs `ring` equivalence test (Phase 6):** byte-identical output on a clip that fits both.
- **Codec gate tests:** assert the transparent+non-alpha combination raises before decode.
- **A frame-counter fixture clip** (burnt-in frame numbers) for visual/manual verification of time direction and off-by-one.

---

## 6. Product constraints & details (carried from spec, do not lose)

- **Default behaviors:** profile `ramp`; axis `x`; slice-width `1`; fill `black`; buffer `auto`; rate unit `cyc`; interpolate off; fit `crop`. (Spec §11.)
- **`max_delay` default** is `extent − 1` (extent = width for axis x, height for axis y). (Spec §5, §11.)
- **Transparent fill** forces RGBA + alpha codec; H.264/H.265 rejected with a suggested alternative. (Spec §9, §15.)
- **Rate units** `hz` (needs fps), `cyc` (needs frame_count), `frames` (needs nothing); all collapse to cycles-per-frame; default `cyc`. (Spec §7.)
- **Modulation application:** `resolved = base + offset + depth × osc(2π·(cyc_per_frame·t + phase))`, then clamp; multiple mods on a destination sum. (Spec §6.)
- **Modulation destinations (v1):** `vanguard`, `max_delay`, `slice_width`, `fill_alpha`. Oscillators (v1): `sine`, `triangle`. (Spec §6.)
- **Buffer selection** is automatic from `max_delay × frame_size` vs `--memory-budget`; overridable. Print projected size; refuse over-budget `full` without `ring`. (Spec §4, §14.)
- **Frame size** = `W × H × channels × dtype_bytes`; RGBA is 4 channels (~33% larger buffer). (Spec §4.)
- **Normalization is a single front-end step**, before buffering; engine sees only fixed dimensions. (Spec §8.)
- **Output extension selects container/codec** per the matrix. (Spec §15.)

## 7. Explicitly OUT of v1 scope (do not build; leave seams)

Per spec §17: `profiles`/`mods` preview commands; oscillators saw/square/sample-hold/noise/envelope; `profile_blend` modulation; profiles radial/sine/`--profile-image`/`--expr`; `beats`/BPM rate unit.

**Leave seams for these:** the profile registry should make adding a profile a one-file drop-in; the oscillator module should make adding an oscillator a one-function addition; `rates.py` should make adding a unit a single mapping entry; `resolve.py` should make `profile_blend` addable as just another destination. Don't build them, but don't wall them out.

## 8. Suggested first PR boundaries

1. PR1: data types + decode/normalize/encode (H.264) + full buffer, no engine — proves I/O round-trips.
2. PR2: gather + ramp + render loop + black fill — Phase 1 complete, first real output.
3. PR3: profiles + axis + band sampling — Phase 2.
4. PR4: fill policies + codec/alpha — Phase 3.
5. PR5: modulation — Phase 4.
6. PR6: interpolation — Phase 5.
7. PR7: ring buffer + equivalence test — Phase 6.
8. PR8: collapse — Phase 7.

Each PR ships its phase's tests. Keep PRs in this order; later phases depend on the invariants established earlier.

---

## 9. Open items to confirm with product owner before/during build

These were resolved in design discussion but are worth a final confirmation as you reach them, since they affect user-visible behavior:

1. **`--vanguard` units:** normalized 0–1 across the axis (spec assumes this). Confirm vs. pixel position — 0–1 is resolution-independent and recommended.
2. **`collapse` output length:** one image always, or allow N images (a strip every K frames)? Spec says one; confirm whether a multi-strip mode is wanted later.
3. **Output fps override** (`--fps`): does changing it resample (drop/dup frames) or just retag? Recommend retag-only in v1 (simpler, predictable); flag if resampling is expected.
4. **Color/pixel format:** spec assumes 8-bit RGB(A). If the user wants 10-bit/ProRes-native high bit depth preserved, that's a buffer-dtype change (`uint16`) affecting size math — out of v1 unless confirmed.
