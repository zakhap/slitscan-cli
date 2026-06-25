"""Generate a synthetic source clip that makes grid slit-scan legible.

Design choices, each so the effect reads at a glance:
  * Whole-frame HUE encodes time (frame N → hue N/total). In grid mode every
    tile reads ONE source frame, so the mosaic becomes a direct color readout
    of "what moment is this cell". Adjacent-in-time tiles are adjacent in hue.
  * A bright disc travels a Lissajous path → obvious spatial motion content,
    so tiles also show the disc at different places/times.
  * The frame number is burned in large, so you can literally read each tile's
    time.

Run:  python examples/make_source_clip.py
"""

from __future__ import annotations

import colorsys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from slitscan.io.encode import open_encoder

W, H = 480, 360
FRAMES = 96
FPS = 24.0


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for cand in (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        try:
            return ImageFont.truetype(cand, size)
        except Exception:
            continue
    return ImageFont.load_default()


def make_frame(n: int, total: int) -> np.ndarray:
    # Background hue sweeps the full spectrum across the clip.
    hue = n / max(total - 1, 1)
    r, g, b = colorsys.hsv_to_rgb(hue, 0.55, 0.95)
    frame = np.empty((H, W, 3), dtype=np.uint8)
    frame[:] = (int(r * 255), int(g * 255), int(b * 255))

    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    # Lissajous-pathed bright disc → clear spatial motion.
    t = n / max(total - 1, 1)
    cx = W * (0.5 + 0.38 * np.sin(2 * np.pi * t))
    cy = H * (0.5 + 0.38 * np.sin(2 * np.pi * 2 * t))
    rad = 34
    draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad],
                 fill=(255, 255, 255), outline=(20, 20, 20), width=3)

    # Burned-in frame number, large, centered.
    label = f"{n:02d}"
    fnt = _font(150)
    bbox = draw.textbbox((0, 0), label, font=fnt)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((W - tw) / 2 - bbox[0], (H - th) / 2 - bbox[1]), label,
              font=fnt, fill=(10, 10, 10))
    return np.asarray(img)


def main() -> None:
    out = Path(__file__).parent / "source.mp4"
    enc = open_encoder(str(out), width=W, height=H, fps=FPS, fill="black")
    for n in range(FRAMES):
        enc.write_frame(make_frame(n, FRAMES))
    enc.close()
    print(f"wrote {out}  ({W}×{H}, {FRAMES} frames @ {FPS}fps)")


if __name__ == "__main__":
    main()
