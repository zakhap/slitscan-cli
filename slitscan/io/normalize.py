"""Frame normalization: resize and fit-mode handling."""

from __future__ import annotations

import numpy as np
from PIL import Image


def normalize_frame(
    frame: np.ndarray,
    target_w: int,
    target_h: int,
    fit: str,
) -> np.ndarray:
    """Resize a frame (H×W×C uint8) to (target_h × target_w × C).

    Parameters
    ----------
    frame:
        Source frame as numpy array, shape (H, W, C), dtype uint8.
    target_w:
        Output pixel width.
    target_h:
        Output pixel height.
    fit:
        Scaling strategy.
        - ``"crop"``      — scale so the smaller dimension fills the target,
                           then center-crop. (Phase 1)
        - ``"letterbox"`` — scale preserving aspect ratio with black bars.
                           (Phase 2+)
        - ``"stretch"``   — ignore aspect ratio, stretch to fill.
                           (Phase 2+)
    """
    if fit == "crop":
        return _fit_crop(frame, target_w, target_h)
    elif fit == "letterbox":
        raise NotImplementedError("fit=letterbox is not implemented until Phase 2")
    elif fit == "stretch":
        raise NotImplementedError("fit=stretch is not implemented until Phase 2")
    else:
        raise ValueError(f"Unknown fit mode: {fit!r}. Choose from: crop, letterbox, stretch")


def _fit_crop(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Scale so the smaller dimension matches target, then center-crop."""
    src_h, src_w = frame.shape[:2]

    # Determine scale factor: whichever axis requires a LARGER scale to fill
    scale = max(target_w / src_w, target_h / src_h)

    scaled_w = int(round(src_w * scale))
    scaled_h = int(round(src_h * scale))

    img = Image.fromarray(frame)
    img = img.resize((scaled_w, scaled_h), Image.LANCZOS)
    scaled = np.array(img)

    # Center crop
    x0 = (scaled_w - target_w) // 2
    y0 = (scaled_h - target_h) // 2
    cropped = scaled[y0: y0 + target_h, x0: x0 + target_w, :]
    return cropped
