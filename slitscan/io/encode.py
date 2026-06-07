"""PyAV-based video encoder and PIL-based image-sequence encoder.

Phase 1: H.264 for .mp4; placeholder for other formats.
Phase 3: Full codec/container matrix with alpha support and image sequences.
"""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np
from PIL import Image


# Image-sequence extensions — handled by PIL, not PyAV
_IMAGE_SEQUENCE_EXTS: frozenset[str] = frozenset({".png", ".tiff", ".tif"})

# GIF extension
_GIF_EXT: str = ".gif"

# Codec selection matrix for video containers
_CODEC_MAP: dict[str, str] = {
    ".mp4": "libx264",
    ".mov": "prores_ks",
    ".mkv": "libx264",
    ".webm": "libvpx-vp9",
}

# Codecs that support an alpha channel
_ALPHA_CODECS: frozenset[str] = frozenset({"libvpx-vp9", "prores_ks"})


def codec_name_for_path(path: "str | Path") -> str:
    """Return the codec name that will be used for the given output path.

    Image-sequence extensions return ``"image_sequence"`` as a sentinel.
    GIF returns ``"gif"`` as a sentinel.
    """
    suffix = Path(path).suffix.lower()
    if suffix in _IMAGE_SEQUENCE_EXTS:
        return "image_sequence"
    if suffix == _GIF_EXT:
        return "gif"
    return _CODEC_MAP.get(suffix, "libx264")


def open_encoder(
    path: str,
    width: int,
    height: int,
    fps: float,
    fill: str = "black",
) -> "Encoder | ImageSequenceEncoder":
    """Open an output encoder for the given path.

    Parameters
    ----------
    path:
        Output file path. Extension selects the codec.
    width, height:
        Frame dimensions in pixels.
    fps:
        Output frame rate.
    fill:
        Fill mode. ``"transparent"`` requires an alpha-capable codec/format.

    Raises
    ------
    ValueError
        If ``fill="transparent"`` is requested but the output format
        does not support an alpha channel.
    """
    p = Path(path)
    suffix = p.suffix.lower()

    # Image sequences
    if suffix in _IMAGE_SEQUENCE_EXTS:
        return ImageSequenceEncoder(output_path=p, fill=fill)

    # Animated GIF
    if suffix == _GIF_EXT:
        return GifEncoder(output_path=p, fps=fps)

    codec = _CODEC_MAP.get(suffix, "libx264")

    if fill == "transparent" and codec not in _ALPHA_CODECS:
        raise ValueError(
            f"fill='transparent' requires an alpha-capable codec, but the output "
            f"format '{suffix}' maps to codec '{codec}' which does not support alpha. "
            f"Use .mov (ProRes 4444) or .png/.tiff (image sequence) for transparency."
        )

    return Encoder(path=path, width=width, height=height, fps=fps, codec=codec, fill=fill)


class Encoder:
    """Stateful encoder wrapper around a PyAV output container."""

    def __init__(
        self,
        path: str,
        width: int,
        height: int,
        fps: float,
        codec: str,
        fill: str,
    ) -> None:
        self._path = path
        self._width = width
        self._height = height
        self._fps = fps
        self._codec = codec
        self._fill = fill
        self._closed = False

        # Determine pixel format based on codec and fill
        if codec == "prores_ks":
            if fill == "transparent":
                # ProRes 4444 with alpha
                self._pix_fmt_in = "rgba"
                self._pix_fmt_enc = "yuva444p10le"
            else:
                # ProRes 422
                self._pix_fmt_in = "rgb24"
                self._pix_fmt_enc = "yuv422p10le"
        elif fill == "transparent" and codec in _ALPHA_CODECS:
            # VP9 alpha
            self._pix_fmt_in = "rgba"
            self._pix_fmt_enc = "yuva420p"
        else:
            self._pix_fmt_in = "rgb24"
            self._pix_fmt_enc = "yuv420p"

        from fractions import Fraction
        self._container = av.open(path, mode="w")
        self._stream = self._container.add_stream(codec, rate=Fraction(fps).limit_denominator(10000))
        self._stream.width = width
        self._stream.height = height
        self._stream.pix_fmt = self._pix_fmt_enc

        # Codec-specific options
        if codec == "libx264":
            self._stream.options = {"crf": "18", "preset": "fast"}
        elif codec == "prores_ks":
            if fill == "transparent":
                # Profile 4 = ProRes 4444
                self._stream.options = {"profile": "4"}
            else:
                # Profile 2 = ProRes 422 HQ (default)
                self._stream.options = {"profile": "2"}

    def write_frame(self, frame: np.ndarray) -> None:
        """Write a single frame (H×W×C uint8 numpy array)."""
        if self._closed:
            raise RuntimeError("Encoder is already closed.")

        av_frame = av.VideoFrame.from_ndarray(frame, format=self._pix_fmt_in)
        av_frame = av_frame.reformat(format=self._pix_fmt_enc)
        for packet in self._stream.encode(av_frame):
            self._container.mux(packet)

    def close(self) -> None:
        """Flush the encoder and close the output file."""
        if self._closed:
            return
        self._closed = True
        for packet in self._stream.encode():
            self._container.mux(packet)
        self._container.close()

    def __enter__(self) -> "Encoder":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


class GifEncoder:
    """Encode frames into an animated GIF using Pillow.

    Frames are palette-quantized (256 colours) per frame with adaptive
    dithering. The output loops infinitely (``loop=0``) by default — ideal
    for installation use. Large frames will produce large files; pass
    ``--resize`` to downscale before rendering.
    """

    def __init__(self, output_path: Path, fps: float, loop: int = 0) -> None:
        self._path = output_path
        self._duration_ms = int(round(1000.0 / max(fps, 1.0)))
        self._loop = loop
        self._frames: list[Image.Image] = []

    def write_frame(self, frame: np.ndarray) -> None:
        img = Image.fromarray(frame).convert(
            "P", palette=Image.ADAPTIVE, colors=256, dither=Image.FLOYDSTEINBERG
        )
        self._frames.append(img)

    def close(self) -> None:
        if not self._frames:
            return
        self._frames[0].save(
            str(self._path),
            format="GIF",
            save_all=True,
            append_images=self._frames[1:],
            duration=self._duration_ms,
            loop=self._loop,
            optimize=False,
        )

    def __enter__(self) -> "GifEncoder":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


class ImageSequenceEncoder:
    """Write individual image files (PNG or TIFF) per frame using Pillow."""

    def __init__(self, output_path: Path, fill: str) -> None:
        self._path = output_path
        self._fill = fill
        self._frame_num = 0
        self._stem = output_path.stem
        self._suffix = output_path.suffix
        self._dir = output_path.parent

    def write_frame(self, frame: np.ndarray) -> None:
        """Write one frame as a numbered image file."""
        filename = self._dir / f"{self._stem}_{self._frame_num:04d}{self._suffix}"
        img = Image.fromarray(frame)
        img.save(str(filename))
        self._frame_num += 1

    def close(self) -> None:
        """No-op: nothing to flush for image sequences."""
        pass

    def __enter__(self) -> "ImageSequenceEncoder":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
