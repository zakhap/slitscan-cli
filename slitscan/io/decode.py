"""PyAV-based video decoder.

Decodes frames to normalized RGB (or RGBA) numpy arrays (H×W×C, uint8).
"""

from __future__ import annotations

from typing import Iterator

import av
import numpy as np

from slitscan.meta import ClipMeta


def open_video(path: str, rgba: bool = False) -> tuple[ClipMeta, Iterator[np.ndarray]]:
    """Open a video file and return (ClipMeta, frame_iterator).

    The iterator yields numpy arrays of shape (H, W, 3) for RGB or
    (H, W, 4) for RGBA, dtype uint8.

    Parameters
    ----------
    path:
        Filesystem path to the source video.
    rgba:
        If True, decode to RGBA (4 channels); otherwise RGB (3 channels).
    """
    container = av.open(path)
    stream = container.streams.video[0]

    # fps
    fps: float
    if stream.average_rate is not None and float(stream.average_rate) > 0:
        fps = float(stream.average_rate)
    elif stream.base_rate is not None and float(stream.base_rate) > 0:
        fps = float(stream.base_rate)
    else:
        fps = 25.0  # fallback

    # frame_count: prefer the container-level duration → derived count
    frame_count: int
    if stream.frames and stream.frames > 0:
        frame_count = int(stream.frames)
    elif stream.duration and stream.time_base:
        frame_count = int(float(stream.duration) * float(stream.time_base) * fps)
    elif container.duration:
        frame_count = int(container.duration / 1_000_000 * fps)
    else:
        # Will count during decode; use 0 as placeholder
        frame_count = 0

    width = stream.width
    height = stream.height
    channels = 4 if rgba else 3

    meta = ClipMeta(
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
        channels=channels,
    )

    pixel_format = "rgba" if rgba else "rgb24"

    def _frame_iter() -> Iterator[np.ndarray]:
        try:
            for packet in container.demux(stream):
                for av_frame in packet.decode():
                    rgb_frame = av_frame.reformat(format=pixel_format)
                    arr = rgb_frame.to_ndarray()  # shape: H × W × C
                    yield arr
        finally:
            container.close()

    return meta, _frame_iter()


def probe_video(path: str) -> ClipMeta:
    """Return ClipMeta for a video without decoding frames.

    Useful for planning buffer sizes before committing to a full decode.
    """
    container = av.open(path)
    stream = container.streams.video[0]

    fps: float
    if stream.average_rate is not None and float(stream.average_rate) > 0:
        fps = float(stream.average_rate)
    elif stream.base_rate is not None and float(stream.base_rate) > 0:
        fps = float(stream.base_rate)
    else:
        fps = 25.0

    frame_count: int
    if stream.frames and stream.frames > 0:
        frame_count = int(stream.frames)
    elif stream.duration and stream.time_base:
        frame_count = int(float(stream.duration) * float(stream.time_base) * fps)
    elif container.duration:
        frame_count = int(container.duration / 1_000_000 * fps)
    else:
        frame_count = 0

    meta = ClipMeta(
        fps=fps,
        frame_count=frame_count,
        width=stream.width,
        height=stream.height,
        channels=3,
    )
    container.close()
    return meta
