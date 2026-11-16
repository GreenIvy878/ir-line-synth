"""Frames from polls: how a recording of codes becomes the frames of a dataset.

1. the mean of every `pool` consecutive polls, counted from the first poll;
2. consecutive tiles of `length` such samples, without overlap; polls after the last
   whole tile are dropped;
3. normalization of every tile: floor((x - min) / (max - min) * 255) over the whole tile
   (`tile_minmax`, 8-bit levels), or none (the samples in codes).

For the reference line this reproduces the frames of its dataset pixel for pixel (see
docs/sensor_validation.md), so the arithmetic is kept exactly: float64 means, the
difference divided by the span, then multiplied by 255.
"""

import numpy as np


FRAME_LEVELS = 255


def read_recording(path, channels):
    """[polls, channels] codes of a recording CSV: a row per poll, no header.

    A byte order mark, empty cells and cells after the last channel are ignored.
    """
    rows = []
    with open(path, encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            cells = [cell for cell in line.strip().split(",") if cell.strip() != ""]
            if not cells:
                continue
            if len(cells) < channels:
                raise ValueError(f"{path}:{line_number}: {len(cells)} values, expected {channels}")
            rows.append([float(cell) for cell in cells[:channels]])
    return np.asarray(rows, np.float64).reshape(-1, channels)


def tiles(polls, pool, length):
    """Samples [tiles, length, channels]: means of `pool` polls, cut into tiles."""
    polls = np.asarray(polls, np.float64)
    count = len(polls) // (pool * length)
    channels = polls.shape[1]
    return polls[:count * pool * length].reshape(count, length, pool, channels).mean(axis=2)


def normalize(samples, mode):
    """Frames from samples [tiles, length, channels]; with tile_minmax also the low and span of every tile."""
    if mode == "none":
        return samples, None, None
    if mode != "tile_minmax":
        raise ValueError("normalize must be tile_minmax or none")
    flat = samples.reshape(len(samples), -1)
    low = flat.min(axis=1, keepdims=True)
    span = flat.max(axis=1, keepdims=True) - low
    pixels = np.floor((flat - low) / np.where(span > 0, span, 1) * FRAME_LEVELS)
    frames = np.clip(pixels, 0, FRAME_LEVELS).astype(np.uint8).reshape(samples.shape)
    return frames, low[:, 0], span[:, 0]


def frames_from_polls(polls, spec):
    """Frames [tiles, length, channels] (time-major) of a recording by a FrameSpec."""
    frames, _, _ = normalize(tiles(polls, spec.pool, spec.length), spec.normalize)
    return frames
