"""New empty backgrounds made from the empty parts of real frames.

Each column of a frame is one sensor channel over time, and channels differ: an
own offset, gain and noise give real frames their vertical stripes. A new
background takes blocks of adjacent columns from a random real frame where those
columns contain no annotated object, keeping every channel at its own position.
Blocks are shifted in time (circularly) so the same frame never contributes the
same rows twice in the same place.

A tape (a long background) belongs to one set of sensors, so a channel keeps its
offset along the whole tape: the first piece of a block sets the offsets of its
channels, and every later piece, taken from another frame, only brings its noise:
its own mean is replaced by those offsets. Pieces join without a step.

The whole background is finally limited to a spread (largest minus smallest value)
of at most 3% of the full scale, never more, in whole 8-bit levels.

Generated backgrounds are referenced as {"split": ..., "generated": {options}} and
rebuilt from the options (seed included) whenever a scene is rendered. The frames that
gave pieces are recorded (`used`), so the provenance of a scene names them.
"""

import numpy as np

from .dataset import scale_boxes
from .scene import LEVELS, integer, number
from .sensor import MAX_CHANNELS


# The largest allowed spread of a generated background, as a part of the full scale.
MAX_SPREAD = 0.03
# Percentiles of the background kept inside the spread; the rest is clipped (limit_spread).
SPREAD_PERCENTILES = (0.5, 99.5)
# Pixels around an annotated box that are not empty either: the soft edge of the object.
OBJECT_MARGIN = 1
DEFAULTS = dict(seed=0, block_min=2, block_max=8, level=1.0, roll=True, spread=MAX_SPREAD)


def background_options(options):
    if not isinstance(options, dict) or set(options) - DEFAULTS.keys():
        raise ValueError("Unknown background option")
    result = {**DEFAULTS, **options}
    result["seed"] = integer(result["seed"], "background seed", 0, 2**32 - 1)
    # A block wider than the line is cut at its last channel.
    result["block_min"] = integer(result["block_min"], "block_min", 1, MAX_CHANNELS)
    result["block_max"] = integer(result["block_max"], "block_max", 1, MAX_CHANNELS)
    if result["block_min"] > result["block_max"]:
        raise ValueError("block_min must not exceed block_max")
    result["level"] = number(result["level"], "level", 0, 3)
    result["spread"] = number(result["spread"], "spread", 0, MAX_SPREAD)
    if not isinstance(result["roll"], bool):
        raise ValueError("roll: expected a boolean")
    return result


def identifier(options):
    return f"generated-{options['seed']}"


def limit_spread(pixels, spread):
    """Pixels with a spread (max - min) of at most `spread`, also after rounding to 8-bit levels.

    The values are squeezed towards their median so that all between SPREAD_PERCENTILES fit,
    and the rest is clipped.
    """
    # Whole levels: a range of n levels apart never rounds to more than n.
    span = np.floor(spread * LEVELS + 1e-9) / LEVELS
    pixels = np.asarray(pixels, dtype=np.float64)
    center = float(np.median(pixels))
    low, high = np.percentile(pixels, SPREAD_PERCENTILES)
    scale = min(1.0, span / (high - low)) if high > low else 1.0
    half = max(span / 2 - 1e-6, 0.0)
    return np.clip(center + (pixels - center) * scale, center - half, center + half)


class BackgroundBank:
    """Real frames of one split (time-major) and which of their columns are empty."""

    def __init__(self, frames):
        self.sensor_axis = frames.sensor_axis
        # File names of the frames (any frame set without paths gets positional names).
        paths = getattr(frames, "paths", None) or []
        self.names = [path.name for path in paths] or [f"frame_{index}" for index in range(len(frames))]
        stack, free = [], []
        for index in range(len(frames)):
            pixels = np.clip(frames.pixels(index), 0, 1)
            boxes = scale_boxes(frames.boxes[index], pixels.shape)
            if self.sensor_axis == "y":
                pixels, boxes = pixels.T, boxes[:, [1, 0, 3, 2]]
            occupied = np.zeros(pixels.shape, bool)
            for x1, y1, x2, y2 in boxes:
                # A margin: the soft edge of an object lies outside its box.
                m = OBJECT_MARGIN
                occupied[max(0, int(np.floor(y1)) - m):int(np.ceil(y2)) + m,
                         max(0, int(np.floor(x1)) - m):int(np.ceil(x2)) + m] = True
            stack.append(pixels.astype(np.float32))
            free.append(occupied)
        self.frames = np.stack(stack)  # [frames, polls, channels]
        self.occupied = np.stack(free)
        self.frame_length, self.channels = self.frames.shape[1:]
        self.free_columns = ~self.occupied.any(1)  # [frames, columns]

    def column_block(self, columns, rng, used=None):
        """A block of columns from one frame where all of them are empty, if any.

        `used`, a set, receives the index of the frame the block comes from.
        """
        candidates = np.nonzero(self.free_columns[:, columns].all(1))[0]
        if len(candidates):
            frame = int(rng.choice(candidates))
            if used is not None:
                used.add(frame)
            return self.frames[frame][:, columns].copy()
        return None

    def filled_column(self, column, rng, used=None):
        """Fallback for a channel that always holds an object: its emptiest frame,
        occupied rows replaced by rows of the same column that are empty."""
        free_rows = (~self.occupied[:, :, column]).sum(1)
        frame = int(rng.choice(np.nonzero(free_rows == free_rows.max())[0]))
        if used is not None:
            used.add(frame)
        values = self.frames[frame, :, column].copy()
        empty = ~self.occupied[frame, :, column]
        if empty.any() and not empty.all():
            values[~empty] = rng.choice(values[empty], int((~empty).sum()))
        return values

    def pick_block(self, x, options, rng, used=None):
        """The next block of channels from column x: (width, rows, from_free_frames).

        A block of several channels comes from one frame where all of them are empty;
        a channel that is never empty falls back to filled_column.
        """
        width = min(int(rng.integers(options["block_min"], options["block_max"] + 1)), self.channels - x)
        block = None
        while block is None and width >= 1:
            block = self.column_block(list(range(x, x + width)), rng, used)
            if block is None:
                width = width // 2 if width > 1 else 0
        if block is None:
            return 1, self.filled_column(x, rng, used)[:, None], False
        return width, block, True

    @staticmethod
    def finish(pixels, options):
        """The level of the background, then the limit of its spread."""
        return np.clip(limit_spread(pixels * options["level"], options["spread"]), 0, 1).astype(np.float32)

    def frame_names(self, used):
        """File names of the frames with the indices in `used`, in frame order."""
        return [self.names[index] for index in sorted(used)]

    def generate(self, options, used=None):
        """A new background in image orientation; `used` receives the indices of its frames."""
        options = background_options(options)
        rng = np.random.default_rng(options["seed"])
        out = np.zeros((self.frame_length, self.channels), np.float32)
        x = 0
        while x < self.channels:
            width, block, _ = self.pick_block(x, options, rng, used)
            if options["roll"]:
                block = np.roll(block, int(rng.integers(self.frame_length)), axis=0)
            out[:, x:x + width] = block
            x += width
        out = self.finish(out, options)
        return out if self.sensor_axis == "x" else out.T.copy()

    def generate_tape(self, options, length, used=None):
        """A long background, time-major: `length` rows (polls) of all channels.

        The blocks of channels are chosen once for the whole tape, so a channel keeps
        its own offset along it. In time every block is a chain of pieces of a frame's length,
        each from another real frame where those channels are empty; the pieces bring
        their noise, and the offsets stay those of the first piece.
        """
        options = background_options(options)
        length = integer(length, "length", 1, 10 ** 6)
        rng = np.random.default_rng(options["seed"])
        out = np.zeros((length, self.channels), np.float32)
        x = 0
        while x < self.channels:
            width, block, free = self.pick_block(x, options, rng, used)
            columns = list(range(x, x + width))
            # The offsets of these channels for the whole tape.
            offsets = block.mean(0, keepdims=True)
            row = 0
            while row < length:
                if options["roll"]:
                    block = np.roll(block, int(rng.integers(self.frame_length)), axis=0)
                rows = min(self.frame_length, length - row)
                out[row:row + rows, x:x + width] = block[:rows]
                row += rows
                if row < length:
                    piece = (self.column_block(columns, rng, used) if free
                             else self.filled_column(x, rng, used)[:, None])
                    block = piece - piece.mean(0, keepdims=True) + offsets
            x += width
        return self.finish(out, options)
