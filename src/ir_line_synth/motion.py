"""How an object moves under the line: its passage stretched in time, shifted across and dimmed
(docs/objects.md, section 6).

A motion is a JSON-friendly mapping in the units of a frame: rows (samples of `pool` polls)
along time and channels across the line.

    kind        blob (calm), long (slow, may stall), short (fast), rolling (drifts across),
                swinging (rocks, loses contact)
    speed       knots of the speed profile over the natural passage: rows of the natural passage
                per output row (0.5 = twice as slow); evenly spaced, linear between knots
    lateral     shift across the line in every output row, channels
    gain        factor of the reflected signal in every output row (lost contact: near 0)
    flip, own, width, brightness, mode
                used by real objects only (morph.render): mirror across the belt, the share of
                the object's own lateral motion kept, its width and brightness; mode notes the
                kind of rolling

The same motions move real objects (morph.render resamples their pixels) and objects described
by shape (shapes.passage moves the shape under the line). The ranges below were tuned on the
reference line against the real objects of every kind, in its rows (REFERENCE_ROW_MS) and channels
(REFERENCE_PITCH_MM). On another line they keep their duration and distance: a LineScale turns them
into the rows and channels of that line (for the reference line both factors are exactly 1).
"""

from dataclasses import dataclass
import math

import numpy as np


# The reference line: rows of 10 polls at 200 Hz (50 ms), channels 9 mm apart (reference_line_40).
REFERENCE_ROW_MS = 50.0
REFERENCE_PITCH_MM = 9.0
KINDS = ("blob", "rolling", "swinging", "long", "short")
CALM_KINDS = ("blob", "long", "short")
# Draws that change only how a real object is resampled: mirrored with this probability,
# width and brightness factors.
LOOK = dict(flip=0.5, width=(0.92, 1.1), brightness=(0.85, 1.12))
# Duration (rows) of the passage: the natural one times a log-uniform factor, limited to the rows ...
DURATION_SCALE = dict(blob=(0.8, 1.25), rolling=(0.7, 1.4), swinging=(1.0, 1.6))
DURATION_ROWS = dict(blob=(5, 10), long=(11, 16), short=(2.6, 4.4), rolling=(3, 14), swinging=(6, 15))
# ... with a smooth wobble of the speed: knots, correlation (knots), sd of the log speed.
SPEED_KNOTS = 8
SPEED_WOBBLE = dict(correlation=2, size=0.12)
# A slow object may stall: probability, first knot (from, to exclusive), knots slowed, factor.
STALL = dict(share=0.4, first=(1, 7), knots=2, factor=(0.4, 0.7))
# Smooth wandering across (channels) and of the contact (gain) in every motion: correlation (rows), sd.
WANDER = dict(correlation=4, size=0.15)
CONTACT_WOBBLE = dict(correlation=3, size=0.05)
# Calm kinds: share of the own lateral motion kept, steady slant (channels per row).
CALM = dict(own=(0.5, 1.15), slant=(-0.12, 0.12))
# Rolling: sideways speed (channels per row) and its course: straight, a turn from one speed to
# another (still before the turn with probability still_share) or a steady change (arc).
ROLLING = dict(speed=(0.3, 1.6), straight=0.3, turn=0.45, arc=0.25, still_share=0.6, still=(-0.15, 0.15),
               turn_at=(0.25, 0.75), turn_width=(0.5, 2), wobble_correlation=3, wobble_size=0.1, own=(0.7, 1.3))
# Swinging: swing amplitude (channels), swings over the passage, trace width, depth of the contact
# changing with the swing, number of contact losses (from, to exclusive), their length (rows, from,
# to exclusive) and gain, the share of the own lateral motion kept.
SWINGING = dict(amplitude=(0.7, 2.0), swings=(0.7, 2.0), width=(0.7, 0.95), contact_depth=(0, 0.35),
                losses=(0, 3), loss_rows=(1, 3), loss_gain=(0.0, 0.3), own=(0.5, 1.3))
MAX_GAIN = 1.3
# Motions are stored rounded to this many decimals, so a replay from the JSON is exactly the one drawn.
DIGITS = 4


@dataclass(frozen=True)
class LineScale:
    """Rows and channels of a line against those of the reference line.

    time: rows of the line in the duration of one reference row; space: channels of the line in the
    distance of one reference channel. Durations in reference rows are multiplied by `time`,
    distances in reference channels by `space`, sideways speeds (channels per row) by space / time.
    """

    time: float = 1.0
    space: float = 1.0

    @classmethod
    def of(cls, pool, poll_hz, pitch_mm):
        """The scale of a line whose rows are `pool` polls at `poll_hz` and whose channels are `pitch_mm` apart."""
        return cls(REFERENCE_ROW_MS / (pool * 1000.0 / poll_hz), REFERENCE_PITCH_MM / pitch_mm)

    @property
    def speed(self):
        return self.space / self.time

    @staticmethod
    def scaled(pair, factor):
        return pair[0] * factor, pair[1] * factor


def smooth_noise(rng, length, correlation, size=1.0):
    """Zero-mean noise with a correlation length in samples."""
    if length <= 0:
        return np.zeros(0, np.float32)
    raw = rng.standard_normal(length + 6 * max(1, int(correlation)))
    sigma = max(correlation, 0.3)
    radius = max(1, math.ceil(3 * sigma))
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    kernel /= np.sqrt((kernel ** 2).sum())
    noise = np.convolve(raw, kernel, mode="same")[3 * max(1, int(correlation)):][:length]
    return (size * noise).astype(np.float32)


def output_times(start, length, speed):
    """Rows of the natural passage sampled by each output row for a speed profile.

    The natural passage covers rows start ... start + length; `speed` holds rows of it per output
    row at evenly spaced points of that span (with one row of margin either side).
    """
    grid = np.linspace(start - 1.0, start + length, 64)
    v = np.interp(np.linspace(0, 1, len(grid)), np.linspace(0, 1, len(speed)), speed)
    elapsed = np.concatenate([[0], np.cumsum(np.diff(grid) * 2 / (v[1:] + v[:-1]))])
    # The tolerance keeps the length stable against rounding of the speeds.
    count = max(1, int(math.ceil(elapsed[-1] - 1e-6)) + 1)
    return np.interp(np.arange(count, dtype=np.float32), elapsed, grid).astype(np.float32)


def sample(kind, rows, span, rng, source_kind=None, scale=LineScale()):
    """Random motion of `kind` for an object whose natural passage lasts `rows` rows.

    `span` (start, length) is the natural passage in rows, as output_times takes it;
    `source_kind` is the kind of the real object replayed (None for an object described by shape);
    `scale` turns the ranges of the reference line into rows and channels of the line.
    The order of the draws is part of the result: the same seed gives the same motion.
    """
    T, S, V = scale.time, scale.space, scale.speed
    scaled = LineScale.scaled
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {', '.join(KINDS)}")
    log_uniform = lambda low, high: float(math.exp(rng.uniform(math.log(low), math.log(high))))
    params = dict(kind=kind, flip=bool(rng.random() < LOOK["flip"]), own=1.0,
                  width=float(rng.uniform(*LOOK["width"])), brightness=float(rng.uniform(*LOOK["brightness"])))
    # A duration is drawn for every kind, in this order, and the one of `kind` is used.
    targets = {}
    for name in ("blob", "long", "short", "rolling", "swinging"):
        if name in DURATION_SCALE:
            targets[name] = float(np.clip(rows * log_uniform(*DURATION_SCALE[name]), *scaled(DURATION_ROWS[name], T)))
        else:
            targets[name] = rng.uniform(*scaled(DURATION_ROWS[name], T))
    speed = (rows / targets[kind]) * np.exp(smooth_noise(rng, SPEED_KNOTS, SPEED_WOBBLE["correlation"],
                                                         SPEED_WOBBLE["size"]))
    if kind == "long" and rng.random() < STALL["share"]:
        # The object stalls for a while: the same cross-section is polled many times.
        at = rng.integers(*STALL["first"])
        speed[at:at + STALL["knots"]] *= rng.uniform(*STALL["factor"])
    speed = np.round(speed, DIGITS)
    params["speed"] = speed.tolist()
    n = len(output_times(span[0], span[1], speed))
    t = np.arange(n, dtype=np.float32)

    lateral = smooth_noise(rng, n, WANDER["correlation"] * T, WANDER["size"] * S)
    gain = 1 + smooth_noise(rng, n, CONTACT_WOBBLE["correlation"] * T, CONTACT_WOBBLE["size"])
    if kind in CALM_KINDS:
        params["own"] = float(rng.uniform(*CALM["own"]))
        lateral += float(rng.uniform(*scaled(CALM["slant"], V))) * (t - n / 2)
    elif kind == "rolling":
        # Sideways speed in channels per row: constant, a turn from one speed to another
        # (hooks: the object starts or stops rolling) or a steady change (arcs).
        sign = rng.choice([-1, 1])
        first, second = sign * rng.uniform(*scaled(ROLLING["speed"], V)), sign * rng.uniform(*scaled(ROLLING["speed"], V))
        mode = str(rng.choice(["straight", "turn", "arc"], p=[ROLLING["straight"], ROLLING["turn"], ROLLING["arc"]]))
        if mode == "straight":
            drift = np.full(n, first, np.float32)
        elif mode == "turn":
            if rng.random() < ROLLING["still_share"]:
                first = rng.uniform(*scaled(ROLLING["still"], V))
            at, width = rng.uniform(*ROLLING["turn_at"]) * n, rng.uniform(*scaled(ROLLING["turn_width"], T))
            drift = first + (second - first) / (1 + np.exp(-(t - at) / width))
        else:
            drift = first + (second * rng.choice([-1, 1]) - first) * t / max(n - 1, 1)
        drift = drift + smooth_noise(rng, n, ROLLING["wobble_correlation"] * T, ROLLING["wobble_size"] * V)
        lateral += np.cumsum(drift) - drift.sum() / 2
        params["mode"] = mode
        params["own"] = float(rng.uniform(*ROLLING["own"])) if source_kind == "rolling" else 1.0
    else:
        # One or two swings over the passage, a thinner trace and lost contact.
        amplitude = rng.uniform(*scaled(SWINGING["amplitude"], S))
        period = n / rng.uniform(*SWINGING["swings"])
        params["width"] = float(rng.uniform(*SWINGING["width"]))
        lateral += amplitude * np.sin(2 * np.pi * t / period + rng.uniform(0, 2 * np.pi))
        # Rocking: contact (and signal) varies with the swing; some rows lose it.
        gain *= 1 - rng.uniform(*SWINGING["contact_depth"]) * (
            0.5 + 0.5 * np.sin(2 * np.pi * t / period + rng.uniform(0, 2 * np.pi)))
        for _ in range(rng.integers(*SWINGING["losses"])):
            shortest = max(1, round(SWINGING["loss_rows"][0] * T))
            at = int(rng.integers(1, max(2, n - 1)))
            length = int(rng.integers(shortest, max(shortest + 1, round(SWINGING["loss_rows"][1] * T))))
            gain[at:at + length] *= rng.uniform(*SWINGING["loss_gain"])
        params["own"] = float(rng.uniform(*SWINGING["own"])) if source_kind == "swinging" else 1.0
    params["lateral"] = np.round(lateral.astype(float), DIGITS).tolist()
    params["gain"] = np.round(np.clip(gain, 0, MAX_GAIN).astype(float), DIGITS).tolist()
    return params
