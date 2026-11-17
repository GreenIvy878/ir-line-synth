"""New objects from real ones by changing how they moved under the sensor line.

A frame is a space-time scan: in the *time-major* layout used here, rows are
successive polls of the line (time) and columns are the sensors. The shape of
an egg therefore records its motion as much as its size:

- a calm egg passing under the line is a short vertical ellipse (``blob``);
- a slow or stopped egg is stretched in time (``long``), a fast one squashed (``short``);
- an egg drifting across the belt moves across sensors from row to row (``rolling``);
- a rocking or bouncing egg jitters sideways and loses contact for a few polls
  (``swinging``: zigzags and torn blobs).

So a real object is split into its *core* (the cross-section profile it shows the
sensors) and its *motion* (lateral track, speed, contact), and a new object is the
same core rendered with a new motion. Every step is a single resampling of the real
pixels, so texture and levels stay those of the sensor, not of a drawing.

Not every result is plausible; :func:`judge` rejects the ones that leave the range
of real objects, that are not the requested kind, or that are almost a copy.
"""

from dataclasses import dataclass, field
import math

import numpy as np

from .dataset import scale_boxes
from .scene import PreparedObject, integer, number, transform_crop, transform_options
from .sensor import MAX_CHANNELS


TAU = 0.1  # object/background threshold (99 % of background pixels lie below 0.086)
# Replayed and imperfect objects: values below this are interpolation tails and are set to 0.
SIGNAL_FLOOR = 0.012
KINDS = ("blob", "rolling", "swinging", "long", "short")
# Real kinds a new object of each kind may be built from.
SOURCE_KINDS = {
    "blob": ("blob", "long", "short"),
    "long": ("blob", "long"),
    "short": ("blob", "short"),
    "rolling": ("blob", "long", "short", "rolling"),
    "swinging": ("blob", "long", "swinging"),
}
FEATURES = ("h", "w", "area", "peak", "mean", "excursion", "wiggle", "row_width")


# --------------------------------------------------------------------------- pixels

def dilate(mask):
    padded = np.pad(mask, 1)
    result = mask.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            result |= padded[1 + dy:1 + dy + mask.shape[0], 1 + dx:1 + dx + mask.shape[1]]
    return result


def components(mask, minimum=2):
    """Number of 8-connected components with at least `minimum` pixels."""
    seen = np.zeros(mask.shape, bool)
    count = 0
    height, width = mask.shape
    for start in zip(*np.nonzero(mask)):
        if seen[start]:
            continue
        stack, size = [start], 0
        seen[start] = True
        while stack:
            y, x = stack.pop()
            size += 1
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
        count += size >= minimum
    return count


def smooth(values):
    """3 x 3 binomial filter with edge replication."""
    padded = np.pad(values, 1, mode="edge")
    kernel = np.array([1, 2, 1], np.float32) / 4
    rows = sum(k * padded[i:i + values.shape[0], :] for i, k in enumerate(kernel))
    return sum(k * rows[:, i:i + values.shape[1]] for i, k in enumerate(kernel))


def blur_rows(values, sigma):
    """Gaussian blur along time (axis 0); anti-aliasing before a faster replay."""
    if sigma < 0.05:
        return values
    radius = max(1, math.ceil(3 * sigma))
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(values, ((radius, radius), (0, 0)))
    return sum(k * padded[i:i + values.shape[0]] for i, k in enumerate(kernel)).astype(np.float32)


def bilinear(image, ys, xs):
    """Sample `image` at fractional pixel coordinates; zero outside."""
    height, width = image.shape
    padded = np.pad(image, 1)
    ys = np.clip(ys + 1, 0, height + 1 - 1e-6)
    xs = np.clip(xs + 1, 0, width + 1 - 1e-6)
    y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
    y1, x1 = np.minimum(y0 + 1, height + 1), np.minimum(x0 + 1, width + 1)
    fy, fx = ys - y0, xs - x0
    return ((1 - fy) * (1 - fx) * padded[y0, x0] + (1 - fy) * fx * padded[y0, x1]
            + fy * (1 - fx) * padded[y1, x0] + fy * fx * padded[y1, x1]).astype(np.float32)


def trim(layer, margin=1):
    """Crop a layer to its nonzero support plus a margin; returns the layer and its offset."""
    support = layer > 1e-4
    if not support.any():
        return layer[:0, :0], (0, 0)
    ys, xs = np.nonzero(support)
    y0, x0 = max(0, ys.min() - margin), max(0, xs.min() - margin)
    y1, x1 = min(layer.shape[0], ys.max() + 1 + margin), min(layer.shape[1], xs.max() + 1 + margin)
    return layer[y0:y1, x0:x1].copy(), (y0, x0)


# ------------------------------------------------------------------------- analysis

def lateral_track(layer):
    """Weighted sensor centroid of every row in the object's time span (NaN rows interpolated)."""
    mask = layer >= TAU
    rows = np.nonzero(mask.any(1))[0]
    if not len(rows):
        return None, 0
    t0, t1 = rows.min(), rows.max()
    xs = np.arange(layer.shape[1], dtype=np.float32)
    track = np.full(t1 - t0 + 1, np.nan, np.float32)
    for t in range(t0, t1 + 1):
        weights = layer[t] * mask[t]
        if weights.sum() > 0:
            track[t - t0] = float((weights * xs).sum() / weights.sum())
    known = ~np.isnan(track)
    track = np.interp(np.arange(len(track)), np.nonzero(known)[0], track[known]).astype(np.float32)
    return track, t0


def describe(layer):
    """Shape features of an object layer, or None when nothing reaches the threshold."""
    mask = layer >= TAU
    if not mask.any():
        return None
    rows, cols = np.nonzero(mask.any(1))[0], np.nonzero(mask.any(0))[0]
    track, t0 = lateral_track(layer)
    h, w = int(rows.max() - rows.min() + 1), int(cols.max() - cols.min() + 1)
    times = np.arange(h, dtype=np.float32)
    wiggle = 0.0
    if h >= 4:
        fit = np.polyval(np.polyfit(times, track, 2), times)
        wiggle = float(np.std(track - fit))
    span = mask[rows.min():rows.max() + 1]
    return dict(
        h=h, w=w, area=int(mask.sum()), peak=float(layer.max()), mean=float(layer[mask].mean()),
        excursion=float(track.max() - track.min()), wiggle=wiggle,
        components=components(mask), gaps=int((~span.any(1)).sum()),
        row_width=float(span.sum(1).mean()),
    )


# Thresholds of the kind of an object (classify), in channels and polls of the reference line.
SWINGING_MIN_LENGTH = 5     # polls: a wiggle is judged only on a trace at least this long
SWINGING_WIGGLE = 0.55      # channels: std of the lateral track around its quadratic fit
ROLLING_EXCURSION = 2.5     # channels: lateral travel of the track over the passage
ROLLING_WIDTH = 7           # channels: a trace this wide has drifted across the line
LONG_LENGTH = 11            # polls
SHORT_LENGTH = 4            # polls


def classify(features):
    """Heuristic kind of an object from its shape features (see the module docstring)."""
    f = features
    if f["components"] >= 2 or f["gaps"] >= 1 or (f["h"] >= SWINGING_MIN_LENGTH and f["wiggle"] >= SWINGING_WIGGLE):
        return "swinging"
    if f["excursion"] >= ROLLING_EXCURSION or f["w"] >= ROLLING_WIDTH:
        return "rolling"
    if f["h"] >= LONG_LENGTH:
        return "long"
    if f["h"] <= SHORT_LENGTH:
        return "short"
    return "blob"


@dataclass(frozen=True)
class RealObject:
    """One annotated object cut from a real frame, in time-major layout."""
    layer: np.ndarray        # float32 intensities, zero outside the object
    box: np.ndarray          # annotated box in layer pixels, xyxy (x = sensor, y = time)
    source: str
    box_index: int
    split: str
    features: dict
    kind: str
    track: np.ndarray = field(repr=False)  # lateral centroid per row of the signal span
    track_start: int = 0
    clipped: bool = False    # cut by the frame edge, so its shape is incomplete (see is_clipped)


# A box closer than this to an edge of the frame (in cells) is cut by it, as in SensorLine
# (TOUCH_TOLERANCE of sensorline/border.py).
EDGE_TOLERANCE = 1.0


def box_near_edge(box, shape):
    """Whether an xyxy box (x = columns) comes closer than EDGE_TOLERANCE to an edge of a frame of `shape`."""
    size_y, size_x = shape
    gap = min(box[0], box[1], size_x - box[2], size_y - box[3])
    return bool(gap < EDGE_TOLERANCE - 1e-6)


def is_clipped(box, strong, offset, shape):
    """Cut by the frame edge: the box is near an edge, or a pixel of the object above TAU
    (`strong`, a mask whose corner lies at `offset` = (row, column) of the frame) is on the
    first or last row or column of the frame."""
    if box_near_edge(box, shape):
        return True
    ys, xs = np.nonzero(strong)
    if not len(ys):
        return False
    ys, xs = ys + offset[0], xs + offset[1]
    return bool(ys.min() == 0 or xs.min() == 0 or ys.max() == shape[0] - 1 or xs.max() == shape[1] - 1)


def cut_object(pixels, boxes, index, source="", split="train", margin=1):
    """Cut box `index` from a time-major frame; other objects' pixels are left out."""
    # One precision for every caller, so the same box always cuts the same pixels.
    boxes = np.asarray(boxes, np.float64)
    box = boxes[index]
    size_y, size_x = pixels.shape
    x1 = max(0, math.floor(box[0] + 1e-5) - margin)
    y1 = max(0, math.floor(box[1] + 1e-5) - margin)
    x2 = min(size_x, math.ceil(box[2] - 1e-5) + margin)
    y2 = min(size_y, math.ceil(box[3] - 1e-5) + margin)
    crop = pixels[y1:y2, x1:x2].astype(np.float32)
    cy, cx = np.mgrid[y1:y2, x1:x2].astype(np.float32) + 0.5

    def inside(b, grow=0.0):
        return (cx >= b[0] - grow) & (cx <= b[2] + grow) & (cy >= b[1] - grow) & (cy <= b[3] + grow)

    keep = inside(box, margin)
    own = inside(box)
    for other_index, other in enumerate(boxes):
        if other_index != index:
            keep &= ~(inside(other) & ~own)
    core = (crop >= TAU) & keep & own
    if not core.any():
        return None
    layer = np.where(dilate(core) & keep, crop, 0).astype(np.float32)
    features = describe(layer)
    if features is None:
        return None
    track, t0 = lateral_track(layer)
    local_box = np.asarray([box[0] - x1, box[1] - y1, box[2] - x1, box[3] - y1], np.float32)
    clipped = is_clipped(box, layer >= TAU, (y1, x1), pixels.shape)
    return RealObject(layer, local_box, source, index, split, features, classify(features),
                      track, t0, clipped)


def time_major(pixels, boxes, sensor_axis):
    """A frame and its pixel xyxy boxes with rows = time and columns = sensors."""
    if sensor_axis == "x":
        return pixels, boxes
    return pixels.T.copy(), boxes[:, [1, 0, 3, 2]]


def extract_objects(frames, split="train"):
    """All usable objects of a dataset.FrameSet, with kind and features."""
    objects = []
    for index in range(len(frames)):
        pixels = frames.pixels(index)
        boxes = scale_boxes(frames.boxes[index], pixels.shape)
        pixels, boxes = time_major(pixels, boxes, frames.sensor_axis)
        for box_index in range(len(boxes)):
            obj = cut_object(np.clip(pixels, 0, 1), boxes, box_index, frames.paths[index].name, split)
            if obj is not None:
                objects.append(obj)
    return objects


# ----------------------------------------------------------------------- statistics

def tight_box(layer):
    mask = layer >= TAU
    ys, xs = np.nonzero(mask)
    return np.asarray([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], np.float32)


def nearest_distances(points, others=None):
    """Distance from each point to its nearest neighbour (in `others`, or among themselves)."""
    alone = others is None
    others = points if alone else others
    nearest = np.empty(len(points))
    for start in range(0, len(points), 256):
        chunk = np.sqrt(((points[start:start + 256, None] - others[None]) ** 2).sum(-1))
        if alone:
            chunk[np.arange(len(chunk)), start + np.arange(len(chunk))] = np.inf
        nearest[start:start + 256] = chunk.min(1)
    return nearest


# Statistics of the real objects of a kind (fit_statistics): at least this many objects,
# the range of each feature between these percentiles, and the nearest-neighbour limit at this one.
MIN_KIND_OBJECTS = 5
RANGE_PERCENTILES = (1, 99)
NEAREST_PERCENTILE = 95


def fit_statistics(objects):
    """What judging and labelling need from the real objects.

    Objects cut by the frame edge are left out: a replay is always complete. Ranges
    and the nearest-neighbour limit are per kind, since the rare kinds (rolling,
    swinging) are a few percent of the objects and would fall outside global ranges.
    """
    whole = [o for o in objects if not o.clipped]
    table = np.asarray([[o.features[k] for k in FEATURES] for o in whole], np.float64)
    center = np.median(table, 0)
    spread = np.percentile(table, 75, 0) - np.percentile(table, 25, 0)
    spread = np.where(spread > 1e-6, spread, np.maximum(np.std(table, 0), 1e-3))
    scaled = (table - center) / spread
    kinds = np.asarray([o.kind for o in whole])
    by_kind = {}
    for kind in KINDS:
        rows = scaled[kinds == kind]
        if len(rows) < MIN_KIND_OBJECTS:
            continue
        values = table[kinds == kind]
        by_kind[kind] = dict(
            count=int(len(rows)),
            low=np.percentile(values, RANGE_PERCENTILES[0], 0).tolist(),
            high=np.percentile(values, RANGE_PERCENTILES[1], 0).tolist(),
            # Typical distance from a real object of this kind to its nearest real neighbour.
            nearest_limit=float(np.percentile(nearest_distances(rows), NEAREST_PERCENTILE)),
            scaled=rows.tolist(),
        )
    # Annotators' margins: annotated box minus the box of pixels above the threshold.
    margins = np.asarray([o.box - tight_box(o.layer) for o in whole], np.float64)
    return dict(
        features=list(FEATURES),
        center=center.tolist(), spread=spread.tolist(), by_kind=by_kind,
        box_margin=np.median(margins, 0).tolist(),
        texture=texture_profile([o.layer for o in whole]),
        # Cut objects have no kind worth counting: a stub of a long egg reads as blob or short.
        kinds={kind: sum(o.kind == kind for o in whole) for kind in KINDS},
        clipped=sum(o.clipped for o in objects),
    )


TEXTURE_BINS = (0.1, 0.25, 0.5, 0.75, 1.01)


def texture_profile(layers):
    """Std of the fine-scale residual (pixel minus its 3 x 3 smoothing) per intensity band."""
    values, residuals = [], []
    for layer in layers:
        inner = layer >= TAU
        if inner.any():
            values.append(layer[inner])
            residuals.append((layer - smooth(layer))[inner])
    values, residuals = np.concatenate(values), np.concatenate(residuals)
    profile = []
    for low, high in zip(TEXTURE_BINS[:-1], TEXTURE_BINS[1:]):
        band = (values >= low) & (values < high)
        profile.append(float(residuals[band].std()) if band.sum() > 20 else 0.0)
    return profile


# -------------------------------------------------------------------------- motion

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


def output_times(obj, speed):
    """Source rows sampled by each output row for a speed profile over the source span.

    `speed` holds source rows per output row at evenly spaced points of the span
    (with one row of margin either side); 0.5 replays the egg twice as slowly.
    """
    grid = np.linspace(obj.track_start - 1.0, obj.track_start + len(obj.track), 64)
    v = np.interp(np.linspace(0, 1, len(grid)), np.linspace(0, 1, len(speed)), speed)
    elapsed = np.concatenate([[0], np.cumsum(np.diff(grid) * 2 / (v[1:] + v[:-1]))])
    # The tolerance keeps the length stable against rounding of the speeds.
    count = max(1, int(math.ceil(elapsed[-1] - 1e-6)) + 1)
    return np.interp(np.arange(count, dtype=np.float32), elapsed, grid).astype(np.float32)


def sample_motion(kind, obj, rng):
    """Random motion parameters that turn `obj` into an object of `kind` (JSON-friendly)."""
    log_uniform = lambda low, high: float(math.exp(rng.uniform(math.log(low), math.log(high))))
    params = dict(kind=kind, flip=bool(rng.random() < 0.5), own=1.0,
                  width=float(rng.uniform(0.92, 1.1)), brightness=float(rng.uniform(0.85, 1.12)))
    # Speed is chosen for a target duration (height in rows) of the replayed object.
    h = obj.features["h"]
    target = {"blob": float(np.clip(h * log_uniform(0.8, 1.25), 5, 10)),
              "long": rng.uniform(11, 16), "short": rng.uniform(2.6, 4.4),
              "rolling": float(np.clip(h * log_uniform(0.7, 1.4), 3, 14)),
              "swinging": float(np.clip(h * log_uniform(1.0, 1.6), 6, 15))}[kind]
    speed = (h / target) * np.exp(smooth_noise(rng, 8, 2, 0.12))
    if kind == "long" and rng.random() < 0.4:
        # The egg stalls for a while: the same cross-section is polled many times.
        at = rng.integers(1, 7)
        speed[at:at + 2] *= rng.uniform(0.4, 0.7)
    # Rounded before use, so a replay from the JSON parameters is exactly this one.
    speed = np.round(speed, 4)
    params["speed"] = speed.tolist()
    times = output_times(obj, speed)
    n = len(times)
    t = np.arange(n, dtype=np.float32)

    lateral = smooth_noise(rng, n, 4, 0.15)
    gain = 1 + smooth_noise(rng, n, 3, 0.05)
    if kind in ("blob", "long", "short"):
        params["own"] = float(rng.uniform(0.5, 1.15))
        lateral += float(rng.uniform(-0.12, 0.12)) * (t - n / 2)
    elif kind == "rolling":
        # Sideways speed in sensors per poll: constant, a turn from one speed to
        # another (hooks: the egg starts or stops rolling) or a steady change (arcs).
        sign = rng.choice([-1, 1])
        first, second = sign * rng.uniform(0.3, 1.6), sign * rng.uniform(0.3, 1.6)
        mode = str(rng.choice(["straight", "turn", "arc"], p=[0.3, 0.45, 0.25]))
        if mode == "straight":
            drift = np.full(n, first, np.float32)
        elif mode == "turn":
            if rng.random() < 0.6:
                first = rng.uniform(-0.15, 0.15)  # still before the turn
            at, width = rng.uniform(0.25, 0.75) * n, rng.uniform(0.5, 2)
            drift = first + (second - first) / (1 + np.exp(-(t - at) / width))
        else:
            drift = first + (second * rng.choice([-1, 1]) - first) * t / max(n - 1, 1)
        drift = drift + smooth_noise(rng, n, 3, 0.1)
        lateral += np.cumsum(drift) - drift.sum() / 2
        params["mode"] = mode
        params["own"] = float(rng.uniform(0.7, 1.3)) if obj.kind == "rolling" else 1.0
    elif kind == "swinging":
        # One or two swings over the passage, a thinner trace and lost contact.
        amplitude = rng.uniform(0.7, 2.0)
        period = n / rng.uniform(0.7, 2.0)
        params["width"] = float(rng.uniform(0.7, 0.95))
        lateral += amplitude * np.sin(2 * np.pi * t / period + rng.uniform(0, 2 * np.pi))
        # Rocking: contact (and signal) varies with the swing; some polls lose it.
        gain *= 1 - rng.uniform(0, 0.35) * (0.5 + 0.5 * np.sin(2 * np.pi * t / period + rng.uniform(0, 2 * np.pi)))
        for _ in range(rng.integers(0, 3)):
            at, length = int(rng.integers(1, max(2, n - 1))), int(rng.integers(1, 3))
            gain[at:at + length] *= rng.uniform(0.0, 0.3)
        params["own"] = float(rng.uniform(0.5, 1.3)) if obj.kind == "swinging" else 1.0
    params["lateral"] = np.round(lateral.astype(float), 4).tolist()
    params["gain"] = np.round(np.clip(gain, 0, 1.3).astype(float), 4).tolist()
    return params


def render(obj, params, texture=None, rng=None):
    """The object replayed with `params`: one bilinear resampling of the real pixels.

    For output row t' and sensor x', the source row is s = T(t') from the speed
    profile and the source sensor is
        x = c + (x' - c - lateral(t') - (own - 1) * dev(s)) / width,
    where dev is the object's own lateral motion around its mean position c.
    `texture` (real and replayed residual std per band) restores the fine noise
    that interpolation averages away.
    """
    layer, track = obj.layer, obj.track
    if params.get("flip"):  # mirrored across the belt: the egg drifts the other way
        layer, track = layer[:, ::-1], layer.shape[1] - 1 - track
    times = output_times(obj, np.asarray(params["speed"], np.float64))
    lateral = np.asarray(params["lateral"], np.float32)
    gain = np.asarray(params["gain"], np.float32)
    if len(lateral) != len(times) or len(gain) != len(times):
        raise ValueError("Motion arrays do not match the replay length")
    v_max = float(np.max(params["speed"]))
    source = blur_rows(layer, 0.45 * math.sqrt(max(v_max ** 2 - 1, 0)))
    center = float(track.mean())
    deviation = np.interp(times - obj.track_start, np.arange(len(track)), track) - center
    shift = lateral + (params.get("own", 1.0) - 1) * deviation
    pad = int(math.ceil(np.abs(shift).max() + layer.shape[1] * abs(params.get("width", 1) - 1))) + 2
    xs_out = np.arange(layer.shape[1] + 2 * pad, dtype=np.float32) - pad
    xs = center + (xs_out[None, :] - center - shift[:, None]) / params.get("width", 1.0)
    ys = np.repeat(times[:, None], len(xs_out), 1)
    out = bilinear(source, ys, xs) * gain[:, None] * params.get("brightness", 1.0)
    if texture is not None and rng is not None:
        out = add_texture(out, texture, rng)
    out = np.clip(out, 0, 1)
    out[out < SIGNAL_FLOOR] = 0  # interpolation tails below the background level
    layer, _ = trim(out)
    return layer


def add_texture(layer, texture, rng):
    """Add noise so the residual std per band matches the real objects again."""
    real, replayed = texture
    result = layer.copy()
    noise = rng.standard_normal(layer.shape).astype(np.float32)
    for index, (low, high) in enumerate(zip(TEXTURE_BINS[:-1], TEXTURE_BINS[1:])):
        missing = math.sqrt(max(real[index] ** 2 - replayed[index] ** 2, 0))
        band = (layer >= low) & (layer < high)
        result[band] += missing * noise[band]
    return result


# ------------------------------------------------------------------------ judging

def similarity(first, second, size=16):
    """Correlation of two objects resampled to a common square; 1 = same picture.

    Each object is first cut to its threshold box, so empty margins do not count.
    """
    def norm(layer):
        if (layer >= TAU).any():
            x1, y1, x2, y2 = tight_box(layer).astype(int)
            layer = layer[y1:y2, x1:x2]
        h, w = layer.shape
        ys = (np.arange(size) + 0.5) * h / size - 0.5
        xs = (np.arange(size) + 0.5) * w / size - 0.5
        sampled = bilinear(layer, *np.meshgrid(ys, xs, indexing="ij"))
        sampled = sampled - sampled.mean()
        return sampled / (np.linalg.norm(sampled) + 1e-9)
    return float((norm(first) * norm(second)).sum())


# Default range of a good blob's length (time) to width (sensors); the caller may set its own.
# Real blobs: median 2.0, 5-95 % between 1.5 and 3.0, never above 5.
BLOB_RATIO = (1.25, 5.0)
MAX_VALLEY = 0.5        # a trace may not dip below this share of the weaker of its two brighter sides
UNBROKEN_KINDS = ("rolling", "swinging")
# Reasons that remove an object from the offered variants instead of only marking it.
HARD_REASONS = ("ratio", "broken")


def trace_valley(layer):
    """Deepest dip of the object's signal along its time axis, as a share (1 = no dip).

    Each row's strongest pixel is the signal on the object's spine (rows follow the
    lateral track, so this also works for a trace that drifts across the sensors).
    A row counts as a dip against the weaker of the two brightest rows on either side
    of it, so the dim, tapering ends of an egg are not dips and a thin neck or a
    lost-contact tear in the middle is.
    """
    mask = layer >= TAU
    rows = np.nonzero(mask.any(1))[0]
    if not len(rows):
        return 1.0
    peaks = np.asarray([layer[t][mask[t]].max() if mask[t].any() else 0.0
                        for t in range(rows.min(), rows.max() + 1)])
    before = np.maximum.accumulate(peaks)
    after = np.maximum.accumulate(peaks[::-1])[::-1]
    sides = np.minimum(before[:-2], after[2:])
    return float(np.min(peaks[1:-1] / np.maximum(sides, 1e-6))) if len(sides) else 1.0


def blob_ratio_range(low=BLOB_RATIO[0], high=BLOB_RATIO[1]):
    """A checked (low, high) range of blob length to width."""
    low, high = number(low, "Blob ratio from", 0.1, 20), number(high, "Blob ratio to", 0.1, 20)
    if low > high:
        raise ValueError("Blob ratio: the lower bound exceeds the upper one")
    return low, high


def shape_problem(features, layer, kind, blob_ratio=BLOB_RATIO):
    """Why an object of `kind` is not allowed at all (see HARD_REASONS), or None."""
    if kind == "blob" and not blob_ratio[0] <= features["h"] / features["w"] <= blob_ratio[1]:
        return "ratio"
    if kind in UNBROKEN_KINDS and (features["gaps"] or features["components"] >= 2
                                   or trace_valley(layer) < MAX_VALLEY):
        return "broken"
    return None


# The filter (judge): a feature may lie outside the real range of its kind by this share of
# the bound plus this share of the feature's spread; the nearest real object may be this many
# times farther than is typical among real ones; above this similarity to the source it is a copy.
RANGE_MARGIN_BOUND = 0.15
RANGE_MARGIN_SPREAD = 0.25
NEAREST_FACTOR = 1.5
COPY_SIMILARITY = 0.97


def judge(layer, kind, stats, source_layer=None, blob_ratio=BLOB_RATIO):
    """(accepted, reason, features) for a replayed object.

    Reasons: ``empty``; ``ratio`` (a blob whose length to width is outside `blob_ratio`);
    ``broken`` (a rolling or swinging trace torn in the middle);
    ``kind:<other>`` (the replay reads as another kind);
    ``range:<features>`` (outside what real objects of this kind show);
    ``unlike_real`` (no real object of this kind nearby in feature space);
    ``copy`` (same shape as the source up to a resize: only scaled, nothing new).
    """
    features = describe(layer)
    if features is None:
        return False, "empty", None
    problem = shape_problem(features, layer, kind, blob_ratio)
    if problem:
        return False, problem, features
    if classify(features) != kind:
        return False, "kind:" + classify(features), features
    real = stats["by_kind"].get(kind)
    if real is None:  # too few real objects of this kind to compare with
        return True, "ok", features
    vector = np.asarray([features[k] for k in FEATURES], np.float64)
    # A share of the bound plus a share of the feature's spread, so that features
    # near zero (excursion, wiggle) get a margin too.
    a, b = RANGE_MARGIN_BOUND, RANGE_MARGIN_SPREAD
    outside = [name for name, value, lo, hi, spread
               in zip(FEATURES, vector, real["low"], real["high"], stats["spread"])
               if not lo - a * abs(lo) - b * spread <= value <= hi + a * abs(hi) + b * spread]
    if outside:
        return False, "range:" + ",".join(outside), features
    scaled = (vector - np.asarray(stats["center"])) / np.asarray(stats["spread"])
    nearest = float(nearest_distances(scaled[None], np.asarray(real["scaled"]))[0])
    if nearest > NEAREST_FACTOR * real["nearest_limit"]:
        return False, "unlike_real", features
    if source_layer is not None and similarity(layer, source_layer) > COPY_SIMILARITY:
        return False, "copy", features
    return True, "ok", features


def label_box(layer, stats):
    """Box in layer pixels as annotators would draw it: threshold box plus their margins."""
    box = tight_box(layer) + np.asarray(stats["box_margin"], np.float32)
    box[:2] = np.maximum(box[:2], 0)
    box[2:] = np.minimum(box[2:], [layer.shape[1], layer.shape[0]])
    return box


# -------------------------------------------------------------------- imperfections
#
# A replay keeps the pixels of a real egg, and real eggs are often smoother than
# the hardest cases on the belt. These seeded changes are applied after the replay,
# so a candidate is shown, chosen and rendered in a scene with them.

NEUTRAL_IMPERFECTIONS = dict(noise=0.0, hotspot=0.0, hotspot_angle=0.0, hotspot_offset=0.6,
                             stripes=0.0, dropout=0.0, seed=0)
IMPERFECTION_BOUNDS = dict(noise=(0, 1), hotspot=(0, 1), hotspot_angle=(-360, 360),
                           hotspot_offset=(0, 1), stripes=(0, 1), dropout=(0, 1))
LIMIT_KEYS = ("noise_max", "hotspot_max", "stripes_max", "dropout_max")


def imperfection_options(options):
    if not isinstance(options, dict):
        raise ValueError("imperfections must be an object")
    if set(options) - NEUTRAL_IMPERFECTIONS.keys():
        raise ValueError("Unknown imperfection")
    result = {**NEUTRAL_IMPERFECTIONS, **options}
    for key, bounds in IMPERFECTION_BOUNDS.items():
        result[key] = number(result[key], key, *bounds)
    result["seed"] = integer(result["seed"], "imperfection seed", 0, 2**32 - 1)
    return result


def imperfection_limits(limits):
    if not isinstance(limits, dict) or set(limits) - set(LIMIT_KEYS):
        raise ValueError("Unknown imperfection limit")
    return {key: number(limits.get(key, 0.0), key, 0, 1) for key in LIMIT_KEYS}


def sample_imperfections(limits, rng):
    """Strengths drawn uniformly up to each limit, so a batch shows mild and strong cases."""
    return dict(
        noise=float(rng.uniform(0, limits["noise_max"])),
        hotspot=float(rng.uniform(0, limits["hotspot_max"])),
        hotspot_angle=float(rng.uniform(-180, 180)),
        hotspot_offset=float(rng.uniform(0.5, 1.0)),
        stripes=float(rng.uniform(0, limits["stripes_max"])),
        dropout=float(rng.uniform(0, limits["dropout_max"])),
        seed=int(rng.integers(2**32)),
    )


def imperfect(layer, options):
    """Apply imperfections to a time-major layer (zero outside the object).

    - hotspot: the brightest part moves from the centre towards an edge (direction
      `hotspot_angle`, distance `hotspot_offset` of the half size), as when the egg
      is tilted or one end is closer to the sensors; the peak level is kept;
    - stripes: an own gain for every sensor column, like uneven channels;
    - dropout: single pixels lose part of their signal;
    - noise: signal-dependent pixel noise.
    Every effect has its own random stream, so changing one leaves the others as they were.
    """
    options = imperfection_options(options)
    out = layer.astype(np.float32).copy()
    if not (out > 0).any():
        return out
    streams = [np.random.default_rng([options["seed"], k]) for k in range(3)]
    height, width = out.shape
    if options["hotspot"]:
        strong = out >= TAU
        ys, xs = np.nonzero(strong if strong.any() else out > 0)
        weights = out[ys, xs]
        cy, cx = float((ys * weights).sum() / weights.sum()), float((xs * weights).sum() / weights.sum())
        half_y, half_x = max((ys.max() - ys.min() + 1) / 2, 0.5), max((xs.max() - xs.min() + 1) / 2, 0.5)
        angle = math.radians(options["hotspot_angle"])
        hx = cx + options["hotspot_offset"] * half_x * math.cos(angle)
        hy = cy + options["hotspot_offset"] * half_y * math.sin(angle)
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        spot = np.exp(-0.5 * (((xx - hx) / max(0.7, 0.5 * half_x)) ** 2 + ((yy - hy) / max(0.7, 0.5 * half_y)) ** 2))
        peak = float(out.max())
        out *= (1 - options["hotspot"]) + options["hotspot"] * spot
        if out.max() > 0:
            out *= peak / float(out.max())
    if options["stripes"]:
        gains = 1 + 0.6 * options["stripes"] * streams[0].standard_normal(width).astype(np.float32)
        out *= np.clip(gains, 0.3, 2)[None, :]
    if options["dropout"]:
        lost = (out >= TAU / 2) & (streams[1].random(out.shape) < 0.45 * options["dropout"])
        out[lost] *= streams[1].uniform(0.1, 0.6, int(lost.sum())).astype(np.float32)
    if options["noise"]:
        inside = out > 0.015
        noise = streams[2].standard_normal(out.shape).astype(np.float32)
        out[inside] += options["noise"] * 0.25 * np.sqrt(out[inside]) * noise[inside]
    out = np.clip(out, 0, 1)
    out[out < SIGNAL_FLOOR] = 0
    return out


# ---------------------------------------------------------------- scene objects

MOTION_KEYS = {"kind", "flip", "own", "width", "brightness", "speed", "lateral", "gain", "mode"}


def number_list(values, name, low, high, longest):
    if not isinstance(values, list) or not 1 <= len(values) <= longest:
        raise ValueError(f"{name}: expected a list of 1 to {longest} numbers")
    return [number(value, name, low, high) for value in values]


def motion_options(motion):
    """A checked copy of motion parameters as sample_motion produces them."""
    if not isinstance(motion, dict) or set(motion) - MOTION_KEYS:
        raise ValueError("Unknown motion parameter")
    if motion.get("kind") not in KINDS:
        raise ValueError(f"kind must be one of {', '.join(KINDS)}")
    if not isinstance(motion.get("flip", False), bool):
        raise ValueError("flip: expected a boolean")
    result = dict(kind=motion["kind"], flip=motion.get("flip", False),
                  own=number(motion.get("own", 1.0), "own", 0, 3),
                  width=number(motion.get("width", 1.0), "width", 0.5, 2),
                  brightness=number(motion.get("brightness", 1.0), "brightness", 0, 4),
                  speed=number_list(motion.get("speed"), "speed", 0.05, 8, 64),
                  lateral=number_list(motion.get("lateral"), "lateral", -MAX_CHANNELS, MAX_CHANNELS, 256),
                  gain=number_list(motion.get("gain"), "gain", 0, 2, 256))
    if "mode" in motion:
        result["mode"] = str(motion["mode"])[:20]
    return result


def morph_options(spec):
    if not isinstance(spec, dict) or set(spec) - {"kind", "motion", "imperfections"}:
        raise ValueError("A generated object needs kind, motion and imperfections")
    motion = motion_options(spec.get("motion"))
    return dict(kind=motion["kind"], motion=motion,
                imperfections=imperfection_options(spec.get("imperfections", {})))


def morph_layer(pixels, boxes, box_index, sensor_axis, spec):
    """A generated object from box `box_index` of a frame (image orientation in and out)."""
    spec = morph_options(spec)
    frame, frame_boxes = time_major(np.clip(pixels, 0, 1), np.asarray(boxes, np.float64), sensor_axis)
    obj = cut_object(frame, frame_boxes, box_index)
    if obj is None:
        raise ValueError("The source object is too faint to generate from")
    layer = imperfect(render(obj, spec["motion"]), spec["imperfections"])
    layer, _ = trim(layer)
    if not (layer >= TAU).any():
        raise ValueError("The generated object has no pixels above the threshold")
    return layer if sensor_axis == "x" else layer.T.copy()


def prepare_morph(obj, load_source, sensor_axis, margins):
    """A generated scene object: replay + imperfections + the usual transform.

    `margins` are the annotators' box margins in time-major order; the label box is
    the threshold box of the final pixels plus these margins, so it follows any
    brightness, scale or rotation change of the transform.
    """
    source = load_source(obj.get("source"))
    index = integer(obj.get("box_index"), "box_index", 0, len(source.boxes) - 1)
    layer = morph_layer(source.pixels, source.boxes, index, sensor_axis, obj.get("morph"))
    options = transform_options(obj.get("transform", {}))
    pixels, active = transform_crop(layer, layer > 0, options)
    strong = active & (pixels >= TAU)
    if not strong.any():
        raise ValueError("The generated object is below the threshold after its transform")
    ys, xs = np.nonzero(strong)
    margins = np.asarray(margins, np.float64)
    if sensor_axis == "y":
        margins = margins[[1, 0, 3, 2]]
    box = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], np.float64) + margins
    box = np.clip(box, 0, [pixels.shape[1], pixels.shape[0]] * 2)
    if not (box[2:] > box[:2]).all():
        raise ValueError("The generated object has an empty box")
    return PreparedObject(pixels, active, int(source.labels[index]), tuple(float(v) for v in box))


# Draws of new objects before giving up: at least this many, or this many per object asked for.
DRAW_ATTEMPTS_MIN = 200
DRAW_ATTEMPTS_PER_OBJECT = 30


def draw_attempts(count):
    return max(DRAW_ATTEMPTS_MIN, DRAW_ATTEMPTS_PER_OBJECT * count)


@dataclass
class Library:
    """Real objects of one split with their statistics, for generating new ones."""
    split: str
    sensor_axis: str
    objects: list
    stats: dict

    @classmethod
    def build(cls, frames, split):
        objects = extract_objects(frames, split)
        if not objects:
            raise ValueError(f"No usable objects in {split}")
        return cls(split, frames.sensor_axis, objects, fit_statistics(objects))

    def counts(self):
        """Whole (not clipped) real objects of every kind."""
        return dict(self.stats["kinds"])

    def find(self, identifier, box_index):
        return next((o for o in self.objects if o.source == identifier and o.box_index == box_index), None)

    def candidates(self, kind, count, seed, limits, sources=None, blob_ratio=BLOB_RATIO):
        """`count` new objects of `kind`, each with the filter's opinion (see `generate`)."""
        return self.generate(kind, count, seed, limits, sources, blob_ratio)["candidates"]

    def source_pool(self, kind, sources=None):
        """Source objects for `kind` and notes on the ones left out.

        Objects cut by the frame edge are never sources, whatever their kind reads as:
        a stub of a long egg looks like a blob or a short one. Without `sources`, every
        whole object of this split whose kind suits `kind`; with `sources`, the whole
        ones among them, which must all belong to this split.
        """
        notes = []
        if sources is None:
            pool = [o for o in self.objects if not o.clipped and o.kind in SOURCE_KINDS[kind]]
            if not pool:
                raise ValueError(f"No whole source objects for {kind} in {self.split}")
            return pool, notes
        sources = list(sources)
        foreign = sorted({o.split for o in sources if o.split != self.split})
        if foreign:
            raise ValueError(f"Objects for {self.split} are made only from {self.split} objects; "
                             f"the sources include {', '.join(foreign)}")
        pool = [o for o in sources if not o.clipped]
        if not pool:
            raise ValueError("Every chosen source object is cut by the frame edge (a stub); choose whole objects")
        if len(pool) < len(sources):
            notes.append(f"{len(sources) - len(pool)} of {len(sources)} chosen source objects are cut by "
                         "the frame edge (stubs) and were left out")
        return pool, notes

    def generate(self, kind, count, seed, limits, sources=None, blob_ratio=BLOB_RATIO):
        """`count` new objects of `kind`, each with the filter's opinion (not applied).

        Objects that break the hard shape rules (HARD_REASONS) are not offered: they are
        replaced by new draws until `count` are collected. Then the smallest are sifted
        out (see `sift_small`). Only when the attempts run out (the rules suit almost
        nothing from these sources) are there fewer. Returns the candidates, notes and
        the report of the sifting.
        """
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {', '.join(KINDS)}")
        limits = imperfection_limits(limits)
        blob_ratio = blob_ratio_range(*blob_ratio)
        pool, notes = self.source_pool(kind, sources)
        rng = np.random.default_rng(seed)

        def draw(attempts, floor=None):
            """New candidates from the pool, one per attempt that passes the hard rules and
            reaches every threshold of `floor` ({measure: smallest value}, see SIFT_MEASURES)."""
            for _ in range(attempts):
                obj = pool[int(rng.integers(len(pool)))]
                motion = sample_motion(kind, obj, rng)
                imperfections = sample_imperfections(limits, rng)
                layer, _ = trim(imperfect(render(obj, motion), imperfections))
                accepted, reason, features = judge(layer, kind, self.stats, obj.layer, blob_ratio)
                if reason in HARD_REASONS:
                    continue
                f = features or dict(h=0, w=0, area=0, mean=0.0)
                measures = dict(length=int(f["h"]), width=int(f["w"]), area=int(f["area"]),
                                brightness=round(float(f["mean"]), 4))
                if any(measures[key] < value for key, value in (floor or {}).items()):
                    continue
                if layer.size and self.sensor_axis == "y":
                    layer = layer.T.copy()
                yield dict(
                    source=dict(split=obj.split, id=obj.source), box_index=obj.box_index, source_kind=obj.kind,
                    morph=dict(kind=kind, motion=motion, imperfections=imperfections),
                    matrix=layer.tolist(), filter=dict(accepted=accepted, reason=reason),
                    **measures)

        result = []
        for item in draw(draw_attempts(count)):
            result.append(item)
            if len(result) == count:
                break
        result, sifting = sift_small(result, seed, draw, sift_shares(kind))
        if sifting["missing"]:
            floor = ", ".join(f"{key} {value}" for key, value in sifting["floor"].items())
            notes.append(f"Only {sifting['replaced']} of the {sifting['removed']} smallest objects sifted out "
                         f"could be replaced by objects reaching the thresholds ({floor}); "
                         f"the batch has {len(result)} objects")
        return dict(candidates=result, notes=notes, size_filter=sifting, split=self.split, kind=kind)


# The share of a batch sifted out by each measure: the shortest (length = rows above TAU)
# and, separately, the narrowest (width = sensors).
SIFT_SHARE = 0.1
SIFT_SHARES = dict(length=SIFT_SHARE, width=SIFT_SHARE)
# Rolling and swinging traces may still come out small and faint, so for them also the
# smallest by area (pixels above TAU) and the dimmest (mean of the pixels above TAU).
FAINT_SIFT_SHARES = dict(area=0.1, brightness=0.2)
SIFT_MEASURES = ("length", "width", "area", "brightness")
# Report keys of the number removed by each measure (the first two kept from before).
SIFT_REMOVED_KEYS = dict(length="removed_short", width="removed_narrow", area="removed_small",
                         brightness="removed_dim")
SIFT_MIN_BATCH = 10


def sift_shares(kind):
    """The measures and shares sifted out of a batch of `kind`."""
    return {**SIFT_SHARES, **(FAINT_SIFT_SHARES if kind in UNBROKEN_KINDS else {})}


def sift_small(batch, seed, draw, shares=SIFT_SHARES):
    """Remove the smallest objects of a batch of one kind and replace them.

    For each measure of `shares` ({measure: share}, measures of SIFT_MEASURES), the
    floor(share * N) of the N objects lowest by it are removed, an object in several lists
    once; ties are broken at random from `seed`. The lowest value of each measure left is
    its threshold, and `draw(attempts, {measure: threshold})` must give new objects that
    reach all of them; the thresholds are not updated, so the sifting ends. A batch below
    SIFT_MIN_BATCH objects is returned as it is. Returns the new batch and a report.
    """
    total = len(batch)
    applied = total >= SIFT_MIN_BATCH
    counts = {key: int(math.floor(share * total + 1e-9)) if applied else 0 for key, share in shares.items()}
    report = dict(batch=total, share=SIFT_SHARE, shares=dict(shares), removed=0, replaced=0, missing=0,
                  applied=applied, floor=None, min_length=None, min_width=None)
    report.update({SIFT_REMOVED_KEYS[key]: count for key, count in counts.items()})
    if not applied:
        return batch, report
    ties = np.random.default_rng([seed, 0x51F7])
    removed = set()
    for key, count in counts.items():
        values = np.asarray([item[key] for item in batch])
        # Sorted by the measure, and among equal values by a seeded random key.
        removed |= set(np.lexsort((ties.random(total), values))[:count].tolist())
    kept = [item for index, item in enumerate(batch) if index not in removed]
    floor = {key: min(item[key] for item in kept) for key in shares}
    replacements = []
    for item in draw(draw_attempts(len(removed)), floor):
        replacements.append(item)
        if len(replacements) == len(removed):
            break
    report.update(removed=len(removed), floor=floor, replaced=len(replacements),
                  missing=len(removed) - len(replacements))
    report.update({f"min_{key}": value for key, value in floor.items()})
    return kept + replacements, report
