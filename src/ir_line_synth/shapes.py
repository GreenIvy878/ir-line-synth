"""Objects described by their shape: a superquadric lying on the belt (docs/objects.md).

The chain for every poll t and channel c:

    shape and pose      ->  height h of the top surface above the belt, mm
    motion              ->  the pose in every poll (motion.py; passage)
    surface             ->  roughness added to h; reflectance rho of every point of the object
    installation        ->  the belt point under channel c in poll t: x_t along the belt, y_c across;
                            distance d = H - h(x_t, y_c) from the line to the surface
    sensor model        ->  signal of the channel, noise, ADC, correction (sensor_model.py)

The line looks straight down and every channel sees one point of the surface: the spot of
the light is not modelled (docs/sensor_model.md, section 10). A part of the object under its
widest outline is hidden from the line and does not matter.
"""

from dataclasses import dataclass
import math
from statistics import NormalDist

import numpy as np

from . import motion
from .scene import integer, number


# Roundness of the superquadric: near 0 square edges (a box), 1 an ellipse, 2 a diamond.
ROUNDNESS_RANGE = (0.1, 2.0)
MAX_SIZE_MM = 1000.0
MAX_SPEED_MM_S = 10000.0
MAX_POLL_HZ = 100000.0
# Smooth random fields of the surface (docs/objects.md, section 3).
FIELD_WAVES = 64
FIELD_CLIP = 5.0
MAX_ROUGHNESS_SHARE = 0.05
# Step of the numerical slope of the top surface (tilt_cosine), mm.
TILT_STEP = 0.05
MAX_TILT_POWER = 10.0
# Largest margin of a label box beyond the object, frame cells.
MAX_MARGIN = 100.0


@dataclass(frozen=True)
class Superquadric:
    """A superquadric lying on the belt: full extents in mm and two roundness exponents.

    Its surface: (|u/a|^(2/e2) + |v/b|^(2/e2))^(e2/e1) + |w/c|^(2/e1) = 1 with half extents
    a = length/2 (along u), b = width/2 (along v), c = height/2 (up), the bottom on the belt.
    `vertical` (e1) rounds the profile seen from the side, `horizontal` (e2) the outline seen
    from above: both 1 give an ellipsoid (a sphere for equal extents), both small a box,
    small e1 with e2 = 1 a cylinder standing on its base.
    """

    length: float
    width: float
    height: float
    vertical: float = 1.0
    horizontal: float = 1.0

    def __post_init__(self):
        for name in ("length", "width", "height"):
            number(getattr(self, name), f"shape {name} (mm)", 1e-3, MAX_SIZE_MM)
        for name in ("vertical", "horizontal"):
            number(getattr(self, name), f"shape {name} roundness", *ROUNDNESS_RANGE)

    def top(self, u, v):
        """Height of the top surface above the belt at object coordinates u, v (mm); 0 off the object."""
        a, b, c = self.length / 2, self.width / 2, self.height / 2
        # Points beyond twice the extent are off the object anyway; the cap keeps the powers finite.
        across = (np.minimum(np.abs(u) / a, 2.0) ** (2 / self.horizontal)
                  + np.minimum(np.abs(v) / b, 2.0) ** (2 / self.horizontal))
        radial = across ** (self.horizontal / self.vertical)
        inside = radial <= 1
        return np.where(inside, c + c * np.maximum(1 - radial, 0.0) ** (self.vertical / 2), 0.0)


@dataclass(frozen=True)
class Pose:
    """Where an object lies on the belt: centre x along the belt, y across from the middle of the line (mm),
    and the angle of its length to the direction of the belt (degrees, counter-clockwise from above)."""

    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0


def object_coordinates(pose, x, y):
    """Belt points x, y (mm) in the coordinates of the object: u along its length, v along its width."""
    angle = math.radians(pose.yaw)
    dx, dy = np.asarray(x, np.float64) - pose.x, np.asarray(y, np.float64) - pose.y
    return math.cos(angle) * dx + math.sin(angle) * dy, -math.sin(angle) * dx + math.cos(angle) * dy


def height(shape, pose, x, y):
    """Height of the object above the belt at belt points x, y (mm, any equal shapes)."""
    return shape.top(*object_coordinates(pose, x, y))


class SmoothField:
    """A smooth random field of unit variance on the surface of an object, defined at any point.

    The sum of FIELD_WAVES cosines with random directions, frequencies and phases: its correlation
    between points r mm apart is exp(-r^2 / (2 size^2)), its values are close to normal.
    """

    def __init__(self, size_mm, rng):
        self.waves = rng.standard_normal((FIELD_WAVES, 2)) / size_mm
        self.phases = rng.uniform(0, 2 * math.pi, FIELD_WAVES)

    def __call__(self, u, v):
        argument = np.multiply.outer(u, self.waves[:, 0]) + np.multiply.outer(v, self.waves[:, 1]) + self.phases
        return math.sqrt(2 / FIELD_WAVES) * np.cos(argument).sum(axis=-1)


@dataclass(frozen=True)
class Spots:
    """Patches of the surface whose reflectance is `factor` times that around them.

    coverage: the share of the surface in patches; size_mm: the correlation length of the field
    that outlines them. factor < 1: dark patches (0: dropouts, the light is lost); > 1: glints.
    """

    coverage: float
    factor: float
    size_mm: float

    def __post_init__(self):
        number(self.coverage, "spots coverage", 0, 0.99)
        number(self.factor, "spots factor", 0, 100)
        number(self.size_mm, "spots size_mm", 1e-3, MAX_SIZE_MM)


@dataclass(frozen=True)
class Surface:
    """What the line sees of the surface of an object besides its shape.

    reflectance: relative to the white paper of the calibration, the median over the surface;
    reflectance_spread: the sd of its natural logarithm, smooth over spread_size_mm;
    roughness_mm: the sd of the relief added to the height, smooth over roughness_size_mm;
    spots: patches of other reflectance; seed: of all the fields of the surface;
    tilt_power: the reflected signal falls as cos(tilt) ** tilt_power, the tilt being the angle of the
    surface to the flat paper of the calibration (0: no dependence, 1: a Lambertian surface).
    """

    reflectance: float = 1.0
    reflectance_spread: float = 0.0
    spread_size_mm: float | None = None
    roughness_mm: float = 0.0
    roughness_size_mm: float | None = None
    spots: tuple = ()
    seed: int = 0
    tilt_power: float = 0.0

    def __post_init__(self):
        number(self.reflectance, "surface reflectance", 0, 100)
        number(self.reflectance_spread, "surface reflectance_spread", 0, 3)
        number(self.roughness_mm, "surface roughness_mm", 0, MAX_SIZE_MM)
        for amount, size in (("reflectance_spread", "spread_size_mm"), ("roughness_mm", "roughness_size_mm")):
            if getattr(self, amount) > 0:
                if getattr(self, size) is None:
                    raise ValueError(f"surface {amount} needs {size}")
                number(getattr(self, size), f"surface {size}", 1e-3, MAX_SIZE_MM)
        if not all(isinstance(item, Spots) for item in self.spots):
            raise ValueError("surface spots must be Spots")
        integer(self.seed, "surface seed", 0, 2**32 - 1)
        number(self.tilt_power, "surface tilt_power", 0, MAX_TILT_POWER)


def surface(shape, pose, finish, x, y):
    """Heights above the belt (mm) and reflectance of the object at belt points x, y.

    The fields of the surface are fixed to the object: they move and turn with it.
    The reflectance off the object is that of the object too; line_signal takes the belt's there.
    """
    return surface_at(shape, finish, *object_coordinates(pose, x, y))


def tilt_cosine(shape, u, v):
    """cos of the tilt of the top surface of `shape` at object coordinates u, v: 1 where it is level,
    towards 0 at the outline, where it turns vertical. The slope is the central difference over TILT_STEP;
    the relief of roughness is not part of it."""
    du = (shape.top(u + TILT_STEP, v) - shape.top(u - TILT_STEP, v)) / (2 * TILT_STEP)
    dv = (shape.top(u, v + TILT_STEP) - shape.top(u, v - TILT_STEP)) / (2 * TILT_STEP)
    return 1 / np.sqrt(1 + du ** 2 + dv ** 2)


def surface_at(shape, finish, u, v):
    """Heights above the belt (mm) and reflectance of the object at object coordinates u, v."""
    if finish.roughness_mm > MAX_ROUGHNESS_SHARE * shape.height:
        raise ValueError(f"Roughness {finish.roughness_mm:g} mm is more than {MAX_ROUGHNESS_SHARE:g} "
                         f"of the height of the object ({shape.height:g} mm)")
    heights = shape.top(u, v)
    on = heights > 0
    u, v = u[on], v[on]
    streams = [np.random.default_rng(child) for child in np.random.SeedSequence(finish.seed).spawn(2 + len(finish.spots))]
    if finish.roughness_mm > 0:
        relief = np.clip(SmoothField(finish.roughness_size_mm, streams[0])(u, v), -FIELD_CLIP, FIELD_CLIP)
        heights[on] += finish.roughness_mm * relief
    reflectance = np.full(heights.shape, finish.reflectance)
    if finish.reflectance_spread > 0:
        spread = SmoothField(finish.spread_size_mm, streams[1])(u, v)
        reflectance[on] *= np.exp(finish.reflectance_spread * spread)
    for spots, stream in zip(finish.spots, streams[2:]):
        if spots.coverage > 0:
            inside = SmoothField(spots.size_mm, stream)(u, v) < NormalDist().inv_cdf(spots.coverage)
            reflectance[on] *= np.where(inside, spots.factor, 1.0)
    if finish.tilt_power > 0:
        reflectance[on] *= tilt_cosine(shape, u, v) ** finish.tilt_power
    return heights, reflectance


@dataclass(frozen=True)
class Installation:
    """The line over the belt.

    height_mm: from the line to the belt surface; pitch_mm: between neighbouring channels;
    poll_hz: polls of the line per second; belt_reflectance: of the belt relative to the white
    paper of the calibration (the reflectance of the response law, docs/sensor_model.md, section 3).
    """

    height_mm: float
    pitch_mm: float
    poll_hz: float
    belt_reflectance: float

    def __post_init__(self):
        number(self.height_mm, "installation height_mm", 1e-3, MAX_SIZE_MM)
        number(self.pitch_mm, "installation pitch_mm", 1e-3, MAX_SIZE_MM)
        number(self.poll_hz, "installation poll_hz", 1e-3, MAX_POLL_HZ)
        number(self.belt_reflectance, "installation belt_reflectance", 0, 100)

    def channel_positions(self, channels):
        """y of every channel across the belt, mm; the middle of the line is 0."""
        return (np.arange(channels) - (channels - 1) / 2) * self.pitch_mm


def belt_points(installation, channels, polls, speed_mm_s):
    """Belt coordinates x, y [polls, channels] under every channel in every poll, mm.

    The belt moves at a constant speed; in poll 0 the line is over x = 0.
    """
    number(speed_mm_s, "belt speed (mm/s)", 0, MAX_SPEED_MM_S)
    x = np.arange(polls, dtype=np.float64) * (speed_mm_s / installation.poll_hz)
    y = installation.channel_positions(channels)
    return np.broadcast_to(x[:, None], (polls, channels)), np.broadcast_to(y[None, :], (polls, channels))


def belt_base(model, installation):
    """The part of the signal of every channel that is not reflected by the surface under it, codes.

    With a measured bare belt (the operation section of the profile) the base makes the bare belt
    give exactly its measured level: base_c = belt_level_c - rho_belt * reflected_c(H). Without one
    it is the offset of the calibration.
    """
    if model.operation is None:
        return model.response.offset
    belt = model.response.reflected(np.full(model.channels, installation.height_mm), installation.belt_reflectance)
    return model.operation.belt_level - belt


def line_signal(model, installation, heights, reflectance):
    """Noise-free signal [polls, channels] of surfaces at `heights` above the belt (mm, 0 = bare belt).

    `reflectance` is that of the object (a number or an array like `heights`); the bare belt has the
    reflectance of the installation.
    """
    heights = np.asarray(heights, np.float64)
    if heights.ndim != 2 or heights.shape[1] != model.channels:
        raise ValueError(f"Expected heights [polls, {model.channels}]")
    distance = installation.height_mm - heights
    closest = model.response.valid_mm[0]
    if distance.min() < closest:
        raise ValueError(f"An object {heights.max():g} mm high comes within {distance.min():g} mm of the line; "
                         f"the response is valid from {closest:g} mm (line {installation.height_mm:g} mm over the belt)")
    rho = np.where(heights > 0, np.asarray(reflectance, np.float64), installation.belt_reflectance)
    return belt_base(model, installation) + model.response.reflected(distance, rho)


def line_codes(model, signal, rng):
    """Codes [polls, channels]: sensor noise, the noise of the installation when the profile has one,
    ADC and correction."""
    noisy = model.noise.sample(signal, rng)
    if model.operation is not None:
        noisy = noisy + model.operation.sample(len(noisy), model.boards, rng)
    return model.correct(model.adc(noisy))


def natural_rows(shape, model, installation, speed_mm_s, yaw=0.0):
    """Rows of `model.frames.pool` polls from the row the object reaches the line to the row it leaves,
    at the belt speed: the natural passage that a motion stretches (motion.sample, span (0, rows)).

    The extent along the belt is that of the rectangle around the outline: exact for a box, a little
    longer for rounded outlines (empty rows at the ends).
    """
    number(speed_mm_s, "belt speed (mm/s)", 1e-3, MAX_SPEED_MM_S)
    angle = math.radians(yaw)
    extent = shape.length * abs(math.cos(angle)) + shape.width * abs(math.sin(angle))
    return extent / (speed_mm_s / installation.poll_hz * model.frames.pool)


def passage(shape, finish, move, model, installation, speed_mm_s, y=0.0, yaw=0.0):
    """Heights (mm) and reflectance [polls, channels] of an object passing under the line with motion `move`.

    `move` is a motion (motion.sample) drawn for the natural passage of the object (natural_rows,
    span (0, rows)). In every output row it gives the row of the natural passage under the line, the
    shift across the line (channels, times the pitch) and the gain of the contact, which multiplies
    the reflectance of the object; between rows the polls are interpolated. `flip` mirrors the
    passage across the belt. `own`, `width` and `brightness` concern real objects only and are not used.
    The object does not turn while it moves: rolling is a drift across, not a rotation of the surface.
    """
    rows = natural_rows(shape, model, installation, speed_mm_s, yaw)
    times = motion.output_times(0.0, rows, np.asarray(move["speed"], np.float64))
    count = len(times)
    if len(move["lateral"]) != count or len(move["gain"]) != count:
        raise ValueError(f"The motion was drawn for another passage ({len(move['lateral'])} rows, this one "
                         f"takes {count}); draw it with motion.sample for {rows:g} rows")
    pool = model.frames.pool
    row = np.arange(count * pool, dtype=np.float64) / pool
    natural = np.interp(row, np.arange(count), times)
    shift = np.interp(row, np.arange(count), move["lateral"]) * installation.pitch_mm
    gain = np.interp(row, np.arange(count), move["gain"])
    sign = -1.0 if move.get("flip") else 1.0
    angle = math.radians(sign * yaw)
    row_mm = speed_mm_s / installation.poll_hz * pool
    # The belt point under every channel relative to the centre of the object. Like the rows of a real
    # trace, row k of the natural passage covers k - 0.5 ... k + 0.5: the object lies over -0.5 ... rows - 0.5,
    # and output_times adds half a row of margin either side.
    dx = ((natural + 0.5) * row_mm - rows * row_mm / 2)[:, None]
    dy = installation.channel_positions(model.channels)[None, :] - sign * (y + shift)[:, None]
    u = math.cos(angle) * dx + math.sin(angle) * dy
    v = -math.sin(angle) * dx + math.cos(angle) * dy
    heights, reflectance = surface_at(shape, finish, u, v)
    return heights, reflectance * gain[:, None]


def geometric_box(heights, pool):
    """The box of the object in the frame (x1, y1, x2, y2), time-major: x across the channels, y along
    the rows of `pool` polls, in cells (cell (row, channel) spans row ... row + 1, channel ... channel + 1).

    A cell belongs to the object when the object is under that channel in at least one poll of the row:
    the box is exact for what the line samples, whatever the reflectance (a dropout is still the object).
    None when the object is under no channel.
    """
    heights = np.asarray(heights)
    if heights.ndim != 2 or len(heights) % pool:
        raise ValueError(f"Expected heights [polls, channels] with whole rows of {pool} polls")
    covered = (heights > 0).reshape(-1, pool, heights.shape[1]).any(axis=1)
    if not covered.any():
        return None
    rows, channels = np.nonzero(covered)
    return np.asarray([channels.min(), rows.min(), channels.max() + 1, rows.max() + 1], np.float64)


def checked_margins(margins):
    """Four margins (x1, y1, x2, y2) in cells, each within MAX_MARGIN."""
    if not isinstance(margins, (list, tuple, np.ndarray)) or len(margins) != 4:
        raise ValueError("margins: expected four numbers (x1, y1, x2, y2), cells")
    return [number(float(value), "margin", -MAX_MARGIN, MAX_MARGIN) for value in margins]


def label_box(box, margins, shape):
    """The label box: the geometric box plus annotators' margins, inside a frame of `shape` (rows, channels).

    `margins` (x1, y1, x2, y2) are added to the box in cells, as morph.fit_statistics measures them on
    real objects (annotated box minus the threshold box): a negative x1 or y1 and a positive x2 or y2
    widen the box. Zero margins give the exact geometric box.
    """
    result = np.asarray(box, np.float64) + checked_margins(margins)
    result[:2] = np.maximum(result[:2], 0)
    result[2:] = np.minimum(result[2:], [shape[1], shape[0]])
    if not (result[2:] > result[:2]).all():
        raise ValueError("The margins leave the label box empty")
    return result
