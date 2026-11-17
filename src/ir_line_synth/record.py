"""A synthetic recording of the line without real data: objects described by shape pass under it
(docs/objects.md, section 8).

A record is described by a JSON-friendly mapping and renders again from it exactly:

    profile         name of the sensor profile (the model is passed to render_record)
    installation    height_mm, pitch_mm, poll_hz, belt_reflectance (shapes.Installation)
    speed_mm_s      speed of the belt
    rows            length of the record in rows of `pool` polls (the pool of the profile)
    seed            of the noise of the sensor and of the installation
    margins         annotators' margins of the label boxes (shapes.label_box), cells
    objects         every object: shape, surface, motion (drawn for its natural passage),
                    row (where its passage starts), y (mm across, 0 = middle of the line), yaw (degrees),
                    name (of the shape rule it was drawn from; not used in rendering)

The chain: every object passes under the line (shapes.passage); in every poll and channel the
line sees the surface nearest to it, the highest one; the signal of that surface, the noise,
the ADC and the correction give the codes (shapes.line_signal, line_codes); frames are made of
the codes as the profile says (framing.py). Label boxes come from the geometry of every object
(shapes.geometric_box), not from the signal.
"""

import csv
from dataclasses import dataclass
import hashlib
import io
from pathlib import Path

import numpy as np

from . import framing, motion as motions
from .morph import motion_options
from .sensor_model import load_profile, profile_file
from .scene import box_iou, integer, number
from .shapes import (
    MAX_ROUGHNESS_SHARE, MAX_SIZE_MM, MAX_SPEED_MM_S, MAX_TILT_POWER, ROUNDNESS_RANGE, Installation, Spots, Superquadric, Surface, checked_margins, geometric_box,
    label_box, line_codes, line_signal, natural_rows, passage,
)


RECORD_KEYS = {"profile", "installation", "speed_mm_s", "rows", "seed", "margins", "objects"}
OBJECT_KEYS = {"shape", "surface", "motion", "row", "y", "yaw", "name"}
SURFACE_KEYS = {"reflectance", "reflectance_spread", "spread_size_mm", "roughness_mm", "roughness_size_mm",
                "spots", "seed", "tilt_power"}
MAX_ROWS = 100000
MAX_RECORD_OBJECTS = 10000


def keys(value, allowed, where, required=()):
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be a mapping")
    unknown, missing = set(value) - set(allowed), set(required) - set(value)
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}")
    if missing:
        raise ValueError(f"{where}: missing {sorted(missing)}")
    return value


def installation_from(spec):
    keys(spec, {"height_mm", "pitch_mm", "poll_hz", "belt_reflectance"}, "installation",
         {"height_mm", "pitch_mm", "poll_hz", "belt_reflectance"})
    return Installation(**spec)


def shape_from(spec, where):
    keys(spec, {"length", "width", "height", "vertical", "horizontal"}, where, {"length", "width", "height"})
    return Superquadric(**spec)


def surface_from(spec, where):
    keys(spec, SURFACE_KEYS, where)
    spots = spec.get("spots", [])
    if not isinstance(spots, list):
        raise ValueError(f"{where}.spots must be a list")
    layers = tuple(Spots(**keys(item, {"coverage", "factor", "size_mm"}, f"{where}.spots",
                                {"coverage", "factor", "size_mm"})) for item in spots)
    return Surface(**{**spec, "spots": layers})


@dataclass(frozen=True)
class RecordObject:
    shape: Superquadric
    surface: Surface
    motion: dict
    row: int
    y: float
    yaw: float


def object_from(spec, index):
    where = f"objects[{index}]"
    keys(spec, OBJECT_KEYS, where, {"shape", "motion", "row"})
    return RecordObject(shape_from(spec["shape"], f"{where}.shape"),
                        surface_from(spec.get("surface", {}), f"{where}.surface"),
                        motion_options(spec["motion"]),
                        integer(spec["row"], f"{where}.row", 0, MAX_ROWS),
                        number(spec.get("y", 0.0), f"{where}.y", -MAX_SIZE_MM, MAX_SIZE_MM),
                        number(spec.get("yaw", 0.0), f"{where}.yaw", -360, 360))


@dataclass(frozen=True)
class RenderedRecord:
    """codes [polls, channels]; boxes [objects, 4] label boxes over the whole record and geometry
    [objects, 4] the geometric ones, time-major (x channels, y rows); frames [tiles, length, channels];
    frame_boxes: for every tile the (object, box in the tile, cut by the tile edge) of the objects in it."""

    codes: np.ndarray
    heights: np.ndarray
    reflectance: np.ndarray
    boxes: np.ndarray
    geometry: np.ndarray
    frames: np.ndarray
    frame_boxes: tuple


def record_surface(spec, model):
    """Heights (mm) and reflectance [polls, channels] of the surface the line sees, and the objects' boxes.

    Where objects overlap, the highest surface is nearest to the line and hides the others.
    """
    keys(spec, RECORD_KEYS, "record", RECORD_KEYS - {"margins"})
    if spec["profile"] != model.name:
        raise ValueError(f"The record is described for the profile {spec['profile']}, not {model.name}")
    installation = installation_from(spec["installation"])
    speed = number(spec["speed_mm_s"], "speed_mm_s", 1e-3, MAX_SPEED_MM_S)
    rows = integer(spec["rows"], "rows", 1, MAX_ROWS)
    if not isinstance(spec["objects"], list) or len(spec["objects"]) > MAX_RECORD_OBJECTS:
        raise ValueError(f"objects: a list of at most {MAX_RECORD_OBJECTS}")
    pool, channels = model.frames.pool, model.channels
    heights = np.zeros((rows * pool, channels))
    reflectance = np.zeros((rows * pool, channels))
    geometry = []
    for index, item in enumerate(spec["objects"]):
        obj = object_from(item, index)
        own_heights, own_reflectance = passage(obj.shape, obj.surface, obj.motion, model, installation, speed,
                                               obj.y, obj.yaw)
        start, length = obj.row * pool, len(own_heights)
        if start + length > len(heights):
            raise ValueError(f"objects[{index}] passes rows {obj.row} ... {obj.row + length // pool - 1}; "
                             f"the record has {rows} rows")
        window = slice(start, start + length)
        nearer = own_heights > heights[window]
        heights[window] = np.where(nearer, own_heights, heights[window])
        reflectance[window] = np.where(nearer, own_reflectance, reflectance[window])
        box = geometric_box(own_heights, pool)
        if box is None:
            raise ValueError(f"objects[{index}] passes under no channel of the line")
        geometry.append(box + [0, obj.row, 0, obj.row])
    return heights, reflectance, np.asarray(geometry, np.float64).reshape(-1, 4), installation


def tile_boxes(boxes, geometry, tile, length, channels):
    """(object, box in the tile, cut by the tile edge) of every object whose geometry reaches the tile."""
    low, high = tile * length, (tile + 1) * length
    result = []
    for index, (box, shape) in enumerate(zip(boxes, geometry)):
        if shape[3] <= low or shape[1] >= high:
            continue
        inside = box - [0, low, 0, low]
        inside[[1, 3]] = np.clip(inside[[1, 3]], 0, length)
        inside[[0, 2]] = np.clip(inside[[0, 2]], 0, channels)
        if (inside[2:] > inside[:2]).all():
            result.append((index, inside, bool(shape[1] < low or shape[3] > high)))
    return tuple(result)


def render_record(spec, model):
    """The record described by `spec` with the sensor `model` of its profile (RenderedRecord)."""
    heights, reflectance, geometry, installation = record_surface(spec, model)
    margins = checked_margins(spec.get("margins", [0, 0, 0, 0]))
    rows, channels = len(heights) // model.frames.pool, model.channels
    boxes = np.asarray([label_box(box, margins, (rows, channels)) for box in geometry], np.float64).reshape(-1, 4)
    signal = line_signal(model, installation, heights, reflectance)
    rng = np.random.default_rng(integer(spec["seed"], "seed", 0, 2**32 - 1))
    codes = line_codes(model, signal, rng)
    frames, _, _ = framing.normalize(framing.tiles(codes, model.frames.pool, model.frames.length), model.frames.normalize)
    frame_boxes = tuple(tile_boxes(boxes, geometry, tile, model.frames.length, channels) for tile in range(len(frames)))
    return RenderedRecord(codes, heights, reflectance, boxes, geometry, frames, frame_boxes)


def record_digest(rendered):
    """The digest of a rendered record: its codes and its label boxes."""
    digest = hashlib.sha256(np.ascontiguousarray(rendered.codes, "<i8").tobytes())
    digest.update(np.ascontiguousarray(rendered.boxes, "<f8").tobytes())
    return digest.hexdigest()


def codes_csv(codes):
    """The codes as CSV: a row per poll, a column per channel, no header (like the recordings of the line)."""
    return "".join(",".join(str(int(value)) for value in row) + "\n" for row in codes)


LABEL_COLUMNS = ("object", "name", "channel_min", "row_min", "channel_max", "row_max")


def labels_csv(rendered, spec):
    """Label boxes of the whole record: [min, max) in channels and rows of `pool` polls."""
    stream = io.StringIO()
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(LABEL_COLUMNS)
    for index, box in enumerate(rendered.boxes):
        writer.writerow([index, spec["objects"][index].get("name", "")] + [f"{value:.12g}" for value in box])
    return stream.getvalue()


def profile_reference(value, folder="."):
    """(source, model, sha256) of the profile a record is made with.

    `value` is the name of a profile shipped with the program, or a path (relative to `folder`).
    The source is kept in the description of a saved record and the digest of the file is checked
    when the record is verified.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError("profile: the name of a shipped profile or a path to a profile file")
    path = Path(folder) / value
    source = str(path.resolve()) if Path(value).suffix or path.exists() else value
    file = profile_file(source)
    return source, load_profile(file), hashlib.sha256(file.read_bytes()).hexdigest()


def natural_motion(kind, shape, model, installation, speed_mm_s, rng, yaw=0.0):
    """A random motion of `kind` for the natural passage of `shape` (motion.sample)."""
    rows = natural_rows(shape, model, installation, speed_mm_s, yaw)
    scale = motions.LineScale.of(model.frames.pool, installation.poll_hz, installation.pitch_mm)
    return motions.sample(kind, rows, (0.0, rows), rng, scale=scale)


# ------------------------------------------------------------------------ random records

# Tries to place one object of a record (each with a new shape, motion and position).
PLACEMENT_TRIES = 50
SHAPE_KEYS = {"name", "weight", "length", "width", "height", "vertical", "horizontal", "surface"}
SURFACE_RULE_KEYS = {"reflectance", "reflectance_spread", "spread_size_mm", "roughness_mm", "roughness_size_mm",
                     "spots", "tilt_power"}
RULE_KEYS = {"rows", "speed_mm_s", "count", "max_iou", "lateral_mm", "yaw", "margins", "motions", "shapes"}


def span(value, name, low, high):
    """A range (low, high) from a number (a fixed value) or a list of two numbers."""
    pair = value if isinstance(value, list) else [value, value]
    if len(pair) != 2:
        raise ValueError(f"{name}: a number or [low, high]")
    first, second = (number(item, name, low, high) for item in pair)
    if first > second:
        raise ValueError(f"{name}: low must not exceed high")
    return first, second


def spot_rule(item, where):
    keys(item, {"coverage", "factor", "size_mm"}, where, {"coverage", "factor", "size_mm"})
    return dict(coverage=span(item["coverage"], f"{where}.coverage", 0, 0.99),
                factor=span(item["factor"], f"{where}.factor", 0, 100),
                size_mm=number(item["size_mm"], f"{where}.size_mm", 1e-3, MAX_SIZE_MM))


def shape_rule(spec, index):
    """A checked shape rule: ranges of the shape and of its surface, a name and a weight."""
    where = f"shapes[{index}]"
    keys(spec, SHAPE_KEYS, where, {"length", "width", "height"})
    surface = keys(spec.get("surface", {}), SURFACE_RULE_KEYS, f"{where}.surface")
    spots = surface.get("spots", [])
    if not isinstance(spots, list):
        raise ValueError(f"{where}.surface.spots must be a list")
    rule = dict(
        name=str(spec.get("name", f"shape {index}"))[:40],
        weight=number(spec.get("weight", 1.0), f"{where}.weight", 1e-6, 1e6),
        size={key: span(spec[key], f"{where}.{key}", 1e-3, MAX_SIZE_MM) for key in ("length", "width", "height")},
        roundness={key: span(spec.get(key, 1.0), f"{where}.{key}", *ROUNDNESS_RANGE)
                   for key in ("vertical", "horizontal")},
        reflectance=span(surface.get("reflectance", 1.0), f"{where}.surface.reflectance", 0, 100),
        reflectance_spread=span(surface.get("reflectance_spread", 0.0), f"{where}.surface.reflectance_spread", 0, 3),
        roughness_mm=span(surface.get("roughness_mm", 0.0), f"{where}.surface.roughness_mm", 0, MAX_SIZE_MM),
        tilt_power=span(surface.get("tilt_power", 0.0), f"{where}.surface.tilt_power", 0, MAX_TILT_POWER),
        spots=[spot_rule(item, f"{where}.surface.spots[{number_}]") for number_, item in enumerate(spots)])
    for amount, size in (("reflectance_spread", "spread_size_mm"), ("roughness_mm", "roughness_size_mm")):
        if rule[amount][1] > 0:
            if size not in surface:
                raise ValueError(f"{where}.surface: {amount} needs {size}")
            rule[size] = number(surface[size], f"{where}.surface.{size}", 1e-3, MAX_SIZE_MM)
    return rule


def record_rules(rules, model, installation):
    """Checked rules of random records for the line `model` over `installation`."""
    keys(rules, RULE_KEYS, "rules", {"rows", "speed_mm_s", "count", "shapes"})
    kinds = rules.get("motions", {"blob": 1})
    if not isinstance(kinds, dict) or not kinds or set(kinds) - set(motions.KINDS):
        raise ValueError(f"rules.motions: weights of {', '.join(motions.KINDS)}")
    shapes = rules["shapes"]
    if not isinstance(shapes, list) or not shapes:
        raise ValueError("rules.shapes: a list of at least one shape")
    count = span(rules["count"], "rules.count", 0, MAX_RECORD_OBJECTS)
    if not all(value.is_integer() for value in count):
        raise ValueError("rules.count: whole numbers")
    half = model.channels * installation.pitch_mm / 2
    tallest = installation.height_mm - model.response.valid_mm[0]
    for index, item in enumerate(shapes):
        rule = shape_rule(item, index)
        if rule["size"]["height"][1] > tallest:
            raise ValueError(f"shapes[{index}].height: up to {tallest:g} mm; the line is {installation.height_mm:g} mm "
                             f"over the belt and the response is valid from {model.response.valid_mm[0]:g} mm")
    return dict(
        rows=integer(rules["rows"], "rules.rows", 1, MAX_ROWS),
        speed_mm_s=span(rules["speed_mm_s"], "rules.speed_mm_s", 1e-3, MAX_SPEED_MM_S),
        count=tuple(int(value) for value in count),
        max_iou=number(rules.get("max_iou", 0.0), "rules.max_iou", 0, 1),
        lateral_mm=span(rules.get("lateral_mm", [-half, half]), "rules.lateral_mm", -MAX_SIZE_MM, MAX_SIZE_MM),
        yaw=span(rules.get("yaw", 0.0), "rules.yaw", -360, 360),
        margins=checked_margins(rules.get("margins", [0, 0, 0, 0])),
        motions={kind: number(weight, f"rules.motions.{kind}", 0, 1e6) for kind, weight in kinds.items()},
        shapes=[shape_rule(item, index) for index, item in enumerate(shapes)])


def weighted(rng, weights):
    """An index drawn with probabilities proportional to `weights`."""
    weights = np.asarray(weights, np.float64)
    if weights.sum() <= 0:
        raise ValueError("Every weight is 0")
    return int(rng.choice(len(weights), p=weights / weights.sum()))


def drawn_object(rule, rng):
    """Shape and surface of one object drawn from a shape rule (JSON-friendly)."""
    uniform = lambda pair: float(rng.uniform(*pair))
    shape = {key: uniform(pair) for key, pair in rule["size"].items()}
    shape.update({key: uniform(pair) for key, pair in rule["roundness"].items()})
    surface = dict(reflectance=uniform(rule["reflectance"]), seed=int(rng.integers(2**32)))
    tilt = uniform(rule["tilt_power"])
    if tilt > 0:
        surface["tilt_power"] = tilt
    spread, roughness = uniform(rule["reflectance_spread"]), uniform(rule["roughness_mm"])
    if spread > 0:
        surface.update(reflectance_spread=spread, spread_size_mm=rule["spread_size_mm"])
    if roughness > 0:
        # Roughness is limited by the height of the object (shapes.MAX_ROUGHNESS_SHARE).
        surface.update(roughness_mm=min(roughness, MAX_ROUGHNESS_SHARE * shape["height"]),
                       roughness_size_mm=rule["roughness_size_mm"])
    spots = [dict(coverage=uniform(item["coverage"]), factor=uniform(item["factor"]), size_mm=item["size_mm"])
             for item in rule["spots"]]
    if spots:
        surface["spots"] = spots
    return shape, surface


def generate_record(rules, model, installation, rng):
    """A random record by checked `rules` (record_rules): (description, notes).

    One belt speed for the record. Every object gets a shape rule and a kind of motion by their
    weights, a shape and surface from the ranges of its rule, a turn, a motion for its natural
    passage, a position across and a starting row. It is placed when its passage fits into the
    record, passes under the line and its geometric box overlaps every placed one by at most
    max_iou (IoU); otherwise it is drawn again, at most PLACEMENT_TRIES times.
    """
    speed = float(rng.uniform(*rules["speed_mm_s"]))
    count = int(rng.integers(rules["count"][0], rules["count"][1] + 1))
    kinds = list(rules["motions"])
    pool, objects, boxes, notes = model.frames.pool, [], [], []
    for _ in range(count):
        for _ in range(PLACEMENT_TRIES):
            rule = rules["shapes"][weighted(rng, [item["weight"] for item in rules["shapes"]])]
            shape, surface = drawn_object(rule, rng)
            kind = kinds[weighted(rng, [rules["motions"][name] for name in kinds])]
            yaw = float(rng.uniform(*rules["yaw"]))
            move = natural_motion(kind, Superquadric(**shape), model, installation, speed, rng, yaw)
            y = float(rng.uniform(*rules["lateral_mm"]))
            heights, _ = passage(Superquadric(**shape), surface_from(surface, "surface"), move, model, installation,
                                 speed, y, yaw)
            length = len(heights) // pool
            box = geometric_box(heights, pool)
            # Roughness may lift the top of a tall object too close to the line.
            too_close = heights.max() > installation.height_mm - model.response.valid_mm[0]
            if length > rules["rows"] or box is None or too_close:
                continue
            row = int(rng.integers(rules["rows"] - length + 1))
            box = box + [0, row, 0, row]
            if any(box_iou(box, other) > rules["max_iou"] for other in boxes):
                continue
            objects.append(dict(name=rule["name"], shape=shape, surface=surface, motion=move, row=row, y=y,
                                yaw=yaw))
            boxes.append(box)
            break
    if len(objects) < count:
        notes.append(f"{len(objects)} of {count} objects placed: the others did not fit by the rules")
    spec = dict(profile=model.name,
                installation=dict(height_mm=installation.height_mm, pitch_mm=installation.pitch_mm,
                                  poll_hz=installation.poll_hz, belt_reflectance=installation.belt_reflectance),
                speed_mm_s=speed, rows=rules["rows"], seed=int(rng.integers(2**32)), margins=rules["margins"],
                objects=objects)
    return spec, notes
