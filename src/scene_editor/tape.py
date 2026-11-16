"""Tapes: long recordings made of the same objects as scenes.

A tape is what the sensor line records over a long run: all its channels (columns) and
as many rows (successive polls) as you like. A frame is a piece of it. A tape is
described like a scene, by a background and a list of placed objects, and it renders
again from that description. Coordinates are channel (x) and row (y), always time-major,
whatever the sensor axis of the source frames is.

The generator places the objects along the tape with a layout function (see LAYOUTS).
"""

import base64
from collections import Counter
import csv
from dataclasses import dataclass
import hashlib
import io

import numpy as np

from .background import DEFAULTS as BACKGROUND_DEFAULTS, background_options
from .generator import ObjectPicker, generator_options, object_pool, random_object
from .scene import (
    BLEND_MODES, LEVELS, PreparedObject, integer, number, object_provenance, paste_object,
    pixel_levels, placement_box, source_splits, validate_boxes,
)


MIN_LENGTH = 40
MAX_LENGTH = 20000
MAX_OBJECTS = 1000
# Tries to place one object (each with a new random transform), and prepared objects in all.
ATTEMPTS = 40
BUDGET = 30000
VALUE_FORMATS = ("levels", "intensity")


def iou_many(box, others):
    """IoU of one box with every row of `others` (an [N, 4] array)."""
    if not len(others):
        return np.zeros(0)
    low = np.maximum(box[:2], others[:, :2])
    high = np.minimum(box[2:], others[:, 2:])
    intersection = np.prod(np.maximum(0, high - low), axis=1)
    union = np.prod(box[2:] - box[:2]) + np.prod(others[:, 2:] - others[:, :2], axis=1) - intersection
    return intersection / union


class BoxSet:
    """Boxes added one by one; the overlap with all of them is one array operation."""

    def __init__(self):
        self.data = np.zeros((64, 4))
        self.count = 0

    @property
    def boxes(self):
        return self.data[:self.count]

    def add(self, box):
        if self.count == len(self.data):
            self.data = np.concatenate([self.data, np.zeros_like(self.data)])
        self.data[self.count] = box
        self.count += 1

    def max_iou(self, box):
        return float(iou_many(box, self.boxes).max(initial=0.0))


def tape_background_options(background):
    """A tape background: {"kind": "blank"} or generated from the empty channels of real frames."""
    if not isinstance(background, dict):
        raise ValueError("The tape background must be an object")
    kind = background.get("kind")
    if kind == "blank":
        if set(background) != {"kind"}:
            raise ValueError("A blank background has no options")
        return dict(kind="blank")
    if kind != "generated":
        raise ValueError("The tape background must be blank or generated")
    if set(background) - {"kind", "split"} - BACKGROUND_DEFAULTS.keys():
        raise ValueError("Unknown background option")
    split = background.get("split")
    if split not in ("train", "val", "test"):
        raise ValueError("Select a train, val or test source for the generated background")
    options = background_options({key: value for key, value in background.items() if key in BACKGROUND_DEFAULTS})
    return dict(kind="generated", split=split, **options)


def tape_options(spec):
    """The validated parts of a tape description."""
    if not isinstance(spec, dict):
        raise ValueError("tape must be an object")
    length = integer(spec.get("length"), "length", MIN_LENGTH, MAX_LENGTH)
    mode = spec.get("blending", "replace")
    if not isinstance(mode, str) or mode not in BLEND_MODES:
        raise ValueError("blending must be replace, max or additive")
    threshold = number(spec.get("max_iou", 0.15), "max_iou", 0, 1)
    objects = spec.get("objects", [])
    if not isinstance(objects, list) or len(objects) > MAX_OBJECTS:
        raise ValueError(f"Expected at most {MAX_OBJECTS} objects on a tape")
    background = tape_background_options(spec.get("background", dict(kind="blank")))
    return dict(length=length, background=background, blending=mode, max_iou=threshold, objects=objects)


def time_major_object(prepared):
    """An object built in the orientation of frames with sensors on y, turned to time-major."""
    box = prepared.box
    if box is not None:
        box = (box[1], box[0], box[3], box[2])
    return PreparedObject(prepared.pixels.T.copy(), prepared.mask.T.copy(), prepared.label, box)


def tape_statistics(boxes, length, channels):
    maximum, minimum = 0.0, None
    for index in range(len(boxes) - 1):
        first, rest = boxes[index], boxes[index + 1:]
        maximum = max(maximum, float(iou_many(first, rest).max()))
        gap = np.maximum(0, np.maximum(first[:2] - rest[:, 2:], rest[:, :2] - first[2:]))
        distance = float(np.linalg.norm(gap, axis=1).min())
        minimum = distance if minimum is None else min(minimum, distance)
    return dict(length=length, channels=channels, total_objects=len(boxes), added_objects=len(boxes),
                max_iou=maximum, min_distance=minimum)


def tape_digest(levels, boxes, labels):
    """Pixels and annotations, independent of the order of the objects."""
    rows = sorted((int(label), *map(float, box)) for box, label in zip(boxes, labels))
    digest = hashlib.sha256(np.asarray(levels.shape, dtype="<i8").tobytes())
    digest.update(levels.tobytes())
    digest.update(np.asarray(rows, dtype="<f8").reshape(-1, 5).tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class RenderedTape:
    levels: np.ndarray      # uint8 [length, channels]: exactly the values a saved tape holds
    boxes: np.ndarray       # [N, 4] x1, y1, x2, y2 = channel, row, channel, row
    labels: tuple
    provenance: tuple

    @property
    def digest(self):
        return tape_digest(self.levels, self.boxes, self.labels)

    def payload(self):
        length, channels = (int(value) for value in self.levels.shape)
        return dict(length=length, channels=channels,
                    levels_base64=base64.b64encode(self.levels.tobytes()).decode("ascii"),
                    boxes_pixels_xyxy=self.boxes.tolist(), labels=list(self.labels),
                    statistics=tape_statistics(self.boxes, length, channels),
                    train_eligible=all(item["split"] == "train" for item in self.provenance),
                    splits=source_splits(self.provenance), digest=self.digest)


def render_tape(spec, load_source, prepare, make_background, channels):
    """Render a tape description for a line of `channels` sensors.

    `prepare(obj, load_source)` builds an added object in time-major orientation and
    `make_background(background, length)` gives the background pixels and their provenance.
    """
    options = tape_options(spec)
    length = options["length"]
    pixels, provenance = make_background(options["background"], length)
    pixels = np.array(pixels, dtype=np.float32)
    if pixels.shape != (length, channels) or not np.isfinite(pixels).all() or pixels.min() < 0 or pixels.max() > 1:
        raise ValueError("The tape background must be finite intensities in [0, 1]")
    provenance = list(provenance)
    boxes, labels = BoxSet(), []
    for position, obj in enumerate(options["objects"], start=1):
        try:
            if not isinstance(obj, dict):
                raise ValueError("Each added object must be an object")
            source = load_source(obj.get("source"))
            prepared = prepare(obj, load_source)
            blur = number(obj.get("edge_blur", 0), "edge_blur", 0, 3)
            box = placement_box(prepared, obj.get("x"), obj.get("y"), pixels.shape)
            if boxes.max_iou(box) > options["max_iou"] + 1e-9:
                raise ValueError(f"Placement exceeds the maximum IoU ({options['max_iou']:g})")
            paste_object(pixels, prepared, int(obj["x"]), int(obj["y"]), options["blending"],
                         obj.get("morph") is not None, blur)
        except ValueError as error:
            raise ValueError(f"Object {position}: {error}") from error
        boxes.add(box)
        labels.append(prepared.label)
        provenance.append(object_provenance(source, obj["box_index"]))
    boxes = boxes.boxes.copy()
    # The same rounding as scenes: the preview is the signal of the saved file.
    levels = pixel_levels(pixels)
    validate_boxes(boxes, labels, levels.shape)
    return RenderedTape(levels, boxes, tuple(labels), tuple(provenance))


# --- layouts: where along the tape the objects go --------------------------------------------

@dataclass(frozen=True)
class Target:
    """Where one object should go: around row `center` within `tolerance` rows (None: anywhere)."""
    center: float | None
    tolerance: float = 0.0

    def row(self, height, length, attempt, rng):
        top = length - height
        if self.center is None:
            return int(rng.integers(top + 1))
        # The reach grows with failed attempts, so a crowded spot still gets its object nearby.
        reach = self.tolerance * (1 + attempt / 10)
        return int(np.clip(round(self.center + rng.uniform(-reach, reach) - height / 2), 0, top))


def layout_random(rules, rng):
    """Every object at an independent random row."""
    return [Target(None)] * rules["objects"]


def layout_even(rules, rng):
    """Objects at equal steps along the tape, each moved by up to `jitter` of half a step."""
    step = rules["length"] / max(rules["objects"], 1)
    return [Target((index + 0.5) * step, rules["jitter"] * step / 2) for index in range(rules["objects"])]


def layout_clusters(rules, rng):
    """Groups of `cluster_min`..`cluster_max` objects close together, groups at random rows."""
    targets = []
    while len(targets) < rules["objects"]:
        size = min(int(rng.integers(rules["cluster_min"], rules["cluster_max"] + 1)),
                   rules["objects"] - len(targets))
        center = float(rng.uniform(0, rules["length"]))
        targets += [Target(center, rules["cluster_spread"])] * size
    return targets


LAYOUTS = dict(random=layout_random, even=layout_even, clusters=layout_clusters)
TAPE_RULES = dict(length=400, objects=20, layout="random", jitter=0.3, cluster_min=2, cluster_max=4,
                  cluster_spread=6.0, background="generated")
# Rules of single images that make no sense on a tape: they are fixed.
FIXED_RULES = dict(count_min=1, count_max=1, images=1, background_mode="generated",
                   unique_per_image=False, unique_source=False)


def tape_rules(options):
    """Validated rules: the tape ones plus the object rules of the autogenerator (`generator`)."""
    if not isinstance(options, dict):
        raise ValueError("Tape rules must be an object")
    if set(options) & FIXED_RULES.keys():
        raise ValueError("Unknown tape rule")
    rules = {**TAPE_RULES, **{key: value for key, value in options.items() if key in TAPE_RULES}}
    # A tape always uses the least used pool entries first, so no object is repeated while
    # the pool still has unused ones; whatever the caller sends for `balanced` is ignored.
    generator = generator_options({**{key: value for key, value in options.items() if key not in TAPE_RULES},
                                   **FIXED_RULES})
    generator["balanced"] = True
    rules["length"] = integer(rules["length"], "length", MIN_LENGTH, MAX_LENGTH)
    rules["objects"] = integer(rules["objects"], "objects", 0, MAX_OBJECTS)
    rules["cluster_min"] = integer(rules["cluster_min"], "cluster_min", 1, 64)
    rules["cluster_max"] = integer(rules["cluster_max"], "cluster_max", 1, 64)
    if rules["cluster_min"] > rules["cluster_max"]:
        raise ValueError("cluster_min must not exceed cluster_max")
    rules["jitter"] = number(rules["jitter"], "jitter", 0, 1)
    rules["cluster_spread"] = number(rules["cluster_spread"], "cluster_spread", 0, MAX_LENGTH)
    if rules["layout"] not in LAYOUTS:
        raise ValueError(f"layout must be one of {', '.join(LAYOUTS)}")
    if rules["background"] not in ("generated", "blank"):
        raise ValueError("background must be generated or blank")
    return dict(rules, generator=generator)


def generate_tape(entries, options, load_source, prepare, channels, is_clipped=None):
    """A tape of random objects of the pool for a line of `channels` sensors; nothing is written.

    Returns the tape description and how many objects could be placed. An object that finds
    no free place in ATTEMPTS tries is dropped, so the placed count may be lower than asked.
    Every object and the background come from the split of the rules (`split`); objects cut
    by the frame edge (`is_clipped(source, box_index)`) are left out of the pool.
    """
    rules = tape_rules(options)
    generator = rules["generator"]
    notes = []
    pool = object_pool(entries, load_source, generator["split"], is_clipped, notes)
    picker = ObjectPicker(pool, generator)
    rng = np.random.default_rng(generator["seed"])
    if rules["background"] == "blank":
        background = dict(kind="blank")
    else:
        level = generator["background_level_min"]
        if generator["background_level_max"] > level:
            level = float(rng.uniform(level, generator["background_level_max"]))
        background = dict(kind="generated", split=generator["split"], seed=int(rng.integers(2**32)),
                          block_min=generator["background_block_min"], block_max=generator["background_block_max"],
                          level=level, roll=True)
    length = rules["length"]
    boxes, objects, rejected, budget = BoxSet(), [], Counter(), BUDGET
    for target in LAYOUTS[rules["layout"]](rules, rng):
        state = picker.start(None)
        failed = Counter()
        for attempt in range(ATTEMPTS):
            if budget <= 0:
                break
            index = picker.pick(state, failed, rng)
            if index is None:
                break
            budget -= 1
            candidate = random_object(pool[index], generator, rng)
            try:
                prepared = prepare(candidate, load_source)
                if prepared.width > channels or prepared.height > length:
                    raise ValueError("Transformed object does not fit inside the tape")
                x = int(rng.integers(channels - prepared.width + 1))
                y = target.row(prepared.height, length, attempt, rng)
                box = placement_box(prepared, x, y, (length, channels))
                if boxes.max_iou(box) > generator["max_iou"] + 1e-9:
                    raise ValueError(f"Placement exceeds the maximum IoU ({generator['max_iou']:g})")
            except ValueError as error:
                rejected[str(error)] += 1
                failed[index] += 1
                continue
            picker.take(state, index)
            picker.accept(state)
            boxes.add(box)
            objects.append(dict(candidate, x=x, y=y))
            break
    objects.sort(key=lambda obj: (obj["y"], obj["x"]))
    if len(objects) < rules["objects"]:
        notes.append(f"Placed {len(objects)} of {rules['objects']} objects: no free place was found for the rest "
                     "(a longer tape, a lower count or a higher maximum IoU helps)")
    tape = dict(length=length, background=background, blending=generator["blending"],
                max_iou=generator["max_iou"], objects=objects)
    return dict(tape=tape, requested=rules["objects"], placed=len(objects), rejected=dict(rejected),
                budget_exhausted=budget <= 0, rules=rules, usage=picker.usage, notes=notes)


# --- files -----------------------------------------------------------------------------------

def tape_csv(levels, values="levels", header=False):
    """The tape as CSV text: a row per poll, a column per channel."""
    if values not in VALUE_FORMATS:
        raise ValueError("values must be levels or intensity")
    stream = io.StringIO()
    if header:
        stream.write(",".join(f"channel_{index}" for index in range(levels.shape[1])) + "\n")
    if values == "levels":
        np.savetxt(stream, levels, fmt="%d", delimiter=",")
    else:
        np.savetxt(stream, levels.astype(np.float64) / LEVELS, fmt="%.6g", delimiter=",")
    return stream.getvalue()


def labels_csv(rendered, spec):
    """One row per object: its box in channels and rows, class and where it came from."""
    stream = io.StringIO()
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["index", "class", "channel_min", "row_min", "channel_max", "row_max",
                     "source_split", "source_id", "source_box", "kind"])
    for index, (box, label, obj) in enumerate(zip(rendered.boxes, rendered.labels, spec["objects"])):
        morph = obj.get("morph")
        writer.writerow([index, label, *(f"{value:.12g}" for value in box),
                         obj["source"]["split"], obj["source"]["id"], obj["box_index"],
                         morph["kind"] if morph else "real"])
    return stream.getvalue()
