"""Pixel operations and validation of scenes: frames with objects added to them.

Coordinates here are image x/y in pixels, in the orientation of the stored image; a
scene has the size of its background frame (see sensor.FrameGeometry).
Scenes always render again from their background and immutable source crops.
"""

import copy
from dataclasses import dataclass
import math
from pathlib import PurePath

import numpy as np
from PIL import Image


# Real frames are 8-bit, so a scene uses the same 256 intensity levels.
LEVELS = 255
BLEND_MODES = {"replace", "max", "additive"}
# Largest random shift of a variation, pixels (a sanity bound; a frame is far smaller).
MAX_SHIFT = 1024
# Tries to place one object of a variation before the variation is given up.
VARIATION_ATTEMPTS = 80
# Appearance changes that alter the ratios between pixels (see reshape_intensities).
NEUTRAL_APPEARANCE = dict(gamma=1.0, contrast=1.0, gradient=0.0, gradient_angle=0.0,
                          noise=0.0, noise_seed=0,
                          shade=0.0, shade_size=0.25, shade_x=0.5, shade_y=0.5,
                          erode=0.0, erode_seed=0)
APPEARANCE_BOUNDS = dict(gamma=(0.3, 3), contrast=(0, 3), gradient=(-1, 1),
                         gradient_angle=(-360, 360), noise=(0, 1), erode=(0, 1),
                         shade=(0, 1), shade_size=(0.05, 1), shade_x=(0, 1), shade_y=(0, 1))


def pixel_levels(pixels):
    """Intensities in [0, 1] as the 8-bit values stored in the image file."""
    return np.round(np.asarray(pixels, dtype=np.float32) * LEVELS).astype(np.uint8)


def number(value, name, minimum, maximum):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}: expected a number")
    value = float(value)
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name}: expected [{minimum}, {maximum}]")
    return value


def integer(value, name, minimum, maximum):
    value = number(value, name, minimum, maximum)
    if not value.is_integer():
        raise ValueError(f"{name}: expected an integer")
    return int(value)


def validate_pixels(pixels, shape=None):
    """Intensities in [0, 1]; with `shape` (rows, columns), also of that size."""
    if pixels.ndim != 2 or (shape is not None and pixels.shape != tuple(shape)):
        expected = f"{shape[1]} x {shape[0]}" if shape is not None else "two-dimensional"
        raise ValueError(f"Expected a {expected} frame, got {' x '.join(map(str, pixels.shape[::-1]))}")
    if not np.isfinite(pixels).all() or pixels.min() < 0 or pixels.max() > 1:
        raise ValueError("Configured image intensities must be finite and in [0, 1]")


def validate_boxes(boxes, labels, shape):
    if boxes.shape != (len(labels), 4):
        raise ValueError("Bounding box and class counts differ")
    if not np.isfinite(boxes).all() or np.any(boxes < 0) or np.any(boxes > [shape[1], shape[0]] * 2):
        raise ValueError("Bounding boxes must be finite and inside the image")
    if np.any(boxes[:, 2:] <= boxes[:, :2]):
        raise ValueError("Bounding boxes must have positive width and height")
    if any(isinstance(label, bool) or int(label) != label or label < 0 for label in labels):
        raise ValueError("Classes must be nonnegative integers")


@dataclass(frozen=True)
class SourceImage:
    pixels: np.ndarray
    boxes: np.ndarray
    labels: tuple
    split: str
    identifier: str
    # Real frames the image is made of: the frame itself, or the frames that gave the
    # pieces of a generated background (see background.py). Empty: not known.
    frames: tuple = ()


def source_splits(provenance):
    """The splits that the sources of a rendered scene or tape come from."""
    return sorted({row["split"] for row in provenance})


def frame_name(identifier):
    """The frame of an image id, without its extension (scan_0001.png -> scan_0001)."""
    return PurePath(identifier).stem


def background_provenance(source):
    row = dict(split=source.split, id=source.identifier, role="background")
    if source.frames:
        row["frames"] = [frame_name(name) for name in source.frames]
    return row


def object_provenance(source, box_index):
    return dict(split=source.split, id=source.identifier, frame=frame_name(source.identifier),
                box_index=box_index, role="object")


@dataclass(frozen=True)
class PreparedObject:
    pixels: np.ndarray
    mask: np.ndarray
    label: int
    # Annotation box in object pixels (x1, y1, x2, y2). None: the whole crop, as for
    # objects cut from a real box. Generated objects keep a soft edge outside their box.
    box: tuple | None = None

    def label_box(self):
        if self.box is None:
            return np.array([0, 0, self.width, self.height], dtype=np.float64)
        return np.asarray(self.box, dtype=np.float64)

    @property
    def width(self):
        return self.pixels.shape[1]

    @property
    def height(self):
        return self.pixels.shape[0]


def transform_options(options):
    if not isinstance(options, dict):
        raise ValueError("transform must be an object")
    defaults = dict(flip_x=False, flip_y=False, brightness=1.0, angle=0.0,
                    scale=1.0, use_mask=False, threshold=0.1, **NEUTRAL_APPEARANCE)
    if set(options) - defaults.keys():
        raise ValueError("Unknown object transformation")
    result = {**defaults, **options}
    for key in ("flip_x", "flip_y", "use_mask"):
        if not isinstance(result[key], bool):
            raise ValueError(f"{key}: expected a boolean")
    for key, bounds in {"brightness": (0, 4), "angle": (-360, 360),
                        "scale": (0.1, 5), "threshold": (0, 1), **APPEARANCE_BOUNDS}.items():
        result[key] = number(result[key], key, *bounds)
    for key in ("noise_seed", "erode_seed"):
        result[key] = integer(result[key], key, 0, 2**32 - 1)
    return result


def erode_edges(active, probability, seed):
    """Randomly drop boundary pixels so the pasted outline no longer repeats the source's.

    A boundary pixel is an object pixel with a 4-neighbour outside the object. Dropped
    pixels keep the background of the scene, which breaks the clean edge shared by copies.
    """
    padded = np.pad(active, 1)
    inside = (padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:])
    boundary = active & ~inside
    dropped = boundary & (np.random.default_rng(seed).random(active.shape) < probability)
    return active & ~dropped


def reshape_intensities(crop, mask, options):
    """Change the *relations* between pixels, not only their overall level.

    A brightness factor multiplies every pixel, so the ratio of any two pixels never
    changes. These steps (all seeded, so a scene renders identically every time) do:
    gamma and contrast bend the ratios, a gradient imitates uneven lighting, noise
    perturbs each pixel on its own and shade darkens one soft patch like dust or dirt.
    """
    crop = crop.astype(np.float32)
    if options["gamma"] != 1:
        crop = np.clip(crop, 0, 1) ** options["gamma"]
    if options["contrast"] != 1:
        center = float(crop[mask].mean())
        crop = center + (crop - center) * options["contrast"]
    height, width = crop.shape
    if options["gradient"]:
        # Linear lighting ramp across the object, in the direction of gradient_angle.
        angle = math.radians(options["gradient_angle"])
        along = ((np.arange(width) - (width - 1) / 2)[None, :] * math.cos(angle)
                 + (np.arange(height) - (height - 1) / 2)[:, None] * math.sin(angle))
        reach = max(float(np.abs(along).max()), 1e-6)
        crop = crop * (1 + options["gradient"] * along / reach)
    if options["noise"]:
        rng = np.random.default_rng(options["noise_seed"])
        crop = crop * (1 + options["noise"] * rng.standard_normal(crop.shape).astype(np.float32))
    if options["shade"]:
        # A soft Gaussian blob; its size is a fraction of the object's longer side.
        sigma = max(options["shade_size"] * max(width, height), 0.3)
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        center_x = options["shade_x"] * (width - 1)
        center_y = options["shade_y"] * (height - 1)
        blob = np.exp(-((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2 * sigma ** 2))
        crop = crop * (1 - options["shade"] * blob)
    return crop


def prepare_object(source, box_index, options):
    options = transform_options(options)
    index = integer(box_index, "box_index", 0, len(source.boxes) - 1)
    x1, y1 = np.floor(source.boxes[index, :2] + 1e-5).astype(int)
    x2, y2 = np.ceil(source.boxes[index, 2:] - 1e-5).astype(int)
    # Very small annotations still select their intersected pixel.
    rows, columns = source.pixels.shape
    x1, y1 = min(columns - 1, x1), min(rows - 1, y1)
    x2, y2 = min(columns, max(x1 + 1, x2)), min(rows, max(y1 + 1, y2))
    crop = source.pixels[y1:y2, x1:x2].copy()
    mask = crop >= options["threshold"] if options["use_mask"] else np.ones(crop.shape, bool)
    if not mask.any():
        raise ValueError("The mask is empty; reduce the intensity threshold")
    values, active = transform_crop(crop, mask, options)
    return PreparedObject(values, active, int(source.labels[index]))


def transform_crop(crop, mask, options):
    """Flips, intensity changes, scale and rotation of an object crop and its mask.

    Returns the pixels and mask cropped to the pixels that remain in the object.
    """
    if options["flip_x"]:
        crop, mask = crop[:, ::-1], mask[:, ::-1]
    if options["flip_y"]:
        crop, mask = crop[::-1], mask[::-1]
    crop = reshape_intensities(crop, mask, options)
    crop = np.clip(crop * options["brightness"], 0, 1)
    # Premultiplication keeps excluded crop background out of interpolation.
    pixels = Image.fromarray((crop * mask).astype(np.float32))
    alpha = Image.fromarray(mask.astype(np.float32))
    shape = (max(1, round(crop.shape[1] * options["scale"])),
             max(1, round(crop.shape[0] * options["scale"])))
    if shape != pixels.size:
        pixels = pixels.resize(shape, Image.Resampling.BILINEAR)
        alpha = alpha.resize(shape, Image.Resampling.BILINEAR)
    if options["angle"] % 360:
        pixels = pixels.rotate(options["angle"], Image.Resampling.BILINEAR, expand=True)
        alpha = alpha.rotate(options["angle"], Image.Resampling.BILINEAR, expand=True)
    weights = np.asarray(alpha, dtype=np.float32)
    active = weights > 1e-6
    if options["erode"]:
        active = erode_edges(active, options["erode"], options["erode_seed"])
    if not active.any():
        raise ValueError("The transformed object has no pixels")
    values = np.zeros(weights.shape, dtype=np.float32)
    np.divide(np.asarray(pixels), weights, out=values, where=active)
    yy, xx = np.nonzero(active)
    slices = (slice(yy.min(), yy.max() + 1), slice(xx.min(), xx.max() + 1))
    values = np.clip(values[slices], 0, 1).copy()
    return values, active[slices].copy()


def prepare_entry(obj, load_source):
    """The default preparation of a scene object: a real annotated box with its transform."""
    return prepare_object(load_source(obj.get("source")), obj.get("box_index"), obj.get("transform", {}))


def box_iou(first, second):
    intersection = np.maximum(0, np.minimum(first[2:], second[2:]) - np.maximum(first[:2], second[:2]))
    area = float(np.prod(intersection))
    union = float(np.prod(first[2:] - first[:2]) + np.prod(second[2:] - second[:2]) - area)
    return area / union


def scene_statistics(boxes, added):
    maximum, minimum = 0.0, None
    for index, first in enumerate(boxes):
        for second in boxes[index + 1:]:
            maximum = max(maximum, box_iou(first, second))
            gap = np.maximum(0, np.maximum(first[:2] - second[2:], second[:2] - first[2:]))
            distance = float(np.linalg.norm(gap))
            minimum = distance if minimum is None else min(minimum, distance)
    return dict(total_objects=len(boxes), added_objects=added, max_iou=maximum,
                min_distance=minimum)


def placement_box(prepared, x, y, shape):
    """The annotation box of an object whose crop corner is at x, y on a canvas of `shape` (rows, columns)."""
    height, width = shape
    x = integer(x, "x", 0, width)
    y = integer(y, "y", 0, height)
    if x + prepared.width > width or y + prepared.height > height:
        raise ValueError(f"The added object must fit completely inside {width} x {height}")
    return prepared.label_box() + np.array([x, y, x, y], dtype=np.float64)


def check_placement(prepared, x, y, existing, threshold, shape):
    box = placement_box(prepared, x, y, shape)
    if any(box_iou(box, other) > threshold + 1e-9 for other in existing):
        raise ValueError(f"Placement exceeds the maximum IoU ({threshold:g})")
    return box


def feather_mask(mask, radius):
    """Blur alpha in float32, retaining support inside the annotated footprint."""
    radius = number(radius, "edge_blur", 0, 3)
    if not radius:
        return mask.astype(np.float32)
    size = max(1, math.ceil(radius * 3))
    positions = np.arange(-size, size + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (positions / radius) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(mask.astype(np.float32), size)
    horizontal = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), 1, padded)
    alpha = np.apply_along_axis(lambda column: np.convolve(column, kernel, mode="valid"), 0, horizontal)
    return np.clip(alpha, 0, 1) * mask


def paste_object(pixels, prepared, x, y, mode, generated, blur):
    """Blend a prepared object into `pixels` with its crop corner at x, y (in place)."""
    # The crop's corner; the label box may lie inside it (generated objects).
    region = pixels[y:y + prepared.height, x:x + prepared.width]
    # A generated object carries its soft edge, which must never darken the
    # background, so replace acts as max for it.
    object_mode = "max" if mode == "replace" and generated else mode
    if object_mode == "replace":
        composite = prepared.pixels
    elif object_mode == "max":
        composite = np.maximum(region, prepared.pixels)
    else:
        composite = np.minimum(1, region + prepared.pixels)
    if blur:
        alpha = feather_mask(prepared.mask, blur)
        composite = region * (1 - alpha) + composite * alpha
    region[prepared.mask] = composite[prepared.mask]


@dataclass(frozen=True)
class RenderedScene:
    pixels: np.ndarray
    boxes: np.ndarray
    labels: tuple
    provenance: tuple
    added: int

    def payload(self):
        return dict(matrix=self.pixels.tolist(), boxes_pixels_xyxy=self.boxes.tolist(),
                    labels=list(self.labels), statistics=scene_statistics(self.boxes, self.added),
                    train_eligible=all(item["split"] == "train" for item in self.provenance),
                    splits=source_splits(self.provenance))


def scene_options(spec):
    if not isinstance(spec, dict):
        raise ValueError("scene must be an object")
    mode = spec.get("blending", "replace")
    if not isinstance(mode, str) or mode not in BLEND_MODES:
        raise ValueError("blending must be replace, max or additive")
    threshold = number(spec.get("max_iou", 0.15), "max_iou", 0, 1)
    objects = spec.get("objects", [])
    if not isinstance(objects, list) or len(objects) > 128:
        raise ValueError("Expected at most 128 added objects")
    return mode, threshold, objects


def render_scene(spec, load_source, prepare=prepare_entry):
    """Render a scene description; `prepare(obj, load_source)` builds each added object."""
    mode, threshold, objects = scene_options(spec)
    background = load_source(spec.get("background"))
    pixels = background.pixels.copy()
    boxes = list(background.boxes.copy())
    labels = list(background.labels)
    provenance = [background_provenance(background)]
    for obj in objects:
        if not isinstance(obj, dict):
            raise ValueError("Each added object must be an object")
        source = load_source(obj.get("source"))
        prepared = prepare(obj, load_source)
        blur = number(obj.get("edge_blur", 0), "edge_blur", 0, 3)
        box = check_placement(prepared, obj.get("x"), obj.get("y"), boxes, threshold, pixels.shape)
        paste_object(pixels, prepared, int(obj["x"]), int(obj["y"]), mode, obj.get("morph") is not None, blur)
        boxes.append(box)
        labels.append(prepared.label)
        provenance.append(object_provenance(source, obj["box_index"]))
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    validate_pixels(pixels)
    # Brightness, scaling and rotation give values between the 8-bit levels of real
    # frames. Rounding here makes the preview the exact signal the saved file gives.
    pixels = pixel_levels(pixels).astype(np.float32) / LEVELS
    validate_boxes(boxes, labels, pixels.shape)
    return RenderedScene(pixels, boxes, tuple(labels), tuple(provenance), len(objects))


def variation_options(options):
    if not isinstance(options, dict):
        raise ValueError("variation settings must be an object")
    defaults = dict(position=3, brightness=0.1, angle=10.0, scale=0.1)
    if set(options) - defaults.keys():
        raise ValueError("Unknown variation setting")
    settings = {**defaults, **options}
    settings["position"] = integer(settings["position"], "position variation", 0, MAX_SHIFT)
    for key, maximum in (("brightness", 1), ("angle", 180), ("scale", 0.9)):
        settings[key] = number(settings[key], f"{key} variation", 0, maximum)
    return settings


def make_variation(spec, load_source, rng, options, attempts=VARIATION_ATTEMPTS, prepare=prepare_entry):
    """Bounded retries; a failed scene is never returned for saving."""
    _, threshold, objects = scene_options(spec)
    if not objects:
        raise ValueError("Add an object before generating variations")
    settings = variation_options(options)
    variant = copy.deepcopy(spec)
    background = load_source(spec["background"])
    existing = list(background.boxes)
    for obj in variant["objects"]:
        source = load_source(obj["source"])
        base = transform_options(obj.get("transform", {}))
        for _ in range(attempts):
            transform = dict(base)
            transform["brightness"] = float(np.clip(base["brightness"] * rng.uniform(1 - settings["brightness"], 1 + settings["brightness"]), 0, 4))
            transform["angle"] = (base["angle"] + rng.uniform(-settings["angle"], settings["angle"]) + 180) % 360 - 180
            transform["scale"] = float(np.clip(base["scale"] * rng.uniform(1 - settings["scale"], 1 + settings["scale"]), 0.1, 5))
            if base["noise"]:
                transform["noise_seed"] = int(rng.integers(2**32))
            if base["erode"]:
                transform["erode_seed"] = int(rng.integers(2**32))
            if base["shade"]:
                transform["shade_x"], transform["shade_y"] = (float(v) for v in rng.random(2))
            try:
                prepared = prepare({**obj, "transform": transform}, load_source)
                delta = settings["position"]
                x = obj["x"] + int(rng.integers(-delta, delta + 1))
                y = obj["y"] + int(rng.integers(-delta, delta + 1))
                box = check_placement(prepared, x, y, existing, threshold, background.pixels.shape)
            except ValueError:
                continue
            obj.update(transform=transform, x=x, y=y)
            existing.append(box)
            break
        else:
            raise ValueError("No valid variation found within the placement retry budget")
    return variant, render_scene(variant, load_source, prepare)
