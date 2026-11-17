"""Read the source dataset: grayscale frames with YOLO boxes, described by a YAML `data` section.

The section names the `train` and `val` split folders (each with images/ and
labels/), the image axis that holds the sensors (`sensor_axis`: x or y) and how
pixels are stored (`scale`, `offset`: intensity = (stored value - offset) / scale).
`channels` and `frame_length` give the geometry of the frames (see sensor.py); when
they are not given, the first frame of the dataset sets them, and every frame must
have that size. The data.yaml this program writes has the same form, so an exported
dataset can be the source of the next one.
"""

from pathlib import Path
import math
from numbers import Real

import numpy as np
from PIL import Image
import yaml

from .sensor import MAX_CHANNELS, MAX_FRAME_LENGTH, FrameGeometry


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
GRAYSCALE_MODES = {"L", "I", "I;16", "I;16L", "I;16B", "F"}
DATA_DESCRIPTION = "data.yaml"
DATA_OPTIONS = {"type", "train", "val", "test", "sensor_axis", "scale", "offset", "channels", "frame_length"}
DATA_DEFAULTS = {"sensor_axis": "x", "scale": 255.0, "offset": 0.0}
SPLIT_OPTIONS = ("train", "val", "test")


def finite_number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def load_data_description(path: str | Path) -> dict:
    """The `data` section of a YAML file (or of a folder's data.yaml), with paths resolved against it."""
    path = Path(path).resolve()
    if path.is_dir():
        path = path / DATA_DESCRIPTION
    if not path.is_file():
        raise ValueError(f"No dataset description: {path}")
    try:
        described = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"{path}: {error}") from error
    data = described.get("data") if isinstance(described, dict) else None
    if not isinstance(data, dict):
        raise ValueError(f"{path} has no data section")
    unknown = set(data) - DATA_OPTIONS
    if unknown:
        raise ValueError(f"{path}: unknown data options {sorted(unknown)}; allowed: {sorted(DATA_OPTIONS)}")
    data = {**DATA_DEFAULTS, **data}
    if data.get("type", "yolo_images") != "yolo_images":
        raise ValueError(f"{path}: data.type must be yolo_images")
    if data["sensor_axis"] not in ("x", "y"):
        raise ValueError(f"{path}: data.sensor_axis must be x or y")
    if finite_number(data["scale"], "data.scale") <= 0:
        raise ValueError(f"{path}: data.scale must be positive")
    finite_number(data["offset"], "data.offset")
    for key, limit in (("channels", MAX_CHANNELS), ("frame_length", MAX_FRAME_LENGTH)):
        value = data.get(key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= limit):
            raise ValueError(f"{path}: data.{key} must be an integer in [1, {limit}]")
    for key in SPLIT_OPTIONS:
        if key not in data:
            if key != "test":
                raise ValueError(f"{path}: missing data.{key}")
            continue
        if not isinstance(data[key], str) or not data[key].strip():
            raise ValueError(f"{path}: data.{key} must be a nonempty path string")
        data[key] = str((path.parent / data[key]).resolve())
    return data


def find_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory not found: {directory}")
    paths = sorted(
        path for path in directory.iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise ValueError(f"No images in {directory}")
    stems = [path.stem for path in paths]
    if len(set(stems)) != len(stems):
        raise ValueError(f"Ambiguous image stems in {directory}")
    return paths


def frame_size(path: Path) -> tuple[int, int]:
    """(rows, columns) of a stored frame."""
    with Image.open(path) as image:
        return image.height, image.width


def frame_geometry(data: dict) -> FrameGeometry:
    """The geometry of the frames of a dataset: from `channels` and `frame_length` of the
    description, the rest taken from its first frame (of train, val or test)."""
    axis = data.get("sensor_axis", "x")
    channels, length = data.get("channels"), data.get("frame_length")
    if channels is None or length is None:
        first = None
        for split in SPLIT_OPTIONS:
            folder = Path(data[split]) / "images" if split in data else None
            if folder is not None and folder.is_dir():
                first = find_images(folder)[0]
                break
        if first is None:
            raise ValueError("No frames to learn the frame geometry from; give data.channels and data.frame_length")
        found = FrameGeometry.from_frame(frame_size(first), axis)
        if channels is not None and channels != found.channels:
            raise ValueError(f"{first}: {found.channels} channels on {axis}, data.channels says {channels}")
        if length is not None and length != found.frame_length:
            raise ValueError(f"{first}: frame of {found.frame_length} polls, data.frame_length says {length}")
        return found
    return FrameGeometry(channels, length, axis)


def validate_image(path: Path, geometry: FrameGeometry) -> None:
    with Image.open(path) as image:
        if (image.height, image.width) != geometry.frame_shape or image.mode not in GRAYSCALE_MODES:
            rows, columns = geometry.frame_shape
            raise ValueError(f"{path}: expected a grayscale frame of {columns} x {rows} pixels "
                             f"({geometry.channels} channels on {geometry.sensor_axis}, "
                             f"{geometry.frame_length} polls)")


def scale_boxes(boxes, shape) -> np.ndarray:
    """Normalized xyxy boxes in pixels of a frame of `shape` (rows, columns)."""
    rows, columns = shape
    return np.asarray(boxes, dtype=np.float64).reshape(-1, 4) * [columns, rows, columns, rows]


def parse_yolo_box(line: str) -> list[float]:
    """One YOLO line as an xyxy box normalized to the image, rounded to float32."""
    values = np.asarray([float(value) for value in line.split()])
    if values.shape != (5,) or not np.isfinite(values).all():
        raise ValueError("expected five finite numbers")
    class_id, center_x, center_y, width, height = values
    if class_id != 0 or min(width, height) <= 0:
        raise ValueError("expected class 0 and positive width/height")

    box = np.asarray([
        center_x - width / 2,
        center_y - height / 2,
        center_x + width / 2,
        center_y + height / 2,
    ])
    if box.min() < -1e-6 or box.max() > 1 + 1e-6:
        raise ValueError("box extends beyond image")
    box = np.clip(box, 0, 1)
    if not (box[2:] > box[:2]).all():
        raise ValueError("box has nonpositive width/height after clipping")
    box = box.astype(np.float32)
    if not (box[2:] > box[:2]).all():
        raise ValueError("box has nonpositive width/height in float32")
    return box.tolist()


def read_yolo_boxes(path: Path) -> np.ndarray:
    """All boxes of a YOLO label file as a float32 [N, 4] array in image axes."""
    boxes = []
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            boxes.append(parse_yolo_box(line))
        except ValueError as error:
            raise ValueError(f"{path}:{line_number}: {error}") from error
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def orphan_labels(root: Path, image_paths) -> list[Path]:
    """Label files of a split folder that belong to no image."""
    image_stems = {path.stem for path in image_paths}
    return sorted(path for path in (root / "labels").glob("*.txt") if path.stem not in image_stems)


class FrameSet:
    """The frames of one split folder and their boxes, checked once when the folder is opened."""

    def __init__(self, root: str | Path, geometry: FrameGeometry, scale: float = 255.0, offset: float = 0.0):
        self.root = Path(root)
        self.geometry = geometry
        self.sensor_axis = geometry.sensor_axis
        self.scale = float(scale)
        self.offset = float(offset)
        self.paths = find_images(self.root / "images")
        self.boxes = self.load_annotations()

    @property
    def shape(self):
        """(rows, columns) of every frame, in its stored orientation."""
        return self.geometry.frame_shape

    def pixel_boxes(self, index: int) -> np.ndarray:
        """The boxes of frame `index` in its pixels (xyxy, float64)."""
        return scale_boxes(self.boxes[index], self.shape)

    def load_annotations(self) -> list[np.ndarray]:
        boxes = []
        for path in self.paths:
            validate_image(path, self.geometry)
            label_path = self.root / "labels" / f"{path.stem}.txt"
            if not label_path.is_file():
                raise FileNotFoundError(f"Missing YOLO annotation: {label_path}")
            boxes.append(read_yolo_boxes(label_path))
        orphans = orphan_labels(self.root, self.paths)
        if orphans:
            stems = sorted(path.stem for path in orphans)
            raise ValueError(f"Labels without images in {self.root}: {stems[:5]}")
        return boxes

    def __len__(self):
        return len(self.paths)

    def pixels(self, index: int) -> np.ndarray:
        """Intensities of one frame in image orientation, read from the file again."""
        path = self.paths[index]
        with Image.open(path) as image:
            values = np.array(image, dtype=np.float32)
        pixels = (values - self.offset) / self.scale
        if not np.isfinite(pixels).all():
            raise ValueError(f"Nonfinite signal in {path}")
        return pixels


def open_split(data: dict, split: str, geometry: FrameGeometry | None = None) -> FrameSet:
    """The frames of `split`; every one must fit `geometry` (by default the dataset's own)."""
    return FrameSet(data[split], geometry or frame_geometry(data), data["scale"], data["offset"])
