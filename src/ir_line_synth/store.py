"""Isolated 8-bit PNG + YOLO export, provenance and duplicate protection.

Scenes are stored like the real frames: grayscale PNG read with scale 255. An output
has the splits train, val and test, and a scene or tape goes only into the split all
of its sources come from (objects and background alike, see check_split), so nothing
of one split of the source dataset can reach another one.
"""

import hashlib
import io
import json
from pathlib import Path
import re
import threading
import uuid

import numpy as np
from PIL import Image
import yaml

from .dataset import IMAGE_EXTENSIONS, SPLIT_OPTIONS, parse_yolo_box
from . import PROGRAM, __version__
from .scene import LEVELS, pixel_levels


_EXPORT_LOCK = threading.Lock()
TAPE_NAME = re.compile(r"[A-Za-z0-9_-]{1,64}")
OUTPUT_SPLITS = ("train", "val", "test")


def output_split(split):
    if split not in OUTPUT_SPLITS:
        raise ValueError("Output split must be train, val or test")
    return split


def check_split(split, provenance):
    """Refuse a scene or tape whose objects or background come from another split than `split`."""
    output_split(split)
    for row in provenance:
        if row["split"] != split:
            what = "The background" if row.get("role") == "background" else f"Object {row.get('id')}"
            raise ValueError(f"A {split} scene takes every object and its background from {split} only: "
                             f"{what} comes from {row['split']}")
    return split


def yolo_text(boxes, labels, shape):
    """YOLO lines of pixel xyxy boxes on a frame of `shape` (rows, columns)."""
    rows, columns = shape
    lines = []
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = np.asarray(box, np.float64) / [columns, rows, columns, rows]
        lines.append(f"{label} {(x1 + x2) / 2:.12g} {(y1 + y2) / 2:.12g} {x2 - x1:.12g} {y2 - y1:.12g}")
    return "\n".join(lines) + ("\n" if lines else "")


def example_digest(pixels, boxes, labels):
    """Compare pixels and semantic annotations, independent of row order."""
    return file_digest(pixels, yolo_text(boxes, labels, pixels.shape))


def file_digest(pixels, text):
    """The digest of an example as its files hold it: intensities and the YOLO label text.

    Annotations are canonicalized exactly as a dataset reader rounds them to float32. A
    saved example is digested from its label lines directly: turning them into pixel boxes
    and back is not exact in float32, so the same file could give another digest.
    """
    rows = []
    for line in text.splitlines():
        if line.strip():
            rows.append((int(line.split()[0]), *parse_yolo_box(line)))
    digest = hashlib.sha256(pixel_levels(pixels).tobytes())
    digest.update(np.asarray(sorted(rows), dtype="<f8").reshape(-1, 5).tobytes())
    return digest.hexdigest()


def program():
    """The program and version that wrote a file, for its provenance."""
    return dict(name=PROGRAM, version=__version__)


def exclusive_write(path, payload, created):
    with path.open("xb") as stream:
        created.append(path)
        stream.write(payload)


def write_new_files(files):
    """Write {path: bytes}, none of which may exist yet; on a failure remove what was written."""
    created = []
    try:
        for path, payload in files.items():
            exclusive_write(path, payload, created)
    except (OSError, ValueError):
        for path in reversed(created):
            path.unlink(missing_ok=True)
        raise


def provenance_bytes(metadata):
    return json.dumps(metadata, allow_nan=False, indent=2).encode("utf-8")


class SyntheticStore:
    def __init__(self, data, details=None):
        self.axis = data["sensor_axis"]
        # details(row) -> the provenance row as written (the editor adds what it knows of a source).
        self.details = details or dict
        self.protected = [Path(data[key]).resolve() for key in SPLIT_OPTIONS if key in data]

    def output_root(self, directory):
        """The requested output folder, once it is known to lie outside every source dataset."""
        if not isinstance(directory, str) or not directory.strip():
            raise ValueError("Specify a separate synthetic output directory")
        path = Path(directory).resolve()
        for source in self.protected:
            if path == source or path in source.parents or source in path.parents:
                raise ValueError("Synthetic output must be separate from all source datasets")
        return path

    def initialize(self, directory):
        root = self.output_root(directory)
        marker = root / "synthetic_dataset.json"
        metadata = dict(schema=1, sensor_axis=self.axis, scale=float(LEVELS), offset=0.0,
                        image_format="8-bit PNG", annotations="YOLO")
        if marker.exists():
            if json.loads(marker.read_text(encoding="utf-8")) != metadata:
                raise ValueError("Output directory has incompatible synthetic dataset metadata")
        else:
            if root.exists() and any(root.iterdir()):
                raise ValueError("Choose an empty directory or an existing synthetic dataset")
            root.mkdir(parents=True, exist_ok=True)
            with marker.open("x", encoding="utf-8") as stream:
                json.dump(metadata, stream, indent=2)
        for split in OUTPUT_SPLITS:
            for kind in ("images", "labels", "provenance"):
                (root / split / kind).mkdir(parents=True, exist_ok=True)
        config_path = root / "data.yaml"
        data_config = dict(type="yolo_images", train="./train", val="./val", test="./test",
                           sensor_axis=self.axis, scale=float(LEVELS), offset=0.0)
        if config_path.exists():
            found = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            old = {key: value for key, value in data_config.items() if key != "test"}
            if found == {"data": old}:
                # An output written before the test split: the same format, test added.
                config_path.write_text(yaml.safe_dump({"data": data_config}, sort_keys=False), encoding="utf-8")
            elif found != {"data": data_config}:
                raise ValueError("Synthetic data.yaml does not match the 8-bit PNG export format")
        else:
            with config_path.open("x", encoding="utf-8") as stream:
                yaml.safe_dump({"data": data_config}, stream, sort_keys=False)
        return root

    def existing_digests(self, root, shape=None):
        """Digests of the examples already in the output; with `shape`, every one must have it
        (an output holds frames of one size, so it can be read back as one dataset)."""
        result = set()
        for split in OUTPUT_SPLITS:
            for path in (root / split / "images").iterdir():
                if path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                labels_path = root / split / "labels" / f"{path.stem}.txt"
                if not labels_path.is_file():
                    raise ValueError(f"Incomplete synthetic example: {path.name}")
                with Image.open(path) as image:
                    pixels = np.asarray(image, dtype=np.float32) / LEVELS
                if shape is not None and pixels.shape != tuple(shape):
                    raise ValueError(f"The output holds frames of {pixels.shape[1]} x {pixels.shape[0]} "
                                     f"({path.name}); a {shape[1]} x {shape[0]} frame cannot join them")
                result.add(file_digest(pixels, labels_path.read_text(encoding="utf-8-sig")))
        return result

    def save(self, directory, split, rendered, scene):
        check_split(split, rendered.provenance)
        digest = example_digest(rendered.pixels, rendered.boxes, rendered.labels)
        buffer = io.BytesIO()
        Image.fromarray(pixel_levels(rendered.pixels)).save(buffer, format="PNG")
        with _EXPORT_LOCK:
            root = self.initialize(directory)
            if digest in self.existing_digests(root, rendered.pixels.shape):
                raise ValueError("An identical image and annotations have already been saved")
            name = "synthetic_" + uuid.uuid4().hex
            image_path = root / split / "images" / f"{name}.png"
            labels_path = root / split / "labels" / f"{name}.txt"
            provenance_path = root / split / "provenance" / f"{name}.json"
            metadata = dict(program=program(), digest=digest, sources=[self.details(row) for row in rendered.provenance],
                            scene=scene, statistics=rendered.statistics(), output_split=split)
            labels = yolo_text(rendered.boxes, rendered.labels, rendered.pixels.shape)
            write_new_files({image_path: buffer.getvalue(), labels_path: labels.encode("utf-8"),
                             provenance_path: provenance_bytes(metadata)})
        return dict(image=str(image_path), annotations=str(labels_path), provenance=str(provenance_path), digest=digest)

    def save_tape(self, directory, split, rendered, spec, name=None, values="levels", header=False):
        """A tape as `tapes/<split>/<name>.csv` (signal), `.labels.csv` (boxes) and `.json` (how it was made)."""
        # tape imports the generator, which imports this module.
        from .tape import labels_csv, tape_csv
        check_split(split, rendered.provenance)
        if name is not None and not (isinstance(name, str) and TAPE_NAME.fullmatch(name)):
            raise ValueError("The tape name may hold only letters, digits, _ and - (at most 64)")
        signal = tape_csv(rendered.levels, values, header).encode("utf-8")
        labels = labels_csv(rendered, spec).encode("utf-8")
        digest = rendered.digest
        with _EXPORT_LOCK:
            root = self.initialize(directory)
            folder = root / "tapes" / split
            folder.mkdir(parents=True, exist_ok=True)
            for path in folder.glob("*.json"):
                try:
                    known = json.loads(path.read_text(encoding="utf-8")).get("digest")
                except (OSError, ValueError):
                    continue
                if known == digest:
                    raise ValueError(f"An identical tape has already been saved: {path.stem}")
            name = name or "tape_" + uuid.uuid4().hex[:12]
            signal_path = folder / f"{name}.csv"
            labels_path = folder / f"{name}.labels.csv"
            provenance_path = folder / f"{name}.json"
            metadata = dict(
                schema=1, program=program(), digest=digest, output_split=split, sensor_axis=self.axis,
                layout="signal CSV: a row per poll (time), a column per channel; labels CSV: boxes "
                       "[min, max) in channels (x) and rows (y)",
                length=int(rendered.levels.shape[0]), channels=int(rendered.levels.shape[1]), values=values,
                scale=float(LEVELS) if values == "levels" else 1.0, header=bool(header),
                files=dict(signal=signal_path.name, labels=labels_path.name),
                statistics=rendered.statistics(),
                sources=[self.details(row) for row in rendered.provenance], tape=spec)
            write_new_files({signal_path: signal, labels_path: labels, provenance_path: provenance_bytes(metadata)})
        return dict(signal=str(signal_path), labels=str(labels_path), provenance=str(provenance_path), digest=digest)

    def save_record(self, directory, split, rendered, spec, profile):
        """A synthetic record (record.py) and its frames.

        The record goes to `records/<split>/<name>.csv` (codes), `.labels.csv` (boxes) and `.json`
        (its description, the profile it was made with and the program); every frame of it becomes an
        example of `split` whose provenance points at the record and the tile. A frame identical to
        one already saved is skipped. `profile` is {"source", "sha256"} (record.profile_reference).
        """
        from .record import codes_csv, labels_csv, record_digest
        output_split(split)
        if rendered.frames.dtype != np.uint8:
            raise ValueError("The frames of this profile are not normalized (frames.normalize: none) and cannot be "
                             "saved as 8-bit PNG")
        digest = record_digest(rendered)
        with _EXPORT_LOCK:
            root = self.initialize(directory)
            folder = root / "records" / split
            folder.mkdir(parents=True, exist_ok=True)
            for path in folder.glob("*.json"):
                if json.loads(path.read_text(encoding="utf-8")).get("digest") == digest:
                    raise ValueError(f"An identical record has already been saved: {path.stem}")
            existing = self.existing_digests(root, rendered.frames.shape[1:])
            name = "record_" + uuid.uuid4().hex[:12]
            files, frames, skipped = {}, [], 0
            for tile, (pixels, items) in enumerate(zip(rendered.frames, rendered.frame_boxes)):
                values = pixels.astype(np.float32) / LEVELS
                boxes = np.asarray([box for _, box, _ in items], np.float64).reshape(-1, 4)
                labels = tuple(0 for _ in items)
                tile_digest = example_digest(values, boxes, labels)
                if tile_digest in existing:
                    skipped += 1
                    continue
                existing.add(tile_digest)
                frame = f"{name}_{tile:04d}"
                buffer = io.BytesIO()
                Image.fromarray(pixels).save(buffer, format="PNG")
                metadata = dict(program=program(), digest=tile_digest, output_split=split,
                                record=dict(name=name, tile=tile, digest=digest),
                                objects=[dict(object=index, name=spec["objects"][index].get("name", ""), cut=cut)
                                         for index, _, cut in items])
                files[root / split / "images" / f"{frame}.png"] = buffer.getvalue()
                files[root / split / "labels" / f"{frame}.txt"] = yolo_text(boxes, labels, values.shape).encode("utf-8")
                files[root / split / "provenance" / f"{frame}.json"] = provenance_bytes(metadata)
                frames.append(frame)
            codes_path, labels_path, record_path = (folder / f"{name}{suffix}"
                                                    for suffix in (".csv", ".labels.csv", ".json"))
            metadata = dict(
                schema=1, program=program(), digest=digest, output_split=split, profile=profile,
                layout="codes CSV: a row per poll, a column per channel; labels CSV: boxes [min, max) in "
                       "channels and rows of `pool` polls; frames: tiles of `frame_length` rows",
                polls=int(rendered.codes.shape[0]), channels=int(rendered.codes.shape[1]),
                pool=int(rendered.codes.shape[0] // spec["rows"]),
                frame_length=int(rendered.frames.shape[1]),
                files=dict(codes=codes_path.name, labels=labels_path.name), frames=frames, skipped_frames=skipped,
                record=spec)
            files[codes_path] = codes_csv(rendered.codes).encode("utf-8")
            files[labels_path] = labels_csv(rendered, spec).encode("utf-8")
            files[record_path] = provenance_bytes(metadata)
            write_new_files(files)
        return dict(record=str(record_path), codes=str(codes_path), frames=len(frames), skipped_frames=skipped,
                    digest=digest)
