"""The work session over one source dataset, shared by the web interface and the command line.

A Workspace opens the splits of the dataset on demand, loads source frames (real or
generated backgrounds), builds the objects of a scene (real boxes or replays, see
morph.py), renders scenes and tapes and saves them through its store. It holds no
state of an editing session: every call takes the full description of what to build.
"""

import json
from pathlib import Path
import threading

import numpy as np

from . import morph
from .background import BackgroundBank, background_options, identifier as background_identifier
from .dataset import open_split, read_yolo_boxes, scale_boxes, frame_geometry
from .scene import SourceImage, prepare_entry, render_scene, validate_boxes, validate_pixels
from .store import SyntheticStore, example_digest
from .tape import render_tape, time_major_object


class Workspace:
    def __init__(self, data, geometry=None):
        """`data`: a dataset description (dataset.load_data_description). The frame geometry is
        the dataset's own (given in it or read from its first frame) unless `geometry` is given."""
        self.data = data
        self.geometry = geometry or frame_geometry(data)
        self.datasets = {}
        self.lock = threading.Lock()
        self.store = SyntheticStore(data, self.source_details)
        # Built on first use from a split: real objects (for new objects) and empty columns (for backgrounds).
        self.libraries = {}
        self.background_banks = {}
        self.build_lock = threading.Lock()

    def library(self, split):
        dataset = self.dataset(split)
        with self.build_lock:
            if split not in self.libraries:
                self.libraries[split] = morph.Library.build(dataset, split)
            return self.libraries[split]

    def background_bank(self, split):
        dataset = self.dataset(split)
        with self.build_lock:
            if split not in self.background_banks:
                self.background_banks[split] = BackgroundBank(dataset)
            return self.background_banks[split]

    def generated_background(self, reference):
        if set(reference) - {"split", "generated"}:
            raise ValueError("A generated background has only split and generated options")
        split = reference.get("split")
        if split not in self.source_splits():
            raise ValueError("Select a source split for the generated background: " + ", ".join(self.source_splits()))
        options = background_options(reference["generated"])
        bank, used = self.background_bank(split), set()
        pixels = bank.generate(options, used)
        validate_pixels(pixels, self.geometry.frame_shape)
        return SourceImage(pixels, np.zeros((0, 4)), (), split, background_identifier(options),
                           tuple(bank.frame_names(used)))

    def is_clipped(self, source, box_index):
        """Whether box `box_index` of a real frame is cut by the frame edge (morph.is_clipped)."""
        obj = self.library(source.split).find(source.identifier, box_index)
        if obj is not None:
            return obj.clipped
        # Too faint to cut out: only the box can tell (the edge rule is the same on both axes).
        return morph.box_near_edge(source.boxes[box_index], source.pixels.shape)

    def source_details(self, row):
        """A provenance row as saved: objects also say whether their box is cut by the frame edge."""
        row = dict(row)
        if row.get("role") == "object":
            obj = self.library(row["split"]).find(row["id"], row["box_index"])
            if obj is not None:
                row.update(clipped=obj.clipped, kind=obj.kind)
        return row

    def prepare(self, obj, load_source):
        """Build an added object: a real box, or a generated one (morph) from a real box."""
        if not isinstance(obj, dict):
            raise ValueError("Each added object must be an object")
        if obj.get("morph") is None:
            return prepare_entry(obj, load_source)
        # Box margins of the annotators of the source split itself: nothing of another split is used.
        margins = self.library(load_source(obj.get("source")).split).stats["box_margin"]
        return morph.prepare_morph(obj, load_source, self.data["sensor_axis"], margins)

    def prepare_tape_object(self, obj, load_source):
        """An added object turned to the time-major orientation of tapes."""
        prepared = self.prepare(obj, load_source)
        return time_major_object(prepared) if self.data["sensor_axis"] == "y" else prepared

    def tape_background(self, background, length):
        """Background pixels [length, channels] (time-major) and their provenance."""
        if background["kind"] == "blank":
            return np.zeros((length, self.geometry.channels), np.float32), []
        options = background_options({key: value for key, value in background.items() if key not in ("kind", "split")})
        bank, used = self.background_bank(background["split"]), set()
        pixels = bank.generate_tape(options, length, used)
        frames = [Path(name).stem for name in bank.frame_names(used)]
        return pixels, [dict(split=background["split"], id=background_identifier(options), role="background",
                             frames=frames)]

    def render_tape(self, spec, load_source):
        return render_tape(spec, load_source, self.prepare_tape_object, self.tape_background, self.geometry.channels)

    def source_splits(self):
        """Splits that can supply real frames and objects: train and val, and test when data.yaml names it."""
        return tuple(split for split in ("train", "val", "test") if split in self.data)

    def dataset(self, split):
        if split not in self.source_splits():
            raise ValueError("Select a source split: " + ", ".join(self.source_splits()))
        with self.lock:
            if split not in self.datasets:
                self.datasets[split] = open_split(self.data, split, self.geometry)
            return self.datasets[split]

    def source(self, reference):
        if not isinstance(reference, dict):
            raise ValueError("Choose a source/background image")
        if "generated" in reference:
            return self.generated_background(reference)
        dataset = self.dataset(reference.get("split"))
        identifier = reference.get("id")
        index = next((i for i, path in enumerate(dataset.paths) if path.name == identifier), None)
        if index is None:
            raise ValueError("Image is not in the configured source dataset")
        # Fetch pixels and labels again so an external source edit cannot go unnoticed.
        path = dataset.paths[index]
        label_path = dataset.root / "labels" / f"{path.stem}.txt"
        matrix = dataset.pixels(index)
        boxes = scale_boxes(read_yolo_boxes(label_path), matrix.shape)
        validate_pixels(matrix, self.geometry.frame_shape)
        # One class per box as the folder held when it was opened; a changed count fails below.
        labels = tuple(0 for _ in dataset.boxes[index])
        validate_boxes(boxes, labels, matrix.shape)
        return SourceImage(matrix, boxes, labels, reference["split"], path.name, (path.name,))

    def source_loader(self):
        """Freeze each distinct source once during a preview/save operation."""
        sources = {}

        def load(reference):
            if isinstance(reference, dict) and "generated" in reference:
                key = ("generated", str(reference.get("split")), json.dumps(reference["generated"], sort_keys=True))
            elif not isinstance(reference, dict) or not all(isinstance(reference.get(key), str) for key in ("split", "id")):
                raise ValueError("A source needs a split and image id")
            else:
                key = (reference["split"], reference["id"])
            if key not in sources:
                sources[key] = self.source(reference)
            return sources[key]

        return load

    def preview(self, spec, load_source):
        rendered = render_scene(spec, load_source, self.prepare)
        payload = rendered.payload()
        payload["digest"] = example_digest(rendered.pixels, rendered.boxes, rendered.labels)
        return rendered, payload

