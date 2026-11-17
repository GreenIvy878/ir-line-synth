"""Check that saved examples render again exactly from their own descriptions.

Every saved scene keeps its description and digest in provenance/<name>.json, every tape
in tapes/<split>/<name>.json, every synthetic record without real data in
records/<split>/<name>.json (its frames point at it from provenance/). Rendering a
description again (from the source dataset, or from the sensor profile for a record) must
give the recorded digest, and the files on disk must hold that same signal and those boxes.
A difference means the source frames, the profile, the files or the program changed since
the save.
"""

import csv
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .record import LABEL_COLUMNS, record_digest, render_record
from .scene import LEVELS, pixel_levels
from .sensor_model import load_profile, profile_file
from .store import OUTPUT_SPLITS, example_digest, file_digest
from .workspace import Workspace


@dataclass(frozen=True)
class Check:
    path: str
    kind: str        # scene, tape, record or frame (of a record)
    ok: bool
    problem: str = ""


def provenance_files(path):
    """The descriptions to check: one file, or every scene and tape of an output folder."""
    path = Path(path)
    if path.is_file():
        return [path]
    if not (path / "synthetic_dataset.json").is_file():
        raise ValueError(f"{path} is neither a provenance file nor a synthetic output")
    files = []
    for split in OUTPUT_SPLITS:
        files += sorted((path / split / "provenance").glob("*.json"))
        files += sorted((path / "tapes" / split).glob("*.json"))
        files += sorted((path / "records" / split).glob("*.json"))
    return files


def check_scene(workspace, path, metadata):
    load = workspace.source_loader()
    rendered, preview = workspace.preview(metadata["scene"], load)
    if preview["digest"] != metadata["digest"]:
        return "renders differently from its description"
    root = path.parent.parent
    image_path = root / "images" / f"{path.stem}.png"
    labels_path = root / "labels" / f"{path.stem}.txt"
    with Image.open(image_path) as image:
        stored = np.asarray(image)
    if not np.array_equal(stored, pixel_levels(rendered.pixels)):
        return f"{image_path.name} differs from the rendered scene"
    if file_digest(stored.astype(np.float32) / LEVELS, labels_path.read_text(encoding="utf-8")) != metadata["digest"]:
        return f"{labels_path.name} differs from the rendered boxes"
    return ""


def check_tape(workspace, path, metadata):
    rendered = workspace.render_tape(metadata["tape"], workspace.source_loader())
    if rendered.digest != metadata["digest"]:
        return "renders differently from its description"
    files = metadata["files"]
    text = (path.parent / files["signal"]).read_text(encoding="utf-8")
    signal = np.loadtxt(io.StringIO(text), delimiter=",", skiprows=1 if metadata.get("header") else 0, ndmin=2)
    if metadata.get("values") == "intensity":
        signal = np.round(signal * LEVELS)
    if signal.shape != rendered.levels.shape or not np.array_equal(signal, rendered.levels):
        return f"{files['signal']} differs from the rendered signal"
    with (path.parent / files["labels"]).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    columns = ("channel_min", "row_min", "channel_max", "row_max")
    labels = np.asarray([[float(row[key]) for key in columns] for row in rows], np.float64).reshape(-1, 4)
    if labels.shape != rendered.boxes.shape or not np.allclose(labels, rendered.boxes, rtol=1e-9, atol=1e-9):
        return f"{files['labels']} differs from the rendered boxes"
    return ""


class Records:
    """Records rendered again from their descriptions, once each, for the record and its frames."""

    def __init__(self):
        self.rendered = {}

    def get(self, path):
        path = Path(path)
        if path not in self.rendered:
            metadata = json.loads(path.read_text(encoding="utf-8"))
            profile = metadata["profile"]
            file = profile_file(profile["source"])
            if hashlib.sha256(file.read_bytes()).hexdigest() != profile["sha256"]:
                raise ValueError(f"the sensor profile {profile['source']} changed since the save")
            self.rendered[path] = render_record(metadata["record"], load_profile(file))
        return self.rendered[path]


def check_record(records, path, metadata):
    rendered = records.get(path)
    if record_digest(rendered) != metadata["digest"]:
        return "renders differently from its description"
    files = metadata["files"]
    codes = np.loadtxt(path.parent / files["codes"], delimiter=",", dtype=np.int64, ndmin=2)
    if codes.shape != rendered.codes.shape or not np.array_equal(codes, rendered.codes):
        return f"{files['codes']} differs from the rendered codes"
    with (path.parent / files["labels"]).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    boxes = np.asarray([[float(row[key]) for key in LABEL_COLUMNS[2:]] for row in rows], np.float64).reshape(-1, 4)
    if boxes.shape != rendered.boxes.shape or not np.allclose(boxes, rendered.boxes, rtol=1e-9, atol=1e-9):
        return f"{files['labels']} differs from the rendered boxes"
    return ""


def check_record_frame(records, path, metadata):
    """A frame of a record: the tile of the record rendered again, against the files of the frame."""
    root = path.parent.parent
    reference = metadata["record"]
    rendered = records.get(root.parent / "records" / root.name / f"{reference['name']}.json")
    tile = reference["tile"]
    pixels = rendered.frames[tile].astype(np.float32) / LEVELS
    boxes = np.asarray([box for _, box, _ in rendered.frame_boxes[tile]], np.float64).reshape(-1, 4)
    if example_digest(pixels, boxes, tuple(0 for _ in boxes)) != metadata["digest"]:
        return "renders differently from its record"
    with Image.open(root / "images" / f"{path.stem}.png") as image:
        stored = np.asarray(image)
    if not np.array_equal(stored, rendered.frames[tile]):
        return f"{path.stem}.png differs from the rendered frame"
    text = (root / "labels" / f"{path.stem}.txt").read_text(encoding="utf-8")
    if file_digest(stored.astype(np.float32) / LEVELS, text) != metadata["digest"]:
        return f"{path.stem}.txt differs from the rendered boxes"
    return ""


def verify(path, data=None):
    """Checks of every description under `path`; scenes and tapes need the dataset described by `data`."""
    workspace, records, results = None, Records(), []
    for file in provenance_files(path):
        metadata = json.loads(file.read_text(encoding="utf-8"))
        if "tape" in metadata:
            kind = "tape"
        elif "record" in metadata:
            kind = "record" if "files" in metadata else "frame"
        else:
            kind = "scene"
        try:
            if kind == "record":
                problem = check_record(records, file, metadata)
            elif kind == "frame":
                problem = check_record_frame(records, file, metadata)
            elif data is None:
                problem = "needs the source dataset (--data) to be rendered again"
            else:
                workspace = workspace or Workspace(data)
                problem = (check_tape if kind == "tape" else check_scene)(workspace, file, metadata)
        except (ValueError, OSError, KeyError) as error:
            problem = f"cannot be rendered again: {error}"
        results.append(Check(str(file), kind, not problem, problem))
    return results
