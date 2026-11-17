"""Check that saved examples render again exactly from their own descriptions.

Every saved scene keeps its description and digest in provenance/<name>.json, every tape
in tapes/<split>/<name>.json. Rendering a description again from the source dataset must
give the recorded digest, and the files on disk must hold that same signal and those boxes.
A difference means the source frames, the files or the program changed since the save.
"""

import csv
from dataclasses import dataclass
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .scene import LEVELS, pixel_levels
from .store import OUTPUT_SPLITS, file_digest
from .workspace import Workspace


@dataclass(frozen=True)
class Check:
    path: str
    kind: str        # scene or tape
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


def verify(path, data):
    """Checks of every description under `path` against the dataset described by `data`."""
    workspace = Workspace(data)
    results = []
    for file in provenance_files(path):
        metadata = json.loads(file.read_text(encoding="utf-8"))
        kind = "tape" if "tape" in metadata else "scene"
        try:
            problem = (check_tape if kind == "tape" else check_scene)(workspace, file, metadata)
        except (ValueError, OSError, KeyError) as error:
            problem = f"cannot be rendered again: {error}"
        results.append(Check(str(file), kind, not problem, problem))
    return results
