"""Reading the source dataset: its description, frames and boxes."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import yaml

from scene_editor.dataset import load_data_description, open_split, read_yolo_boxes, sensor_profile
from scene_editor.sensor import SensorProfile


def write_frames(split_root: Path, labels="0 0.1 0.3 0.05 0.1\n", size=(40, 40)):
    (split_root / "images").mkdir(parents=True)
    (split_root / "labels").mkdir()
    pixels = np.zeros(size[::-1], dtype=np.uint8)
    pixels[10:14, 3:5] = 150
    Image.fromarray(pixels).save(split_root / "images/a.png")
    (split_root / "labels/a.txt").write_text(labels)


def describe(path: Path, **data) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"data": data}), encoding="utf-8")
    return path


class DataDescriptionTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)

    def tearDown(self):
        self.folder.cleanup()

    def test_paths_resolve_against_the_file_and_defaults_apply(self):
        describe(self.root / "set" / "data.yaml", train="./train", val="../other/val")
        for given in (self.root / "set", self.root / "set" / "data.yaml"):
            with self.subTest(given=given):
                data = load_data_description(given)
                self.assertEqual(data["train"], str((self.root / "set" / "train").resolve()))
                self.assertEqual(data["val"], str((self.root / "other" / "val").resolve()))
                self.assertEqual((data["sensor_axis"], data["scale"], data["offset"]), ("x", 255.0, 0.0))

    def test_any_yaml_with_a_data_section_is_accepted(self):
        # For example a model configuration that names its training data.
        path = self.root / "model.yaml"
        path.write_text(yaml.safe_dump({"data": {"type": "yolo_images", "train": "t", "val": "v"},
                                        "model": {"backbone": []}}), encoding="utf-8")
        self.assertEqual(load_data_description(path)["train"], str((self.root / "t").resolve()))

    def test_invalid_descriptions_are_rejected_with_the_file_named(self):
        cases = {
            "no section": {"train": "t"},
            "unknown option": {"data": {"train": "t", "val": "v", "names": ["egg"]}},
            "missing val": {"data": {"train": "t"}},
            "axis": {"data": {"train": "t", "val": "v", "sensor_axis": "z"}},
            "scale": {"data": {"train": "t", "val": "v", "scale": 0}},
            "type": {"data": {"train": "t", "val": "v", "type": "coco"}},
        }
        for name, content in cases.items():
            with self.subTest(name):
                path = self.root / f"{name}.yaml"
                path.write_text(yaml.safe_dump(content), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "yaml"):
                    load_data_description(path)
        with self.assertRaisesRegex(ValueError, "No dataset description"):
            load_data_description(self.root / "missing")


class FrameSetTests(unittest.TestCase):
    def test_frames_scale_and_boxes_in_image_axes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_frames(root / "train")
            frames = open_split({"train": str(root / "train"), "sensor_axis": "x",
                                 "scale": 255.0, "offset": 0.0}, "train")
            self.assertEqual([path.name for path in frames.paths], ["a.png"])
            pixels = frames.pixels(0)
            self.assertEqual(pixels.dtype, np.float32)
            self.assertAlmostEqual(float(pixels[11, 3]), 150 / 255, places=7)
            np.testing.assert_allclose(frames.boxes[0] * 40, [[3, 10, 5, 14]], atol=1e-5)

    def test_broken_folders_fail_when_opened(self):
        cases = {
            "orphan label": lambda root: (root / "labels/b.txt").write_text(""),
            "missing label": lambda root: (root / "labels/a.txt").unlink(),
            "bad box": lambda root: (root / "labels/a.txt").write_text("0 0.1 0.3 0.05\n"),
        }
        for name, damage in cases.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as folder:
                root = Path(folder) / "train"
                write_frames(root)
                damage(root)
                with self.assertRaises((ValueError, FileNotFoundError)):
                    open_split({"train": str(root), "sensor_axis": "x", "scale": 255.0, "offset": 0.0}, "train")

    def test_the_first_frame_sets_the_profile_on_either_axis(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "train"
            write_frames(root, size=(24, 64))  # 24 columns, 64 rows
            data = {"train": str(root), "scale": 255.0, "offset": 0.0}
            self.assertEqual(sensor_profile({**data, "sensor_axis": "x"}), SensorProfile(24, 64, "x"))
            self.assertEqual(sensor_profile({**data, "sensor_axis": "y"}), SensorProfile(64, 24, "y"))
            frames = open_split({**data, "sensor_axis": "x"}, "train")
            self.assertEqual(frames.shape, (64, 24))
            np.testing.assert_allclose(frames.pixel_boxes(0), [[1.8, 16.0, 3.0, 22.4]], atol=1e-5)

    def test_frames_must_match_the_described_profile(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "train"
            write_frames(root, size=(24, 64))
            data = {"train": str(root), "scale": 255.0, "offset": 0.0, "sensor_axis": "x"}
            self.assertEqual(sensor_profile({**data, "channels": 24}), SensorProfile(24, 64, "x"))
            with self.assertRaisesRegex(ValueError, "24 channels on x, data.channels says 40"):
                sensor_profile({**data, "channels": 40})
            with self.assertRaisesRegex(ValueError, "40 x 40 pixels"):
                open_split(data, "train", SensorProfile(40, 40, "x"))
            # A described profile needs no frame at all.
            self.assertEqual(sensor_profile({"channels": 16, "frame_length": 8, "sensor_axis": "y"}),
                             SensorProfile(16, 8, "y"))
            self.assertEqual(SensorProfile(16, 8, "y").frame_shape, (16, 8))
            for bad in (dict(channels=0, frame_length=8), dict(channels=16, frame_length=True),
                        dict(channels=16, frame_length=8, sensor_axis="z")):
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    SensorProfile(**bad)

    def test_label_errors_name_file_and_line(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "a.txt"
            path.write_text("0 0.5 0.5 0.2 0.2\n\n1 0.5 0.5 0.2 0.2\n")
            with self.assertRaisesRegex(ValueError, r"a\.txt:3: expected class 0"):
                read_yolo_boxes(path)


if __name__ == "__main__":
    unittest.main()
