"""Lines of any size: a dataset of 48 channels and 32-poll frames, stored on either axis,
goes through the editor, the generators and the export like the 40 x 40 reference."""

import base64
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from scene_editor.dataset import load_data_description, open_split, read_yolo_boxes, scale_boxes, sensor_profile
from scene_editor.sensor import SensorProfile
from scene_editor.web.views import EditorViews
from test_morph import draw_egg


CHANNELS, LENGTH = 48, 32


def line_frame(seed):
    """A time-major frame (polls x channels) with an egg of every kind, away from the edges."""
    rng = np.random.default_rng(seed)
    frame = np.full((LENGTH, CHANNELS), 0.004, np.float32)
    jitter = lambda: float(rng.uniform(-0.3, 0.3))
    boxes = [
        draw_egg(frame, 6 + jitter(), 2, int(rng.integers(6, 10)), width=float(rng.uniform(2.6, 3.6))),
        draw_egg(frame, 14 + jitter(), 2, 14),
        draw_egg(frame, 6 + jitter(), 18, 3, width=3),
        draw_egg(frame, 24 + jitter(), 2, 8, track=lambda t: 1.1 * t),
        draw_egg(frame, 42 + jitter(), 14, 10, track=lambda t: 1.5 * math.sin(t),
                 gain=lambda t: 0.0 if t in (4, 5) else 1.0),
    ]
    return np.clip(frame, 0, 1), np.asarray(boxes, np.float64)


def write_line_dataset(root, axis, frames=8):
    for split in ("train", "val"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir()
        for seed in range(frames):
            pixels, boxes = line_frame(seed + (100 if split == "val" else 0))
            if axis == "y":
                pixels, boxes = pixels.T, boxes[:, [1, 0, 3, 2]]
            rows, columns = pixels.shape
            Image.fromarray(np.round(pixels * 255).astype(np.uint8)).save(root / split / "images" / f"f{seed}.png")
            lines = [f"0 {(b[0] + b[2]) / 2 / columns:.6f} {(b[1] + b[3]) / 2 / rows:.6f} "
                     f"{(b[2] - b[0]) / columns:.6f} {(b[3] - b[1]) / rows:.6f}" for b in boxes]
            (root / split / "labels" / f"f{seed}.txt").write_text("\n".join(lines) + "\n")
    (root / "data.yaml").write_text(f"data:\n  train: train\n  val: val\n  sensor_axis: {axis}\n")
    return load_data_description(root)


class LineGeometryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def test_every_tool_follows_the_profile_of_the_dataset(self):
        for axis in ("x", "y"):
            with self.subTest(axis=axis):
                self.check_axis(axis)

    def check_axis(self, axis):
        data = write_line_dataset(self.root / axis / "raw", axis)
        out = self.root / axis / "out"
        views = EditorViews(data)
        profile = SensorProfile(CHANNELS, LENGTH, axis)
        shape = profile.frame_shape
        self.assertEqual(views.workspace.profile, profile)
        sensor = views.get("/api/synthetic/sensor", {})
        self.assertEqual((sensor["frame_height"], sensor["frame_width"]), shape)

        # The editor: a frame and its boxes in their own orientation.
        sample = views.get("/api/synthetic/sample", dict(split=["train"], id=["f0.png"]))
        self.assertEqual(np.asarray(sample["matrix"]).shape, shape)
        self.assertEqual(sample["clipped"], [False] * 5)

        # The autogenerator on generated backgrounds, then the export.
        pool = [dict(source=dict(split="train", id=f"f{seed}.png"), box_index=box, transform={})
                for seed in range(4) for box in range(5)]
        batch = views.post("/api/synthetic/generate", dict(background=None, pool=pool, rules=dict(
            images=3, count_min=2, count_max=4, background_mode="generated", split="train", seed=3)))
        self.assertEqual(len(batch["candidates"]), 3)
        for candidate in batch["candidates"]:
            self.assertEqual(np.asarray(candidate["preview"]["matrix"]).shape, shape)
        chosen = batch["candidates"][0]
        saved = views.post("/api/synthetic/generated-save", dict(
            items=[dict(scene=chosen["scene"], expected_digest=chosen["preview"]["digest"])],
            directory=str(out), split="train"))["saved"][0]
        with Image.open(saved["image"]) as image:
            self.assertEqual((image.height, image.width), shape)
        np.testing.assert_allclose(scale_boxes(read_yolo_boxes(Path(saved["annotations"])), shape),
                                   chosen["preview"]["boxes_pixels_xyxy"], atol=1e-4)
        # The output reads back as a dataset of the same line.
        output = load_data_description(out)
        self.assertEqual(sensor_profile(output), profile)
        self.assertEqual(len(open_split(output, "train")), 1)

        # New objects from the real ones of this line.
        objects = views.post("/api/synthetic/morph-generate", dict(kind="blob", count=4, seed=1, split="train"))
        self.assertTrue(objects["candidates"])

        # A tape has all the channels of the line.
        tape = views.post("/api/synthetic/tape-generate", dict(pool=pool, rules=dict(
            length=240, objects=6, split="train", seed=2)))
        self.assertEqual(tape["preview"]["channels"], CHANNELS)
        levels = np.frombuffer(base64.b64decode(tape["preview"]["levels_base64"]), np.uint8)
        self.assertEqual(levels.size, 240 * CHANNELS)
        record = views.post("/api/synthetic/tape-save", dict(
            tape=tape["tape"], expected_digest=tape["preview"]["digest"], directory=str(out), split="train"))
        rows = Path(record["signal"]).read_text(encoding="utf-8").splitlines()
        self.assertEqual((len(rows), len(rows[0].split(","))), (240, CHANNELS))

    def test_an_output_holds_frames_of_one_size(self):
        small = write_line_dataset(self.root / "small", "x")
        out = self.root / "out"
        for data in (small, None):
            if data is None:
                # A 40 x 40 dataset may not add its frames to an output of 48 x 32 frames.
                from test_splits import write_split_dataset
                data = write_split_dataset(self.root / "reference")
            views = EditorViews(data)
            entry = dict(source=dict(split="train", id=views.get("/api/synthetic/catalog", {})["train"][0]["id"]),
                         box_index=0, transform={})
            scene = dict(background=entry["source"], objects=[dict(entry, x=0, y=0)], max_iou=1.0)
            preview = views.post("/api/synthetic/preview", dict(scene=scene))
            body = dict(scene=scene, expected_digest=preview["digest"], directory=str(out), split="train")
            if data is small:
                views.post("/api/synthetic/save", body)
            else:
                with self.assertRaisesRegex(ValueError, "48 x 32"):
                    views.post("/api/synthetic/save", body)


if __name__ == "__main__":
    unittest.main()
