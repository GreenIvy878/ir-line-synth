"""Motion replay: object analysis, replay geometry, judging and the demo export."""

from pathlib import Path
import json
import math
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from scene_editor import morph
from scene_editor.dataset import read_yolo_boxes


def draw_egg(frame, x0, y0, rows, width=3.5, peak=0.9, track=None, gain=None):
    """A smooth egg trace: row t is centred at x0 + track(t); returns its box (xyxy)."""
    xs = np.arange(frame.shape[1]) + 0.5
    lefts, rights = [], []
    for t in range(rows):
        cx = x0 + (track(t) if track else 0.0)
        envelope = math.sin(math.pi * (t + 0.5) / rows) * (gain(t) if gain else 1.0)
        row = peak * envelope * np.exp(-0.5 * ((xs - cx) / (width / 2.5)) ** 2)
        frame[y0 + t] = np.maximum(frame[y0 + t], row)
        lefts.append(cx - width / 2 - 0.5)
        rights.append(cx + width / 2 + 0.5)
    return [min(lefts), y0, max(rights), y0 + rows]


def kinds_frame(seed):
    """One frame with an egg of every kind, each away from the frame edges."""
    rng = np.random.default_rng(seed)
    frame = np.full((40, 40), 0.004, np.float32)
    jitter = lambda: float(rng.uniform(-0.3, 0.3))
    tilt = float(rng.uniform(-0.15, 0.15))
    boxes = [
        draw_egg(frame, 4 + jitter(), 2, int(rng.integers(6, 10)), width=float(rng.uniform(2.6, 3.6)),
                 peak=float(rng.uniform(0.6, 1.0)), track=lambda t: tilt * t),  # blob
        draw_egg(frame, 12 + jitter(), 2, 14),                                 # long
        draw_egg(frame, 4 + jitter(), 14, 3, width=3),                         # short
        draw_egg(frame, 20 + jitter(), 2, 8, track=lambda t: 1.1 * t),        # rolling
        draw_egg(frame, 33 + jitter(), 22, 10, track=lambda t: 1.5 * math.sin(t),
                 gain=lambda t: 0.0 if t in (4, 5) else 1.0),                  # swinging
    ]
    return np.clip(frame, 0, 1), np.asarray(boxes, np.float32)


def all_objects(frames=8):
    objects = []
    for seed in range(frames):
        pixels, boxes = kinds_frame(seed)
        for index in range(len(boxes)):
            objects.append(morph.cut_object(pixels, boxes, index, f"f{seed}.png"))
    return objects


def still(obj, speed=1.0, lateral=None, gain=1.0):
    """Motion parameters that replay `obj` with a constant speed and given lateral shift."""
    params = dict(kind="blob", flip=False, own=1.0, width=1.0, brightness=1.0, speed=[speed] * 8)
    n = len(morph.output_times(obj, np.asarray(params["speed"])))
    params["lateral"] = (np.zeros(n) if lateral is None else lateral(np.arange(n))).tolist()
    params["gain"] = [gain] * n
    return params


class AnalysisTests(unittest.TestCase):
    def test_kinds_of_drawn_eggs(self):
        pixels, boxes = kinds_frame(0)
        kinds = [morph.cut_object(pixels, boxes, i).kind for i in range(len(boxes))]
        self.assertEqual(kinds, ["blob", "long", "short", "rolling", "swinging"])

    def test_cut_leaves_out_neighbours_and_marks_edge_objects(self):
        frame = np.zeros((40, 40), np.float32)
        first = draw_egg(frame, 10, 10, 7)
        second = draw_egg(frame, 13.5, 12, 7)  # overlaps the first box
        edge = draw_egg(frame, 30, 0, 6)
        boxes = np.asarray([first, second, edge], np.float32)
        obj = morph.cut_object(frame, boxes, 0)
        # Pixels that only the neighbour's box covers are zero in the cut layer.
        x1, y1 = math.floor(first[0]) - 1, math.floor(first[1]) - 1
        cy, cx = np.mgrid[0:obj.layer.shape[0], 0:obj.layer.shape[1]] + 0.5
        cy, cx = cy + y1, cx + x1
        neighbour_only = ((cx >= second[0]) & (cx <= second[2]) & (cy >= second[1]) & (cy <= second[3])
                          & ~((cx >= first[0]) & (cx <= first[2]) & (cy >= first[1]) & (cy <= first[3])))
        self.assertTrue(neighbour_only.any())
        self.assertTrue((obj.layer[neighbour_only] == 0).all())
        self.assertFalse(obj.clipped)
        self.assertTrue(morph.cut_object(frame, boxes, 2).clipped)

    def test_faint_object_is_skipped(self):
        frame = np.zeros((40, 40), np.float32)
        box = draw_egg(frame, 10, 10, 7, peak=0.08)
        self.assertIsNone(morph.cut_object(frame, np.asarray([box], np.float32), 0))

    def test_sensor_axis_y_is_transposed_to_time_major(self):
        pixels, boxes = kinds_frame(0)
        turned, turned_boxes = morph.time_major(pixels.T.copy(), boxes[:, [1, 0, 3, 2]], "y")
        np.testing.assert_array_equal(turned, pixels)
        np.testing.assert_array_equal(turned_boxes, boxes)


class ReplayTests(unittest.TestCase):
    def setUp(self):
        pixels, boxes = kinds_frame(0)
        self.blob = morph.cut_object(pixels, boxes, 0)

    def test_identity_replay_reproduces_the_object(self):
        layer = morph.render(self.blob, still(self.blob))
        expected, _ = morph.trim(np.where(self.blob.layer < 0.012, 0, self.blob.layer), margin=0)
        np.testing.assert_allclose(morph.trim(layer, margin=0)[0], expected, atol=1e-6)
        self.assertGreater(morph.similarity(layer, self.blob.layer), 0.999)

    def test_speed_changes_duration_not_width(self):
        h, w = self.blob.features["h"], self.blob.features["w"]
        slow = morph.describe(morph.render(self.blob, still(self.blob, speed=0.5)))
        fast = morph.describe(morph.render(self.blob, still(self.blob, speed=2.0)))
        self.assertAlmostEqual(slow["h"], 2 * h, delta=2)
        self.assertAlmostEqual(fast["h"], h / 2, delta=1.5)
        self.assertLessEqual(abs(slow["w"] - w), 1)

    def test_lateral_drift_makes_a_rolling_trace(self):
        layer = morph.render(self.blob, still(self.blob, lateral=lambda t: 0.9 * t))
        features = morph.describe(layer)
        self.assertEqual(morph.classify(features), "rolling")
        self.assertAlmostEqual(features["excursion"], 0.9 * (features["h"] - 1), delta=1.5)

    def test_lost_contact_tears_the_trace(self):
        params = still(self.blob)
        params["gain"][3:5] = [0.0, 0.0]
        features = morph.describe(morph.render(self.blob, params))
        self.assertGreaterEqual(features["gaps"], 1)
        self.assertEqual(morph.classify(features), "swinging")

    def test_flip_mirrors_the_objects_own_motion(self):
        pixels, boxes = kinds_frame(0)
        rolling = morph.cut_object(pixels, boxes, 3)
        right = morph.lateral_track(morph.render(rolling, still(rolling)))[0]
        left = morph.lateral_track(morph.render(rolling, dict(still(rolling), flip=True)))[0]
        self.assertGreater(right[-1] - right[0], 3)
        self.assertAlmostEqual(left[-1] - left[0], -(right[-1] - right[0]), delta=0.5)

    def test_motion_is_seeded_and_lengths_match(self):
        for kind in morph.KINDS:
            first = morph.sample_motion(kind, self.blob, np.random.default_rng(5))
            second = morph.sample_motion(kind, self.blob, np.random.default_rng(5))
            self.assertEqual(first, second)
            json.dumps(first)
            n = len(morph.output_times(self.blob, np.asarray(first["speed"])))
            self.assertEqual((len(first["lateral"]), len(first["gain"])), (n, n))
            np.testing.assert_array_equal(morph.render(self.blob, first), morph.render(self.blob, second))

    def test_mismatched_motion_is_refused(self):
        params = still(self.blob)
        params["lateral"].append(0.0)
        with self.assertRaises(ValueError):
            morph.render(self.blob, params)

    def test_texture_restores_missing_noise_only(self):
        layer = np.full((6, 6), 0.5, np.float32)
        rng = np.random.default_rng(0)
        same = morph.add_texture(layer, ([0.05] * 4, [0.05] * 4), rng)
        np.testing.assert_array_equal(same, layer)
        noisy = morph.add_texture(layer, ([0.05] * 4, [0.0] * 4), rng)
        self.assertGreater(float(np.std(noisy)), 0.02)


class JudgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.objects = all_objects()
        cls.stats = morph.fit_statistics(cls.objects)

    def test_statistics_per_kind(self):
        self.assertEqual(set(self.stats["by_kind"]), set(morph.KINDS))
        self.assertEqual(self.stats["kinds"], {kind: 8 for kind in morph.KINDS})

    def test_copy_wrong_kind_and_out_of_range_are_rejected(self):
        blob = self.objects[0]
        ok, reason, _ = morph.judge(morph.render(blob, still(blob)), "blob", self.stats, blob.layer)
        self.assertEqual((ok, reason), (False, "copy"))
        rolled = morph.render(blob, still(blob, lateral=lambda t: 0.9 * t))
        self.assertEqual(morph.judge(rolled, "blob", self.stats)[1], "ratio")
        self.assertEqual(morph.judge(rolled, "long", self.stats)[1], "kind:rolling")
        dim = morph.render(blob, still(blob, gain=0.3))
        self.assertTrue(morph.judge(dim, "blob", self.stats)[1].startswith("range:"))
        self.assertEqual(morph.judge(np.zeros((0, 0), np.float32), "blob", self.stats)[1], "empty")

    def test_slightly_changed_object_is_accepted_and_labelled(self):
        blob = self.objects[0]
        stretched = morph.render(blob, still(blob, speed=0.85))
        # A plain resize is still the same shape for the copy check.
        self.assertEqual(morph.judge(stretched, "blob", self.stats, blob.layer)[1], "copy")
        layer = morph.render(blob, still(blob, speed=0.85, lateral=lambda t: 0.6 * np.sin(t / 2)))
        ok, reason, _ = morph.judge(layer, "blob", self.stats, blob.layer)
        self.assertTrue(ok, reason)
        box = morph.label_box(layer, self.stats)
        tight = morph.tight_box(layer)
        self.assertTrue((box[2:] > box[:2]).all())
        self.assertLessEqual(float(np.abs(box - tight).max()), 1.0)
        self.assertTrue((box >= 0).all() and box[2] <= layer.shape[1] and box[3] <= layer.shape[0])


class ShapeRuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.objects = all_objects()
        cls.stats = morph.fit_statistics(cls.objects)
        cls.library = morph.Library("train", "x", cls.objects, cls.stats)
        cls.limits = dict(noise_max=0.0, hotspot_max=0.0, stripes_max=0.0, dropout_max=0.0)

    def test_round_blob_is_refused_and_elongated_one_is_not(self):
        blob = self.objects[0]
        wide = morph.render(blob, dict(still(blob), width=2.0))
        features = morph.describe(wide)
        self.assertLess(features["h"], features["w"] * morph.BLOB_RATIO[0])
        self.assertEqual(morph.judge(wide, "blob", self.stats)[1], "ratio")
        self.assertIsNone(morph.shape_problem(morph.describe(blob.layer), blob.layer, "blob"))
        # The rule belongs to blobs only.
        self.assertIsNone(morph.shape_problem(features, wide, "long"))

    def test_blob_ratio_range_is_set_by_the_caller(self):
        blob = self.objects[0]
        features = morph.describe(blob.layer)
        ratio = features["h"] / features["w"]
        narrow = (ratio + 0.2, ratio + 1)
        self.assertEqual(morph.shape_problem(features, blob.layer, "blob", narrow), "ratio")
        self.assertEqual(morph.judge(blob.layer, "blob", self.stats, blob_ratio=narrow)[1], "ratio")
        self.assertIsNone(morph.shape_problem(features, blob.layer, "blob", (ratio - 0.1, ratio + 0.1)))
        self.assertEqual(morph.shape_problem(features, blob.layer, "blob", (0.1, ratio - 0.1)), "ratio")
        for bad in ((2, 1), (0, 3), (1, 21), ("a", 2)):
            with self.assertRaises(ValueError):
                morph.blob_ratio_range(*bad)

    def test_candidates_stay_inside_the_requested_ratio_range(self):
        for low, high in ((1.5, 2.2), (2.0, 3.0)):
            result = self.library.candidates("blob", 20, 4, self.limits, None, (low, high))
            self.assertEqual(len(result), 20)
            for item in result:
                f = morph.describe(np.asarray(item["matrix"], np.float32))
                self.assertTrue(low <= f["h"] / f["w"] <= high, (f["h"], f["w"]))

    def test_trace_torn_in_the_middle_is_refused_for_rolling_and_swinging(self):
        rolling = self.objects[3]
        smooth_layer = morph.render(rolling, still(rolling))
        self.assertGreater(morph.trace_valley(smooth_layer), 0.9)
        for kind in morph.UNBROKEN_KINDS:
            self.assertIsNone(morph.shape_problem(morph.describe(smooth_layer), smooth_layer, kind))
        # A tear (empty polls) and a thin neck (weak but present polls) are both breaks.
        for gain in (0.0, 0.18):
            params = still(rolling, lateral=lambda t: 0.9 * t)
            n = len(params["gain"])
            params["gain"][n // 2 - 1:n // 2 + 1] = [gain, gain]
            torn = morph.render(rolling, params)
            for kind in morph.UNBROKEN_KINDS:
                self.assertEqual(morph.shape_problem(morph.describe(torn), torn, kind), "broken", gain)
            self.assertIsNone(morph.shape_problem(morph.describe(torn), torn, "long"))

    def test_candidates_are_refilled_to_the_requested_count(self):
        for kind in ("blob", "rolling", "swinging"):
            result = self.library.candidates(kind, 25, 3, self.limits)
            self.assertEqual(len(result), 25, kind)
            for item in result:
                reason = item["filter"]["reason"]
                self.assertNotIn(reason, morph.HARD_REASONS)
                layer = np.asarray(item["matrix"], np.float32)
                features = morph.describe(layer)
                if kind == "blob":
                    self.assertGreaterEqual(features["h"], features["w"] * morph.BLOB_RATIO[0])
                else:
                    self.assertEqual((features["gaps"], features["components"]), (0, 1))
                    self.assertGreaterEqual(morph.trace_valley(layer), morph.MAX_VALLEY)
        self.assertEqual(self.library.candidates("blob", 8, 3, self.limits),
                         self.library.candidates("blob", 8, 3, self.limits))


class DemoTests(unittest.TestCase):
    def test_demo_writes_pictures_report_and_a_valid_train_dataset(self):
        from scene_editor import morph_demo
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ("train", "val"):
                (root / "raw" / split / "images").mkdir(parents=True)
                (root / "raw" / split / "labels").mkdir()
                for seed in range(8):
                    pixels, boxes = kinds_frame(seed)
                    Image.fromarray(np.round(pixels * 255).astype(np.uint8)).save(
                        root / "raw" / split / "images" / f"f{seed}.png")
                    lines = [f"0 {(b[0] + b[2]) / 80:.6f} {(b[1] + b[3]) / 80:.6f} "
                             f"{(b[2] - b[0]) / 40:.6f} {(b[3] - b[1]) / 40:.6f}"
                             for b in np.clip(boxes, 0, 40)]
                    (root / "raw" / split / "labels" / f"f{seed}.txt").write_text("\n".join(lines) + "\n")
            (root / "raw" / "data.yaml").write_text("data:\n  train: train\n  val: val\n")
            out = root / "demo"
            with patch("builtins.print"):
                morph_demo.main(["--data", str(root / "raw"), "--out", str(out), "--sources", "2",
                                 "--variants", "3", "--trials", "20", "--scenes", "4"])
            for kind in morph.KINDS:
                self.assertTrue((out / f"variants_{kind}.png").is_file())
            report = json.loads((out / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(set(report["kinds"]), set(morph.KINDS))
            images = sorted((out / "scenes" / "train" / "images").glob("*.png"))
            self.assertEqual(len(images), report["scenes_saved"])
            self.assertGreater(len(images), 0)
            self.assertEqual(list((out / "scenes" / "val" / "images").iterdir()), [])
            for image in images:
                boxes = read_yolo_boxes(out / "scenes" / "train" / "labels" / f"{image.stem}.txt")
                provenance = json.loads((out / "scenes" / "train" / "provenance" / f"{image.stem}.json")
                                        .read_text(encoding="utf-8"))
                added = [row for row in provenance["sources"] if row["role"] == "object"]
                self.assertTrue(added)
                self.assertTrue(all(row["split"] == "train" for row in provenance["sources"]))
                self.assertEqual(len(boxes), 5 + len(added))
                with Image.open(image) as picture:
                    self.assertEqual((picture.size, picture.mode), ((40, 40), "L"))

    def test_demo_refuses_output_inside_the_source(self):
        from scene_editor import morph_demo
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw"
            for split in ("train", "val"):
                (raw / split / "images").mkdir(parents=True)
                (raw / split / "labels").mkdir()
                Image.fromarray(np.zeros((40, 40), np.uint8)).save(raw / split / "images" / "a.png")
                (raw / split / "labels" / "a.txt").write_text("")
            (raw / "data.yaml").write_text("data:\n  train: train\n  val: val\n")
            with self.assertRaises(ValueError):
                morph_demo.main(["--data", str(raw), "--out", str(raw / "train" / "demo")])


if __name__ == "__main__":
    unittest.main()
