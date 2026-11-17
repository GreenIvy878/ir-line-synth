"""Generated objects and backgrounds inside the editor: imperfections, scenes, API, export."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from ir_line_synth import morph
from ir_line_synth.background import BackgroundBank, background_options
from ir_line_synth.dataset import load_data_description, open_split, read_yolo_boxes
from ir_line_synth.scene import PreparedObject, SourceImage, check_placement, render_scene
from ir_line_synth.web.views import EditorViews
from test_morph import kinds_frame, still


def write_kinds_dataset(root, frames=8, splits=("train", "val")):
    """A dataset whose frames hold one egg of every kind."""
    for split in splits:
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir()
        for seed in range(frames):
            pixels, boxes = kinds_frame(seed)
            Image.fromarray(np.round(pixels * 255).astype(np.uint8)).save(root / split / "images" / f"f{seed}.png")
            lines = [f"0 {(b[0] + b[2]) / 80:.6f} {(b[1] + b[3]) / 80:.6f} {(b[2] - b[0]) / 40:.6f} {(b[3] - b[1]) / 40:.6f}"
                     for b in np.clip(boxes, 0, 40)]
            (root / split / "labels" / f"f{seed}.txt").write_text("\n".join(lines) + "\n")
    (root / "data.yaml").write_text("data:\n" + "".join(f"  {split}: {split}\n" for split in splits))
    return load_data_description(root)


def blob_layer():
    pixels, boxes = kinds_frame(0)
    obj = morph.cut_object(pixels, boxes, 0)
    return obj, morph.render(obj, still(obj))


def centroid(layer):
    ys, xs = np.mgrid[0:layer.shape[0], 0:layer.shape[1]]
    return float((ys * layer).sum() / layer.sum()), float((xs * layer).sum() / layer.sum())


class ImperfectionTests(unittest.TestCase):
    def setUp(self):
        self.obj, self.layer = blob_layer()

    def test_neutral_options_change_nothing(self):
        np.testing.assert_array_equal(morph.imperfect(self.layer, {}), self.layer)

    def test_every_distortion_is_described_by_stage(self):
        from ir_line_synth.scene import NEUTRAL_APPEARANCE
        text = (Path(__file__).parents[1] / "docs" / "synthetic_data_method.md").read_text(encoding="utf-8")
        section = text[text.index("### 8.1."):text.index("## 9.")]
        rows = {}
        for line in section.splitlines():
            cells = [cell.strip() for cell in line.split("|")[1:-1]]
            if len(cells) == 5 and cells[1] in ("transform", "imperfections"):
                for key in cells[0].replace("`", "").split(", "):
                    rows[(cells[1], key)] = cells[2]
        expected = {("transform", key) for key in NEUTRAL_APPEARANCE}
        expected |= {("imperfections", key) for key in morph.NEUTRAL_IMPERFECTIONS}
        self.assertEqual(set(rows), expected)
        self.assertTrue(set(rows.values()) <= {"поверхность", "датчик", "кадр", "контур"})

    def test_seeded_and_each_effect_has_its_own_stream(self):
        options = dict(noise=0.5, stripes=0.5, dropout=0.5, seed=3)
        np.testing.assert_array_equal(morph.imperfect(self.layer, options), morph.imperfect(self.layer, options))
        self.assertFalse(np.array_equal(morph.imperfect(self.layer, options),
                                        morph.imperfect(self.layer, dict(options, seed=4))))
        # Stripes alone and stripes with noise share the same column gains.
        striped = morph.imperfect(self.layer, dict(stripes=0.5, seed=3))
        both = morph.imperfect(self.layer, dict(stripes=0.5, noise=0.001, seed=3))
        np.testing.assert_allclose(striped, both, atol=0.01)

    def test_hotspot_moves_the_bright_part_towards_an_edge_and_keeps_the_peak(self):
        cy, cx = centroid(self.layer)
        down = morph.imperfect(self.layer, dict(hotspot=0.9, hotspot_angle=90, hotspot_offset=1.0))
        up = morph.imperfect(self.layer, dict(hotspot=0.9, hotspot_angle=-90, hotspot_offset=1.0))
        self.assertGreater(centroid(down)[0], cy + 0.5)
        self.assertLess(centroid(up)[0], cy - 0.5)
        self.assertAlmostEqual(float(down.max()), float(self.layer.max()), places=5)

    def test_stripes_dropout_and_noise_roughen_the_object(self):
        roughness = lambda layer: float(np.abs(layer - morph.smooth(layer))[layer >= morph.TAU].mean())
        base = roughness(self.layer)
        for key in ("stripes", "dropout", "noise"):
            self.assertGreater(roughness(morph.imperfect(self.layer, {key: 1.0, "seed": 1})), base * 1.3, key)

    def test_limits_give_strengths_up_to_the_limit(self):
        limits = morph.imperfection_limits(dict(noise_max=0.2, hotspot_max=0, stripes_max=1, dropout_max=0.5))
        rng = np.random.default_rng(0)
        for _ in range(50):
            values = morph.imperfection_options(morph.sample_imperfections(limits, rng))
            self.assertLessEqual(values["noise"], 0.2)
            self.assertEqual(values["hotspot"], 0)
            self.assertLessEqual(values["dropout"], 0.5)

    def test_invalid_descriptions_are_refused(self):
        motion = morph.sample_motion("blob", self.obj, np.random.default_rng(1))
        good = dict(kind="blob", motion=motion, imperfections={})
        morph.morph_options(good)
        bad = [dict(good, extra=1), dict(good, motion=dict(motion, speed=[])),
               dict(good, motion=dict(motion, speed=[float("nan")])), dict(good, motion=dict(motion, kind="cube")),
               dict(good, motion=dict(motion, lateral=[0.0] * 300)), dict(good, imperfections=dict(noise=2)),
               dict(good, imperfections=dict(paint=1)), dict(good, motion=dict(motion, flip="yes"))]
        for spec in bad:
            with self.assertRaises(ValueError):
                morph.morph_options(spec)
        with self.assertRaises(ValueError):
            morph.imperfection_limits(dict(glow_max=1))


class SceneTests(unittest.TestCase):
    def setUp(self):
        self.pixels, self.boxes = kinds_frame(0)
        self.source = SourceImage(self.pixels, self.boxes.astype(np.float64), (0,) * len(self.boxes), "train", "f0.png")
        obj = morph.cut_object(self.pixels, self.boxes, 0)
        self.spec = dict(kind="blob", motion=still(obj, speed=0.8), imperfections=dict(noise=0.2, seed=2))
        self.background = SourceImage(np.full((40, 40), 0.05, np.float32), np.zeros((0, 4)), (), "train", "bg.png")

    def load(self, reference):
        return self.background if reference["id"] == "bg.png" else self.source

    def prepare(self, obj, load_source):
        return morph.prepare_morph(obj, load_source, "x", [0.0, 0.0, 0.0, 0.0])

    def test_prepared_object_has_a_threshold_box_inside_its_soft_edge(self):
        entry = dict(source=dict(split="train", id="f0.png"), box_index=0, morph=self.spec)
        prepared = self.prepare(entry, self.load)
        box = prepared.label_box()
        self.assertEqual(prepared.label, 0)
        self.assertTrue((box[:2] >= 0).all() and box[2] <= prepared.width and box[3] <= prepared.height)
        self.assertTrue((box[:2] > 0).any() or (box[2:] < [prepared.width, prepared.height]).any())
        inside = prepared.pixels[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
        self.assertAlmostEqual(float(inside.max()), float(prepared.pixels.max()))
        np.testing.assert_array_equal(prepared.pixels, self.prepare(entry, self.load).pixels)

    def test_placement_uses_the_label_box(self):
        prepared = PreparedObject(np.ones((6, 4), np.float32), np.ones((6, 4), bool), 0, (1.0, 1.0, 3.0, 5.0))
        np.testing.assert_array_equal(check_placement(prepared, 10, 20, [], 0.15, (40, 40)), [11, 21, 13, 25])
        with self.assertRaises(ValueError):
            check_placement(prepared, 37, 20, [], 0.15, (40, 40))  # the crop, not only the box, must fit

    def test_scene_places_the_crop_at_x_y_and_never_darkens_the_background(self):
        scene = dict(background=dict(split="train", id="bg.png"), blending="replace", max_iou=0.15,
                     objects=[dict(source=dict(split="train", id="f0.png"), box_index=0, morph=self.spec,
                                   transform={}, x=10, y=12)])
        result = render_scene(scene, self.load, self.prepare)
        prepared = self.prepare(scene["objects"][0], self.load)
        expected_box = prepared.label_box() + [10, 12, 10, 12]
        np.testing.assert_allclose(result.boxes[0], expected_box)
        self.assertGreaterEqual(float(result.pixels.min()), 0.05 - 1 / 255)
        region = result.pixels[12:12 + prepared.height, 10:10 + prepared.width]
        np.testing.assert_allclose(region, np.round(np.maximum(0.05, prepared.pixels) * 255) / 255, atol=1e-6)


class BackgroundTests(unittest.TestCase):
    def test_blocks_come_from_empty_columns_of_real_frames_and_the_spread_is_limited(self):
        with tempfile.TemporaryDirectory() as directory:
            data = write_kinds_dataset(Path(directory) / "raw")
            frames = open_split(data, "train")
            bank = BackgroundBank(frames)
            options = dict(seed=4, roll=False)
            first, second = bank.generate(options), bank.generate(options)
            np.testing.assert_array_equal(first, second)
            self.assertFalse(np.array_equal(first, bank.generate(dict(seed=5, roll=False))))
            stacked = bank.frames
            rng = np.random.default_rng(0)
            for x in range(40):
                if not bank.free_columns[:, x].any():
                    continue  # never empty in this dataset: filled from its own empty rows
                _, block, free = bank.pick_block(x, background_options(options), rng)
                sources = [n for n in range(len(stacked)) if bank.free_columns[n, x]
                           and any(np.array_equal(stacked[n][:, x], block[:, k]) for k in range(block.shape[1]))]
                self.assertTrue(free and sources, f"column {x} is not an empty real column")
            # Every frame has eggs at the same place, so those columns use the fallback, without egg pixels.
            self.assertLess(float(first.max()), 0.1)
            for seed in range(5):
                for level in (0.0, 1.0, 3.0):
                    levels = np.round(bank.generate(dict(seed=seed, level=level)) * 255)
                    self.assertLessEqual(levels.max() - levels.min(), 7)  # under 3% of 255

    def test_options_are_checked(self):
        for bad in (dict(block_min=5, block_max=2), dict(level=-1), dict(roll="no"), dict(colour=1),
                    dict(spread=0.031), dict(spread=-0.01)):
            with self.assertRaises(ValueError):
                background_options(bad)


class TestSplitTests(unittest.TestCase):
    """A test split named in data.yaml can supply objects, but never training data."""

    def test_generation_from_the_test_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            views = EditorViews(write_kinds_dataset(root / "raw", splits=("train", "val", "test")))
            self.assertEqual(set(views.get("/api/synthetic/catalog", {})), {"train", "val", "test"})
            body = dict(kind="blob", count=5, seed=1, split="test")
            result = views.post("/api/synthetic/morph-generate", body)
            self.assertEqual(len(result["candidates"]), 5)
            self.assertEqual({c["source"]["split"] for c in result["candidates"]}, {"test"})
            candidate = result["candidates"][0]
            scene = dict(background=dict(split="train", generated=dict(seed=7)), blending="replace", max_iou=0.15,
                         objects=[dict(source=candidate["source"], box_index=candidate["box_index"],
                                       morph=candidate["morph"], transform={}, x=20, y=20)])
            preview = views.post("/api/synthetic/preview", dict(scene=scene))
            self.assertFalse(preview["train_eligible"])
            with self.assertRaises(ValueError):
                views.post("/api/synthetic/save", dict(scene=scene, expected_digest=preview["digest"],
                                                       directory=str(root / "out"), split="train"))
            bank = [dict(source=dict(split="test", id="f1.png"), box_index=3)]
            self.assertTrue(all(c["source"]["id"] == "f1.png" for c in views.post(
                "/api/synthetic/morph-generate", dict(body, kind="rolling", sources=bank))["candidates"]))

    def test_without_a_test_split_it_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            views = EditorViews(write_kinds_dataset(Path(directory) / "raw"))
            self.assertEqual(set(views.get("/api/synthetic/catalog", {})), {"train", "val"})
            with self.assertRaises(ValueError):
                views.post("/api/synthetic/morph-generate", dict(kind="blob", count=2, split="test"))


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.views = EditorViews(write_kinds_dataset(cls.root / "raw"))

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def generate(self, **body):
        return self.views.post("/api/synthetic/morph-generate", dict(dict(kind="blob", count=6, seed=2), **body))

    def test_generation_is_seeded_and_carries_the_filter_opinion(self):
        first, second = self.generate(), self.generate()
        self.assertEqual(json.dumps(first), json.dumps(second))
        self.assertEqual(len(first["candidates"]), 6)
        for candidate in first["candidates"]:
            self.assertIn(candidate["filter"]["reason"].split(":")[0], {"ok", "kind", "range", "unlike_real", "copy", "empty"})
            self.assertEqual(candidate["source"]["split"], "train")
        self.assertNotEqual(json.dumps(first), json.dumps(self.generate(seed=3)))

    def test_blob_ratio_range_comes_from_the_request(self):
        result = self.generate(count=8, ratio_min=1.6, ratio_max=2.4)
        self.assertEqual(len(result["candidates"]), 8)
        for candidate in result["candidates"]:
            matrix = np.asarray(candidate["matrix"], np.float32)
            f = morph.describe(matrix.T if self.views.workspace.library("train").sensor_axis == "y" else matrix)
            self.assertTrue(1.6 <= f["h"] / f["w"] <= 2.4)
        with self.assertRaises(ValueError):
            self.generate(ratio_min=3, ratio_max=2)

    def test_bank_sources_and_errors(self):
        result = self.generate(sources=[dict(source=dict(split="train", id="f1.png"), box_index=3)], kind="rolling")
        self.assertTrue(all(c["source"]["id"] == "f1.png" and c["box_index"] == 3 for c in result["candidates"]))
        with self.assertRaises(ValueError):
            self.generate(sources=[dict(source=dict(split="train", id="f1.png"), box_index=99)])
        with self.assertRaises(ValueError):
            self.generate(kind="cube")
        with self.assertRaises(ValueError):
            self.generate(limits=dict(noise_max=5))

    def test_chosen_candidate_is_the_prepared_object(self):
        for kind in morph.KINDS:
            limits = dict(noise_max=0.4, hotspot_max=0.6, stripes_max=0.3, dropout_max=0.3)
            for candidate in self.generate(kind=kind, limits=limits)["candidates"]:
                prepared = self.views.post("/api/synthetic/prepare", dict(
                    source=candidate["source"], box_index=candidate["box_index"], morph=candidate["morph"], transform={}))
                shown = np.asarray(candidate["matrix"], np.float32)
                ys, xs = np.nonzero(shown > 0)
                np.testing.assert_allclose(np.asarray(prepared["matrix"]) * np.asarray(prepared["mask"]),
                                           shown[ys.min():ys.max() + 1, xs.min():xs.max() + 1], atol=1e-6)
                self.assertEqual(len(prepared["box"]), 4)

    def test_generated_background_scene_and_its_split(self):
        candidate = self.generate()["candidates"][0]
        obj = dict(source=candidate["source"], box_index=candidate["box_index"], morph=candidate["morph"],
                   transform={}, x=20, y=20)
        scene = dict(background=dict(split="train", generated=dict(seed=7)), objects=[obj], blending="replace", max_iou=0.15)
        preview = self.views.post("/api/synthetic/preview", dict(scene=scene))
        self.assertEqual(preview["statistics"]["total_objects"], 1)
        self.assertTrue(preview["train_eligible"])
        self.assertEqual(preview, self.views.post("/api/synthetic/preview", dict(scene=scene)))
        val_background = copy.deepcopy(scene)
        val_background["background"]["split"] = "val"
        self.assertFalse(self.views.post("/api/synthetic/preview", dict(scene=val_background))["train_eligible"])
        with self.assertRaises(ValueError):
            self.views.post("/api/synthetic/preview", dict(scene=dict(scene, background=dict(split="train", generated=dict(seed=-1)))))

    def test_autogenerator_with_generated_backgrounds_saves_what_it_showed(self):
        pool = [dict(source=c["source"], box_index=c["box_index"], morph=c["morph"], transform={})
                for c in self.generate(count=3)["candidates"]]
        pool.append(dict(source=dict(split="train", id="f2.png"), box_index=0, transform={}))
        rules = dict(background_mode="generated", images=6, count_min=1, count_max=3, rotation_degrees=0, seed=4)
        result = self.views.post("/api/synthetic/generate", dict(background=None, pool=pool, rules=rules))
        self.assertEqual(len(result["candidates"]), 6)
        seeds = set()
        for candidate in result["candidates"]:
            background = candidate["scene"]["background"]
            self.assertIn("generated", background)
            seeds.add(background["generated"]["seed"])
            for obj in candidate["scene"]["objects"]:
                if "morph" in obj:
                    self.assertFalse(obj["transform"]["use_mask"])
        self.assertEqual(len(seeds), 6)
        reseeded = [obj["morph"]["imperfections"]["seed"] for c in result["candidates"]
                    for obj in c["scene"]["objects"] if "morph" in obj]
        self.assertEqual(len(set(reseeded)), len(reseeded))
        real = self.views.post("/api/synthetic/generate", dict(background=None, pool=pool,
                                                               rules=dict(rules, background_mode="real")))
        self.assertTrue(all(c["scene"]["background"].get("id", "").startswith("f") for c in real["candidates"]))
        with self.assertRaises(ValueError):
            self.views.post("/api/synthetic/generate", dict(background=None, pool=pool, rules=dict(rules, background_mode="fixed")))
        items = [dict(scene=c["scene"], expected_digest=c["preview"]["digest"]) for c in result["candidates"][:3]]
        out = self.root / "out"
        saved = self.views.post("/api/synthetic/generated-save", dict(items=items, directory=str(out), split="train"))
        self.assertEqual(len(saved["saved"]), 3)
        for row, candidate in zip(saved["saved"], result["candidates"]):
            with Image.open(row["image"]) as image:
                pixels = np.asarray(image, np.float32) / 255
            np.testing.assert_allclose(pixels, np.asarray(candidate["preview"]["matrix"]), atol=1e-6)
            np.testing.assert_allclose(read_yolo_boxes(Path(row["annotations"])) * 40,
                                       np.asarray(candidate["preview"]["boxes_pixels_xyxy"]), atol=1e-4)


if __name__ == "__main__":
    unittest.main()
