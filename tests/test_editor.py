"""Editor regressions: image axes, geometry, provenance and lossless export."""

import copy
import dataclasses
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image
import yaml

from scene_editor.dataset import open_split
from scene_editor.scene import (RenderedScene, SourceImage, box_iou, make_variation,
    prepare_object, render_scene, scene_statistics)
from scene_editor.web.server import make_server
from scene_editor.store import SyntheticStore, example_digest
from scene_editor.web.views import EditorViews


def source(split="train"):
    pixels = np.full((40, 40), 0.1, dtype=np.float32)
    pixels[2:4, 3:6] = [[0.2, 0.4, 0.6], [0.8, 1.0, 0.3]]
    return SourceImage(pixels, np.array([[3., 2., 6., 4.]]), (0,), split, "donor.png")


def background():
    return SourceImage(np.full((40, 40), 0.25, dtype=np.float32),
                       np.array([[0., 0., 2., 2.]]), (0,), "train", "background.png")


def stored(pixels):
    """Intensities as a saved scene holds them: rounded to the 8-bit levels of real frames."""
    return np.round(np.asarray(pixels, dtype=np.float32) * 255).astype(np.float32) / 255


def scene(split="train", transform=None):
    return dict(background=dict(split="train", id="background.png"),
                blending="replace", max_iou=0.15,
                objects=[dict(source=dict(split=split, id="donor.png"), box_index=0,
                              transform=transform or {}, x=10, y=12)])


def load_source(reference):
    return background() if reference["id"] == "background.png" else source(reference["split"])


class SyntheticGeometryTests(unittest.TestCase):
    def test_unchanged_crop_and_both_flips(self):
        prepared = prepare_object(source(), 0, {})
        np.testing.assert_array_equal(prepared.pixels, source().pixels[2:4, 3:6])
        flipped = prepare_object(source(), 0, dict(flip_x=True, flip_y=True))
        np.testing.assert_array_equal(flipped.pixels, prepared.pixels[::-1, ::-1])
        self.assertTrue(prepared.mask.all())

    def test_arbitrary_rotation_scale_and_intensity_clipping(self):
        transformed = prepare_object(source(), 0, dict(angle=37, scale=2, brightness=3))
        self.assertGreater(transformed.width, 3)
        self.assertGreater(transformed.height, 2)
        self.assertTrue(np.isfinite(transformed.pixels).all())
        self.assertGreaterEqual(float(transformed.pixels.min()), 0)
        self.assertLessEqual(float(transformed.pixels.max()), 1)
        self.assertTrue(transformed.mask[0].any())
        self.assertTrue(transformed.mask[-1].any())
        self.assertTrue(transformed.mask[:, 0].any())
        self.assertTrue(transformed.mask[:, -1].any())

    def test_brightness_keeps_pixel_ratios_but_reshaping_changes_them(self):
        def ratio(options):
            pixels = prepare_object(source(), 0, options).pixels
            return pixels[0, 2] / pixels[0, 1]
        base = ratio({})
        self.assertAlmostEqual(ratio({"brightness": 0.5}), base, places=5)
        self.assertGreater(abs(ratio({"gamma": 2}) - base), 0.1)
        self.assertGreater(abs(ratio({"contrast": 0.5}) - base), 0.05)
        self.assertGreater(abs(ratio({"gradient": 0.8}) - base), 0.05)

    def test_noise_is_seeded_and_shade_darkens_only_near_its_center(self):
        noisy = {"noise": 0.3, "noise_seed": 5}
        first = prepare_object(source(), 0, noisy).pixels
        np.testing.assert_array_equal(first, prepare_object(source(), 0, noisy).pixels)
        self.assertFalse(np.array_equal(first, prepare_object(source(), 0, {**noisy, "noise_seed": 6}).pixels))
        plain = prepare_object(source(), 0, {}).pixels
        shaded = prepare_object(source(), 0, {"shade": 1, "shade_size": 0.1,
                                              "shade_x": 0, "shade_y": 0}).pixels
        self.assertLess(shaded[0, 0], plain[0, 0] * 0.1)
        self.assertAlmostEqual(float(shaded[-1, -1]), float(plain[-1, -1]), places=3)

    def test_edge_erosion_removes_only_seeded_boundary_pixels(self):
        full = prepare_object(source(), 0, {})
        options = {"erode": 0.7, "erode_seed": 3}
        eroded = prepare_object(source(), 0, options)
        np.testing.assert_array_equal(eroded.mask, prepare_object(source(), 0, options).mask)
        self.assertFalse(np.array_equal(eroded.mask, prepare_object(source(), 0, {**options, "erode_seed": 4}).mask))
        self.assertLess(eroded.mask.sum(), full.mask.sum())

    def test_edge_erosion_keeps_interior_pixels(self):
        pixels = np.full((40, 40), 0.5, dtype=np.float32)
        big = SourceImage(pixels, np.array([[5., 5., 15., 15.]]), (0,), "train", "big.png")
        prepared = prepare_object(big, 0, {"erode": 1.0, "erode_seed": 1})
        self.assertEqual(prepared.mask.shape, (8, 8))
        self.assertTrue(prepared.mask.all())

    def test_mask_tail_threshold_and_tight_bbox(self):
        data = source()
        strong = prepare_object(data, 0, dict(use_mask=True, threshold=0.7))
        self.assertEqual((strong.width, strong.height), (2, 1))
        tail = prepare_object(data, 0, dict(use_mask=True, threshold=0.2, flip_x=True, angle=90))
        self.assertEqual((tail.width, tail.height), (2, 3))
        self.assertEqual(int(tail.mask.sum()), 6)
        with self.assertRaisesRegex(ValueError, "empty"):
            dim = SourceImage(data.pixels * 0.5, data.boxes, data.labels, data.split, data.identifier)
            prepare_object(dim, 0, dict(use_mask=True, threshold=0.9))

    def test_excluded_crop_pixels_do_not_bleed_during_interpolation(self):
        data = source()
        pixels = np.zeros((40, 40), dtype=np.float32)
        pixels[2:4, 3:6] = [[0.8, 0, 0], [0, 0, 0]]
        data = SourceImage(pixels, data.boxes, data.labels, data.split, data.identifier)
        obj = prepare_object(data, 0, dict(use_mask=True, threshold=0.5, scale=3, angle=23))
        np.testing.assert_allclose(obj.pixels[obj.mask], 0.8, atol=1e-6)

    def test_replace_max_additive_and_mask_preserve_background(self):
        for mode in ("replace", "max", "additive"):
            with self.subTest(mode=mode):
                spec = scene(transform=dict(use_mask=True, threshold=0.5))
                spec["blending"] = mode
                result = render_scene(spec, load_source)
                prepared = prepare_object(source(), 0, spec["objects"][0]["transform"])
                region = result.pixels[12:12 + prepared.height, 10:10 + prepared.width]
                expected = prepared.pixels if mode == "replace" else np.maximum(0.25, prepared.pixels) if mode == "max" else np.minimum(1, 0.25 + prepared.pixels)
                np.testing.assert_array_equal(region[prepared.mask], stored(expected)[prepared.mask])
                np.testing.assert_array_equal(region[~prepared.mask], stored(0.25))
                np.testing.assert_array_equal(result.boxes[0], background().boxes[0])
                self.assertEqual(result.labels, (0, 0))
                self.assertEqual(result.payload()["statistics"]["added_objects"], 1)

    def test_iou_enforced_for_background_and_added_objects(self):
        spec = scene()
        spec["objects"][0].update(x=0, y=0)
        with self.assertRaisesRegex(ValueError, "IoU"):
            render_scene(spec, load_source)
        spec = scene()
        spec["objects"].append(copy.deepcopy(spec["objects"][0]))
        with self.assertRaisesRegex(ValueError, "IoU"):
            render_scene(spec, load_source)
        spec["max_iou"] = 1
        result = render_scene(spec, load_source)
        self.assertEqual(result.payload()["statistics"]["max_iou"], 1)

    def test_bounds_reject_clipping_and_noninteger_coordinates(self):
        for value in (-1, 39, 1.5, True, float("nan")):
            with self.subTest(x=value):
                spec = scene()
                spec["objects"][0]["x"] = value
                with self.assertRaises(ValueError):
                    render_scene(spec, load_source)
        spec = scene()
        spec["objects"][0].update(x=37, y=38)
        render_scene(spec, load_source)

    def test_tiny_annotation_at_last_pixel_is_not_lost(self):
        data = source()
        edge = SourceImage(data.pixels, np.array([[39.999999, 39.999999, 40., 40.]]), (0,), data.split, data.identifier)
        obj = prepare_object(edge, 0, {})
        self.assertEqual((obj.width, obj.height), (1, 1))

    def test_statistics_bbox_edge_distance_and_original_overlap_allowed(self):
        boxes = np.array([[0., 0., 2., 2.], [2., 0., 4., 2.]])
        self.assertEqual(scene_statistics(boxes, 1)["min_distance"], 0)
        boxes[1] = [5, 6, 7, 8]
        self.assertEqual(scene_statistics(boxes, 1)["min_distance"], 5)
        self.assertIsNone(scene_statistics(boxes[:1], 0)["min_distance"])
        self.assertAlmostEqual(box_iou(np.array([0, 0, 4, 4]), np.array([2, 2, 6, 6])), 1/7)
        bg = background()
        overlapping = SourceImage(bg.pixels, np.array([[0, 0, 4, 4], [0, 0, 4, 4.]]), (0, 0), "train", bg.identifier)
        result = render_scene(scene(), lambda ref: overlapping if ref["id"] == bg.identifier else source())
        self.assertEqual(result.payload()["statistics"]["max_iou"], 1)

    def test_variations_reproducible_preserve_background_and_validate_each(self):
        spec = scene()
        first, rendered = make_variation(spec, load_source, np.random.default_rng(17), {})
        second, other = make_variation(spec, load_source, np.random.default_rng(17), {})
        self.assertEqual(first, second)
        np.testing.assert_array_equal(rendered.pixels, other.pixels)
        np.testing.assert_array_equal(rendered.boxes[0], background().boxes[0])
        self.assertEqual(spec, scene())
        self.assertEqual(rendered.labels, (0, 0))
        for seed in range(30):
            _, result = make_variation(spec, load_source, np.random.default_rng(seed), {})
            self.assertLessEqual(box_iou(result.boxes[0], result.boxes[1]), 0.15)
            self.assertTrue((result.boxes >= 0).all() and (result.boxes <= 40).all())

    def test_impossible_variation_fails_with_bounded_retries(self):
        spec = scene(transform=dict(scale=5))
        spec["objects"][0].update(x=39, y=39)
        with self.assertRaisesRegex(ValueError, "retry budget"):
            make_variation(spec, load_source, np.random.default_rng(1),
                           dict(position=0, scale=0, brightness=0, angle=0), attempts=3)


class SyntheticExportTests(unittest.TestCase):
    def test_scene_uses_the_8_bit_levels_of_real_frames(self):
        result = render_scene(scene(transform=dict(scale=1.5, angle=17, brightness=0.73)), load_source)
        levels = result.pixels * 255
        np.testing.assert_array_equal(levels, np.round(levels))
        self.assertEqual(result.pixels.dtype, np.float32)
        # The background keeps its level; only the rounding of the test value 0.25 is seen.
        np.testing.assert_array_equal(result.pixels[30:, 30:], stored(0.25))

    def test_png_yolo_roundtrip_both_axes_and_duplicate_detection(self):
        for axis in ("x", "y"):
            with self.subTest(axis=axis), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                data = dict(train=str(root / "raw/train"), val=str(root / "raw/val"), sensor_axis=axis)
                store = SyntheticStore(data)
                spec = scene(transform=dict(scale=1.5, angle=17, brightness=0.73))
                result = render_scene(spec, load_source)
                saved = store.save(str(root / "synthetic"), "train", result, spec)
                with Image.open(saved["image"]) as image:
                    self.assertEqual((image.mode, image.format), ("L", "PNG"))
                    np.testing.assert_array_equal(np.asarray(image), np.round(result.pixels * 255))
                # The dataset describes itself with the storage of the real frames.
                description = yaml.safe_load((root / "synthetic/data.yaml").read_text(encoding="utf-8"))["data"]
                self.assertEqual((description["scale"], description["offset"]), (255.0, 0.0))
                normalized = dict(description, train=str(root / "synthetic/train"),
                                  val=str(root / "synthetic/val"))
                self.assertEqual(description["sensor_axis"], axis)
                frames = open_split(normalized, "train")
                np.testing.assert_array_equal(frames.pixels(0), result.pixels)
                np.testing.assert_allclose(frames.boxes[0], result.boxes / 40, atol=1e-7)
                self.assertEqual(len(frames.boxes[0]), len(result.labels))
                with self.assertRaisesRegex(ValueError, "identical"):
                    SyntheticStore(data).save(str(root / "synthetic"), "train", result, spec)
                self.assertEqual(len(list((root / "synthetic/train/images").iterdir())), 1)

    def test_duplicate_fingerprint_independent_of_annotation_order(self):
        result = render_scene(scene(), load_source)
        self.assertEqual(example_digest(result.pixels, result.boxes, result.labels),
                         example_digest(result.pixels, result.boxes[::-1], result.labels[::-1]))

    def test_saved_duplicates_are_found_whatever_their_boxes(self):
        # A box from a real run whose YOLO line, read back to pixels and normalized again,
        # rounds to another float32: the saved file must still be known as a duplicate.
        box = [24.757261514663696, 0.0, 36.19363880157471, 6.962655067443848]
        rendered = RenderedScene(np.zeros((40, 40), np.float32), np.asarray([box]), (0,),
                                 (dict(split="train", id="a.png", role="background"),), 0)
        with tempfile.TemporaryDirectory() as directory:
            store = SyntheticStore(dict(train=str(Path(directory) / "raw"), sensor_axis="x"))
            store.save(str(Path(directory) / "out"), "train", rendered, {})
            with self.assertRaisesRegex(ValueError, "identical"):
                store.save(str(Path(directory) / "out"), "train", rendered, {})

    def test_validation_origin_blocked_before_any_training_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = SyntheticStore(dict(train=root / "raw/train", val=root / "raw/val", sensor_axis="x"))
            spec = scene("val")
            result = render_scene(spec, load_source)
            self.assertFalse(result.payload()["train_eligible"])
            self.assertEqual(result.payload()["splits"], ["train", "val"])
            # A val object on a train background belongs to no split at all.
            for split in ("train", "val", "test"):
                with self.assertRaisesRegex(ValueError, f"A {split} scene takes every object"):
                    store.save(str(root / "output"), split, result, spec)
            self.assertFalse((root / "output").exists())
            own = lambda reference: dataclasses.replace(load_source(reference), split=reference["split"])
            spec["background"]["split"] = "val"
            store.save(str(root / "output"), "val", render_scene(spec, own), spec)

    def test_protected_paths_and_existing_directories_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = SyntheticStore(dict(train=root / "raw/train", val=root / "raw/val", sensor_axis="x"))
            for path in (root, root / "raw", root / "raw/train", root / "raw/val/images/new"):
                with self.subTest(path=path), self.assertRaisesRegex(ValueError, "separate"):
                    store.initialize(str(path))
            output = root / "existing"
            output.mkdir()
            (output / "keep.txt").write_text("original")
            with self.assertRaisesRegex(ValueError, "empty"):
                store.initialize(str(output))
            self.assertEqual((output / "keep.txt").read_text(), "original")

    def test_save_rollback_on_annotation_write_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = SyntheticStore(dict(train=root / "raw/train", val=root / "raw/val", sensor_axis="x"))
            from scene_editor import store as synthetic_store
            original = synthetic_store.exclusive_write
            def fail(path, payload, created):
                if path.suffix == ".txt":
                    raise OSError("simulated write failure")
                original(path, payload, created)
            with patch.object(synthetic_store, "exclusive_write", side_effect=fail):
                with self.assertRaises(OSError):
                    store.save(str(root / "output"), "train", render_scene(scene(), load_source), scene())
            self.assertFalse(any((root / "output/train/images").iterdir()))
            self.assertFalse(any((root / "output/train/labels").iterdir()))


def write_dataset(root):
    for split in ("train", "val"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir()
        pixels = np.zeros((40, 40), dtype=np.uint8)
        pixels[10:14, 3:5] = 150
        Image.fromarray(pixels).save(root / split / "images/a.png")
        (root / split / "labels/a.txt").write_text("0 0.1 0.3 0.05 0.1\n")
    return dict(type="yolo_images", train=str(root / "train"), val=str(root / "val"),
                sensor_axis="x", scale=255., offset=0.)


class SyntheticAPITests(unittest.TestCase):
    def test_existing_dataset_api_orientation_and_stale_preview(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            views = EditorViews(write_dataset(root / "raw"))
            catalog = views.get("/api/synthetic/catalog", {})
            self.assertEqual(catalog["train"], [dict(id="a.png", objects=1)])
            data = views.get("/api/synthetic/sample", dict(split=["train"], id=["a.png"]))
            np.testing.assert_allclose(data["boxes_pixels_xyxy"], [[3, 10, 5, 14]], atol=1e-5)
            self.assertAlmostEqual(data["matrix"][11][3], 150/255, places=7)
            with self.assertRaises(ValueError):
                views.workspace.source(dict(split="train", id="../../val/images/a.png"))
            spec = dict(background=dict(split="train", id="a.png"), objects=[])
            preview = views.post("/api/synthetic/preview", dict(scene=spec))
            body = dict(scene=spec, expected_digest=preview["digest"], directory=str(root / "out"), split="train")
            Image.fromarray(np.ones((40, 40), dtype=np.uint8)).save(root / "raw/train/images/a.png")
            with self.assertRaisesRegex(ValueError, "changed"):
                views.post("/api/synthetic/save", body)
            self.assertFalse((root / "out").exists())

    def test_batch_duplicates_skipped_and_validation_background_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            views = EditorViews(write_dataset(root / "raw"))
            spec = dict(background=dict(split="train", id="a.png"), objects=[
                dict(source=dict(split="train", id="a.png"), box_index=0, x=20, y=20, transform={})])
            preview = views.post("/api/synthetic/preview", dict(scene=spec))
            body = dict(scene=spec, expected_digest=preview["digest"], directory=str(root / "out"), split="train",
                        count=3, seed=17, variation=dict(position=0, angle=0, scale=0, brightness=0))
            result = views.post("/api/synthetic/variations", body)
            self.assertEqual(len(result["saved"]), 1)
            self.assertEqual(len(result["skipped"]), 2)
            spec["background"]["split"] = "val"
            body["expected_digest"] = views.post("/api/synthetic/preview", dict(scene=spec))["digest"]
            with self.assertRaisesRegex(ValueError, "The background comes from val"):
                views.post("/api/synthetic/variations", body)

    def test_http_assets_preview_save_and_same_origin(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            server = make_server(EditorViews(write_dataset(root / "raw")), 0)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}"
            try:
                originals = {path: path.read_bytes() for path in (root / "raw").rglob("*") if path.is_file()}
                types = {"/": "text/html", "/app.js": "text/javascript", "/editor.js": "text/javascript",
                         "/base.css": "text/css", "/editor.css": "text/css"}
                for path in (*types, "/api/synthetic/catalog"):
                    with urlopen(base + path) as response:
                        self.assertEqual(response.status, 200)
                        if path in types:
                            # Not the type of the Windows registry, which a browser may refuse to run.
                            self.assertEqual(response.headers["Content-Type"], types[path] + "; charset=utf-8")
                body = json.dumps(dict(scene=dict(background=dict(split="train", id="a.png"), objects=[]))).encode()
                request = Request(base + "/api/synthetic/preview", data=body, headers={"Content-Type": "application/json"})
                with urlopen(request) as response:
                    preview = json.load(response)
                self.assertEqual(preview["statistics"]["total_objects"], 1)
                request.add_header("Origin", "https://untrusted.example")
                with self.assertRaises(HTTPError) as caught:
                    urlopen(request)
                self.assertEqual(caught.exception.code, 403)
                rebinding_request = Request(base + "/api/synthetic/preview", data=body,
                                            headers={"Host": "untrusted.example"})
                with self.assertRaises(HTTPError) as caught:
                    urlopen(rebinding_request)
                self.assertEqual(caught.exception.code, 403)
                save_body = json.dumps(dict(scene=dict(background=dict(split="train", id="a.png"), objects=[]),
                    expected_digest=preview["digest"], directory=str(root / "out"), split="train")).encode()
                with urlopen(Request(base + "/api/synthetic/save", data=save_body)) as response:
                    self.assertTrue(Path(json.load(response)["image"]).is_file())
                self.assertEqual(originals, {path: path.read_bytes() for path in originals})
            finally:
                server.shutdown()
                thread.join(timeout=5)
                server.server_close()


if __name__ == "__main__":
    unittest.main()
