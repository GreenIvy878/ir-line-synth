"""Generator regressions: rules, preview-first workflow and exact selective export."""

from collections import Counter
import copy
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

from scene_editor.generator import generate_candidates, generator_options
from scene_editor.scene import SourceImage, box_iou, feather_mask, render_scene
from scene_editor.web.server import make_server
from scene_editor.web.views import EditorViews
from test_editor import background, load_source, scene, source, stored, write_dataset


def pool(split="train"):
    return [dict(source=dict(split=split, id="donor.png"), box_index=0, transform={})]


class GeneratorTests(unittest.TestCase):
    def test_random_count_seed_provenance_bounds_and_iou(self):
        ref = dict(split="train", id="background.png")
        # One pool object several times per image: the repeat rules are off here.
        rules = dict(images=20, count_min=2, count_max=4, seed=17, unique_per_image=False, unique_source=False)
        first = generate_candidates(ref, pool(), rules, load_source)
        second = generate_candidates(ref, pool(), rules, load_source)
        self.assertEqual(first, second)
        self.assertEqual(len(first["candidates"]), 20)
        counts = {len(row["scene"]["objects"]) for row in first["candidates"]}
        self.assertEqual(counts, {2, 3, 4})
        self.assertEqual(len({row["preview"]["digest"] for row in first["candidates"]}), 20)
        for row in first["candidates"]:
            preview = row["preview"]
            self.assertTrue(preview["train_eligible"])
            boxes = np.asarray(preview["boxes_pixels_xyxy"])
            np.testing.assert_array_equal(boxes[0], background().boxes[0])
            self.assertTrue(np.all(boxes >= 0) and np.all(boxes <= 40))
            for index, box in enumerate(boxes[1:], 1):
                self.assertTrue(all(box_iou(box, other) <= 0.15 + 1e-9 for other in boxes[:index]))

    def test_current_edits_relative_limits_reflections_and_insertion(self):
        entries = pool()
        entries[0]["transform"] = dict(brightness=0.5, angle=20, scale=1.2, threshold=0.4)
        before = copy.deepcopy(entries)
        result = generate_candidates(dict(split="train", id="background.png"), entries,
            dict(images=20, count_min=1, count_max=1, flip_x=True, flip_y=True,
                 rotation_degrees=5, brightness_min=0.8, brightness_max=1.2, insertion="mask"), load_source)
        self.assertEqual(entries, before)
        transforms = [row["scene"]["objects"][0]["transform"] for row in result["candidates"]]
        self.assertEqual({transform["flip_x"] for transform in transforms}, {False, True})
        self.assertEqual({transform["flip_y"] for transform in transforms}, {False, True})
        for transform in transforms:
            self.assertTrue(15 <= transform["angle"] <= 25)
            self.assertTrue(0.4 <= transform["brightness"] <= 0.6)
            self.assertEqual(transform["scale"], 1.2)
            self.assertEqual(transform["threshold"], 0.4)
            self.assertTrue(transform["use_mask"])
        fixed = generate_candidates(dict(split="train", id="background.png"), entries,
            dict(images=1, flip_x=False, flip_y=False, rotation_degrees=0,
                 brightness_min=1, brightness_max=1, insertion="box"), load_source)
        transform = fixed["candidates"][0]["scene"]["objects"][0]["transform"]
        self.assertEqual(transform["angle"], 20)
        self.assertEqual(transform["brightness"], 0.5)
        self.assertFalse(transform["use_mask"])

    def test_feathering_softens_boundary_and_retains_annotation_support(self):
        spec = scene()
        spec["objects"][0]["edge_blur"] = 0.8
        hard = render_scene(scene(), load_source)
        soft = render_scene(spec, load_source)
        np.testing.assert_array_equal(hard.boxes, soft.boxes)
        self.assertEqual(hard.labels, soft.labels)
        region = soft.pixels[12:14, 10:13]
        alpha = feather_mask(np.ones((2, 3), bool), 0.8)
        expected = background().pixels[12:14, 10:13] * (1 - alpha) + source().pixels[2:4, 3:6] * alpha
        np.testing.assert_array_equal(region, stored(expected))
        self.assertTrue(np.all((alpha > 0) & (alpha < 1)))
        outside = np.ones((40, 40), bool)
        outside[12:14, 10:13] = False
        np.testing.assert_array_equal(soft.pixels[outside], stored(background().pixels)[outside])
        masked = scene(transform=dict(use_mask=True, threshold=0.5))
        masked["objects"][0]["edge_blur"] = 1
        result = render_scene(masked, load_source)
        self.assertEqual(result.pixels.dtype, np.float32)
        self.assertTrue((result.pixels >= 0).all() and (result.pixels <= 1).all())

    def test_impossible_scene_returns_partial_report_and_no_invalid_candidates(self):
        entries = pool()
        entries[0]["transform"] = dict(scale=5)
        # A background box covering the image leaves no permitted insertion.
        bg = background()
        full = SourceImage(bg.pixels, np.array([[0., 0., 40., 40.]]), (0,), "train", bg.identifier)
        result = generate_candidates(dict(split="train", id="background.png"), entries,
            dict(images=2, max_iou=0, insertion="box"),
            lambda ref: full if ref["id"] == bg.identifier else source())
        self.assertEqual(result["candidates"], [])
        self.assertLessEqual(result["attempts"], 40)
        self.assertTrue(result["rejected"])

    def test_invalid_rules_and_pool_fail_before_generation(self):
        for options in (dict(count_min=4, count_max=2), dict(brightness_min=2, brightness_max=1),
                        dict(images=True), dict(images=1001), dict(flip_x=1), dict(insertion="bad"),
                        dict(edges="bad"), dict(blur_radius=0), dict(max_iou=float("nan"))):
            with self.subTest(options=options), self.assertRaises(ValueError):
                generator_options(options)
        with self.assertRaises(ValueError):
            generate_candidates(dict(split="train", id="background.png"), [], {}, load_source)


def multi_source():
    """A frame with six small objects in a row."""
    pixels = np.full((40, 40), 0.05, dtype=np.float32)
    boxes = []
    for k in range(6):
        pixels[2:4, 3 + 6 * k:6 + 6 * k] = 0.6 + 0.05 * k
        boxes.append([3 + 6 * k, 2, 6 + 6 * k, 4])
    return SourceImage(pixels, np.asarray(boxes, np.float64), (0,) * 6, "train", "multi.png")


def multi_load(reference):
    return multi_source() if reference["id"] == "multi.png" else load_source(reference)


def multi_pool(*indices):
    return [dict(source=dict(split="train", id="multi.png"), box_index=i, transform={}) for i in indices]


def keys(candidate):
    return [(obj["source"]["id"], obj["box_index"]) for obj in candidate["scene"]["objects"]]


class RepeatRuleTests(unittest.TestCase):
    background = dict(split="train", id="background.png")
    plain = dict(rotation_degrees=0, flip_x=False, seed=3)

    def test_no_object_twice_in_an_image(self):
        result = generate_candidates(self.background, multi_pool(0, 1, 2, 3),
                                     dict(self.plain, images=15, count_min=3, count_max=3), multi_load)
        self.assertEqual(len(result["candidates"]), 15)
        for candidate in result["candidates"]:
            self.assertEqual(len(set(keys(candidate))), 3)
        # Without the rules two pool objects cannot fill three places without a repeat.
        free = dict(self.plain, images=5, count_min=3, count_max=3, unique_per_image=False,
                    unique_source=False, balanced=False)
        result = generate_candidates(self.background, multi_pool(0, 1), free, multi_load)
        self.assertTrue(all(len(set(keys(c))) < 3 for c in result["candidates"]))
        with self.assertRaisesRegex(ValueError, "without repeats"):
            generate_candidates(self.background, multi_pool(0, 1), dict(free, unique_per_image=True), multi_load)

    def test_count_above_the_distinct_objects_is_lowered_with_a_note(self):
        result = generate_candidates(self.background, multi_pool(0, 1, 2),
                                     dict(self.plain, images=10, count_min=1, count_max=5), multi_load)
        self.assertTrue(result["notes"])
        self.assertEqual(result["rules"]["count_max"], 3)
        self.assertTrue(all(len(c["scene"]["objects"]) <= 3 for c in result["candidates"]))

    def test_one_object_per_source_box_and_none_from_the_background_frame(self):
        entries = multi_pool(0, 0, 1)
        entries[1]["transform"] = dict(brightness=0.8)  # another entry of the same real box
        result = generate_candidates(self.background, entries,
                                     dict(self.plain, images=10, count_min=2, count_max=2), multi_load)
        for candidate in result["candidates"]:
            self.assertEqual(sorted(box for _, box in keys(candidate)), [0, 1])
        with self.assertRaisesRegex(ValueError, "different source boxes"):
            generate_candidates(self.background, entries, dict(self.plain, count_min=3, count_max=3), multi_load)
        frame = dict(split="train", id="multi.png")
        mixed = multi_pool(0, 1) + [dict(source=dict(split="train", id="donor.png"), box_index=0, transform={})]
        result = generate_candidates(frame, mixed, dict(self.plain, images=5, count_min=1, count_max=1), multi_load)
        self.assertEqual(len(result["candidates"]), 5)
        self.assertTrue(all(keys(c) == [("donor.png", 0)] for c in result["candidates"]))
        only_frame = generate_candidates(frame, multi_pool(0, 1), dict(self.plain, images=2), multi_load)
        self.assertEqual(only_frame["candidates"], [])
        self.assertTrue(any("background frame" in reason for reason in only_frame["rejected"]))

    def test_balanced_use_over_the_batch(self):
        rules = dict(self.plain, images=30, count_min=1, count_max=1)
        result = generate_candidates(self.background, multi_pool(*range(6)), rules, multi_load)
        self.assertEqual(result["usage"], [5] * 6)
        counted = Counter(box for c in result["candidates"] for _, box in keys(c))
        self.assertEqual([counted[i] for i in range(6)], result["usage"])
        loose = generate_candidates(self.background, multi_pool(*range(6)), dict(rules, balanced=False), multi_load)
        self.assertEqual(sum(loose["usage"]), 30)
        self.assertGreater(max(loose["usage"]) - min(loose["usage"]), 1)


class GeneratorAPITests(unittest.TestCase):
    def request_body(self, split="train"):
        return dict(background=dict(split=split, id="a.png"),
                    pool=[dict(source=dict(split=split, id="a.png"), box_index=0, transform={})],
                    # Object and background come from the same frame, two per image: repeats allowed.
                    rules=dict(images=4, count_min=1, count_max=2, edges="blur", insertion="mask",
                               unique_per_image=False, unique_source=False, split=split))

    def test_generate_writes_nothing_and_selection_saves_exact_previews(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            views = EditorViews(write_dataset(root / "raw"))
            originals = {path: path.read_bytes() for path in (root / "raw").rglob("*") if path.is_file()}
            result = views.post("/api/synthetic/generate", self.request_body())
            self.assertEqual(len(result["candidates"]), 4)
            self.assertFalse((root / "output").exists())
            rows = [result["candidates"][0], result["candidates"][2]]
            body = dict(directory=str(root / "output"), split="train", items=[
                dict(scene=row["scene"], expected_digest=row["preview"]["digest"]) for row in rows])
            saved = views.post("/api/synthetic/generated-save", body)
            self.assertEqual(len(saved["saved"]), 2)
            self.assertEqual(len(list((root / "output/train/images").iterdir())), 2)
            for record, row in zip(saved["saved"], rows):
                with Image.open(record["image"]) as image:
                    np.testing.assert_array_equal(np.asarray(image, dtype=np.float32) / 255,
                                                  np.asarray(row["preview"]["matrix"], dtype=np.float32))
                self.assertEqual(record["digest"], row["preview"]["digest"])
            repeated = views.post("/api/synthetic/generated-save", body)
            self.assertEqual(len(repeated["saved"]), 0)
            self.assertEqual(len(repeated["skipped"]), 2)
            self.assertEqual(originals, {path: path.read_bytes() for path in originals})

    def test_val_and_stale_preview_rejected_before_any_batch_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            views = EditorViews(write_dataset(root / "raw"))
            train = views.post("/api/synthetic/generate", self.request_body())["candidates"][0]
            val = views.post("/api/synthetic/generate", self.request_body("val"))["candidates"][0]
            items = [dict(scene=row["scene"], expected_digest=row["preview"]["digest"]) for row in (train, val)]
            body = dict(directory=str(root / "output"), split="train", items=items)
            with self.assertRaisesRegex(ValueError, "comes from val"):
                views.post("/api/synthetic/generated-save", body)
            self.assertFalse((root / "output").exists())
            body["split"] = "val"
            with self.assertRaisesRegex(ValueError, "comes from train"):
                views.post("/api/synthetic/generated-save", body)
            self.assertFalse((root / "output").exists())
            body["items"] = body["items"][1:]
            body["items"][0]["expected_digest"] = "changed"
            with self.assertRaisesRegex(ValueError, "Sources changed"):
                views.post("/api/synthetic/generated-save", body)
            self.assertFalse((root / "output").exists())

    def test_partial_output_error_reports_saved_and_remaining_selections(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            views = EditorViews(write_dataset(root / "raw"))
            rows = views.post("/api/synthetic/generate", self.request_body())["candidates"][:3]
            body = dict(directory=str(root / "output"), split="train", items=[
                dict(scene=row["scene"], expected_digest=row["preview"]["digest"]) for row in rows])
            original = views.workspace.store.save
            attempts = 0
            def fail_after_first(*args):
                nonlocal attempts
                attempts += 1
                if attempts == 2:
                    raise OSError("simulated output error")
                return original(*args)
            with patch.object(views.workspace.store, "save", side_effect=fail_after_first):
                result = views.post("/api/synthetic/generated-save", body)
            self.assertEqual(len(result["saved"]), 1)
            self.assertEqual(len(result["skipped"]), 2)
            self.assertEqual(len(list((root / "output/train/images").iterdir())), 1)
            self.assertIn("simulated output error", result["skipped"][0]["reason"])

    def test_http_generator_assets_generation_and_selected_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            server = make_server(EditorViews(write_dataset(root / "raw")), 0)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}"
            try:
                for path in ("/autogen.js", "/autogen.css"):
                    with urlopen(base + path) as response:
                        self.assertEqual(response.status, 200)
                with urlopen(Request(base + "/api/synthetic/generate", data=json.dumps(self.request_body()).encode())) as response:
                    candidate = json.load(response)["candidates"][0]
                body = dict(directory=str(root / "output"), split="train", items=[
                    dict(scene=candidate["scene"], expected_digest=candidate["preview"]["digest"])])
                with urlopen(Request(base + "/api/synthetic/generated-save", data=json.dumps(body).encode())) as response:
                    self.assertEqual(len(json.load(response)["saved"]), 1)
                with self.assertRaises(HTTPError) as caught:
                    urlopen(Request(base + "/api/synthetic/generate", data=b'{}', headers={"Origin": "https://untrusted.example"}))
                self.assertEqual(caught.exception.code, 403)
            finally:
                server.shutdown()
                thread.join(timeout=5)
                server.server_close()


if __name__ == "__main__":
    unittest.main()
