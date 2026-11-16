"""One split at a time: nothing crosses splits, stubs cut by the frame edge are never used,
and the smallest generated objects are sifted out."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from scene_editor import morph
from scene_editor.dataset import load_data_description
from scene_editor.web.views import EditorViews
from test_morph import all_objects, draw_egg, kinds_frame


SPLITS = ("train", "val", "test")


def stub_frame(seed):
    """A frame with an egg of every kind and a stub cut by the bottom edge (the last box)."""
    pixels, boxes = kinds_frame(seed)
    pixels = pixels.copy()
    stub = draw_egg(pixels, 23, 35, 5)  # rows 35..39: the box ends at the frame edge
    return np.clip(pixels, 0, 1), np.vstack([boxes, np.asarray([stub], np.float32)])


def write_split_dataset(root, frames=8):
    """Frames named after their split (train_f0.png, ...), so a frame tells its split."""
    for split in SPLITS:
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir()
        for seed in range(frames):
            pixels, boxes = stub_frame(seed)
            name = f"{split}_f{seed}"
            Image.fromarray(np.round(pixels * 255).astype(np.uint8)).save(root / split / "images" / f"{name}.png")
            lines = [f"0 {(b[0] + b[2]) / 80:.6f} {(b[1] + b[3]) / 80:.6f} {(b[2] - b[0]) / 40:.6f} "
                     f"{(b[3] - b[1]) / 40:.6f}" for b in np.clip(boxes, 0, 40)]
            (root / split / "labels" / f"{name}.txt").write_text("\n".join(lines) + "\n")
    (root / "data.yaml").write_text("data:\n" + "".join(f"  {split}: {split}\n" for split in SPLITS))
    return load_data_description(root)


STUB = 5  # index of the stub box in every frame


def entry(split, seed=0, box_index=0):
    return dict(source=dict(split=split, id=f"{split}_f{seed}.png"), box_index=box_index, transform={})


class SplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.views = EditorViews(write_split_dataset(cls.root / "raw"))

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def pool(self, split):
        return [entry(split, seed, box) for seed in range(4) for box in range(5)]

    def generate(self, split, mode, **rules):
        background = dict(split=split, id=f"{split}_f7.png")
        return self.views.post("/api/synthetic/generate", dict(
            background=background, pool=self.pool(split),
            rules=dict(dict(images=3, count_min=2, count_max=3, background_mode=mode, split=split, seed=5), **rules)))

    def saved_sources(self, record):
        return json.loads(Path(record["provenance"]).read_text(encoding="utf-8"))["sources"]

    def check_sources(self, sources, split):
        for row in sources:
            self.assertEqual(row["split"], split, row)
            if row["role"] == "object":
                self.assertTrue(row["frame"].startswith(split + "_"), row)
                self.assertIsInstance(row["box_index"], int)
                self.assertFalse(row["clipped"])
            else:
                self.assertTrue(row["frames"], row)
                self.assertTrue(all(frame.startswith(split + "_") for frame in row["frames"]), row)

    def test_each_split_takes_objects_and_backgrounds_only_from_itself(self):
        for split in SPLITS:
            for mode in ("fixed", "real", "generated"):
                with self.subTest(split=split, mode=mode):
                    result = self.generate(split, mode)
                    self.assertTrue(result["candidates"])
                    out = str(self.root / f"out_{split}_{mode}")
                    for candidate in result["candidates"]:
                        self.assertEqual(candidate["preview"]["splits"], [split])
                        scene = candidate["scene"]
                        self.assertEqual(scene["background"]["split"], split)
                        self.assertTrue(all(obj["source"]["split"] == split for obj in scene["objects"]))
                    items = [dict(scene=row["scene"], expected_digest=row["preview"]["digest"])
                             for row in result["candidates"]]
                    for other in SPLITS:
                        if other != split:
                            with self.assertRaisesRegex(ValueError, f"comes from {split}"):
                                self.views.post("/api/synthetic/generated-save",
                                                dict(items=items, directory=out, split=other))
                    saved = self.views.post("/api/synthetic/generated-save", dict(items=items, directory=out,
                                                                                    split=split))["saved"]
                    self.assertEqual(len(saved), len(items))
                    for record in saved:
                        self.assertIn(f"{Path(out).name}\\{split}\\".replace("\\", "/"),
                                      record["image"].replace("\\", "/"))
                        self.check_sources(self.saved_sources(record), split)

    def test_each_split_makes_tapes_only_from_itself(self):
        for split in SPLITS:
            with self.subTest(split=split):
                result = self.views.post("/api/synthetic/tape-generate", dict(
                    pool=self.pool(split), rules=dict(length=200, objects=8, seed=2, split=split)))
                self.assertEqual(result["tape"]["background"]["split"], split)
                self.assertEqual(result["preview"]["splits"], [split])
                body = dict(tape=result["tape"], expected_digest=result["preview"]["digest"],
                            directory=str(self.root / f"tapes_{split}"))
                for other in SPLITS:
                    if other != split:
                        with self.assertRaisesRegex(ValueError, f"comes from {split}"):
                            self.views.post("/api/synthetic/tape-save", dict(body, split=other))
                saved = self.views.post("/api/synthetic/tape-save", dict(body, split=split))
                self.check_sources(self.saved_sources(saved), split)

    def test_a_source_of_another_split_is_refused(self):
        with self.assertRaisesRegex(ValueError, "Pool object 1 comes from train: a test batch"):
            self.views.post("/api/synthetic/generate", dict(
                background=dict(split="test", id="test_f7.png"), pool=[entry("train")],
                rules=dict(split="test", images=1)))
        with self.assertRaisesRegex(ValueError, "background comes from val: a test batch"):
            self.views.post("/api/synthetic/generate", dict(
                background=dict(split="val", id="val_f7.png"), pool=[entry("test")],
                rules=dict(split="test", images=1)))
        # A hand-made scene with a train object on a test background goes nowhere, val and test included.
        scene = dict(background=dict(split="test", id="test_f7.png"), max_iou=1,
                     objects=[dict(entry("train"), x=20, y=30), dict(entry("test", 1), x=10, y=30)])
        preview = self.views.post("/api/synthetic/preview", dict(scene=scene))
        self.assertEqual(preview["splits"], ["test", "train"])
        for split in SPLITS:
            with self.assertRaisesRegex(ValueError, f"A {split} scene takes every object"):
                self.views.post("/api/synthetic/save", dict(scene=scene, expected_digest=preview["digest"],
                                                            directory=str(self.root / "mixed"), split=split))
        self.assertFalse((self.root / "mixed").exists())
        with self.assertRaisesRegex(ValueError, "Source object 1 comes from val: objects for test"):
            self.views.post("/api/synthetic/morph-generate", dict(kind="blob", split="test", sources=[entry("val")]))

    def test_output_has_three_splits(self):
        result = self.generate("test", "fixed", images=1)
        out = self.root / "three"
        row = result["candidates"][0]
        self.views.post("/api/synthetic/generated-save", dict(
            items=[dict(scene=row["scene"], expected_digest=row["preview"]["digest"])], directory=str(out),
            split="test"))
        self.assertIn("test: ./test", (out / "data.yaml").read_text(encoding="utf-8"))
        self.assertEqual(len(list((out / "test" / "images").iterdir())), 1)

    def test_old_output_without_test_split_is_extended(self):
        out = self.root / "old"
        out.mkdir()
        (out / "synthetic_dataset.json").write_text(json.dumps(dict(
            schema=1, sensor_axis="x", scale=255.0, offset=0.0, image_format="8-bit PNG", annotations="YOLO")))
        (out / "data.yaml").write_text("data:\n  type: yolo_images\n  train: ./train\n  val: ./val\n"
                                       "  sensor_axis: x\n  scale: 255.0\n  offset: 0.0\n")
        self.views.workspace.store.initialize(str(out))
        self.assertIn("test: ./test", (out / "data.yaml").read_text(encoding="utf-8"))

    # --- stubs ---------------------------------------------------------------------------------

    def test_stubs_are_marked_in_the_editor_and_left_out_of_the_generators(self):
        sample = self.views.get("/api/synthetic/sample", dict(split=["val"], id=["val_f0.png"]))
        self.assertEqual(sample["clipped"], [False] * STUB + [True])
        pool = self.pool("val") + [entry("val", seed, STUB) for seed in range(4)]
        result = self.views.post("/api/synthetic/generate", dict(
            background=dict(split="val", id="val_f7.png"), pool=pool,
            rules=dict(images=4, count_min=3, count_max=5, split="val", unique_source=False)))
        self.assertTrue(any("4 of 24 pool objects are cut" in note for note in result["notes"]))
        self.assertEqual(len(result["usage"]), 20)
        for candidate in result["candidates"]:
            self.assertTrue(all(obj["box_index"] != STUB for obj in candidate["scene"]["objects"]))
        with self.assertRaisesRegex(ValueError, "Every chosen object is cut by the frame edge"):
            self.views.post("/api/synthetic/generate", dict(
                background=dict(split="val", id="val_f7.png"), pool=[entry("val", 0, STUB)],
                rules=dict(images=1, split="val")))
        tape = self.views.post("/api/synthetic/tape-generate", dict(
            pool=pool, rules=dict(length=300, objects=12, seed=4, split="val")))
        self.assertTrue(all(obj["box_index"] != STUB for obj in tape["tape"]["objects"]))
        self.assertTrue(any("cut by the frame edge" in note for note in tape["notes"]))

    def test_stubs_are_never_sources_of_new_objects(self):
        with self.assertRaisesRegex(ValueError, "cut by the frame edge"):
            self.views.post("/api/synthetic/morph-generate", dict(
                kind="short", split="train", count=3, sources=[entry("train", 0, STUB)]))
        result = self.views.post("/api/synthetic/morph-generate", dict(
            kind="short", split="train", count=5, sources=[entry("train", 0, STUB), entry("train", 1, 2)]))
        self.assertTrue(any("1 of 2 chosen source objects" in note for note in result["notes"]))
        self.assertTrue(all(row["box_index"] == 2 for row in result["candidates"]))
        library = self.views.workspace.library("train")
        stubs = [o for o in library.objects if o.clipped]
        self.assertEqual(len(stubs), 8)
        for kind in morph.KINDS:
            for row in library.candidates(kind, 12, 3, {}):
                self.assertNotEqual(row["box_index"], STUB, kind)
        # Counts per kind and the statistics hold whole objects only.
        self.assertEqual(sum(library.counts().values()), len(library.objects) - len(stubs))
        self.assertEqual(library.stats["kinds"], library.counts())


class StubRuleTests(unittest.TestCase):
    def test_box_closer_than_one_cell_to_an_edge(self):
        frame = np.zeros((40, 40), np.float32)
        box = draw_egg(frame, 20, 10, 7)
        for edge, near in ((0, (0.7, None, None, None)), (1, (None, 0.7, None, None)),
                           (2, (None, None, 39.3, None)), (3, (None, None, None, 39.3))):
            moved = np.asarray(box, np.float64)
            moved[edge] = near[edge]
            self.assertTrue(morph.box_near_edge(moved, (40, 40)), edge)
        self.assertFalse(morph.box_near_edge([1.0, 1.0, 39.0, 39.0], (40, 40)))
        frame = np.zeros((40, 40), np.float32)
        box = draw_egg(frame, 20, 1, 7)
        box[1] = 0.7  # the box 0.7 cells from the top
        self.assertTrue(morph.cut_object(frame, np.asarray([box]), 0).clipped)

    def test_pixel_on_the_edge_row_with_the_box_further_than_one_cell(self):
        frame = np.zeros((40, 40), np.float32)
        box = draw_egg(frame, 20, 0, 8)  # pixels from row 0
        self.assertGreaterEqual(frame[0].max(), morph.TAU)
        box[1] = 1.2  # but the box was drawn more than one cell from the edge
        self.assertFalse(morph.box_near_edge(box, frame.shape))
        self.assertTrue(morph.cut_object(frame, np.asarray([box]), 0).clipped)
        whole = np.zeros((40, 40), np.float32)
        self.assertFalse(morph.cut_object(whole, np.asarray([draw_egg(whole, 20, 5, 8)]), 0).clipped)

    def test_a_short_stub_is_not_a_source_of_its_kind(self):
        objects = all_objects()
        frame = np.zeros((40, 40), np.float32)
        stub = morph.cut_object(frame, np.asarray([draw_egg(frame, 20, 34, 6)]), 0, "stub.png")
        self.assertTrue(stub.clipped)
        self.assertIn(stub.kind, ("blob", "short"))
        library = morph.Library("train", "x", objects + [stub], morph.fit_statistics(objects + [stub]))
        self.assertEqual(library.counts(), morph.fit_statistics(objects)["kinds"])
        for kind in ("blob", "short"):
            pool, _ = library.source_pool(kind)
            self.assertFalse(any(obj is stub for obj in pool))
            self.assertTrue(all(row["source"]["id"] != "stub.png" for row in library.candidates(kind, 12, 1, {})))
        with self.assertRaisesRegex(ValueError, "cut by the frame edge"):
            library.candidates("blob", 3, 1, {}, sources=[stub])


class SiftTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        objects = all_objects()
        cls.library = morph.Library("train", "x", objects, morph.fit_statistics(objects))

    def test_no_object_below_the_thresholds_and_at_most_a_tenth_removed_per_feature(self):
        for kind in morph.KINDS:
            with self.subTest(kind=kind):
                batch = self.library.generate(kind, 40, 9, dict(dropout_max=0.4, noise_max=0.3))
                report = batch["size_filter"]
                self.assertTrue(report["applied"])
                self.assertEqual((report["removed_short"], report["removed_narrow"]), (4, 4))
                faint = kind in ("rolling", "swinging")
                if faint:
                    self.assertEqual((report["removed_small"], report["removed_dim"]), (4, 8))
                else:
                    self.assertNotIn("removed_dim", report)
                self.assertLessEqual(report["removed"], 20 if faint else 8)
                self.assertGreaterEqual(report["removed"], 8 if faint else 4)
                rows = batch["candidates"]
                self.assertEqual(len(rows), 40 - report["missing"])
                self.assertEqual(report["replaced"], report["removed"] - report["missing"])
                self.assertEqual(set(report["floor"]), set(morph.sift_shares(kind)))
                self.assertTrue(all(row[key] >= value for row in rows for key, value in report["floor"].items()))
                # The measures are those of the object's pixels above the threshold.
                for row in rows[:5]:
                    features = morph.describe(np.asarray(row["matrix"], np.float32))
                    self.assertEqual((row["length"], row["width"], row["area"]),
                                     (features["h"], features["w"], features["area"]))
                    self.assertAlmostEqual(row["brightness"], features["mean"], places=4)

    def test_the_same_seed_gives_the_same_batch(self):
        first = self.library.generate("blob", 20, 4, dict(dropout_max=0.3))
        self.assertEqual(first, self.library.generate("blob", 20, 4, dict(dropout_max=0.3)))
        self.assertNotEqual(first, self.library.generate("blob", 20, 5, dict(dropout_max=0.3)))

    def test_small_batches_are_not_changed(self):
        batch = self.library.generate("short", 9, 2, {})
        self.assertFalse(batch["size_filter"]["applied"])
        self.assertEqual(batch["size_filter"]["removed"], 0)
        self.assertEqual(len(batch["candidates"]), 9)
        # The batch of 9 is the first 9 of the draws, as without the sifting.
        self.assertEqual(batch["candidates"], self.library.generate("short", 9, 2, {})["candidates"])
        rows = [dict(length=3, width=3)] * 9
        self.assertEqual(morph.sift_small(rows, 1, lambda attempts, smallest: iter(()))[0], rows)

    def test_ties_are_broken_by_the_seed_and_the_thresholds_stay_fixed(self):
        rows = [dict(name=index, length=5 if index % 2 else 6, width=4) for index in range(20)]
        asked = []

        def draw(attempts, smallest):
            asked.append(smallest)
            for index in range(attempts):
                yield dict(name=100 + index, length=5 + index % 3, width=4)

        first, report = morph.sift_small(rows, 3, draw)
        self.assertEqual(first, morph.sift_small(rows, 3, draw)[0])
        # Two shortest (length 5) and two narrowest (all width 4), chosen at random among the ties.
        self.assertEqual((report["removed_short"], report["removed_narrow"]), (2, 2))
        self.assertEqual(report["removed"], len([row for row in rows if row not in first]))
        self.assertEqual((report["min_length"], report["min_width"]), (5, 4))
        self.assertEqual(asked, [dict(length=5, width=4)] * 2)
        self.assertEqual(len(first), 20)
        others = {tuple(row["name"] for row in morph.sift_small(rows, seed, draw)[0]) for seed in range(10)}
        self.assertGreater(len(others), 1)

    def test_faint_kinds_also_lose_the_smallest_and_dimmest(self):
        self.assertEqual(morph.sift_shares("blob"), dict(length=0.1, width=0.1))
        self.assertEqual(morph.sift_shares("rolling"), dict(length=0.1, width=0.1, area=0.1, brightness=0.2))
        self.assertEqual(morph.sift_shares("swinging"), morph.sift_shares("rolling"))
        # All alike in length and width; area falls and brightness rises with the index.
        rows = [dict(name=index, length=6, width=4, area=40 - index, brightness=0.2 + index / 100)
                for index in range(20)]
        asked = []

        def draw(attempts, floor):
            asked.append(floor)
            for index in range(attempts):
                yield dict(name=100 + index, length=6, width=4, area=30, brightness=0.5)

        result, report = morph.sift_small(rows, 0, draw, morph.sift_shares("rolling"))
        names = {row["name"] for row in result}
        # The 2 smallest by area (19, 18) and the 4 dimmest (0-3) go; 2+2 more by the length and width ties.
        self.assertTrue({0, 1, 2, 3, 18, 19}.isdisjoint(names))
        self.assertEqual((report["removed_small"], report["removed_dim"]), (2, 4))
        self.assertLessEqual(report["removed"], 10)
        kept = [row for row in rows if row["name"] in names]
        self.assertEqual(asked, [{key: min(row[key] for row in kept)
                                  for key in ("length", "width", "area", "brightness")}])
        self.assertGreaterEqual(asked[0]["brightness"], 0.24)
        self.assertEqual(len(result), 20)

    def test_too_few_replacements_are_reported(self):
        rows = [dict(length=4 + index, width=3 + index) for index in range(10)]
        result, report = morph.sift_small(rows, 0, lambda attempts, smallest: iter([dict(length=99, width=99)]))
        self.assertEqual((report["removed"], report["replaced"], report["missing"]), (1, 1, 0))
        rows = [dict(length=4 + index, width=13 - index) for index in range(20)]
        result, report = morph.sift_small(rows, 0, lambda attempts, smallest: iter(()))
        self.assertEqual((report["removed"], report["replaced"], report["missing"]), (4, 0, 4))
        self.assertEqual(len(result), 16)


if __name__ == "__main__":
    unittest.main()
