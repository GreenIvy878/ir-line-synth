"""Tape regressions: rendering, layouts, backgrounds and the CSV export."""

import base64
import csv
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np

from scene_editor import morph
from scene_editor.background import BackgroundBank, background_options
from scene_editor.dataset import open_split
from scene_editor.scene import PreparedObject, box_iou, pixel_levels, prepare_entry, scene_statistics
from scene_editor.web.server import make_server
from scene_editor.tape import (
    BoxSet, generate_tape, render_tape, tape_csv, tape_digest, tape_options, tape_rules, tape_statistics,
    time_major_object,
)
from scene_editor.web.views import EditorViews
from test_editor import load_source, source
from test_morph import kinds_frame, still
from test_morph_app import write_kinds_dataset


def blank(background, length):
    return np.zeros((length, 40), np.float32), []


def donor_entry(**extra):
    return dict(source=dict(split="train", id="donor.png"), box_index=0, transform={}, **extra)


def tape(objects, length=200, **extra):
    return {**dict(length=length, background=dict(kind="blank"), blending="replace", max_iou=0.15,
                   objects=objects), **extra}


def render(spec):
    return render_tape(spec, load_source, prepare_entry, blank, 40)


def rules(**extra):
    return {**dict(length=400, objects=20, background="blank", seed=3), **extra}


def generate(options, entries=None):
    return generate_tape(entries or [donor_entry()], options, load_source, prepare_entry, 40)


class RenderTests(unittest.TestCase):
    def test_object_goes_to_channel_x_and_row_y_with_its_box(self):
        result = render(tape([dict(donor_entry(), x=7, y=120)]))
        self.assertEqual(result.levels.shape, (200, 40))
        self.assertEqual(result.levels.dtype, np.uint8)
        np.testing.assert_array_equal(result.boxes, [[7, 120, 10, 122]])
        np.testing.assert_array_equal(result.levels[120:122, 7:10], pixel_levels(source().pixels[2:4, 3:6]))
        self.assertEqual(int(result.levels.sum()), int(pixel_levels(source().pixels[2:4, 3:6]).sum()))

    def test_overlap_and_tape_edges_name_the_object(self):
        crowded = tape([dict(donor_entry(), x=7, y=50), dict(donor_entry(), x=7, y=50)])
        with self.assertRaisesRegex(ValueError, "Object 2: Placement exceeds"):
            render(crowded)
        with self.assertRaisesRegex(ValueError, "Object 1: .*fit completely inside 40 x 200"):
            render(tape([dict(donor_entry(), x=7, y=199)]))
        with self.assertRaisesRegex(ValueError, "Object 1: .*fit completely inside 40 x 200"):
            render(tape([dict(donor_entry(), x=38, y=5)]))
        render(tape([dict(donor_entry(), x=7, y=50), dict(donor_entry(), x=7, y=50)], max_iou=1))

    def test_description_is_validated(self):
        for bad in (dict(tape([]), length=39), dict(tape([]), length=20001), dict(tape([]), blending="x"),
                    dict(tape([]), background=dict(kind="blank", level=1)),
                    dict(tape([]), background=dict(kind="generated", split="other")),
                    dict(tape([]), background=dict(kind="generated", split="train", colour=1)),
                    dict(tape([]), objects="no"), dict(tape([]), max_iou=2)):
            with self.assertRaises(ValueError, msg=str(bad)):
                tape_options(bad)
        with self.assertRaisesRegex(ValueError, "Object 1: Each added object"):
            render(tape([5]))

    def test_digest_ignores_object_order_but_not_pixels_or_boxes(self):
        first = render(tape([dict(donor_entry(), x=7, y=20), dict(donor_entry(), x=20, y=90)]))
        swapped = render(tape([dict(donor_entry(), x=20, y=90), dict(donor_entry(), x=7, y=20)]))
        moved = render(tape([dict(donor_entry(), x=7, y=20), dict(donor_entry(), x=20, y=91)]))
        self.assertEqual(first.digest, swapped.digest)
        self.assertNotEqual(first.digest, moved.digest)
        self.assertEqual(first.payload()["digest"], first.digest)
        self.assertNotEqual(tape_digest(first.levels, first.boxes, first.labels),
                            tape_digest(first.levels[:100], first.boxes, first.labels))

    def test_payload_levels_decode_to_the_rendered_signal(self):
        result = render(tape([dict(donor_entry(), x=7, y=20)]))
        payload = result.payload()
        decoded = np.frombuffer(base64.b64decode(payload["levels_base64"]), np.uint8).reshape(200, 40)
        np.testing.assert_array_equal(decoded, result.levels)
        self.assertTrue(payload["train_eligible"])
        self.assertEqual(payload["statistics"]["total_objects"], 1)

    def test_validation_sources_make_a_tape_val_only(self):
        spec = tape([dict(donor_entry(), x=7, y=20)])
        spec["objects"][0]["source"]["split"] = "val"
        self.assertFalse(render(spec).payload()["train_eligible"])

    def test_box_set_and_statistics_match_the_pairwise_versions(self):
        rng = np.random.default_rng(1)
        corners = rng.uniform(0, 30, (60, 2))
        boxes = np.hstack([corners, corners + rng.uniform(1, 8, (60, 2))])
        cell = BoxSet()
        for box in boxes[:50]:
            cell.add(box)
        self.assertEqual(cell.count, 50)
        for box in boxes[50:]:
            self.assertAlmostEqual(cell.max_iou(box), max(box_iou(box, other) for other in boxes[:50]))
        expected = scene_statistics(boxes, len(boxes))
        found = tape_statistics(boxes, 100, 40)
        self.assertAlmostEqual(found["max_iou"], expected["max_iou"])
        self.assertAlmostEqual(found["min_distance"], expected["min_distance"])
        self.assertEqual(BoxSet().max_iou(boxes[0]), 0.0)

    def test_sensor_axis_y_objects_are_turned_to_time_major(self):
        box = (1.0, 2.0, 3.0, 5.0)
        prepared = PreparedObject(np.arange(12, dtype=np.float32).reshape(4, 3), np.ones((4, 3), bool), 0, box)
        turned = time_major_object(prepared)
        np.testing.assert_array_equal(turned.pixels, prepared.pixels.T)
        self.assertEqual(turned.box, (2.0, 1.0, 5.0, 3.0))
        self.assertEqual((turned.width, turned.height), (prepared.height, prepared.width))
        self.assertIsNone(time_major_object(PreparedObject(prepared.pixels, prepared.mask, 0)).box)


class GenerateTests(unittest.TestCase):
    def positions(self, result):
        return [obj["y"] for obj in result["tape"]["objects"]]

    def test_seeded_count_bounds_iou_and_order(self):
        first = generate(rules(objects=30, layout="random"))
        self.assertEqual(first, generate(rules(objects=30, layout="random")))
        self.assertNotEqual(first["tape"], generate(rules(objects=30, seed=4))["tape"])
        self.assertEqual((first["placed"], first["requested"], first["notes"]), (30, 30, []))
        rendered = render(first["tape"])
        self.assertEqual(len(rendered.boxes), 30)
        self.assertTrue((rendered.boxes >= 0).all() and (rendered.boxes[:, [0, 2]] <= 40).all()
                        and (rendered.boxes[:, [1, 3]] <= 400).all())
        for index, box in enumerate(rendered.boxes):
            self.assertTrue(all(box_iou(box, other) <= 0.15 + 1e-9 for other in rendered.boxes[:index]))
        self.assertEqual(self.positions(first), sorted(self.positions(first)))

    def test_even_layout_follows_equal_steps(self):
        result = generate(rules(objects=10, length=1000, layout="even", jitter=0))
        step = 100
        for index, obj in enumerate(result["tape"]["objects"]):
            centre = obj["y"] + 1  # the donor crop is two rows high
            self.assertAlmostEqual(centre, (index + 0.5) * step, delta=2)
        loose = generate(rules(objects=10, length=1000, layout="even", jitter=1))
        self.assertTrue(any(abs(obj["y"] + 1 - (index + 0.5) * step) > 2
                            for index, obj in enumerate(loose["tape"]["objects"])))

    def test_clusters_keep_their_objects_together(self):
        result = generate(rules(objects=12, length=3000, layout="clusters", cluster_min=4, cluster_max=4,
                                cluster_spread=2, max_iou=1))
        ys = sorted(self.positions(result))
        groups = 1 + sum(second - first > 40 for first, second in zip(ys, ys[1:]))
        self.assertLessEqual(groups, 3)
        scattered = sorted(self.positions(generate(rules(objects=12, length=3000, layout="random"))))
        self.assertGreater(1 + sum(second - first > 40 for first, second in zip(scattered, scattered[1:])), 3)

    def test_a_full_tape_places_what_fits_and_says_so(self):
        result = generate(rules(objects=300, length=40, max_iou=0))
        self.assertLess(result["placed"], 300)
        self.assertGreater(result["placed"], 20)
        self.assertEqual(len(result["tape"]["objects"]), result["placed"])
        self.assertIn(f"Placed {result['placed']} of 300", result["notes"][0])
        self.assertIn("Placement exceeds the maximum IoU (0)", result["rejected"])
        render(result["tape"])

    def test_no_objects_gives_a_blank_tape_and_pool_use_is_balanced(self):
        empty = generate(rules(objects=0))
        self.assertEqual(empty["tape"]["objects"], [])
        self.assertEqual(int(render(empty["tape"]).levels.sum()), 0)
        pool = [donor_entry(), dict(donor_entry(), transform=dict(brightness=0.5))]
        result = generate(rules(objects=20, length=2000), pool)
        self.assertEqual(result["usage"], [10, 10])

    def test_no_object_repeats_while_the_pool_has_unused_ones(self):
        pool = [dict(donor_entry(), transform=dict(brightness=0.5 + i / 100)) for i in range(70)]
        for layout in ("random", "even", "clusters"):
            for balanced in (True, False):  # the client's choice does not matter
                result = generate(rules(objects=50, length=1500, layout=layout, balanced=balanced), pool)
                self.assertEqual(result["placed"], 50)
                self.assertEqual(max(result["usage"]), 1, (layout, balanced))
        # More objects than the pool: every entry is used, as evenly as possible.
        result = generate(rules(objects=100, length=3000), pool)
        self.assertEqual(sorted(set(result["usage"])), [1, 2])
        self.assertEqual(sum(result["usage"]), 100)

    def test_rules_are_checked(self):
        for bad in (dict(layout="spiral"), dict(objects=1001), dict(length=10), dict(jitter=2), dict(background="real"),
                    dict(cluster_min=5, cluster_max=2), dict(count_min=2), dict(images=3), dict(unique_source=True),
                    dict(colour=1), dict(rotation_degrees=500)):
            with self.assertRaises(ValueError, msg=str(bad)):
                tape_rules(bad)
        with self.assertRaises(ValueError):
            generate_tape([], rules(), load_source, prepare_entry, 40)
        self.assertEqual(tape_rules({})["generator"]["unique_per_image"], False)

    def test_generated_background_is_described_by_its_seed(self):
        # The background comes from the split of the tape, as its objects do.
        entries = [dict(donor_entry(), source=dict(split="val", id="donor.png"))]
        options = dict(rules(objects=1), background="generated", split="val", background_block_min=3,
                       background_block_max=5, background_level_min=0.5, background_level_max=1.5)
        result = generate(options, entries)
        background = result["tape"]["background"]
        self.assertEqual((background["kind"], background["split"], background["block_min"]), ("generated", "val", 3))
        self.assertTrue(0.5 <= background["level"] <= 1.5)
        self.assertEqual(background, generate(options, entries)["tape"]["background"])
        # The old name of the rule still works when it names the same split, and only then.
        self.assertEqual(background, generate(dict(options, background_split="val"), entries)["tape"]["background"])
        with self.assertRaisesRegex(ValueError, "background_split train differs"):
            generate(dict(options, background_split="train"), entries)
        with self.assertRaisesRegex(ValueError, "comes from val: a train batch"):
            generate(dict(options, split="train"), entries)


class FileFormatTests(unittest.TestCase):
    def test_csv_values_and_header(self):
        levels = np.array([[0, 255, 51], [1, 2, 3]], np.uint8)
        self.assertEqual(tape_csv(levels), "0,255,51\n1,2,3\n")
        self.assertEqual(tape_csv(levels, header=True).splitlines()[0], "channel_0,channel_1,channel_2")
        rows = [[float(value) for value in row.split(",")] for row in tape_csv(levels, "intensity").splitlines()]
        np.testing.assert_allclose(np.round(np.array(rows) * 255), levels)
        with self.assertRaises(ValueError):
            tape_csv(levels, "floats")


class DatasetTapeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.data = write_kinds_dataset(cls.root / "raw")
        cls.views = EditorViews(cls.data)
        cls.pool = [dict(source=dict(split="train", id=f"f{index}.png"), box_index=index % 3, transform={})
                    for index in range(4)]
        frame, boxes = kinds_frame(0)
        spec = dict(kind="blob", motion=still(morph.cut_object(frame, boxes, 0)), imperfections=dict(noise=0.1, seed=3))
        cls.pool.append(dict(source=dict(split="train", id="f0.png"), box_index=0, transform={}, morph=spec))

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def generate(self, **extra):
        body = dict(pool=self.pool, rules={**dict(length=500, objects=25, seed=5, layout="clusters"), **extra})
        return self.views.post("/api/synthetic/tape-generate", body)

    def test_generated_tape_renders_the_same_again_and_blends_a_real_background(self):
        first = self.generate()
        self.assertEqual(first["placed"], 25)
        preview = self.views.post("/api/synthetic/tape-preview", dict(tape=first["tape"]))
        self.assertEqual(preview, first["preview"])
        levels = np.frombuffer(base64.b64decode(preview["levels_base64"]), np.uint8).reshape(500, 40)
        self.assertGreater(int(levels.max()), 100)
        self.assertEqual(first["tape"]["background"]["kind"], "generated")
        self.assertTrue(any("morph" in obj for obj in first["tape"]["objects"]))
        self.assertEqual(preview, self.views.post("/api/synthetic/tape-preview", dict(tape=first["tape"])))

    def test_edits_are_just_descriptions(self):
        first = self.generate()["tape"]
        edited = json.loads(json.dumps(first))
        removed = edited["objects"].pop(0)
        preview = self.views.post("/api/synthetic/tape-preview", dict(tape=edited))
        self.assertEqual(preview["statistics"]["total_objects"], 24)
        edited["objects"].append(dict(removed, y=first["length"] - 12, x=0))
        added = self.views.post("/api/synthetic/tape-preview", dict(tape=edited))
        self.assertEqual(len(added["boxes_pixels_xyxy"]), 25)
        edited["background"]["seed"] += 1
        other = self.views.post("/api/synthetic/tape-preview", dict(tape=edited))
        self.assertNotEqual(other["levels_base64"], added["levels_base64"])
        edited["objects"][0]["y"] = first["length"]
        with self.assertRaisesRegex(ValueError, "Object 1"):
            self.views.post("/api/synthetic/tape-preview", dict(tape=edited))

    def test_prepare_returns_the_object_in_tape_orientation(self):
        prepared = self.views.post("/api/synthetic/tape-prepare", self.pool[1])
        self.assertEqual(np.array(prepared["matrix"]).shape, (prepared["height"], prepared["width"]))

    def test_tape_background_pieces_come_from_empty_real_columns(self):
        bank = BackgroundBank(open_split(self.data, "train"))
        options = dict(seed=4, roll=False, block_min=2, block_max=4)
        first = bank.generate_tape(options, 130)
        np.testing.assert_array_equal(first, bank.generate_tape(options, 130))
        self.assertEqual(first.shape, (130, 40))
        self.assertFalse(np.array_equal(first, bank.generate_tape(dict(options, seed=5), 130)))
        self.assertEqual(bank.generate_tape(options, 7).shape, (7, 40))
        self.assertLess(float(first.max()), 0.1)

    def test_tape_background_spread_is_at_most_three_percent(self):
        bank = BackgroundBank(open_split(self.data, "train"))
        for seed in range(4):
            for level in (0.0, 1.0, 3.0):
                levels = np.round(bank.generate_tape(dict(seed=seed, level=level), 300) * 255)
                self.assertLessEqual(levels.max() - levels.min(), 7)
        self.assertEqual(background_options({})["spread"], 0.03)
        # Through the API too: the generated tape of a dataset, with objects removed.
        result = self.generate(seed=2)
        spec = dict(result["tape"], objects=[])
        preview = self.views.post("/api/synthetic/tape-preview", dict(tape=spec))
        levels = np.frombuffer(base64.b64decode(preview["levels_base64"]), np.uint8).astype(int)
        self.assertLessEqual(levels.max() - levels.min(), 7)

    def test_tape_background_is_one_set_of_sensors_without_seams(self):
        class Frames:
            """Frames whose channels have fixed offsets, but each frame drifts as a whole."""
            sensor_axis = "x"

            def __init__(self):
                rng = np.random.default_rng(7)
                self.offsets = np.linspace(0.0, 0.02, 40)
                self.images = [np.clip(self.offsets + rng.uniform(0, 0.03) + 0.001 * rng.standard_normal((40, 40)), 0, 1)
                               .astype(np.float32) for _ in range(12)]
                self.boxes = [np.zeros((0, 4)) for _ in self.images]

            def __len__(self):
                return len(self.images)

            def pixels(self, index):
                return self.images[index]

        bank = BackgroundBank(Frames())
        for seed in range(3):
            tape = bank.generate_tape(dict(seed=seed, block_min=3, block_max=6, spread=0.03), 800)
            pieces = tape.reshape(20, 40, 40).mean(1) * 255  # mean of every 40-row piece, per channel
            # Each channel keeps one level the whole length (frame drifts of up to 7 levels are gone) ...
            self.assertLess(float(pieces.std(0).max()), 0.6)
            # ... and the channels still differ from each other, as in real stripes.
            self.assertGreater(float(np.ptp(pieces.mean(0))), 1.0)
            levels = np.round(tape * 255)
            self.assertLessEqual(levels.max() - levels.min(), 7)

    def save(self, tape_description, preview, **extra):
        body = {**dict(tape=tape_description, expected_digest=preview["digest"], directory=str(self.root / "out"),
                       split="train"), **extra}
        return self.views.post("/api/synthetic/tape-save", body)

    def test_save_writes_signal_labels_and_description_and_refuses_duplicates(self):
        first = self.generate(seed=11)
        saved = self.save(first["tape"], first["preview"], name="long_run")
        signal, labels, description = (Path(saved[key]) for key in ("signal", "labels", "provenance"))
        self.assertEqual(signal, self.root / "out" / "tapes" / "train" / "long_run.csv")
        rows = np.loadtxt(signal, delimiter=",", dtype=np.int64)
        expected = np.frombuffer(base64.b64decode(first["preview"]["levels_base64"]), np.uint8).reshape(500, 40)
        np.testing.assert_array_equal(rows, expected)
        with labels.open(newline="") as stream:
            table = list(csv.DictReader(stream))
        self.assertEqual(len(table), 25)
        self.assertEqual(list(table[0])[:6], ["index", "class", "channel_min", "row_min", "channel_max", "row_max"])
        boxes = np.array([[float(row[key]) for key in ("channel_min", "row_min", "channel_max", "row_max")] for row in table])
        np.testing.assert_allclose(boxes, first["preview"]["boxes_pixels_xyxy"])
        expected_kinds = [obj["morph"]["kind"] if "morph" in obj else "real" for obj in first["tape"]["objects"]]
        self.assertEqual([row["kind"] for row in table], expected_kinds)
        self.assertIn("blob", expected_kinds)
        metadata = json.loads(description.read_text(encoding="utf-8"))
        self.assertEqual((metadata["digest"], metadata["length"], metadata["channels"], metadata["values"]),
                         (first["preview"]["digest"], 500, 40, "levels"))
        self.assertEqual(metadata["tape"], first["tape"])
        # The same dataset can still be used for scenes and keeps its marker and splits.
        self.assertTrue((self.root / "out" / "synthetic_dataset.json").is_file())
        with self.assertRaisesRegex(ValueError, "identical tape"):
            self.save(first["tape"], first["preview"], name="again")
        self.assertFalse((self.root / "out" / "tapes" / "train" / "again.csv").exists())

    def test_save_formats_names_and_stale_previews(self):
        second = self.generate(seed=12)
        third = self.generate(seed=13)
        with self.assertRaisesRegex(ValueError, "changed"):
            self.save(second["tape"], third["preview"])
        for name in ("../escape", "a b", "x" * 65, 5):
            with self.assertRaisesRegex(ValueError, "name"):
                self.save(second["tape"], second["preview"], name=name)
        with self.assertRaisesRegex(ValueError, "values"):
            self.save(second["tape"], second["preview"], values="floats")
        saved = self.save(second["tape"], second["preview"], values="intensity", header=True)
        first_lines = Path(saved["signal"]).read_text().splitlines()
        self.assertEqual(first_lines[0].split(",")[:2], ["channel_0", "channel_1"])
        self.assertEqual(len(first_lines), 501)
        self.assertTrue(all(0 <= float(value) <= 1 for value in first_lines[1].split(",")))
        self.assertTrue(Path(saved["signal"]).name.startswith("tape_"))

    def test_validation_sources_cannot_enter_a_training_tape(self):
        pool = [dict(self.pool[0], source=dict(split="val", id="f0.png"))]
        with self.assertRaisesRegex(ValueError, "Pool object 1 comes from val"):
            self.views.post("/api/synthetic/tape-generate",
                            dict(pool=pool, rules=dict(length=100, objects=2, background="blank", seed=1)))
        result = self.views.post("/api/synthetic/tape-generate",
                                 dict(pool=pool, rules=dict(length=100, objects=2, background="blank", seed=1,
                                                            split="val")))
        self.assertFalse(result["preview"]["train_eligible"])
        self.assertEqual(result["preview"]["splits"], ["val"])
        with self.assertRaisesRegex(ValueError, "comes from val"):
            self.save(result["tape"], result["preview"])
        with self.assertRaisesRegex(ValueError, "comes from val"):
            self.save(result["tape"], result["preview"], split="test")
        self.assertTrue(self.save(result["tape"], result["preview"], split="val")["signal"].endswith(".csv"))

    def test_output_cannot_be_inside_a_source_dataset(self):
        result = self.generate(seed=14)
        body = dict(tape=result["tape"], expected_digest=result["preview"]["digest"],
                    directory=str(self.root / "raw" / "train" / "tapes"), split="train")
        with self.assertRaisesRegex(ValueError, "separate"):
            self.views.post("/api/synthetic/tape-save", body)

    def test_http_accepts_a_large_tape_and_serves_the_page_files(self):
        result = self.generate(seed=15)
        server = make_server(self.views, 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            for path in ("/tape.js", "/tape.css"):
                with urlopen(base + path) as response:
                    self.assertEqual(response.status, 200)
            # Far above the 512 KiB limit of ordinary requests.
            body = json.dumps(dict(tape=result["tape"], padding="x" * 700_000)).encode()
            with urlopen(Request(base + "/api/synthetic/tape-preview", data=body)) as response:
                self.assertEqual(json.load(response)["digest"], result["preview"]["digest"])
            with self.assertRaises(HTTPError) as caught:
                urlopen(Request(base + "/api/synthetic/tape-prepare", data=body))
            self.assertEqual(caught.exception.code, 400)
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()


if __name__ == "__main__":
    unittest.main()
