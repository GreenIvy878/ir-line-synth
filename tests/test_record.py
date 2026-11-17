"""A synthetic record without real data: objects described by shape under the reference line."""

import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import yaml

from ir_line_synth.__main__ import main
from ir_line_synth.recipe import load_recipe, run
from ir_line_synth.record import generate_record, natural_motion, record_rules, record_surface, render_record
from ir_line_synth.scene import box_iou
from ir_line_synth.sensor_model import load_profile
from ir_line_synth.verify import verify
from ir_line_synth.shapes import Installation, Superquadric, geometric_box, passage, Surface


MODEL = load_profile("reference_line_40")            # 40 channels, 10 polls per row, frames of 40 rows
INSTALLATION = dict(height_mm=55.0, pitch_mm=10.0, poll_hz=200.0, belt_reflectance=0.3)
EGG = dict(length=57, width=44, height=44)


def steady(rows, kind="blob"):
    """A calm motion at the belt speed for a passage of `rows` rows."""
    from ir_line_synth.motion import output_times
    n = len(output_times(0.0, rows, np.ones(8)))
    return dict(kind=kind, speed=[1.0] * 8, lateral=[0.0] * n, gain=[1.0] * n)


def egg_rows():
    return 57 / (160.0 / 200.0 * 10)


def record(objects, rows=80, seed=3, margins=(0, 0, 0, 0)):
    return dict(profile="reference_line_40", installation=INSTALLATION, speed_mm_s=160.0, rows=rows, seed=seed,
                margins=list(margins), objects=objects)


def egg(row, y=0.0, surface=None, move=None):
    item = dict(shape=EGG, motion=move or steady(egg_rows()), row=row, y=y)
    if surface is not None:
        item["surface"] = surface
    return item


class RecordTests(unittest.TestCase):
    def test_an_empty_record_is_the_bare_belt(self):
        rendered = render_record(record([]), MODEL)
        self.assertEqual(rendered.codes.shape, (800, 40))
        self.assertEqual(rendered.boxes.shape, (0, 4))
        self.assertEqual(rendered.frames.shape, (2, 40, 40))
        self.assertEqual(rendered.frames.dtype, np.uint8)
        self.assertEqual(rendered.frame_boxes, ((), ()))
        offset = rendered.codes.mean(axis=0) - MODEL.operation.belt_level
        self.assertLess(float(np.abs(offset).max()), 8.0)

    def test_one_egg(self):
        rendered = render_record(record([egg(10)]), MODEL)
        own, _ = passage(Superquadric(**EGG), Surface(), steady(egg_rows()), MODEL,
                         Installation(**INSTALLATION), 160.0)
        expected = geometric_box(own, 10) + [0, 10, 0, 10]
        np.testing.assert_array_equal(rendered.geometry, [expected])
        np.testing.assert_array_equal(rendered.boxes, [expected])
        x1, y1, x2, y2 = expected.astype(int)
        self.assertEqual(x2 - x1, 4)                  # 44 mm wide: the channels at -15, -5, 5 and 15 mm
        under = rendered.codes.reshape(80, 10, 40).mean(axis=1)[y1:y2, x1:x2] - MODEL.operation.belt_level[x1:x2]
        self.assertGreater(float(under.max()), 400)   # the egg rises hundreds of codes over the belt
        (index, box, cut), = rendered.frame_boxes[0]
        self.assertEqual((index, cut), (0, False))
        np.testing.assert_array_equal(box, expected)
        self.assertEqual(rendered.frame_boxes[1], ())

    def test_the_same_description_renders_the_same_record(self):
        spec = record([egg(5, surface=dict(spots=[dict(coverage=0.2, factor=0.0, size_mm=4.0)], seed=2))])
        first, second = render_record(spec, MODEL), render_record(copy.deepcopy(spec), MODEL)
        np.testing.assert_array_equal(first.codes, second.codes)
        np.testing.assert_array_equal(first.frames, second.frames)
        other = render_record(dict(spec, seed=4), MODEL)
        self.assertFalse(np.array_equal(first.codes, other.codes))
        np.testing.assert_array_equal(first.boxes, other.boxes)

    def test_the_line_sees_the_highest_surface(self):
        low = dict(shape=dict(length=120, width=120, height=10, vertical=0.1, horizontal=0.1),
                   surface=dict(reflectance=0.5), motion=steady(120 / 8), row=5)
        heights, rho, geometry, _ = record_surface(record([low, egg(8, surface=dict(reflectance=0.9))]), MODEL)
        top = heights > 10.5
        self.assertTrue(top.any())
        np.testing.assert_array_equal(rho[top], 0.9)
        np.testing.assert_array_equal(rho[(heights > 0) & ~top & (heights < 10.5)], 0.5)
        self.assertEqual(len(geometry), 2)

    def test_margins_and_frames_cut_by_the_tile_edge(self):
        rendered = render_record(record([egg(36)], margins=(-0.25, 0, 0.2, 0)), MODEL)
        x1, y1, x2, y2 = rendered.geometry[0]
        np.testing.assert_allclose(rendered.boxes[0], [x1 - 0.25, y1, x2 + 0.2, y2])
        (first, in_first, cut_first), = rendered.frame_boxes[0]
        (second, in_second, cut_second), = rendered.frame_boxes[1]
        self.assertTrue(cut_first and cut_second)
        np.testing.assert_allclose(in_first, [x1 - 0.25, y1, x2 + 0.2, 40])
        np.testing.assert_allclose(in_second, [x1 - 0.25, 0, x2 + 0.2, y2 - 40])

    def test_wrong_descriptions_are_refused(self):
        with self.assertRaisesRegex(ValueError, "the record has 80 rows"):
            render_record(record([egg(75)]), MODEL)
        with self.assertRaisesRegex(ValueError, "not reference_line_40"):
            render_record(dict(record([]), profile="example_line_32"), MODEL)
        with self.assertRaisesRegex(ValueError, "passes under no channel"):
            render_record(record([egg(5, y=900.0)]), MODEL)
        with self.assertRaisesRegex(ValueError, "unknown keys"):
            render_record(record([dict(egg(5), colour="white")]), MODEL)
        with self.assertRaisesRegex(ValueError, "margins"):
            render_record(record([], margins=(0, 0)), MODEL)

    def test_random_motions_of_every_kind_fit_their_passage(self):
        rng = np.random.default_rng(1)
        shape = Superquadric(**EGG)
        objects = []
        for index, kind in enumerate(("blob", "rolling", "swinging", "long", "short")):
            move = natural_motion(kind, shape, MODEL, Installation(**INSTALLATION), 160.0, rng)
            objects.append(dict(shape=EGG, motion=move, row=40 * index + 5, y=-40.0 + 20 * index))
        rendered = render_record(record(objects, rows=200), MODEL)
        self.assertEqual(len(rendered.boxes), 5)
        self.assertEqual(len(rendered.frames), 5)


RULES = dict(rows=80, speed_mm_s=[140, 180], count=[4, 6], max_iou=0.05, yaw=[-20, 20],
             motions=dict(blob=2, rolling=1, long=1),
             shapes=[dict(name="egg", weight=3, length=[52, 62], width=[40, 46], height=[40, 46],
                          surface=dict(reflectance=[0.8, 1.1], roughness_mm=[0, 0.5], roughness_size_mm=4,
                                       spots=[dict(coverage=[0, 0.1], factor=[0, 0.4], size_mm=3)])),
                     dict(name="box", length=[60, 100], width=[40, 60], height=[20, 30], vertical=0.1,
                          horizontal=0.1)])


class RandomRecordTests(unittest.TestCase):
    def setUp(self):
        self.installation = Installation(**INSTALLATION)
        self.rules = record_rules(copy.deepcopy(RULES), MODEL, self.installation)

    def test_a_random_record_follows_its_rules_and_repeats_with_the_seed(self):
        spec, notes = generate_record(self.rules, MODEL, self.installation, np.random.default_rng(8))
        again, _ = generate_record(self.rules, MODEL, self.installation, np.random.default_rng(8))
        self.assertEqual(json.dumps(spec), json.dumps(again))
        self.assertTrue(4 <= len(spec["objects"]) <= 6, notes)
        self.assertTrue(140 <= spec["speed_mm_s"] <= 180)
        self.assertEqual({item["name"] for item in spec["objects"]} - {"egg", "box"}, set())
        rendered = render_record(spec, MODEL)
        for first in range(len(rendered.geometry)):
            for second in range(first):
                self.assertLessEqual(box_iou(rendered.geometry[first], rendered.geometry[second]), 0.05 + 1e-9)
        self.assertEqual(rendered.frames.shape, (2, 40, 40))

    def test_rules_are_checked(self):
        for change, message in (
                (dict(shapes=[dict(length=50, width=40, height=60)]), "up to 50 mm"),
                (dict(motions=dict(hopping=1)), "rules.motions"),
                (dict(count=[3, 2]), "low must not exceed high"),
                (dict(colour="red"), "unknown keys"),
                (dict(shapes=[dict(length=50, width=40, height=30, surface=dict(reflectance_spread=0.2))]),
                 "needs spread_size_mm")):
            with self.assertRaisesRegex(ValueError, message):
                record_rules(dict(copy.deepcopy(RULES), **change), MODEL, self.installation)


class ParametricRecipeTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.recipe = self.root / "recipe.yaml"
        self.recipe.write_text(yaml.safe_dump(dict(
            recipe=1, output="./out", split="val", seed=5,
            parametric=dict(profile="reference_line_40", installation=INSTALLATION, records=2, rules=RULES))),
            encoding="utf-8")

    def tearDown(self):
        self.folder.cleanup()

    def test_a_run_saves_records_and_frames_that_verify(self):
        record = run(self.recipe, report=lambda line: None)
        out = self.root / "out"
        records = sorted((out / "records" / "val").glob("*.json"))
        self.assertEqual(len(records), 2)
        frames = sorted((out / "val" / "images").glob("*.png"))
        self.assertEqual(len(frames), sum(row["saved"]["frames"] for row in record["records"]))
        self.assertTrue((out / "runs").is_dir())
        results = verify(out)
        self.assertTrue(results and all(item.ok for item in results), [item.problem for item in results])
        self.assertEqual({item.kind for item in results}, {"record", "frame"})
        # The same recipe makes the same records.
        again = run(self.recipe, self.root / "again", report=lambda line: None)
        self.assertEqual([row["saved"]["digest"] for row in record["records"]],
                         [row["saved"]["digest"] for row in again["records"]])
        # A changed code and a changed frame are found.
        codes = records[0].with_suffix(".csv")
        lines = codes.read_text(encoding="utf-8").splitlines()
        cells = lines[3].split(",")
        cells[0] = str(int(cells[0]) + 1)
        lines[3] = ",".join(cells)
        codes.write_text("\n".join(lines) + "\n", encoding="utf-8")
        image = frames[0]
        with Image.open(image) as picture:
            pixels = np.asarray(picture).copy()
        pixels[0, 0] = (int(pixels[0, 0]) + 1) % 256
        Image.fromarray(pixels).save(image)
        failed = sorted((item.kind, Path(item.path).stem) for item in verify(out) if not item.ok)
        self.assertEqual(failed, sorted([("record", records[0].stem), ("frame", image.stem)]))

    def test_dry_run_writes_nothing(self):
        run(self.recipe, dry_run=True, report=lambda line: None)
        self.assertFalse((self.root / "out").exists())

    def test_command_line(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(["generate", str(self.recipe)]), 0)
            self.assertEqual(main(["verify", str(self.root / "out")]), 0)
        self.assertIn("examples render again exactly", output.getvalue())

    def test_a_recipe_without_real_data_takes_no_dataset_keys(self):
        spec = yaml.safe_load(self.recipe.read_text(encoding="utf-8"))
        spec["data"] = "../somewhere"
        self.recipe.write_text(yaml.safe_dump(spec), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unknown keys"):
            load_recipe(self.recipe)

    def test_the_documented_example(self):
        docs = Path(__file__).resolve().parents[1] / "docs"
        example = docs / "examples" / "parametric.yaml"
        recipe, _, _ = load_recipe(example)
        self.assertEqual(recipe["parametric"]["profile"], "reference_line_40")
        shown = (docs / "recipes.md").read_text(encoding="utf-8").split("```yaml\n")[2].split("```", 1)[0]
        self.assertEqual(example.read_text(encoding="utf-8"), shown)


class ObjectReportTests(unittest.TestCase):
    """object_report.py: the fit of model eggs to real spots."""

    def test_the_fit_recovers_known_values(self):
        from ir_line_synth.object_report import fit, model_spots
        known = Installation(55.0, 10.0, 200.0, 0.3)
        # "Real" spots made by the model itself with known values, from other eggs than the fit uses.
        real = model_spots(MODEL, known, 120.0, 2.0, 1.2, count=200, seed=99)
        fitted = fit(MODEL, real, 55.0, 200.0, 0.3)
        self.assertLessEqual(abs(fitted["pitch"] - 10.0), 0.5)
        self.assertLessEqual(abs(fitted["tilt"] - 2.0), 0.5)
        self.assertLess(abs(fitted["speed"] / 120.0 - 1), 0.1)
        self.assertLess(abs(fitted["reflectance"] / 1.2 - 1), 0.08)
        self.assertGreater(fitted["errors"][(fitted["pitch"], 0.0)], 5 * fitted["errors"][(fitted["pitch"], fitted["tilt"])])

    def test_real_spots_are_read_from_the_frames_of_recordings(self):
        from ir_line_synth.object_report import real_spots, spot_features
        rendered = render_record(record([egg(10), egg(50, y=-60.0)]), MODEL)
        root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(root, ignore_errors=True))
        (root / "train" / "labels").mkdir(parents=True)
        (root / "data.yaml").write_text("data:\n  train: ./train\n  val: ./train\n  sensor_axis: x\n", encoding="utf-8")
        rows = []
        for tile, items in enumerate(rendered.frame_boxes):
            boxes = np.asarray([box for _, box, _ in items]).reshape(-1, 4) / 40
            lines = [f"0 {(b[0] + b[2]) / 2} {(b[1] + b[3]) / 2} {b[2] - b[0]} {b[3] - b[1]}" for b in boxes]
            (root / "train" / "labels" / f"f{tile}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            rows.append(f"f{tile},train,DATA_7.csv,{tile}")
        (root / "manifest.csv").write_text("name,split,recording,tile\n" + "\n".join(rows) + "\n", encoding="utf-8")
        spots = real_spots(MODEL, root, [rendered.codes.astype(float)], [root / "DATA_7.csv"])
        samples = rendered.codes.reshape(-1, 10, 40).mean(axis=1) - MODEL.operation.belt_level
        expected = [spot_features(samples, box) for box in rendered.boxes]
        self.assertEqual(len(spots), 2)
        for got, want in zip(spots, expected):
            self.assertAlmostEqual(got["peak"], want["peak"], places=6)
            self.assertEqual((got["width"], got["length"]), (want["width"], want["length"]))


class LineScaleTests(unittest.TestCase):
    """Ranges and thresholds tuned on the reference line keep their duration and distance on another line."""

    def test_the_reference_line_is_exactly_the_identity(self):
        from ir_line_synth.motion import LineScale
        installation = MODEL.installation
        self.assertEqual(LineScale.of(MODEL.frames.pool, installation["poll_hz"], installation["pitch_mm"]),
                         LineScale(1.0, 1.0))
        # Rows of 1 poll at 100 Hz last 10 ms: five per reference row; channels 4.5 mm apart: two per reference channel.
        self.assertEqual(LineScale.of(1, 100.0, 4.5), LineScale(5.0, 2.0))

    def test_motions_keep_their_duration_and_distance(self):
        from ir_line_synth.motion import LineScale, output_times, sample
        fine = LineScale(3.0, 2.0)
        lengths, travels = {}, {}
        for name, scale, rows in (("reference", LineScale(), 8.0), ("fine", fine, 24.0)):
            rng = np.random.default_rng(4)
            moves = [sample("long", rows, (0.0, rows), rng, scale=scale) for _ in range(60)]
            # Output rows minus the fixed margin of the passage (one row and the closing sample): the duration.
            lengths[name] = np.mean([(len(output_times(0.0, rows, np.asarray(m["speed"]))) - 1) * rows / (rows + 1)
                                     for m in moves])
            rolls = [sample("rolling", rows, (0.0, rows), rng, scale=scale) for _ in range(60)]
            travels[name] = np.mean([np.ptp(m["lateral"]) for m in rolls])
        self.assertAlmostEqual(lengths["fine"] / lengths["reference"], 3.0, delta=0.3)
        self.assertAlmostEqual(travels["fine"] / travels["reference"], 2.0, delta=0.3)

    def test_kind_thresholds_follow_the_scale(self):
        from ir_line_synth.morph import classify
        from ir_line_synth.motion import LineScale
        features = dict(components=1, gaps=0, h=11, w=3, wiggle=0.1, excursion=1.0)
        self.assertEqual(classify(features), "long")
        self.assertEqual(classify(features, LineScale(2.0, 1.0)), "blob")
        self.assertEqual(classify(dict(features, h=22), LineScale(2.0, 1.0)), "long")
        self.assertEqual(classify(dict(features, h=8, excursion=3.0)), "rolling")
        self.assertEqual(classify(dict(features, h=8, excursion=3.0), LineScale(1.0, 2.0)), "blob")

    def test_a_dataset_names_the_profile_of_its_line(self):
        from ir_line_synth.dataset import load_data_description
        from ir_line_synth.motion import LineScale
        from ir_line_synth.workspace import line_scale
        from test_editor import write_dataset
        with tempfile.TemporaryDirectory() as directory:
            data = write_dataset(Path(directory) / "raw")
            self.assertEqual(line_scale(data), LineScale())
            description = Path(data["train"]).parent / "data.yaml"
            text = "data:\n  train: ./train\n  val: ./val\n"
            description.write_text(text.replace("data:\n", "data:\n  sensor_profile: reference_line_40\n", 1),
                                   encoding="utf-8")
            self.assertEqual(line_scale(load_data_description(description)), LineScale(1.0, 1.0))
            description.write_text(text.replace("data:\n", "data:\n  sensor_profile: example_line_32\n", 1),
                                   encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no installation section"):
                line_scale(load_data_description(description))


class ProfileInstallationTests(unittest.TestCase):
    def test_a_recipe_without_installation_takes_the_one_of_the_profile(self):
        from ir_line_synth.recipe import parametric_section
        section = parametric_section(dict(profile="reference_line_40", records=1, rules=copy.deepcopy(RULES)), ".")
        self.assertEqual(section["installation"], Installation(55.0, 9.0, 200.0, 0.3))
        self.assertIn("object-report", MODEL.installation["source"])
        with self.assertRaisesRegex(ValueError, "no installation section"):
            parametric_section(dict(profile="example_line_32", records=1, rules=copy.deepcopy(RULES)), ".")


class PageTests(unittest.TestCase):
    """The page of records without real data (web/views.py, static/records.js)."""

    def setUp(self):
        from ir_line_synth.web.views import EditorViews
        from test_editor import write_dataset
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.views = EditorViews(write_dataset(self.root / "raw"))
        spec = yaml.safe_load(self.views.get("/api/synthetic/parametric-default", {})["text"])
        spec["records"], spec["rules"]["rows"], spec["rules"]["count"] = 2, 80, [3, 5]
        self.text = yaml.safe_dump(spec, sort_keys=False)

    def tearDown(self):
        self.folder.cleanup()

    def test_the_page_starts_from_the_documented_example(self):
        default = self.views.get("/api/synthetic/parametric-default", {})
        example = Path(__file__).resolve().parents[1] / "docs" / "examples" / "parametric.yaml"
        self.assertEqual(yaml.safe_load(default["text"]), yaml.safe_load(example.read_text(encoding="utf-8"))["parametric"])
        self.assertIn("reference_line_40", default["profiles"])

    def test_preview_save_and_the_same_run_as_a_recipe(self):
        preview = self.views.post("/api/synthetic/parametric-preview", dict(text=self.text, seed=9, split="val"))
        self.assertEqual(preview["records"], 2)
        self.assertEqual(len(preview["frames"]), 2)
        self.assertEqual(len(preview["boxes"]), 2)
        self.assertTrue(preview["normalized"])
        self.assertEqual(sum(len(tile) for tile in preview["boxes"]) >= preview["objects"], True)
        saved = self.views.post("/api/synthetic/parametric-save",
                                dict(text=self.text, seed=9, split="val", directory=str(self.root / "page")))
        self.assertEqual((len(saved["saved"]), saved["skipped"]), (2, []))
        self.assertTrue(all(item.ok for item in verify(self.root / "page")))
        # The recipe the page shows makes the same records.
        recipe = self.root / "recipe.yaml"
        recipe.write_text(preview["recipe"], encoding="utf-8")
        run(recipe, self.root / "recipe_out", report=lambda line: None)
        page = sorted(json.loads(path.read_text(encoding="utf-8"))["digest"]
                      for path in (self.root / "page" / "records" / "val").glob("*.json"))
        recipe_made = sorted(json.loads(path.read_text(encoding="utf-8"))["digest"]
                             for path in (self.root / "recipe_out" / "records" / "val").glob("*.json"))
        self.assertEqual(page, recipe_made)

    def test_records_are_never_saved_into_the_source_dataset(self):
        with self.assertRaisesRegex(ValueError, "separate"):
            self.views.post("/api/synthetic/parametric-save",
                            dict(text=self.text, seed=9, split="train", directory=str(self.root / "raw" / "train")))
        with self.assertRaisesRegex(ValueError, "YAML"):
            self.views.post("/api/synthetic/parametric-preview", dict(text="rules: [", seed=9))

    def test_the_page_files_are_served(self):
        from urllib.request import urlopen
        import threading
        from ir_line_synth.web.server import make_server
        server = make_server(self.views, 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            for path, kind in (("/records.js", "text/javascript"), ("/records.css", "text/css")):
                with urlopen(f"http://127.0.0.1:{server.server_port}{path}") as response:
                    self.assertEqual(response.headers["Content-Type"], kind + "; charset=utf-8")
            with urlopen(f"http://127.0.0.1:{server.server_port}/") as response:
                page = response.read().decode("utf-8")
            for part in ('id="records-open"', 'id="records-dialog"', 'src="/records.js"', 'href="/records.css"'):
                self.assertIn(part, page)
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()


if __name__ == "__main__":
    unittest.main()
