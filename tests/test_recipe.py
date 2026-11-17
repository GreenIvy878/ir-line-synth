"""Recipes run without the page, the same recipe makes the same data, saved examples are
verified against their descriptions, and the registry of constants matches the code."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import yaml

from ir_line_synth import __version__, constants
from ir_line_synth.__main__ import main
from ir_line_synth.dataset import load_data_description
from ir_line_synth.recipe import load_recipe, run, step_seed
from ir_line_synth.verify import verify
from test_splits import write_split_dataset


RECIPE = dict(
    recipe=1, data="raw", output="out", split="train", seed=17,
    objects=dict(real=dict(kinds=["blob", "long", "short"]),
                 generated=[dict(kind="blob", count=4, accept="all")]),
    scenes=dict(images=4, rules=dict(count_min=2, count_max=3, background_mode="generated")),
    tapes=dict(count=1, rules=dict(length=200, objects=6)),
)


def quiet(line):
    pass


class RecipeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        write_split_dataset(cls.root / "raw")

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def recipe(self, name, **changes):
        path = self.root / f"{name}.yaml"
        path.write_text(yaml.safe_dump({**RECIPE, **changes}), encoding="utf-8")
        return path

    def digests(self, output):
        files = list(Path(output).glob("*/provenance/*.json")) + list(Path(output).glob("tapes/*/*.json"))
        return sorted(json.loads(path.read_text(encoding="utf-8"))["digest"] for path in files)

    def test_the_same_recipe_makes_the_same_data_and_records_the_run(self):
        path = self.recipe("same")
        first = run(path, self.root / "first", report=quiet)
        second = run(path, self.root / "second", report=quiet)
        self.assertEqual(len(first["scenes"]["saved"]), 4)
        self.assertEqual(len(first["tapes"]), 1)
        self.assertEqual(self.digests(self.root / "first"), self.digests(self.root / "second"))
        self.assertEqual(len(self.digests(self.root / "first")), 5)
        # Another seed makes other data.
        run(self.recipe("other", seed=18), self.root / "third", report=quiet)
        self.assertNotEqual(self.digests(self.root / "first"), self.digests(self.root / "third"))
        # The record: program, recipe and its digest, the line, and a copy of the recipe file.
        record = json.loads(Path(first["record"]).read_text(encoding="utf-8"))
        self.assertEqual(record["program"]["version"], __version__)
        self.assertEqual(record["recipe_sha256"], load_recipe(path)[2])
        self.assertEqual(record["geometry"]["channels"], 40)
        self.assertTrue(Path(first["record"].replace(".json", ".recipe.yaml")).is_file())
        # Every saved file names the program that wrote it.
        provenance = next((self.root / "first" / "train" / "provenance").glob("*.json"))
        self.assertEqual(json.loads(provenance.read_text(encoding="utf-8"))["program"]["version"], __version__)
        # A second run into the same output finds only duplicates and says so.
        again = run(path, self.root / "first", report=quiet)
        self.assertEqual(again["scenes"]["saved"], [])
        self.assertEqual(len(again["scenes"]["skipped"]), 4)

    def test_a_dry_run_writes_nothing(self):
        record = run(self.recipe("dry"), self.root / "dry", dry_run=True, report=quiet)
        self.assertEqual(record["scenes"]["made"], 4)
        self.assertFalse((self.root / "dry").exists())

    def test_steps_have_their_own_seeds(self):
        seeds = {step_seed(17, step, index) for step in ("real", "generated", "scenes", "tapes") for index in range(3)}
        self.assertEqual(len(seeds), 12)
        self.assertEqual(step_seed(17, "scenes"), step_seed(17, "scenes"))

    def test_recipes_are_checked(self):
        cases = {
            "unknown key": dict(colour="red"),
            "schema": dict(recipe=2),
            "split": dict(split="dev"),
            "nothing to make": dict(scenes=None, tapes=None),
            "seed in rules": dict(scenes=dict(images=2, rules=dict(seed=3))),
            "split in rules": dict(tapes=dict(count=1, rules=dict(split="val"))),
            "kind": dict(objects=dict(real=dict(kinds=["cube"]))),
            "accept": dict(objects=dict(generated=[dict(kind="blob", count=2, accept="some")])),
            "limits": dict(objects=dict(generated=[dict(kind="blob", count=2, limits=dict(glow=1))])),
            "no objects": dict(objects={}),
        }
        for name, change in cases.items():
            recipe = {**RECIPE, **change}
            recipe = {key: value for key, value in recipe.items() if value is not None}
            path = self.root / "bad.yaml"
            path.write_text(yaml.safe_dump(recipe), encoding="utf-8")
            with self.subTest(name), self.assertRaises(ValueError):
                load_recipe(path)
        # The output may not lie inside the source dataset.
        with self.assertRaisesRegex(ValueError, "separate"):
            run(self.recipe("inside", output="raw/train/out"), report=quiet)

    def test_verify_finds_changed_files(self):
        run(self.recipe("verified"), self.root / "verified", report=quiet)
        data = load_data_description(self.root / "raw")
        results = verify(self.root / "verified", data)
        self.assertEqual(len(results), 5)
        self.assertTrue(all(item.ok for item in results), results)
        # A changed pixel of a scene ...
        image_path = next((self.root / "verified" / "train" / "images").glob("*.png"))
        with Image.open(image_path) as image:
            pixels = np.asarray(image).copy()
        pixels[0, 0] = (int(pixels[0, 0]) + 1) % 256
        Image.fromarray(pixels).save(image_path)
        # ... and a changed value of a tape are found, each in its own example.
        signal = next((self.root / "verified" / "tapes" / "train").glob("*[!s].csv"))
        lines = signal.read_text(encoding="utf-8").splitlines()
        values = lines[5].split(",")
        values[3] = str((int(values[3]) + 1) % 256)
        lines[5] = ",".join(values)
        signal.write_text("\n".join(lines) + "\n", encoding="utf-8")
        failed = sorted((item.kind, Path(item.path).stem) for item in verify(self.root / "verified", data)
                        if not item.ok)
        self.assertEqual(failed, sorted([("scene", image_path.stem), ("tape", signal.stem)]))

    def test_the_documented_example_is_a_valid_recipe(self):
        docs = Path(__file__).resolve().parents[1] / "docs"
        example = docs / "examples" / "recipe.yaml"
        recipe, _, _ = load_recipe(example)
        self.assertEqual(recipe["split"], "train")
        # The example file is the one shown in docs/recipes.md.
        shown = (docs / "recipes.md").read_text(encoding="utf-8").split("```yaml\n", 1)[1].split("```", 1)[0]
        self.assertEqual(example.read_text(encoding="utf-8"), shown)

    def test_command_line(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(["generate", str(self.recipe("cli")), "--output", str(self.root / "cli")]), 0)
            self.assertEqual(main(["verify", str(self.root / "cli"), "--data", str(self.root / "raw")]), 0)
        self.assertIn("5 of 5 examples render again exactly", output.getvalue())
        errors = io.StringIO()
        with contextlib.redirect_stderr(errors):
            self.assertEqual(main(["generate", str(self.root / "missing.yaml")]), 2)
        self.assertIn("missing.yaml", errors.getvalue())


class ConstantsTests(unittest.TestCase):
    def test_every_constant_is_registered_and_described(self):
        self.assertEqual(constants.unregistered(), ([], []))
        for item in constants.REGISTRY:
            self.assertIn(item.origin, constants.ORIGINS)
            self.assertTrue(item.unit and item.meaning, item.name)
            self.assertTrue(constants.is_numeric(item.value), item.name)

    def test_the_document_matches_the_code(self):
        document = Path(__file__).resolve().parents[1] / "docs" / "constants.md"
        self.assertEqual(document.read_text(encoding="utf-8"), constants.markdown(),
                         "Run: python -m ir_line_synth constants --output docs/constants.md")

    def test_numbers_are_told_from_other_constants(self):
        self.assertTrue(constants.is_numeric((1, 2.5)))
        self.assertTrue(constants.is_numeric(dict(a=(0, 1))))
        for value in (True, "x", (), {}, ("a", 1), dict(a=True)):
            self.assertFalse(constants.is_numeric(value), value)


if __name__ == "__main__":
    unittest.main()
