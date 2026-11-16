"""Recipes: a whole generation run described in one YAML file and run without the page.

A recipe names the source dataset, the split every source and output belongs to, one
seed, the pool of objects (whole real objects by kind, and new objects replayed from
them) and what to make from it: a batch of scenes by the rules of the autogenerator,
tapes by the rules of the tape generator, or both. The rules are the same ones the page
sends, so a run of the page can be written down as a recipe.

    recipe: 1
    data: ../datasets/main_flow      # relative to this file
    output: ./out                    # relative to this file
    split: train
    seed: 17
    objects:
      real: {kinds: [blob, long, short], limit: 200}
      generated:
        - {kind: rolling, count: 30, limits: {noise_max: 0.1}, accept: filter}
    scenes:
      images: 100
      rules: {count_min: 3, count_max: 8, max_iou: 0.15, background_mode: generated}
    tapes:
      count: 2
      rules: {length: 2000, objects: 100, layout: clusters}

Without the review of the page, every scene the rules allow is saved: the rules, the
filter of new objects (accept: filter) and the duplicate check are the selection. The
seeds of the steps are derived from the one seed of the recipe (step_seed), so the same
recipe on the same data makes the same files. A run writes a record of itself into
`runs/` of the output: the recipe, its digest, the program version, the dataset, what
was made and what was refused.
"""

import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from . import PROGRAM, __version__, morph
from .dataset import load_data_description
from .generator import generate_candidates
from .scene import integer
from .store import OUTPUT_SPLITS
from .tape import generate_tape
from .workspace import Workspace


SCHEMA = 1
RECIPE_KEYS = {"recipe", "data", "output", "split", "seed", "objects", "scenes", "tapes", "background"}
# Rules a recipe sets once at its top level, so a step may not set them again.
TOP_LEVEL_RULES = {"seed", "split", "background_split", "images"}
ACCEPT = ("filter", "all")
# Steps of a run, each with its own random stream (step_seed).
STEPS = ("real", "generated", "scenes", "tapes")


def step_seed(seed, step, index=0):
    """The seed of one step of a run, derived from the seed of the recipe."""
    return int(np.random.SeedSequence([seed, STEPS.index(step), index]).generate_state(1)[0])


def keys(value, allowed, where):
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be a mapping")
    unknown = set(value) - set(allowed)
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}; allowed: {sorted(allowed)}")
    return value


def rules_of(step, where):
    """The rules of a step; those the recipe sets once (TOP_LEVEL_RULES) are refused here."""
    rules = step.get("rules", {})
    if not isinstance(rules, dict):
        raise ValueError(f"{where}.rules must be a mapping")
    repeated = set(rules) & TOP_LEVEL_RULES
    if repeated:
        raise ValueError(f"{where}.rules: {sorted(repeated)} are set once for the whole recipe "
                         "(split, seed; scenes.images)")
    return copy.deepcopy(rules)


def load_recipe(path):
    """A checked recipe: (the recipe as written, its folder, its SHA-256)."""
    path = Path(path).resolve()
    text = path.read_bytes()
    try:
        recipe = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise ValueError(f"{path}: {error}") from error
    keys(recipe, RECIPE_KEYS, "recipe")
    if recipe.get("recipe") != SCHEMA:
        raise ValueError(f"{path}: expected recipe: {SCHEMA}")
    for key in ("data", "output"):
        if not isinstance(recipe.get(key), str) or not recipe[key].strip():
            raise ValueError(f"recipe.{key} must be a path")
    if recipe.get("split") not in OUTPUT_SPLITS:
        raise ValueError("recipe.split must be train, val or test")
    integer(recipe.get("seed"), "recipe.seed", 0, 2**32 - 1)
    objects = keys(recipe.get("objects"), {"real", "generated"}, "recipe.objects")
    if not objects:
        raise ValueError("recipe.objects needs real or generated objects")
    if "real" in objects:
        real = keys(objects["real"], {"kinds", "limit"}, "objects.real")
        kinds = real.get("kinds", list(morph.KINDS))
        if not isinstance(kinds, list) or not kinds or set(kinds) - set(morph.KINDS):
            raise ValueError(f"objects.real.kinds: a list of {', '.join(morph.KINDS)}")
        if "limit" in real:
            integer(real["limit"], "objects.real.limit", 1, 10000)
    generated = objects.get("generated", [])
    if not isinstance(generated, list):
        raise ValueError("objects.generated must be a list")
    for number, item in enumerate(generated, start=1):
        where = f"objects.generated[{number}]"
        keys(item, {"kind", "count", "limits", "ratio", "accept"}, where)
        if item.get("kind") not in morph.KINDS:
            raise ValueError(f"{where}.kind: one of {', '.join(morph.KINDS)}")
        integer(item.get("count"), f"{where}.count", 1, 1000)
        morph.imperfection_limits(item.get("limits", {}))
        if item.get("accept", "filter") not in ACCEPT:
            raise ValueError(f"{where}.accept: filter or all")
        ratio = item.get("ratio", list(morph.BLOB_RATIO))
        if not isinstance(ratio, list) or len(ratio) != 2:
            raise ValueError(f"{where}.ratio: [low, high]")
        morph.blob_ratio_range(*ratio)
    if "scenes" not in recipe and "tapes" not in recipe:
        raise ValueError("A recipe makes scenes, tapes or both")
    if "scenes" in recipe:
        scenes = keys(recipe["scenes"], {"images", "rules"}, "recipe.scenes")
        integer(scenes.get("images"), "scenes.images", 1, 1000)
        rules_of(scenes, "scenes")
    if "tapes" in recipe:
        tapes = keys(recipe["tapes"], {"count", "rules", "values", "header"}, "recipe.tapes")
        integer(tapes.get("count"), "tapes.count", 1, 1000)
        rules_of(tapes, "tapes")
    if "background" in recipe:
        keys(recipe["background"], {"id"}, "recipe.background")
    return recipe, path.parent, hashlib.sha256(text).hexdigest()


def object_pool(workspace, recipe, notes):
    """The pool of a run as entries of the page's object bank."""
    split, seed, objects = recipe["split"], recipe["seed"], recipe["objects"]
    library = workspace.library(split)
    pool = []
    if "real" in objects:
        kinds = objects["real"].get("kinds", list(morph.KINDS))
        whole = [obj for obj in library.objects if not obj.clipped and obj.kind in kinds]
        limit = objects["real"].get("limit")
        if limit is not None and len(whole) > limit:
            order = np.random.default_rng(step_seed(seed, "real")).permutation(len(whole))[:limit]
            whole = [whole[index] for index in sorted(order)]
        pool += [dict(source=dict(split=split, id=obj.source), box_index=obj.box_index, transform={})
                 for obj in whole]
        notes.append(f"real objects: {len(whole)} whole objects of {split} of the kinds {', '.join(kinds)}")
    for index, item in enumerate(objects.get("generated", [])):
        batch = library.generate(item["kind"], item["count"], step_seed(seed, "generated", index),
                                 item.get("limits", {}), None, tuple(item.get("ratio", morph.BLOB_RATIO)))
        chosen = [row for row in batch["candidates"]
                  if item.get("accept", "filter") == "all" or row["filter"]["accepted"]]
        pool += [dict(source=row["source"], box_index=row["box_index"], transform={}, morph=row["morph"])
                 for row in chosen]
        notes.append(f"new objects {item['kind']}: {len(chosen)} of {len(batch['candidates'])} made "
                     f"({'accepted by the filter' if item.get('accept', 'filter') == 'filter' else 'all'})")
        notes += batch["notes"]
    if not pool:
        raise ValueError("The recipe gives no objects: no real object of these kinds, no new object accepted")
    return pool


def run(path, output=None, dry_run=False, report=print):
    """Run a recipe; returns the record of the run (also written to `runs/` unless dry_run).

    `output` replaces the output folder of the recipe. With `dry_run`, everything is made
    and checked but nothing is written.
    """
    recipe, folder, digest = load_recipe(path)
    data_path = (folder / recipe["data"]).resolve()
    output = Path(output).resolve() if output else (folder / recipe["output"]).resolve()
    workspace = Workspace(load_data_description(data_path))
    if dry_run:
        workspace.store.output_root(str(output))
    else:
        workspace.store.initialize(str(output))
    split, seed = recipe["split"], recipe["seed"]
    notes = []
    pool = object_pool(workspace, recipe, notes)
    report(f"Pool: {len(pool)} objects")
    record = dict(program=dict(name=PROGRAM, version=__version__), recipe=recipe, recipe_sha256=digest,
                  recipe_file=str(Path(path).resolve()), data=str(data_path), geometry=workspace.geometry.payload(),
                  output=str(output), split=split, pool=len(pool), notes=notes, scenes=None, tapes=None,
                  started=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    load = workspace.source_loader()

    if "scenes" in recipe:
        rules = dict(rules_of(recipe["scenes"], "scenes"), split=split, seed=step_seed(seed, "scenes"),
                     images=recipe["scenes"]["images"])
        background = None
        if rules.get("background_mode", "fixed") == "fixed":
            if "background" not in recipe:
                raise ValueError("background_mode fixed needs recipe.background: {id: <frame>}")
            background = dict(split=split, id=recipe["background"]["id"])
        names = lambda part: [image.name for image in workspace.dataset(part).paths]
        batch = generate_candidates(background, pool, rules, load, workspace.prepare, names, workspace.is_clipped)
        saved, skipped = [], []
        for candidate in batch["candidates"]:
            rendered, preview = workspace.preview(candidate["scene"], load)
            if dry_run:
                continue
            try:
                saved.append(workspace.store.save(str(output), split, rendered, candidate["scene"]))
            except ValueError as error:
                skipped.append(dict(candidate=candidate["index"], reason=str(error)))
        record["scenes"] = dict(requested=batch["requested"], made=len(batch["candidates"]), saved=saved,
                                skipped=skipped, rejected=batch["rejected"], notes=batch["notes"],
                                rules=batch["rules"])
        report(f"Scenes: {len(batch['candidates'])} of {batch['requested']} made, {len(saved)} saved, "
               f"{len(skipped)} skipped")

    if "tapes" in recipe:
        tapes = recipe["tapes"]
        made = []
        for index in range(tapes["count"]):
            rules = dict(rules_of(tapes, "tapes"), split=split, seed=step_seed(seed, "tapes", index))
            result = generate_tape(pool, rules, load, workspace.prepare_tape_object, workspace.geometry.channels,
                                   workspace.is_clipped)
            rendered = workspace.render_tape(result["tape"], load)
            row = dict(requested=result["requested"], placed=result["placed"], rejected=result["rejected"],
                       notes=result["notes"])
            if not dry_run:
                try:
                    row["saved"] = workspace.store.save_tape(str(output), split, rendered, result["tape"], None,
                                                             tapes.get("values", "levels"),
                                                             tapes.get("header", False) is True)
                except ValueError as error:
                    row["skipped"] = str(error)
            made.append(row)
            report(f"Tape {index + 1}: {result['placed']} of {result['requested']} objects placed")
        record["tapes"] = made

    record["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if not dry_run:
        runs = output / "runs"
        runs.mkdir(parents=True, exist_ok=True)
        stamp = record["started"].replace(":", "").replace("-", "").replace("+0000", "Z")
        name = f"run_{stamp}_{digest[:8]}"
        (runs / f"{name}.json").write_text(json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False),
                                           encoding="utf-8")
        (runs / f"{name}.recipe.yaml").write_bytes(Path(path).read_bytes())
        record["record"] = str(runs / f"{name}.json")
    return record
