"""Generate reviewable scenes from the editor's object bank; no file writes."""

from collections import Counter
import copy

import numpy as np

from .background import background_options
from .morph import morph_options
from .scene import (
    BLEND_MODES, check_placement, integer, number, prepare_entry, render_scene,
    transform_options,
)
from .store import OUTPUT_SPLITS, example_digest


BACKGROUND_MODES = ("fixed", "real", "generated")
# Largest object pool (bank entries) and largest batch of images of one request.
MAX_POOL = 10000
MAX_IMAGES = 1000
# Retry budgets that keep impossible requests finite: tries to place one object, failures of
# one pool entry before it is skipped for that object, whole-scene tries per image asked for,
# and prepared objects in a whole batch.
PLACEMENT_ATTEMPTS = 80
ENTRY_FAILURES = 10
SCENE_ATTEMPTS_PER_IMAGE = 20
BATCH_BUDGET = 20000


def generator_options(options):
    if not isinstance(options, dict):
        raise ValueError("Generator rules must be an object")
    defaults = dict(flip_x=True, flip_y=False, rotation_degrees=10.,
                    brightness_min=0.9, brightness_max=1.1, insertion="mask",
                    edges="hard", blur_radius=0.8, count_min=1, count_max=3,
                    images=12, seed=17, max_iou=0.15, blending="replace",
                    # Neutral defaults: nothing below changes a scene unless it is enabled.
                    gamma_min=1.0, gamma_max=1.0, contrast_min=1.0, contrast_max=1.0,
                    noise_max=0.0, gradient_max=0.0, shade_probability=0.0, shade_max=0.5,
                    erode_max=0.0,
                    # The split of the batch: every object of the pool and every background
                    # (real frame or the empty columns of a generated one) come from it, and
                    # so the batch can only be saved into it.
                    split="train",
                    # Background of each image: the chosen one, a random real frame of the
                    # split or a new generated one (see background.py). background_split is
                    # the old name of the split and, when given, must be the same.
                    background_mode="fixed", background_split=None,
                    background_block_min=2, background_block_max=8,
                    background_level_min=1.0, background_level_max=1.0,
                    # Generated objects: new random imperfections (same strengths) per insertion.
                    morph_reseed=True,
                    # Fewer repeats (see ObjectPicker): no pool entry twice in an image, no two
                    # objects of one real box (nor of the background frame) in an image, and
                    # the least used entries first over the whole batch.
                    unique_per_image=True, unique_source=True, balanced=True)
    if set(options) - defaults.keys():
        raise ValueError("Unknown generator rule")
    if "split" not in options and options.get("background_split") is not None:
        options = dict(options, split=options["background_split"])
    rules = {**defaults, **options}
    for key in ("flip_x", "flip_y", "morph_reseed", "unique_per_image", "unique_source", "balanced"):
        if not isinstance(rules[key], bool):
            raise ValueError(f"{key}: expected a boolean")
    if rules["background_mode"] not in BACKGROUND_MODES:
        raise ValueError("background_mode must be fixed, real or generated")
    if rules["split"] not in OUTPUT_SPLITS:
        raise ValueError("split must be train, val or test")
    if rules["background_split"] not in (None, rules["split"]):
        raise ValueError(f"The background comes from the split of the batch ({rules['split']}); "
                         f"background_split {rules['background_split']} differs")
    rules["background_split"] = rules["split"]
    for key in ("background_level_min", "background_level_max"):
        rules[key] = number(rules[key], key, 0, 3)
    if rules["background_level_min"] > rules["background_level_max"]:
        raise ValueError("background_level_min must not exceed background_level_max")
    background_options(dict(block_min=rules["background_block_min"], block_max=rules["background_block_max"]))
    for key, bounds in {"rotation_degrees": (0, 180), "brightness_min": (0, 4),
                        "brightness_max": (0, 4), "blur_radius": (0.1, 3),
                        "max_iou": (0, 1), "gamma_min": (0.3, 3), "gamma_max": (0.3, 3),
                        "contrast_min": (0, 3), "contrast_max": (0, 3), "noise_max": (0, 1),
                        "gradient_max": (0, 1), "shade_probability": (0, 1),
                        "shade_max": (0, 1), "erode_max": (0, 1)}.items():
        rules[key] = number(rules[key], key, *bounds)
    for low, high in (("gamma_min", "gamma_max"), ("contrast_min", "contrast_max")):
        if rules[low] > rules[high]:
            raise ValueError(f"{low} must not exceed {high}")
    for key, bounds in {"count_min": (1, 128), "count_max": (1, 128),
                        "images": (1, MAX_IMAGES), "seed": (0, 2**32 - 1)}.items():
        rules[key] = integer(rules[key], key, *bounds)
    if rules["count_min"] > rules["count_max"]:
        raise ValueError("Minimum object count must not exceed maximum")
    if rules["brightness_min"] > rules["brightness_max"]:
        raise ValueError("Minimum brightness must not exceed maximum")
    if rules["insertion"] not in ("box", "mask"):
        raise ValueError("Insertion must be box or mask")
    if rules["edges"] not in ("hard", "blur"):
        raise ValueError("Edges must be hard or blur")
    if not isinstance(rules["blending"], str) or rules["blending"] not in BLEND_MODES:
        raise ValueError("Invalid blending mode")
    return rules


def object_pool(entries, load_source, split=None, is_clipped=None, notes=None):
    """The checked pool entries.

    With `split`, every entry must come from that split. `is_clipped(source, box_index)`
    tells whether a real box is cut by the frame edge; such entries (and objects generated
    from them) are left out with a note in `notes`.
    """
    if not isinstance(entries, list) or not 1 <= len(entries) <= MAX_POOL:
        raise ValueError(f"Choose between 1 and {MAX_POOL} objects from the editor bank")
    result, clipped = [], 0
    for number_, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise ValueError("Each bank entry must be an object")
        source = load_source(entry.get("source"))
        if split is not None and source.split != split:
            raise ValueError(f"Pool object {number_} comes from {source.split}: a {split} batch takes "
                             f"objects only from {split}")
        index = integer(entry.get("box_index"), "box_index", 0, len(source.boxes) - 1)
        if is_clipped is not None and is_clipped(source, index):
            clipped += 1
            continue
        item = dict(source=copy.deepcopy(entry["source"]), box_index=index,
                    transform=transform_options(entry.get("transform", {})))
        if entry.get("morph") is not None:
            item["morph"] = morph_options(entry["morph"])
        result.append(item)
    if not result:
        raise ValueError("Every chosen object is cut by the frame edge (a stub); choose whole objects")
    if clipped and notes is not None:
        notes.append(f"{clipped} of {len(entries)} pool objects are cut by the frame edge (stubs) and were left out")
    return result


def choose_background(fixed_ref, rules, rng, real_ids):
    """The background reference of one generated image."""
    mode = rules["background_mode"]
    if mode == "fixed":
        return copy.deepcopy(fixed_ref)
    split = rules["split"]
    if mode == "real":
        ids = real_ids(split)
        if not ids:
            raise ValueError(f"No {split} frames to use as backgrounds")
        return dict(split=split, id=ids[int(rng.integers(len(ids)))])
    level = rules["background_level_min"]
    if rules["background_level_max"] > level:
        level = float(rng.uniform(level, rules["background_level_max"]))
    return dict(split=split, generated=dict(
        seed=int(rng.integers(2**32)), block_min=rules["background_block_min"],
        block_max=rules["background_block_max"], level=level, roll=True))


def random_transform(base, rules, rng):
    transform = dict(base)
    for key in ("flip_x", "flip_y"):
        if rules[key] and rng.random() < 0.5:
            transform[key] = not transform[key]
    angle = base["angle"] + rng.uniform(-rules["rotation_degrees"], rules["rotation_degrees"])
    transform["angle"] = float((angle + 180) % 360 - 180)
    gain = rng.uniform(rules["brightness_min"], rules["brightness_max"])
    transform["brightness"] = float(np.clip(base["brightness"] * gain, 0, 4))
    transform["use_mask"] = rules["insertion"] == "mask"
    if rules["gamma_min"] != rules["gamma_max"]:
        transform["gamma"] = float(rng.uniform(rules["gamma_min"], rules["gamma_max"]))
    elif rules["gamma_min"] != 1:
        transform["gamma"] = rules["gamma_min"]
    if rules["contrast_min"] != rules["contrast_max"]:
        transform["contrast"] = float(rng.uniform(rules["contrast_min"], rules["contrast_max"]))
    elif rules["contrast_min"] != 1:
        transform["contrast"] = rules["contrast_min"]
    if rules["noise_max"]:
        transform["noise"] = float(rng.uniform(0, rules["noise_max"]))
        transform["noise_seed"] = int(rng.integers(2**32))
    if rules["erode_max"]:
        transform["erode"] = float(rng.uniform(0, rules["erode_max"]))
        transform["erode_seed"] = int(rng.integers(2**32))
    if rules["gradient_max"]:
        transform["gradient"] = float(rng.uniform(-rules["gradient_max"], rules["gradient_max"]))
        transform["gradient_angle"] = float(rng.uniform(-180, 180))
    if rules["shade_probability"] and rng.random() < rules["shade_probability"]:
        transform["shade"] = float(rng.uniform(0, rules["shade_max"]))
        transform["shade_size"] = float(rng.uniform(0.1, 0.4))
        transform["shade_x"] = float(rng.random())
        transform["shade_y"] = float(rng.random())
    return transform


def random_object(entry, rules, rng):
    """A pool entry with a fresh random transform, not yet placed (no x, y).

    Generated (space-time) objects keep their mask, because a box cut would remove
    the soft edge, and may get new random imperfections.
    """
    transform = random_transform(entry["transform"], rules, rng)
    morph = copy.deepcopy(entry.get("morph"))
    if morph is not None:
        transform["use_mask"] = False
        if rules["morph_reseed"]:
            morph["imperfections"]["seed"] = int(rng.integers(2**32))
    obj = dict(source=copy.deepcopy(entry["source"]), box_index=entry["box_index"], transform=transform)
    if morph is not None:
        obj["morph"] = morph
    if rules["edges"] == "blur":
        obj["edge_blur"] = rules["blur_radius"]
    return obj


def source_key(entry):
    """The real annotated box an entry comes from (generated objects included)."""
    return (entry["source"].get("split"), entry["source"].get("id"), entry["box_index"])


class ObjectPicker:
    """Chooses the pool entry of every insertion under the repeat rules.

    unique_per_image: an entry is used at most once in an image.
    unique_source: at most one object per real box in an image, and none from the
        background frame itself (it already shows the original egg).
    balanced: among the allowed entries, one of the least used so far in the batch.
    Entries that failed to fit ENTRY_FAILURES times are skipped for the current insertion.
    """

    def __init__(self, pool, rules):
        self.rules = rules
        self.keys = [source_key(entry) for entry in pool]
        self.usage = [0] * len(pool)

    def capacity(self):
        """Most objects an image can hold under the rules (ignoring the background frame)."""
        if self.rules["unique_source"]:
            return len(set(self.keys))
        return len(self.keys) if self.rules["unique_per_image"] else None

    def start(self, background_ref):
        frame = None
        if isinstance(background_ref, dict) and "id" in background_ref:
            frame = (background_ref.get("split"), background_ref.get("id"))
        return dict(entries=[], keys=set(), frame=frame)

    def allowed(self, state):
        result = []
        for index, key in enumerate(self.keys):
            if self.rules["unique_per_image"] and index in state["entries"]:
                continue
            if self.rules["unique_source"] and (key in state["keys"] or key[:2] == state["frame"]):
                continue
            result.append(index)
        return result

    def pick(self, state, failed, rng):
        """An entry index, or None when every allowed entry already failed to fit."""
        allowed = self.allowed(state)
        if not allowed:
            if state["frame"] and not state["entries"] and all(key[:2] == state["frame"] for key in self.keys):
                raise ValueError("Every pool object comes from the background frame; "
                                 "choose another background or allow objects of the same source")
            raise ValueError(f"No unused object left for object {len(state['entries']) + 1} of this image; "
                             "add objects to the pool, lower the count or allow repeats")
        allowed = [index for index in allowed if failed[index] < ENTRY_FAILURES]
        if not allowed:
            return None
        if self.rules["balanced"]:
            local = Counter(state["entries"])
            used = [self.usage[index] + local[index] for index in allowed]
            lowest = min(used)
            allowed = [index for index, count in zip(allowed, used) if count == lowest]
        return allowed[int(rng.integers(len(allowed)))]

    def take(self, state, index):
        state["entries"].append(index)
        state["keys"].add(self.keys[index])

    def accept(self, state):
        for index in state["entries"]:
            self.usage[index] += 1


def random_scene(background_ref, background, pool, rules, rng, load_source, budget, prepare=prepare_entry,
                 picker=None):
    picker = picker or ObjectPicker(pool, rules)
    state = picker.start(background_ref)
    scene = dict(background=copy.deepcopy(background_ref), objects=[],
                 blending=rules["blending"], max_iou=rules["max_iou"])
    count = int(rng.integers(rules["count_min"], rules["count_max"] + 1))
    existing = list(background.boxes)
    rows, columns = background.pixels.shape
    for _ in range(count):
        last_error = "No valid placement"
        failed = Counter()
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            if budget[0] <= 0:
                raise ValueError("Placement budget exhausted; reduce count or transform limits")
            index = picker.pick(state, failed, rng)
            if index is None:
                break
            budget[0] -= 1
            entry = pool[index]
            candidate = random_object(entry, rules, rng)
            try:
                prepared = prepare(candidate, load_source)
                if prepared.width > columns or prepared.height > rows:
                    raise ValueError(f"Transformed object does not fit inside {columns} x {rows}")
                x = int(rng.integers(columns - prepared.width + 1))
                y = int(rng.integers(rows - prepared.height + 1))
                box = check_placement(prepared, x, y, existing, rules["max_iou"], background.pixels.shape)
            except ValueError as error:
                last_error = str(error)
                failed[index] += 1
                continue
            picker.take(state, index)
            obj = dict(candidate, x=x, y=y)
            scene["objects"].append(obj)
            existing.append(box)
            placed = True
            break
        if not placed:
            raise ValueError(last_error)
    return scene, render_scene(scene, load_source, prepare), state


def generate_candidates(background_ref, entries, options, load_source, prepare=prepare_entry,
                        real_ids=lambda split: [], is_clipped=None):
    """Candidate scenes for review; nothing is written.

    `prepare(obj, load_source)` builds an added object (real or generated),
    `real_ids(split)` lists the frames a "real" background may be drawn from and
    `is_clipped(source, box_index)` tells the boxes cut by the frame edge, which are
    left out of the pool. Every object and background comes from `rules["split"]`.
    """
    rules = generator_options(options)
    if rules["background_mode"] == "fixed":
        background = load_source(background_ref)
        if background.split != rules["split"]:
            raise ValueError(f"The chosen background comes from {background.split}: a {rules['split']} "
                             f"batch needs a background of {rules['split']}")
    notes = []
    pool = object_pool(entries, load_source, rules["split"], is_clipped, notes)
    picker = ObjectPicker(pool, rules)
    capacity = picker.capacity()
    if capacity is not None and rules["count_min"] > capacity:
        what = "different source boxes" if rules["unique_source"] else "objects"
        raise ValueError(f"{rules['count_min']} objects per image without repeats need at least "
                         f"{rules['count_min']} {what}; the pool has {capacity}")
    if capacity is not None and rules["count_max"] > capacity:
        notes.append(f"count_max lowered from {rules['count_max']} to {capacity}: "
                     f"only {capacity} objects can appear in one image without repeats")
        rules = dict(rules, count_max=capacity)
    rng = np.random.default_rng(rules["seed"])
    candidates, seen = [], set()
    rejected = Counter()
    budget = [BATCH_BUDGET]
    attempts = 0
    # Bounded whole-scene and placement retries keep impossible requests finite.
    while len(candidates) < rules["images"] and attempts < rules["images"] * SCENE_ATTEMPTS_PER_IMAGE and budget[0] > 0:
        attempts += 1
        try:
            scene_ref = choose_background(background_ref, rules, rng, real_ids)
            scene, rendered, state = random_scene(scene_ref, load_source(scene_ref), pool, rules, rng,
                                                  load_source, budget, prepare, picker)
        except ValueError as error:
            rejected[str(error)] += 1
            continue
        digest = example_digest(rendered.pixels, rendered.boxes, rendered.labels)
        if digest in seen:
            rejected["Duplicate candidate"] += 1
            continue
        seen.add(digest)
        picker.accept(state)
        preview = rendered.payload()
        preview["digest"] = digest
        candidates.append(dict(index=len(candidates), scene=scene, preview=preview))
    return dict(requested=rules["images"], candidates=candidates, attempts=attempts,
                rejected=dict(rejected), budget_exhausted=budget[0] <= 0, rules=rules,
                usage=picker.usage, notes=notes)
