"""The JSON API of the web interface: requests mapped onto a Workspace."""

import numpy as np

from .. import morph
from ..generator import generate_candidates
from ..scene import integer, make_variation, variation_options
from ..store import check_split, output_split
from ..tape import generate_tape
from ..workspace import Workspace


def prepared_payload(prepared):
    return dict(matrix=prepared.pixels.tolist(), mask=prepared.mask.tolist(),
                width=prepared.width, height=prepared.height, label=prepared.label,
                box=prepared.label_box().tolist())


class EditorViews:
    def __init__(self, data):
        self.workspace = Workspace(data)

    def get(self, path, query):
        if path == "/api/synthetic/sensor":
            return self.workspace.geometry.payload()
        if path == "/api/synthetic/catalog":
            catalog = {}
            for split in self.workspace.source_splits():
                dataset = self.workspace.dataset(split)
                catalog[split] = [dict(id=image.name, objects=len(boxes))
                                  for image, boxes in zip(dataset.paths, dataset.boxes)]
            return catalog
        if path == "/api/synthetic/sample":
            source = self.workspace.source(dict(split=query.get("split", [""])[0], id=query.get("id", [""])[0]))
            # Boxes cut by the frame edge (stubs) are shown marked; the generators never take them.
            clipped = [self.workspace.is_clipped(source, index) for index in range(len(source.boxes))]
            return dict(matrix=source.pixels.tolist(), boxes_pixels_xyxy=source.boxes.tolist(),
                        labels=list(source.labels), split=source.split, id=source.identifier, clipped=clipped)
        raise ValueError("Unknown synthetic editor endpoint")

    def post(self, path, body):
        if path == "/api/synthetic/generate":
            return generate_candidates(body.get("background"), body.get("pool"),
                                       body.get("rules", {}), self.workspace.source_loader(), self.workspace.prepare,
                                       self.workspace.frame_ids, self.workspace.is_clipped)
        if path.startswith("/api/synthetic/tape-"):
            return self.post_tape(path, body)
        if path == "/api/synthetic/morph-generate":
            return self.generate_objects(body)
        if path == "/api/synthetic/generated-save":
            return self.save_generated(body)
        if path == "/api/synthetic/prepare":
            return prepared_payload(self.workspace.prepare(body, self.workspace.source_loader()))
        if path not in ("/api/synthetic/preview", "/api/synthetic/save", "/api/synthetic/variations"):
            raise ValueError("Unknown synthetic editor endpoint")
        spec = body.get("scene")
        load_source = self.workspace.source_loader()
        rendered, preview = self.workspace.preview(spec, load_source)
        if path == "/api/synthetic/preview":
            return preview
        if body.get("expected_digest") != preview["digest"]:
            raise ValueError("The scene changed; refresh its preview before saving")
        directory, split = body.get("directory"), body.get("split")
        check_split(split, rendered.provenance)
        if path == "/api/synthetic/save":
            return self.workspace.store.save(directory, split, rendered, spec)
        count = integer(body.get("count", 1), "Variation count", 1, 100)
        seed = integer(body.get("seed", 17), "Variation seed", 0, 2**32 - 1)
        options = variation_options(body.get("variation", {}))
        if not spec.get("objects"):
            raise ValueError("Add an object before generating variations")
        # Validate the destination before any expensive generation.
        self.workspace.store.output_root(directory)
        rng = np.random.default_rng(seed)
        saved, skipped = [], []
        for index in range(count):
            try:
                variant, result = make_variation(spec, load_source, rng, options, prepare=self.workspace.prepare)
                saved.append(self.workspace.store.save(directory, split, result, variant))
            except ValueError as error:
                skipped.append(dict(variation=index + 1, reason=str(error)))
        return dict(requested=count, saved=saved, skipped=skipped)

    def post_tape(self, path, body):
        load_source = self.workspace.source_loader()
        if path == "/api/synthetic/tape-prepare":
            return prepared_payload(self.workspace.prepare_tape_object(body, load_source))
        if path == "/api/synthetic/tape-generate":
            result = generate_tape(body.get("pool"), body.get("rules", {}), load_source, self.workspace.prepare_tape_object,
                                   self.workspace.geometry.channels, self.workspace.is_clipped)
            result["preview"] = self.workspace.render_tape(result["tape"], load_source).payload()
            return result
        if path not in ("/api/synthetic/tape-preview", "/api/synthetic/tape-save"):
            raise ValueError("Unknown synthetic editor endpoint")
        spec = body.get("tape")
        rendered = self.workspace.render_tape(spec, load_source)
        preview = rendered.payload()
        if path == "/api/synthetic/tape-preview":
            return preview
        if body.get("expected_digest") != preview["digest"]:
            raise ValueError("The tape changed; refresh its preview before saving")
        return self.workspace.store.save_tape(body.get("directory"), body.get("split"), rendered, spec,
                                              body.get("name") or None, body.get("values", "levels"),
                                              body.get("header", False) is True)

    def generate_objects(self, body):
        """New objects for review: replays of real objects with random imperfections.

        Sources are random whole real objects of `split` suited to the kind, or the given
        list (the editor's bank), which must come from `split` too: the statistics of the
        filter are those of the split. Objects cut by the frame edge are never sources.
        Each candidate carries the filter's opinion only; the smallest of the batch are
        replaced (morph.sift_small), and the response says how.
        """
        split = body.get("split", "train")
        if split not in self.workspace.source_splits():
            raise ValueError("Select a source split: " + ", ".join(self.workspace.source_splits()))
        library = self.workspace.library(split)
        count = integer(body.get("count", 10), "count", 1, 1000)
        seed = integer(body.get("seed", 1), "seed", 0, 2**32 - 1)
        sources = body.get("sources")
        chosen = None
        if sources is not None:
            if not isinstance(sources, list) or not 1 <= len(sources) <= 1000:
                raise ValueError("Choose between 1 and 1000 source objects")
            chosen = []
            for number, entry in enumerate(sources, start=1):
                reference = entry.get("source") if isinstance(entry, dict) else None
                if not isinstance(reference, dict) or reference.get("split") not in self.workspace.source_splits():
                    raise ValueError(f"Source object {number} has no image of a source split")
                if reference["split"] != split:
                    raise ValueError(f"Source object {number} comes from {reference['split']}: objects for "
                                     f"{split} are made only from {split} objects")
                obj = library.find(reference.get("id"), entry.get("box_index"))
                if obj is None:
                    raise ValueError(f"Source object {number} is too faint to generate from")
                chosen.append(obj)
        ratio = (body.get("ratio_min", morph.BLOB_RATIO[0]), body.get("ratio_max", morph.BLOB_RATIO[1]))
        batch = library.generate(body.get("kind"), count, seed, body.get("limits", {}), chosen, ratio)
        if not batch["candidates"]:
            raise ValueError("No object met the shape rules (blob length to width, unbroken traces) "
                             "from these sources; try other sources or another kind")
        return dict(candidates=batch["candidates"], requested=count, library=library.counts(), split=split,
                    notes=batch["notes"], size_filter=batch["size_filter"])

    def save_generated(self, body):
        items = body.get("items")
        if not isinstance(items, list) or not 1 <= len(items) <= 100:
            raise ValueError("Select between 1 and 100 generated images")
        directory, split = body.get("directory"), body.get("split")
        self.workspace.store.output_root(directory)
        output_split(split)
        load_source = self.workspace.source_loader()
        checked = []
        # Check every selected preview before writing any member of the batch.
        for item in items:
            if not isinstance(item, dict):
                raise ValueError("Invalid generated image selection")
            rendered, preview = self.workspace.preview(item.get("scene"), load_source)
            if item.get("expected_digest") != preview["digest"]:
                raise ValueError("Sources changed; generate new previews before saving")
            check_split(split, rendered.provenance)
            checked.append((item, rendered))
        saved, skipped = [], []
        for index, (item, rendered) in enumerate(checked):
            try:
                result = self.workspace.store.save(directory, split, rendered, item["scene"])
                saved.append(dict(selection=index, **result))
            except ValueError as error:
                skipped.append(dict(selection=index, reason=str(error)))
            except OSError as error:
                skipped.append(dict(selection=index, reason=str(error)))
                for remaining in range(index + 1, len(checked)):
                    skipped.append(dict(selection=remaining, reason="Not saved after an output error"))
                break
        return dict(selected=len(items), saved=saved, skipped=skipped)
