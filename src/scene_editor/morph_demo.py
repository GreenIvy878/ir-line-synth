"""Demonstrate motion replay (morph.py) on a real dataset.

    python -m scene_editor.morph_demo --data real_data --out morph_demo

Writes into --out:
  real_kinds.png         real training objects grouped by the detected kind
  variants_<kind>.png    rows: real source | 10 replays (green = accepted, red = rejected)
  report.json            acceptance rate and rejection reasons per kind, real statistics
  scenes/                a YOLO dataset of scenes: real frames + accepted new objects
  scenes.png             preview of the scenes (gray = real boxes, colour = new objects)
"""

import argparse
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from . import morph
from .dataset import load_data_description, open_split, scale_boxes
from .scene import LEVELS, RenderedScene, box_iou, pixel_levels
from .store import SyntheticStore


COLORS = dict(blob=(80, 200, 255), rolling=(255, 170, 40), swinging=(255, 80, 200),
              long=(120, 255, 120), short=(255, 255, 90))
ZOOM = 6
CELL = 20  # object cell, pixels before zoom


def to_image(layer, zoom=ZOOM):
    values = pixel_levels(np.clip(layer, 0, 1))
    return Image.fromarray(values).resize((layer.shape[1] * zoom, layer.shape[0] * zoom),
                                          Image.Resampling.NEAREST).convert("RGB")


def cell(layer, border, text=""):
    """An object centred in a fixed cell with a coloured frame and a caption."""
    size = CELL * ZOOM
    tile = Image.new("RGB", (size + 6, size + 20), (25, 25, 25))
    picture = to_image(layer[:CELL, :CELL]) if layer.size else Image.new("RGB", (1, 1))
    tile.paste(picture, (3 + (size - picture.width) // 2, 3 + (size - picture.height) // 2))
    draw = ImageDraw.Draw(tile)
    draw.rectangle([0, 0, size + 5, size + 5], outline=border, width=3)
    draw.text((3, size + 7), text[:20], fill=border)
    return tile


def grid(rows, title):
    width = max(len(row) for row in rows) * (CELL * ZOOM + 10)
    height = len(rows) * (CELL * ZOOM + 26) + 24
    canvas = Image.new("RGB", (width, height), (15, 15, 15))
    ImageDraw.Draw(canvas).text((6, 5), title, fill=(255, 255, 255))
    for r, row in enumerate(rows):
        for c, tile in enumerate(row):
            canvas.paste(tile, (c * (CELL * ZOOM + 10), 24 + r * (CELL * ZOOM + 26)))
    return canvas


class Replayer:
    def __init__(self, objects, stats, seed):
        self.stats = stats
        self.rng = np.random.default_rng(seed)
        usable = [o for o in objects if not o.clipped]
        self.pools = {kind: [o for o in usable if o.kind in morph.SOURCE_KINDS[kind]]
                      for kind in morph.KINDS}
        self.texture = None
        # Interpolation averages the sensor noise away; measure how much, once.
        layers = []
        for kind in morph.KINDS:
            for _ in range(80):
                obj = self.pick(kind)
                layers.append(morph.render(obj, morph.sample_motion(kind, obj, self.rng)))
        self.texture = (stats["texture"], morph.texture_profile([l for l in layers if l.size]))

    def pick(self, kind):
        pool = self.pools[kind]
        return pool[int(self.rng.integers(len(pool)))]

    def replay(self, kind, obj):
        params = morph.sample_motion(kind, obj, self.rng)
        layer = morph.render(obj, params, self.texture, self.rng)
        accepted, reason, _ = morph.judge(layer, kind, self.stats, obj.layer)
        return layer, params, accepted, reason

    def accepted(self, kind, attempts=60):
        for _ in range(attempts):
            obj = self.pick(kind)
            layer, params, ok, _ = self.replay(kind, obj)
            if ok:  # judge already refuses round blobs and torn traces
                return obj, layer, params
        return None


def real_kinds(objects, rng):
    rows = []
    for kind in morph.KINDS:
        pool = [o for o in objects if o.kind == kind]
        chosen = rng.permutation(len(pool))[:12]
        rows.append([cell(pool[i].layer, COLORS[kind], f"{kind} {pool[i].source[:10]}") for i in chosen])
    return grid(rows, "Real training objects by detected kind")


def place(frame_boxes, box, max_iou):
    return all(box_iou(box, other) <= max_iou for other in frame_boxes)


def make_scene(frames, index, replayer, rng, max_iou=0.15):
    """A real training frame with 1-3 accepted new objects (max blending)."""
    pixels = np.clip(frames.pixels(index), 0, 1)
    pixels, boxes = morph.time_major(pixels, scale_boxes(frames.boxes[index], pixels.shape), frames.sensor_axis)
    pixels = pixels.copy()
    rows, columns = pixels.shape
    boxes = [np.asarray(b, np.float64) for b in boxes]
    real_count = len(boxes)
    added, kinds = [], []
    for _ in range(int(rng.integers(1, 4))):
        kind = morph.KINDS[int(rng.integers(len(morph.KINDS)))]
        found = replayer.accepted(kind)
        if found is None:
            continue
        obj, layer, params = found
        box = morph.label_box(layer, replayer.stats)
        height, width = layer.shape
        for _ in range(60):
            oy = int(rng.integers(-int(box[1]), rows - int(math.ceil(box[3])) + 1))
            ox = int(rng.integers(-int(box[0]), columns - int(math.ceil(box[2])) + 1))
            placed = box + np.asarray([ox, oy, ox, oy], np.float32)
            if placed.min() < 0 or placed[2] > columns or placed[3] > rows:
                continue
            if not place(boxes, placed.astype(np.float64), max_iou):
                continue
            # Paste the part of the layer that falls inside the frame.
            y0, x0 = max(0, oy), max(0, ox)
            y1, x1 = min(rows, oy + height), min(columns, ox + width)
            region = layer[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
            pixels[y0:y1, x0:x1] = np.maximum(pixels[y0:y1, x0:x1], region)
            boxes.append(placed.astype(np.float64))
            kinds.append(kind)
            added.append(dict(kind=kind, source=obj.source, box_index=obj.box_index,
                              x=ox, y=oy, motion=params))
            break
    if not added:
        return None
    boxes = np.asarray(boxes, np.float64)
    if frames.sensor_axis == "y":
        pixels, boxes = pixels.T.copy(), boxes[:, [1, 0, 3, 2]]
    pixels = pixel_levels(pixels).astype(np.float32) / LEVELS
    provenance = [dict(split="train", id=frames.paths[index].name, role="background")]
    provenance += [dict(split="train", id=row["source"], box_index=row["box_index"],
                        role="object", morph=row["kind"]) for row in added]
    rendered = RenderedScene(pixels, boxes, tuple(0 for _ in boxes), tuple(provenance), len(added))
    scene = dict(method="motion_replay", background=dict(split="train", id=frames.paths[index].name),
                 objects=added)
    return rendered, scene, real_count, kinds


def scene_tile(rendered, real_count, kinds, zoom=5):
    image = to_image(rendered.pixels, zoom)
    draw = ImageDraw.Draw(image)
    for i, box in enumerate(rendered.boxes):
        color = (110, 110, 110) if i < real_count else COLORS[kinds[i - real_count]]
        draw.rectangle([box[0] * zoom, box[1] * zoom, box[2] * zoom - 1, box[3] * zoom - 1], outline=color)
    return image


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default="morph_demo")
    parser.add_argument("--sources", type=int, default=6, help="real sources per kind in the grids")
    parser.add_argument("--variants", type=int, default=10)
    parser.add_argument("--trials", type=int, default=400, help="replays per kind for the acceptance rate")
    parser.add_argument("--scenes", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    data = load_data_description(args.data)
    frames = open_split(data, "train")
    out = Path(args.out).resolve()
    scenes_root = SyntheticStore(data).output_root(str(out / "scenes"))
    out.mkdir(parents=True, exist_ok=True)

    objects = morph.extract_objects(frames, "train")
    stats = morph.fit_statistics(objects)
    rng = np.random.default_rng(args.seed)
    replayer = Replayer(objects, stats, args.seed)
    real_kinds(objects, rng).save(out / "real_kinds.png")
    print(f"{len(objects)} real objects:", stats["kinds"])

    report = dict(real_kinds=stats["kinds"], box_margin=stats["box_margin"],
                  texture=dict(real=replayer.texture[0], replayed=replayer.texture[1]), kinds={})
    for kind in morph.KINDS:
        rows = []
        for _ in range(args.sources):
            obj = replayer.pick(kind)
            row = [cell(obj.layer, (230, 230, 230), f"src {obj.kind}")]
            for _ in range(args.variants):
                layer, _, ok, reason = replayer.replay(kind, obj)
                row.append(cell(layer, (60, 220, 60) if ok else (230, 50, 50), "ok" if ok else reason))
            rows.append(row)
        grid(rows, f"{kind}: real source (white) -> replays; green accepted, red rejected (reason)"
             ).save(out / f"variants_{kind}.png")
        reasons = Counter(replayer.replay(kind, replayer.pick(kind))[3].split(":")[0]
                          for _ in range(args.trials))
        report["kinds"][kind] = dict(sources=len(replayer.pools[kind]),
                                     acceptance=reasons["ok"] / args.trials, reasons=dict(reasons))
        print(f"{kind:9s} accepted {reasons['ok'] / args.trials:5.1%}  {dict(reasons)}")

    store = SyntheticStore(data)
    tiles, saved = [], 0
    order = rng.permutation(len(frames))
    for index in order:
        if saved >= args.scenes:
            break
        made = make_scene(frames, int(index), replayer, rng)
        if made is None:
            continue
        rendered, scene, real_count, kinds = made
        try:
            store.save(str(scenes_root), "train", rendered, scene)
        except ValueError as error:
            print("skipped scene:", error)
            continue
        saved += 1
        tiles.append(scene_tile(rendered, real_count, kinds))
    if tiles:
        columns = 8
        sheet = Image.new("RGB", (columns * 204, math.ceil(len(tiles) / columns) * 204), (15, 15, 15))
        for i, tile in enumerate(tiles):
            sheet.paste(tile, ((i % columns) * 204, (i // columns) * 204))
        sheet.save(out / "scenes.png")
    report["scenes_saved"] = saved
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"{saved} scenes saved to {scenes_root}; pictures and report in {out}")


if __name__ == "__main__":
    main()
