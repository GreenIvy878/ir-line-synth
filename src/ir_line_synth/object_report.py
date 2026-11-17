"""Objects described by shape against the real objects of a dataset (docs/objects.md, section 9).

Real objects: the labelled boxes of the frames of a dataset made from recordings (manifest.csv:
recording and tile of every frame), in raw codes: the means of `pool` polls of the recording
minus the belt level of every channel. Boxes at the frame edge are left out.

Model objects: ellipsoids of the size of a hen egg (EGG_LENGTH, EGG_WIDTH; lying on its side, so
the height is the width) passing calmly under the line of the profile (motion kind blob), turned
by up to EGG_YAW degrees and shifted across within a pitch. Their boxes are the geometric ones.

Features of a spot, in its box: the peak over the belt (codes); the channels and rows lifted above
LIFT_SHARE of the peak (width, length); the next three channels of the peak row after the peak, as
shares of the peak (side1 ... side3: the profile across the line).

The fit, with the height of the line, the poll rate and the belt reflectance given:
1. the pitch of the channels and the tilt power of the surface (shapes.Surface.tilt_power) on the
   grid PITCH_GRID x TILT_GRID, by the medians of side1 ... side3;
2. the belt speed by the mean length;
3. the reflectance of the egg by the median peak.
The height of the line and the reflectance trade off: a higher line needs a brighter egg for the
same peak. The report shows the reflectance the peak needs at other heights.

Margins of the label boxes: annotators draw their boxes around the signal, the boxes of objects
described by shape are geometric. The annotators' margins (annotated box minus the threshold box,
morph.fit_statistics on the train split of the dataset) plus the offset of the threshold box from the
geometric one on model eggs (alone in a normalized frame) give margins for geometric boxes that
imitate the annotators.

`report` writes the results as Markdown (docs/object_validation.md for the reference line).
"""

import csv
from pathlib import Path

import numpy as np

from . import PROGRAM, __version__, motion
from .calibration import digest
from .dataset import load_data_description, read_yolo_boxes, scale_boxes
from .morph import TAU, box_near_edge, cut_object, tight_box
from .record import render_record
from .shapes import Installation, Superquadric, Surface, geometric_box, line_signal, natural_rows, passage


LIFT_SHARE = 0.2
# Typical hen egg, mm: length and width (the eggs of the reference data were not measured).
EGG_LENGTH = (53.0, 61.0)
EGG_WIDTH = (41.0, 47.0)
EGG_YAW = 20.0
MODEL_EGGS = 40
REPORT_SEED = 2026
PITCH_GRID = tuple(6.0 + 0.5 * step for step in range(17))
TILT_GRID = tuple(0.25 * step for step in range(17))
START_SPEED = 160.0
HEIGHTS_SHOWN = (55.0, 60.0, 65.0, 70.0)
FEATURES = ("peak", "width", "length", "side1", "side2", "side3")


def spot_features(samples, box):
    """Features of the spot in `box` (x1, y1, x2, y2 in cells) of samples [rows, channels] over the belt."""
    x1, y1, x2, y2 = (int(value) for value in (np.floor(box[0]), np.floor(box[1]), np.ceil(box[2]), np.ceil(box[3])))
    region = samples[y1:y2, x1:x2]
    peak = float(region.max())
    row = np.sort(region[np.unravel_index(np.argmax(region), region.shape)[0]])[::-1] / peak
    sides = [float(row[k]) if len(row) > k else 0.0 for k in (1, 2, 3)]
    lifted = region > LIFT_SHARE * peak
    return dict(peak=peak, width=int(lifted.any(axis=0).sum()), length=int(lifted.any(axis=1).sum()),
                side1=sides[0], side2=sides[1], side3=sides[2])


def real_spots(model, dataset, recordings, recording_files):
    """Features of the labelled whole objects of the frames of `dataset` made from these recordings."""
    by_name = {Path(path).name: polls for polls, path in zip(recordings, recording_files)}
    pool, length, channels = model.frames.pool, model.frames.length, model.channels
    belt = model.operation.belt_level
    dataset = Path(dataset)
    axis = load_data_description(dataset)["sensor_axis"]
    spots = []
    with open(dataset / "manifest.csv", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        polls = by_name.get(row["recording"])
        if polls is None:
            continue
        tile = int(row["tile"])
        window = polls[tile * pool * length:(tile + 1) * pool * length]
        if len(window) < pool * length:
            continue
        samples = window.reshape(length, pool, channels).mean(axis=1) - belt
        labels = read_yolo_boxes(dataset / row["split"] / "labels" / f"{row['name']}.txt")
        if axis == "y":  # channels along the rows of the image: turn the boxes to rows of time
            labels = np.asarray(labels, np.float64).reshape(-1, 4)[:, [1, 0, 3, 2]]
        boxes = scale_boxes(labels, (length, channels))
        for box in boxes:
            if not box_near_edge(box, (length, channels)):
                spots.append(spot_features(samples, box))
    return spots


def model_spots(model, installation, speed, tilt, reflectance=1.0, count=MODEL_EGGS, seed=REPORT_SEED):
    """Features of `count` egg-sized ellipsoids passing calmly under the line (the same draws for the same seed)."""
    rng = np.random.default_rng(seed)
    pool, spots = model.frames.pool, []
    for _ in range(count):
        width = float(rng.uniform(*EGG_WIDTH))
        shape = Superquadric(float(rng.uniform(*EGG_LENGTH)), width, width)
        yaw = float(rng.uniform(-EGG_YAW, EGG_YAW))
        y = float(rng.uniform(-installation.pitch_mm / 2, installation.pitch_mm / 2))
        rows = natural_rows(shape, model, installation, speed, yaw)
        scale = motion.LineScale.of(pool, installation.poll_hz, installation.pitch_mm)
        move = motion.sample("blob", rows, (0.0, rows), rng, scale=scale)
        heights, rho = passage(shape, Surface(reflectance, tilt_power=tilt), move, model, installation, speed, y, yaw)
        excess = line_signal(model, installation, heights, rho) - model.operation.belt_level
        samples = excess.reshape(-1, pool, model.channels).mean(axis=1)
        spots.append(spot_features(samples, geometric_box(heights, pool)))
    return spots


def threshold_offsets(model, installation, speed, tilt, reflectance, count=4 * MODEL_EGGS, seed=REPORT_SEED):
    """Mean of (threshold box - geometric box) of model eggs, each alone in one normalized frame (cells).

    The egg is cut with its geometric box as the real ones are cut with the annotated box
    (morph.cut_object), so the threshold box is measured the same way as for the annotators' margins."""
    rng = np.random.default_rng(seed)
    length, offsets = model.frames.length, []
    for _ in range(count):
        width = float(rng.uniform(*EGG_WIDTH))
        shape = dict(length=float(rng.uniform(*EGG_LENGTH)), width=width, height=width)
        yaw = float(rng.uniform(-EGG_YAW, EGG_YAW))
        rows = natural_rows(Superquadric(**shape), model, installation, speed, yaw)
        scale = motion.LineScale.of(model.frames.pool, installation.poll_hz, installation.pitch_mm)
        move = motion.sample("blob", rows, (0.0, rows), rng, scale=scale)
        start = max(0, (length - len(motion.output_times(0.0, rows, np.asarray(move["speed"])))) // 2)
        spec = dict(profile=model.name, speed_mm_s=speed, rows=length, seed=int(rng.integers(2**32)),
                    installation=dict(height_mm=installation.height_mm, pitch_mm=installation.pitch_mm,
                                      poll_hz=installation.poll_hz, belt_reflectance=installation.belt_reflectance),
                    objects=[dict(shape=shape, surface=dict(reflectance=reflectance, tilt_power=tilt), motion=move,
                                  row=start, y=float(rng.uniform(-installation.pitch_mm / 2, installation.pitch_mm / 2)),
                                  yaw=yaw)])
        try:
            rendered = render_record(spec, model)
        except ValueError:
            continue  # a slowed egg longer than a frame
        obj = cut_object(rendered.frames[0].astype(np.float32) / 255, rendered.geometry, 0)
        if obj is not None:
            offsets.append(tight_box(obj.layer) - obj.box)
    # Both boxes are whole cells, so a median would jump between them: the mean is taken.
    return np.mean(offsets, axis=0)


def annotator_margins(dataset):
    """The annotators' margins of the train split of a dataset (morph.fit_statistics)."""
    from .dataset import load_data_description
    from .workspace import Workspace
    return np.asarray(Workspace(load_data_description(dataset)).library("train").stats["box_margin"], np.float64)


def medians(spots):
    return {key: float(np.median([spot[key] for spot in spots])) for key in FEATURES}


def mean_length(spots):
    return float(np.mean([spot["length"] for spot in spots]))


def fit(model, real, height_mm, poll_hz, belt_reflectance):
    """Pitch, tilt power, speed and reflectance of the egg that bring the model spots to the real ones."""
    target = medians(real)
    sides = ("side1", "side2", "side3")
    errors = {}
    for pitch in PITCH_GRID:
        installation = Installation(height_mm, pitch, poll_hz, belt_reflectance)
        for tilt in TILT_GRID:
            got = medians(model_spots(model, installation, START_SPEED, tilt))
            errors[(pitch, tilt)] = sum((got[key] - target[key]) ** 2 for key in sides)
    pitch, tilt = min(errors, key=errors.get)
    installation = Installation(height_mm, pitch, poll_hz, belt_reflectance)
    speed = START_SPEED
    for _ in range(2):
        speed *= mean_length(model_spots(model, installation, speed, tilt)) / mean_length(real)
    reflectance = 1.0
    for _ in range(2):
        reflectance *= target["peak"] / medians(model_spots(model, installation, speed, tilt, reflectance))["peak"]
    return dict(pitch=pitch, tilt=tilt, speed=speed, reflectance=reflectance, installation=installation,
                errors=errors)


def peak_reflectance(model, real, fitted, height_mm):
    """The reflectance the median peak needs with the line at `height_mm` (other fitted values kept)."""
    installation = Installation(height_mm, fitted["pitch"], fitted["installation"].poll_hz,
                                fitted["installation"].belt_reflectance)
    peak = medians(model_spots(model, installation, fitted["speed"], fitted["tilt"]))["peak"]
    return medians(real)["peak"] / peak


def num(value, digits=2):
    return f"{value:.{digits}f}".replace(".", ",")


def quartiles(spots, key, digits):
    values = [spot[key] for spot in spots]
    return " / ".join(num(np.percentile(values, q), digits) for q in (25, 50, 75))


def report(model, recordings, recording_files, dataset, height_mm, poll_hz, belt_reflectance):
    """The fit and the comparison as a Markdown document."""
    if model.operation is None:
        raise ValueError("The profile has no bare belt (operation section): the spots are measured over it")
    real = real_spots(model, dataset, recordings, recording_files)
    if len(real) < MODEL_EGGS:
        raise ValueError(f"Only {len(real)} whole labelled objects in the frames of these recordings")
    fitted = fit(model, real, height_mm, poll_hz, belt_reflectance)
    installation = fitted["installation"]
    final = model_spots(model, installation, fitted["speed"], fitted["tilt"], fitted["reflectance"], 4 * MODEL_EGGS)
    flat = fitted["errors"][(fitted["pitch"], 0.0)]
    real_peaks = [spot["peak"] for spot in real]
    model_peaks = [spot["peak"] for spot in final]
    spread = [fitted["reflectance"] * np.percentile(real_peaks, q) / np.percentile(model_peaks, q) for q in (25, 75)]
    names = {"peak": ("Пик над лентой, коды", 0), "width": ("Ширина, каналы", 1), "length": ("Длина, строки", 1),
             "side1": ("Соседний канал / пик", 2), "side2": ("Второй канал / пик", 2), "side3": ("Третий канал / пик", 2)}
    lines = [
        f"# Объекты по описанию против реальных: профиль {model.name}",
        "",
        f"Документ создан командой `ir-line-synth object-report` ({PROGRAM} {__version__}) и не правится вручную.",
        "Метод описан в [objects.md](objects.md), раздел 9.",
        "",
        "Исходные данные:",
        "",
        f"- рабочие записи: {len(recordings)} ({', '.join(Path(p).stem for p in recording_files)}); "
        f"SHA-256 `{digest(recording_files)[:16]}…`;",
        f"- набор кадров: `{Path(dataset).name}` (manifest.csv), целых размеченных объектов: {len(real)};",
        f"- заданы: высота линейки {num(height_mm, 1)} мм, частота опроса {num(poll_hz, 0)} Гц, отражательная "
        f"способность ленты {num(belt_reflectance)}.",
        "",
        "## Подобранные значения",
        "",
        "| Величина | Значение | По чему подобрана |",
        "|---|---|---|",
        f"| Шаг каналов | {num(fitted['pitch'], 1)} мм | Профиль пятна поперёк каналов (сетка {num(PITCH_GRID[0], 0)}–"
        f"{num(PITCH_GRID[-1], 0)} мм через {num(PITCH_GRID[1] - PITCH_GRID[0], 1)}) |",
        f"| Показатель наклона `tilt_power` | {num(fitted['tilt'])} | То же (сетка {num(TILT_GRID[0], 0)}–"
        f"{num(TILT_GRID[-1], 0)} через {num(TILT_GRID[1] - TILT_GRID[0])}); ламбертова поверхность — 1 |",
        f"| Скорость ленты | {num(fitted['speed'], 0)} мм/с ({num(fitted['speed'] * 0.06, 1)} м/мин) | Средняя длина пятна |",
        f"| Отражательная способность яйца | {num(fitted['reflectance'])} | Медиана пика; по 25-му и 75-му "
        f"процентилям пика — {num(spread[0])}–{num(spread[1])} |",
        "",
        f"Сумма квадратов ошибок медиан бокового профиля: {num(fitted['errors'][(fitted['pitch'], fitted['tilt'])], 4)} "
        f"при подобранном показателе наклона и {num(flat, 4)} без зависимости от наклона (тот же шаг).",
        "",
        "## Сравнение распределений",
        "",
        f"Квартили (25 / 50 / 75 %) признаков: реальные объекты и {len(final)} модельных яиц при подобранных значениях.",
        "",
        "| Признак | Реальные | Модель |",
        "|---|---|---|",
    ]
    lines += [f"| {title} | {quartiles(real, key, digits)} | {quartiles(final, key, digits)} |"
              for key, (title, digits) in names.items()]
    lines += [
        "",
        f"Признаки считаются в рамке объекта по средним из {model.frames.pool} опросов за вычетом уровня ленты; "
        f"ширина и длина — каналы и строки выше {num(LIFT_SHARE * 100, 0)} % пика.",
        "",
        "## Высота линейки и отражательная способность",
        "",
        "Пик зависит от высоты линейки и отражательной способности яйца совместно: при другой высоте тот же",
        "пик даёт другая отражательная способность (остальные подобранные значения те же).",
        "",
        "| Высота линейки, мм | Отражательная способность яйца для медианы пика |",
        "|---|---|",
    ]
    lines += [f"| {num(height, 1)} | {num(peak_reflectance(model, real, fitted, height))} |" for height in HEIGHTS_SHOWN]
    annotated = annotator_margins(dataset)
    offsets = threshold_offsets(model, installation, fitted["speed"], fitted["tilt"], fitted["reflectance"])
    margins = annotated + offsets
    row = lambda values: " | ".join(num(value) for value in values)
    lines += [
        "",
        "## Поля рамок разметки",
        "",
        "Поля (x1, y1, x2, y2) в ячейках: отрицательные x1, y1 и положительные x2, y2 расширяют рамку.",
        "",
        "| | x1 | y1 | x2 | y2 |",
        "|---|---|---|---|---|",
        f"| Разметчики: рамка минус рамка по порогу (train набора, медиана) | {row(annotated)} |",
        f"| Модельные яйца: рамка по порогу минус геометрическая (среднее) | {row(offsets)} |",
        f"| Поля для геометрических рамок (сумма) | {row(margins)} |",
        "",
        f"Порог — {num(TAU)} шкалы нормированного кадра (`morph.TAU`); модельное яйцо одно в кадре.",
    ]
    lines += [
        "",
        "## Ограничения",
        "",
        f"- Размеры яиц не измерены: взяты типичные для куриного яйца (длина {num(EGG_LENGTH[0], 0)}–"
        f"{num(EGG_LENGTH[1], 0)} мм, ширина {num(EGG_WIDTH[0], 0)}–{num(EGG_WIDTH[1], 0)} мм).",
        "- Показатель наклона подобран вместе с шагом каналов и может частично заменять неучтённое пятно засветки.",
        "- Разброс пиков реальных яиц шире модельного при одной отражательной способности: яйца различаются цветом",
        "  и размером; для случайных записей отражательную способность лучше задавать диапазоном выше.",
        "",
    ]
    return "\n".join(lines)
