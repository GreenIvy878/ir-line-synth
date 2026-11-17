"""The check of a sensor profile against measurements (docs/sensor_model.md, section 10).

C1  response law on held-out distances: every inner distance is left out of the fit in
    turn and predicted;
C2  sensor noise: the spread of integer samples simulated at the calibration nodes
    against the measured spread;
C3  frames: the frames of a dataset made again from its recordings against its files
    (the dataset lists the recording and tile of every frame in manifest.csv);
C4  operation noise: the parameters fitted on one half of the recordings against the
    other half, and the correlation in time against the model;
C5  channel offsets on the belt against the offsets of the calibration.

`report` writes the results as Markdown (docs/sensor_validation.md for the reference line).
"""

import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image

from . import PROGRAM, __version__
from .calibration import belt_statistics, digest, fit_response, operation_parameters
from .dataset import load_data_description
from .framing import frames_from_polls, read_recording


# Replicates of the noise simulation (C2) and its seed; lags of the belt autocorrelation (C4).
NOISE_REPLICATES = 400
REPORT_SEED = 2026
LAGS = (1, 10, 21, 42, 85, 200)


def check_response(calibration):
    """C1: relative errors of the law in the fit and on held-out distances."""
    full = fit_response(calibration)
    means, saturated = calibration.means, calibration.saturated
    grid = np.repeat(calibration.distances[:, None], calibration.channels, 1)
    fitted = np.where(saturated, np.nan, (full.signal(grid) - means) / means)
    rms = np.sqrt(np.nanmean(fitted ** 2, axis=0))
    usable = np.flatnonzero(~saturated.all(1))
    held = []
    for node in usable[1:-1]:
        response = fit_response(calibration, exclude=node)
        prediction = response.signal(np.full(calibration.channels, calibration.distances[node]))
        error = (prediction - means[node]) / means[node]
        held += [abs(value) for value, sat in zip(error, saturated[node]) if not sat]
    held = np.asarray(held)
    return dict(fit_rms_median=float(np.median(rms)), fit_rms_max=float(rms.max()),
                held_median=float(np.median(held)), held_p95=float(np.percentile(held, 95)),
                held_max=float(held.max()), held_count=int(len(held)),
                held_from=float(calibration.distances[usable[1]]), held_to=float(calibration.distances[usable[-2]]),
                power=(float(full.power.min()), float(np.median(full.power)), float(full.power.max())),
                shift=(float(full.shift.min()), float(np.median(full.shift)), float(full.shift.max())),
                offset=(float(full.offset.min()), float(np.median(full.offset)), float(full.offset.max())))


def check_noise(calibration, model):
    """C2: measured sd of the integer samples of every node against the simulated one."""
    rng = np.random.default_rng(REPORT_SEED)
    means, saturated = calibration.means, calibration.saturated
    count = len(calibration.samples[0])
    measured, low, middle, high, zero_simulated = [], [], [], [], []
    for node, channel in zip(*np.nonzero(~saturated)):
        mean = means[node, channel]
        noise = model.noise.sd(mean, channel)
        draws = np.round(mean + noise * rng.standard_normal((NOISE_REPLICATES, count)))
        sds = draws.std(axis=1, ddof=1)
        measured.append(float(np.sqrt(calibration.variances[node, channel])))
        low.append(np.percentile(sds, 5)); middle.append(np.median(sds)); high.append(np.percentile(sds, 95))
        zero_simulated.append(float(np.mean(sds == 0)))
    measured, low, middle, high = (np.asarray(v) for v in (measured, low, middle, high))
    return dict(nodes=int(len(measured)), measured_median=float(np.median(measured)),
                simulated_median=float(np.median(middle)),
                difference_median=float(np.median(np.abs(measured - middle))),
                inside_5_95=float(np.mean((measured >= low) & (measured <= high))),
                zero_measured=float(np.mean(measured == 0)), zero_simulated=float(np.mean(zero_simulated)),
                a=model.noise.a, k=model.noise.k,
                scale=(1.0, 1.0) if model.noise.scale is None else
                (float(model.noise.scale.min()), float(model.noise.scale.max())))


def check_frames(dataset, recordings, spec, channels):
    """C3: frames of a dataset (manifest.csv: name, split, recording, tile from 0) made again."""
    dataset = Path(dataset)
    axis = load_data_description(dataset)["sensor_axis"]
    with open(dataset / "manifest.csv", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    made, checked, differing, pixels = {}, 0, 0, 0
    for row in rows:
        name = row["recording"]
        if name not in made:
            made[name] = frames_from_polls(read_recording(Path(recordings) / name, channels), spec)
        tile = int(row["tile"])
        with Image.open(dataset / row["split"] / "images" / f"{row['name']}.png") as image:
            stored = np.asarray(image)
        if axis == "y":
            stored = stored.T
        frame = made[name][tile] if tile < len(made[name]) else None
        checked += 1
        if frame is None or frame.shape != stored.shape:
            differing += 1
            pixels += stored.size
        elif not np.array_equal(frame, stored):
            differing += 1
            pixels += int((frame != stored).sum())
    return dict(frames=checked, differing=differing, pixels=pixels, recordings=len(made))


def check_operation(recordings, numbers, boards, model):
    """C4: belt noise of the even- and odd-numbered recordings, and its correlation in time."""
    halves = {}
    for name, parity in (("чётные", 0), ("нечётные", 1)):
        part = [polls for polls, number in zip(recordings, numbers) if number % 2 == parity]
        if part:
            belt = belt_statistics(part, boards)
            halves[name] = dict(recordings=len(part), **operation_parameters(belt))
    belt = belt_statistics(recordings, boards)
    lags = {lag: dict(measured=belt.autocorrelation(lag), model=model.operation.autocorrelation(lag)) for lag in LAGS}
    return dict(halves=halves, lags=lags, runs=len(belt.runs), polls=int(sum(len(run) for run in belt.runs)),
                level_spread=float(np.median(belt.levels.std(axis=0))))


def check_offsets(model, belt_levels):
    """C5: channel levels on the belt against the offsets and the signal of the calibration."""
    level = np.median(belt_levels, axis=0)
    at = {d: float(np.corrcoef(level, model.response.signal(np.full(model.channels, d)))[0, 1])
          for d in (30.0, 60.0, 100.0)}
    return dict(offset=float(np.corrcoef(level, model.response.offset)[0, 1]), signal=at,
                belt_range=(float(level.min()), float(level.max())),
                offset_range=(float(model.response.offset.min()), float(model.response.offset.max())))


def pct(value, digits=1):
    return f"{100 * value:.{digits}f} %".replace(".", ",")


def num(value, digits=2):
    return f"{value:.{digits}f}".replace(".", ",").replace("-", "−")


def sci(value):
    """A number as m·10^e in LaTeX, for small coefficients."""
    if value == 0:
        return "0"
    exponent = int(math.floor(math.log10(abs(value))))
    mantissa = f"{value / 10 ** exponent:.2f}".replace(".", "{,}")
    return f"${mantissa} \\cdot 10^{{{exponent}}}$"


def report(model, calibration, recordings, numbers, recording_files, dataset=None, recordings_folder=None):
    """The checks C1-C5 as a Markdown document."""
    c1 = check_response(calibration)
    c2 = check_noise(calibration, model)
    c4 = check_operation(recordings, numbers, model.boards, model) if model.operation else None
    c5 = check_offsets(model, belt_statistics(recordings, model.boards).levels) if recordings else None
    c3 = check_frames(dataset, recordings_folder, model.frames, model.channels) if dataset else None
    lines = [
        f"# Проверка модели датчика: профиль {model.name}",
        "",
        f"Документ создан командой `ir-line-synth sensor-report` ({PROGRAM} {__version__}) и не правится вручную.",
        "Модель описана в [sensor_model.md](sensor_model.md).",
        "",
        "Исходные данные:",
        "",
        f"- калибровочные замеры: {len(calibration.files)} файлов, расстояния "
        f"{num(calibration.distances.min(), 0)}–{num(calibration.distances.max(), 0)} мм, "
        f"{len(calibration.samples[0])} отсчётов в узле, насыщение {calibration.clip}; SHA-256 `{digest(calibration.files)[:16]}…`;",
    ]
    if recordings:
        lines.append(f"- рабочие записи: {len(recordings)} ({', '.join(Path(p).stem for p in recording_files)}), "
                     f"{sum(len(r) for r in recordings)} опросов; SHA-256 `{digest(recording_files)[:16]}…`;")
    if dataset:
        lines.append(f"- набор кадров: `{Path(dataset).name}` (manifest.csv).")
    lines += [
        "",
        "## П1. Закон отклика",
        "",
        "Закон $s = o + A (d + d_0)^{-p}$ подобран для каждого канала по узлам без насыщения.",
        "",
        "| Показатель | Значение |",
        "|---|---|",
        f"| Относительная ошибка в узлах подбора, медиана СКО по каналам | {pct(c1['fit_rms_median'], 2)} |",
        f"| То же, худший канал | {pct(c1['fit_rms_max'], 2)} |",
        f"| Отложенные расстояния {num(c1['held_from'], 0)}–{num(c1['held_to'], 0)} мм "
        f"({c1['held_count']} предсказаний): медиана модуля ошибки | {pct(c1['held_median'], 2)} |",
        f"| То же, 95-й процентиль | {pct(c1['held_p95'], 2)} |",
        f"| То же, максимум | {pct(c1['held_max'], 2)} |",
        f"| Показатель $p$: мин. / медиана / макс. | {' / '.join(num(v) for v in c1['power'])} |",
        f"| Сдвиг $d_0$, мм: мин. / медиана / макс. | {' / '.join(num(v) for v in c1['shift'])} |",
        f"| Смещение $o$, коды: мин. / медиана / макс. | {' / '.join(num(v, 1) for v in c1['offset'])} |",
        "",
        "Отложенное расстояние убирается из подбора всех каналов сразу; крайние расстояния не проверяются,",
        "потому что их предсказание было бы экстраполяцией.",
        "",
        "## П2. Шум датчика",
        "",
        f"Дисперсия шума до округления $\\sigma_c^2 = q_c (a + k s)$: $a$ = {num(c2['a'], 4)}, $k$ = {sci(c2['k'])}, "
        f"множитель канала $q_c$ от {num(c2['scale'][0])} до {num(c2['scale'][1])} (медиана 1). "
        f"В каждом из {c2['nodes']} узлов без насыщения смоделировано {NOISE_REPLICATES} серий по "
        f"{len(calibration.samples[0])} округлённых отсчётов.",
        "",
        "| Показатель | Измерено | Модель |",
        "|---|---|---|",
        f"| СКО целых отсчётов, медиана по узлам, коды | {num(c2['measured_median'], 3)} | {num(c2['simulated_median'], 3)} |",
        f"| Доля узлов с нулевым разбросом | {pct(c2['zero_measured'])} | {pct(c2['zero_simulated'])} |",
        f"| Медиана модуля разности СКО по узлам, коды | {num(c2['difference_median'], 3)} | — |",
        f"| Доля узлов, где измеренное СКО внутри 5–95 % смоделированных | {pct(c2['inside_5_95'])} | 90 % ожидается |",
        "",
    ]
    if c3:
        lines += [
            "## П3. Формирование кадра",
            "",
            f"Кадры набора построены заново из рабочих записей (усреднение {model.frames.pool} опросов, тайлы по "
            f"{model.frames.length} отсчётов, нормировка `{model.frames.normalize}`) и сравнены с файлами.",
            "",
            "| Показатель | Значение |",
            "|---|---|",
            f"| Кадров проверено | {c3['frames']} (из {c3['recordings']} записей) |",
            f"| Кадров с отличиями | {c3['differing']} |",
            f"| Отличающихся пикселей | {c3['pixels']} |",
            "",
        ]
    if c4:
        lines += [
            "## П4. Шум в работе",
            "",
            f"Пустая лента — отрезки не короче 40 опросов, где все каналы в пределах ±12 кодов от уровня записи: "
            f"{c4['runs']} отрезков, {c4['polls']} опросов.",
            "",
            "| Записи | Число | СКО, коды | Период, опросов | Доля периодической составляющей | "
            "Связь соседних опросов (медленная составляющая) | Доля общей составляющей платы |",
            "|---|---|---|---|---|---|---|",
        ]
        rows = list(c4["halves"].items())
        operation = model.operation
        rows.append(("все (профиль)", dict(recordings=len(recordings), sd=operation.sd, period=operation.period,
                                           periodic_share=operation.periodic_share, lag1=operation.lag1,
                                           board_share=operation.board_share)))
        for name, row in rows:
            lines.append(f"| {name} | {row['recordings']} | {num(row['sd'])} | {num(row['period'], 2)} | "
                         f"{pct(row['periodic_share'])} | {num(row['lag1'], 4)} | {pct(row['board_share'])} |")
        lines += [
            "",
            "Автокорреляция шума ленты против модели профиля (периодическая составляющая и авторегрессия):",
            "",
            "| Сдвиг, опросов | Измерено | Модель |",
            "|---|---|---|",
        ]
        lines += [f"| {lag} | {num(row['measured'], 3)} | {num(row['model'], 3)} |" for lag, row in c4["lags"].items()]
        lines += [
            "",
            f"Разброс уровня ленты между записями (медиана по каналам СКО уровней записей): "
            f"{num(c4['level_spread'], 1)} кода.",
            "",
        ]
    if c5:
        lines += [
            "## П5. Смещения каналов в работе",
            "",
            f"Уровень пустой ленты по каналам: {num(c5['belt_range'][0], 0)}–{num(c5['belt_range'][1], 0)} кодов; "
            f"смещения калибровки: {num(c5['offset_range'][0], 1)}–{num(c5['offset_range'][1], 1)} кодов.",
            "",
            "| Корреляция по 40 каналам уровня ленты с | Значение |",
            "|---|---|",
            f"| смещением калибровки $o_c$ | {num(c5['offset'])} |",
        ]
        lines += [f"| сигналом калибровки при {num(d, 0)} мм | {num(value)} |" for d, value in c5["signal"].items()]
        lines.append("")
    return "\n".join(lines)
