"""The registry of the program's fixed numbers: value, unit, origin and meaning of each.

Every numeric constant at the top level of a core module (CORE_MODULES) is listed here;
the tests fail when one is added without an entry, or an entry names no constant. The
values are read from the modules themselves, so the registry cannot drift from the code.
docs/constants.md is this table as text (python -m ir_line_synth constants --output docs/constants.md).

The descriptions are in Russian, like the rest of the user documentation.
"""

import ast
from dataclasses import dataclass
import importlib
from itertools import groupby
import numbers
from pathlib import Path


CORE_MODULES = ("sensor", "dataset", "scene", "morph", "background", "generator", "tape", "store", "workspace",
                "sensor_model", "framing", "calibration", "sensor_report", "shapes", "motion", "record", "object_report")

# Where a value comes from.
MEASURED = "измерено"      # computed from the real data; the source is named
TUNED = "подобрано"        # chosen by hand on the reference data and checked by eye and by tests
EXTERNAL = "внешнее"       # taken from another system so that both agree
BOUND = "ограничение"      # a bound on work or size that keeps requests finite
DEFINITION = "определение" # fixed by a format or by the meaning of a parameter
ORIGINS = (MEASURED, TUNED, EXTERNAL, BOUND, DEFINITION)

REFERENCE = "линия 40 каналов, кадры 40 опросов"


@dataclass(frozen=True)
class Constant:
    module: str
    name: str
    unit: str
    origin: str
    meaning: str

    @property
    def value(self):
        return getattr(importlib.import_module(f"{__package__}.{self.module}"), self.name)


def entry(module, name, unit, origin, meaning):
    if origin not in ORIGINS:
        raise ValueError(f"{module}.{name}: unknown origin {origin}")
    return Constant(module, name, unit, origin, meaning)


REGISTRY = (
    # --- sensor line ------------------------------------------------------------------------
    entry("sensor", "MAX_CHANNELS", "каналы", BOUND,
          "Наибольшее число каналов линии; проверка данных, на порядок выше линий в работе."),
    entry("sensor", "MAX_FRAME_LENGTH", "опросы", BOUND, "Наибольшая длина кадра."),

    # --- scenes -----------------------------------------------------------------------------
    entry("scene", "LEVELS", "уровни", DEFINITION,
          "Наибольший код 8-битного кадра: интенсивность 1 хранится как 255. Сцены округляются до этих уровней, "
          "поэтому предпросмотр совпадает с файлом."),
    entry("scene", "MAX_SHIFT", "пиксели", BOUND, "Наибольший случайный сдвиг объекта в вариациях."),
    entry("scene", "VARIATION_ATTEMPTS", "попытки", BOUND,
          "Попыток разместить один объект вариации, после чего вариация пропускается."),
    entry("scene", "NEUTRAL_APPEARANCE", "—", DEFINITION,
          "Нейтральные значения изменений яркости объекта: при них пиксели не меняются."),
    entry("scene", "APPEARANCE_BOUNDS", "—", BOUND, "Допустимые диапазоны изменений яркости объекта."),

    # --- real objects and new objects (morph.py) --------------------------------------------
    entry("morph", "TAU", "доля шкалы", MEASURED,
          "Порог «объект/фон»: 99 % пикселей фона ниже 0,086 (real_data, train, 18.10.2026; "
          "docs/synthetic_data_method.md, п. 2.3)."),
    entry("morph", "SIGNAL_FLOOR", "доля шкалы", TUNED,
          "Значения нового объекта ниже этого обнуляются как «хвосты» интерполяции: 3 уровня из 255, "
          "на уровне среднего фона (0,013; п. 2.3 методики)."),
    entry("morph", "SWINGING_MIN_LENGTH", "строки эталона", TUNED,
          "Вид «качание»: извилистость следа оценивается только при длине не меньше этой (250 мс)."),
    entry("morph", "SWINGING_WIGGLE", "каналы эталона", TUNED,
          "Вид «качание»: СКО бокового следа от его квадратичного приближения не меньше этого (5 мм)."),
    entry("morph", "ROLLING_EXCURSION", "каналы эталона", TUNED,
          "Вид «перекатывание»: боковой ход следа за время прохода не меньше этого (22,5 мм)."),
    entry("morph", "ROLLING_WIDTH", "каналы эталона", TUNED,
          "Вид «перекатывание»: след такой ширины и шире уже сместился поперёк линии (63 мм)."),
    entry("morph", "LONG_LENGTH", "строки эталона", TUNED,
          "Вид «длинный» (медленный проход): длина не меньше этой (550 мс)."),
    entry("morph", "SHORT_LENGTH", "строки эталона", TUNED,
          "Вид «короткий» (быстрый проход): длина не больше этой (200 мс)."),
    entry("morph", "EDGE_TOLERANCE", "ячейки", EXTERNAL,
          "Рамка ближе этого к краю кадра считается обрезанной краем (TOUCH_TOLERANCE в SensorLine, "
          "sensorline/border.py)."),
    entry("morph", "MIN_KIND_OBJECTS", "объекты", TUNED,
          "Меньше реальных объектов вида — фильтр не сравнивает новые объекты этого вида с реальными."),
    entry("morph", "RANGE_PERCENTILES", "процентили", TUNED,
          "Диапазон признака реальных объектов вида: между этими процентилями (крайние 1 % — выбросы)."),
    entry("morph", "NEAREST_PERCENTILE", "процентиль", TUNED,
          "Типичное расстояние реального объекта до ближайшего реального соседа того же вида (в масштабе признаков)."),
    entry("morph", "TEXTURE_BINS", "доля шкалы", TUNED,
          "Полосы интенсивности, в каждой из которых восстанавливается мелкий шум реальных объектов."),
    entry("morph", "BLOB_RATIO", "длина/ширина", TUNED,
          "Допустимое отношение длины к ширине нового «обычного» объекта, выбрано по реальным: медиана 2,0, "
          "5–95 % между 1,5 и 3,0, не выше 5 (значение по умолчанию, меняется в интерфейсе)."),
    entry("morph", "MAX_VALLEY", "доля", TUNED,
          "След не должен проседать ниже этой доли более слабой из двух сторон: иначе он разорван."),
    entry("morph", "RANGE_MARGIN_BOUND", "доля", TUNED,
          "Фильтр: признак может выйти за реальный диапазон на эту долю границы ..."),
    entry("morph", "RANGE_MARGIN_SPREAD", "доля", TUNED,
          "... плюс эту долю межквартильного размаха признака (запас для признаков около нуля)."),
    entry("morph", "NEAREST_FACTOR", "раз", TUNED,
          "Фильтр: ближайший реальный объект может быть дальше типичного (NEAREST_PERCENTILE) не более чем во "
          "столько раз, иначе объект «не похож на реальные»."),
    entry("morph", "COPY_SIMILARITY", "корреляция", TUNED,
          "Фильтр: при большем сходстве с исходным объектом (после приведения к общему размеру) новый — копия."),
    entry("morph", "NEUTRAL_IMPERFECTIONS", "—", DEFINITION, "Нейтральные значения несовершенств нового объекта."),
    entry("morph", "IMPERFECTION_BOUNDS", "—", BOUND, "Допустимые диапазоны несовершенств нового объекта."),
    entry("morph", "DRAW_ATTEMPTS_MIN", "попытки", BOUND, "Наименьшее число попыток получить новые объекты."),
    entry("morph", "DRAW_ATTEMPTS_PER_OBJECT", "попытки", BOUND, "Попыток на каждый запрошенный новый объект."),
    entry("morph", "SIFT_SHARE", "доля", TUNED,
          "Отсев мелких: доля самых коротких и, отдельно, самых узких новых объектов партии."),
    entry("morph", "SIFT_SHARES", "доля", TUNED, "Отсев мелких по длине и ширине (из SIFT_SHARE)."),
    entry("morph", "FAINT_SIFT_SHARES", "доля", TUNED,
          "Перекатывание и качание: также доля самых малых по площади и самых тусклых."),
    entry("morph", "SIFT_MIN_BATCH", "объекты", TUNED, "Партия меньше этого не отсеивается."),

    # --- motion (motion.py, docs/objects.md, section 6) ----------------------------------------
    entry("motion", "LOOK", "доля; раз", TUNED,
          "Только для реальных объектов: вероятность отражения поперёк ленты, множители ширины и яркости следа."),
    entry("motion", "DURATION_SCALE", "раз", TUNED,
          "Длительность прохода — естественная, умноженная на лог-равномерный множитель из этого диапазона "
          "(обычный, перекатывание, качание)."),
    entry("motion", "DURATION_ROWS", "строки", TUNED,
          "Пределы длительности прохода по видам; для «длинного» и «короткого» — равномерно в пределах, "
          "без множителя."),
    entry("motion", "SPEED_KNOTS", "узлы", TUNED, "Узлов профиля скорости на естественный проход."),
    entry("motion", "SPEED_WOBBLE", "узлы; СКО ln", TUNED,
          "Плавное колебание скорости по узлам: длина корреляции и СКО логарифма скорости."),
    entry("motion", "STALL", "доля; узлы; раз", TUNED,
          "Остановка «длинного» объекта: вероятность, первый замедленный узел (1–6), число узлов, множитель скорости."),
    entry("motion", "WANDER", "строки; каналы", TUNED, "Плавный боковой уход в любом движении: длина корреляции, СКО."),
    entry("motion", "CONTACT_WOBBLE", "строки; доля", TUNED,
          "Плавное колебание контакта (множителя сигнала): длина корреляции, СКО."),
    entry("motion", "CALM", "доля; каналы/строка", TUNED,
          "Спокойные виды: доля собственного бокового движения реального объекта, постоянный косой снос."),
    entry("motion", "ROLLING", "каналы/строка; доли", TUNED,
          "Перекатывание: боковая скорость; вероятности хода «прямо», «поворот», «дуга»; вероятность стоять до "
          "поворота и скорость при этом; место и плавность поворота (доля прохода, строки); колебание сноса; "
          "доля собственного движения."),
    entry("motion", "SWINGING", "каналы; доли; строки", TUNED,
          "Качание: размах (каналы), число качаний за проход, ширина следа, глубина изменения контакта с "
          "качанием, число потерь контакта (0–2), их длина (1–2 строки) и множитель сигнала, доля собственного "
          "движения."),
    entry("motion", "MAX_GAIN", "раз", BOUND, "Наибольший множитель сигнала (контакта) в строке."),
    entry("motion", "REFERENCE_ROW_MS", "мс", MEASURED,
          "Длительность строки эталонной линейки (10 опросов при 200 Гц), в строках которой подобраны диапазоны "
          "движения и пороги видов; на другой линейке они пересчитываются в её строки (LineScale)."),
    entry("motion", "REFERENCE_PITCH_MM", "мм", MEASURED,
          "Шаг каналов эталонной линейки (подбор по реальным яйцам, docs/object_validation.md), в каналах которой "
          "подобраны диапазоны движения и пороги видов."),
    entry("motion", "DIGITS", "знаки", DEFINITION,
          "Движение хранится округлённым до этого числа знаков, чтобы повтор по JSON совпадал с выбранным."),

    # --- generated backgrounds --------------------------------------------------------------
    entry("background", "MAX_SPREAD", "доля шкалы", TUNED,
          "Наибольший размах сгенерированного фона: 3 % шкалы, 7 уровней из 255."),
    entry("background", "SPREAD_PERCENTILES", "процентили", TUNED,
          "Значения фона между этими процентилями укладываются в размах, остальные обрезаются."),
    entry("background", "OBJECT_MARGIN", "пиксели", TUNED,
          "Поле вокруг размеченной рамки, которое тоже не считается пустым: мягкий край объекта."),

    # --- autogenerator ----------------------------------------------------------------------
    entry("generator", "MAX_POOL", "объекты", BOUND, "Наибольший набор объектов автогенератора."),
    entry("generator", "MAX_IMAGES", "кадры", BOUND, "Наибольшая партия кадров за один запрос."),
    entry("generator", "PLACEMENT_ATTEMPTS", "попытки", BOUND, "Попыток разместить один объект в кадре."),
    entry("generator", "ENTRY_FAILURES", "попытки", BOUND,
          "Неудач одного объекта набора, после которых он пропускается для этой вставки."),
    entry("generator", "SCENE_ATTEMPTS_PER_IMAGE", "попытки", BOUND, "Попыток собрать кадр на каждый запрошенный."),
    entry("generator", "BATCH_BUDGET", "объекты", BOUND, "Подготовленных объектов на всю партию."),

    # --- tapes ------------------------------------------------------------------------------
    entry("tape", "MIN_LENGTH", "опросы", BOUND, "Наименьшая длина ленты."),
    entry("tape", "MAX_LENGTH", "опросы", BOUND, "Наибольшая длина ленты."),
    entry("tape", "MAX_OBJECTS", "объекты", BOUND, "Наибольшее число объектов на ленте."),
    entry("tape", "ATTEMPTS", "попытки", BOUND, "Попыток разместить один объект на ленте; иначе он пропускается."),
    entry("tape", "BUDGET", "объекты", BOUND, "Подготовленных объектов на всю ленту."),

    # --- sensor model (docs/sensor_model.md) --------------------------------------------------
    entry("sensor_model", "PROFILE_SCHEMA", "—", DEFINITION, "Версия формата файла профиля датчика."),
    entry("sensor_model", "MAX_ADC", "коды", BOUND, "Наибольший код АЦП в профиле (24 бита)."),
    entry("sensor_model", "POWER_RANGE", "—", BOUND,
          "Допустимый показатель спада отклика $p$; на эталонной линейке подобраны значения 0,91–2,0."),
    entry("sensor_model", "SHIFT_RANGE", "мм", BOUND,
          "Допустимый сдвиг расстояния $d_0$; на эталонной линейке −4,2…8,1 мм."),
    entry("framing", "FRAME_LEVELS", "уровни", DEFINITION,
          "Верх 8-битной шкалы кадра при нормировке тайла: $\\lfloor (x-\\min)/(\\max-\\min) \\cdot 255 \\rfloor$."),
    entry("calibration", "FIT_POWER", "—", BOUND, "Диапазон показателя $p$ при подборе отклика."),
    entry("calibration", "FIT_SHIFT", "мм", BOUND,
          "Диапазон сдвига $d_0$ при подборе; нижняя граница ещё ограничена ближайшим расстоянием замеров."),
    entry("calibration", "FIT_STEPS", "—, мм", TUNED,
          "Шаги трёх проходов сетки (показатель, сдвиг); каждый проход сужает поиск вокруг лучшей точки. "
          "Совпадает с подбором методом Левенберга — Марквардта (2,04 / 3,24 %)."),
    entry("calibration", "NOISE_SD_MAX", "коды", BOUND, "Верх поиска СКО шума до округления в узле."),
    entry("calibration", "BELT_MARGIN", "коды", TUNED,
          "Опрос — пустая лента, если все каналы в пределах этого от уровня записи; объект поднимает сигнал "
          "на десятки и сотни кодов, шум ленты — около 4 кодов."),
    entry("calibration", "BELT_MIN_RUN", "опросы", TUNED, "Самый короткий учитываемый отрезок пустой ленты."),
    entry("calibration", "RHYTHM_LAGS", "опросы", TUNED,
          "Сдвиги автокорреляции шума ленты при подборе; около четырёх периодов эталонной установки."),
    entry("calibration", "RHYTHM_PERIODS", "опросы", BOUND, "Диапазон искомого периода шума ленты."),
    entry("calibration", "RHYTHM_MIN_SHARE", "доля", TUNED,
          "Периодическая составляющая с меньшей долей дисперсии не включается в профиль."),
    entry("sensor_report", "NOISE_REPLICATES", "серии", BOUND, "Смоделированных серий в каждом узле при проверке П2."),
    entry("sensor_report", "REPORT_SEED", "—", DEFINITION, "Зерно моделирования в отчёте проверки, чтобы отчёт повторялся."),
    entry("sensor_report", "LAGS", "опросы", DEFINITION,
          "Сдвиги, на которых отчёт сравнивает автокорреляцию шума ленты с моделью (П4)."),

    # --- objects described by shape (docs/objects.md) ----------------------------------------
    entry("shapes", "ROUNDNESS_RANGE", "—", BOUND,
          "Допустимые показатели скруглённости суперквадрика $e_1$, $e_2$: около 0 — прямые рёбра (брусок), "
          "1 — эллипс, 2 — ромб."),
    entry("shapes", "MAX_SIZE_MM", "мм", BOUND, "Наибольший размер объекта, высота линейки и шаг каналов."),
    entry("shapes", "MAX_SPEED_MM_S", "мм/с", BOUND, "Наибольшая скорость ленты (600 м/мин)."),
    entry("shapes", "MAX_POLL_HZ", "Гц", BOUND, "Наибольшая частота опроса линейки."),
    entry("shapes", "FIELD_WAVES", "волны", TUNED,
          "Число косинусов в гладком случайном поле поверхности: при 64 значения поля близки к нормальным "
          "(доля пятен отличается от заданной не более чем на несколько процентов, тест)."),
    entry("shapes", "FIELD_CLIP", "СКО", DEFINITION,
          "Рельеф шероховатости ограничен этим числом СКО, чтобы поверхность не опускалась к ленте."),
    entry("shapes", "MAX_ROUGHNESS_SHARE", "доля высоты", BOUND,
          "Наибольшее СКО шероховатости в долях высоты объекта: с FIELD_CLIP поверхность не ниже четверти "
          "высоты объекта."),
    entry("shapes", "MAX_MARGIN", "ячейки", BOUND, "Наибольшее поле рамки разметки за пределами объекта."),
    entry("shapes", "TILT_STEP", "мм", DEFINITION,
          "Шаг центральной разности, по которой считается наклон верхней поверхности объекта."),
    entry("shapes", "MAX_TILT_POWER", "—", BOUND, "Наибольший показатель зависимости отражения от наклона."),
    entry("record", "MAX_ROWS", "строки", BOUND, "Наибольшая длина синтетической записи."),
    entry("record", "MAX_RECORD_OBJECTS", "объекты", BOUND, "Наибольшее число объектов в синтетической записи."),
    entry("record", "PLACEMENT_TRIES", "попытки", BOUND,
          "Попыток разместить один объект случайной записи (каждая — с новой формой, движением и положением)."),

    # --- objects against the real ones (object_report.py, docs/object_validation.md) ----------------
    entry("object_report", "LIFT_SHARE", "доля пика", TUNED,
          "Канал или строка пятна считается поднятым выше этой доли его пика (ширина и длина пятна)."),
    entry("object_report", "EGG_LENGTH", "мм", EXTERNAL,
          "Длина модельного яйца: типичная для куриного яйца (яйца эталонных данных не измерялись)."),
    entry("object_report", "EGG_WIDTH", "мм", EXTERNAL,
          "Ширина модельного яйца, она же высота (яйцо лежит на боку): типичная для куриного яйца."),
    entry("object_report", "EGG_YAW", "градусы", TUNED, "Наибольший поворот модельного яйца на ленте."),
    entry("object_report", "MODEL_EGGS", "яйца", BOUND, "Модельных яиц в каждой точке сетки подбора."),
    entry("object_report", "REPORT_SEED", "—", DEFINITION,
          "Зерно модельных яиц: одни и те же яйца во всех точках сетки и при повторном запуске."),
    entry("object_report", "PITCH_GRID", "мм", BOUND, "Сетка шага каналов при подборе."),
    entry("object_report", "TILT_GRID", "—", BOUND, "Сетка показателя зависимости отражения от наклона."),
    entry("object_report", "START_SPEED", "мм/с", TUNED,
          "Скорость ленты при подборе шага и наклона (затем уточняется по длине пятна)."),
    entry("object_report", "HEIGHTS_SHOWN", "мм", DEFINITION,
          "Высоты линейки, для которых отчёт показывает нужную отражательную способность яйца."),
)


def is_numeric(value):
    """A number, or a non-empty tuple, list or dict of numbers (booleans are not numbers)."""
    if isinstance(value, bool):
        return False
    if isinstance(value, numbers.Real):
        return True
    if isinstance(value, (tuple, list)):
        return bool(value) and all(is_numeric(item) for item in value)
    if isinstance(value, dict):
        return bool(value) and all(is_numeric(item) for item in value.values())
    return False


def module_constants(module):
    """Names of the numeric constants assigned at the top level of a core module (not imported)."""
    path = Path(__file__).with_name(f"{module}.py")
    loaded = importlib.import_module(f"{__package__}.{module}")
    names = []
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        for target in targets:
            if isinstance(target, ast.Name) and target.id.isupper() and is_numeric(getattr(loaded, target.id)):
                names.append(target.id)
    return names


def unregistered():
    """(module, name) of numeric constants with no entry, and of entries that name no constant."""
    found = {(module, name) for module in CORE_MODULES for name in module_constants(module)}
    listed = {(item.module, item.name) for item in REGISTRY}
    return sorted(found - listed), sorted(listed - found)


def show(value):
    if isinstance(value, dict):
        return ", ".join(f"{key} {show(item)}" for key, item in value.items())
    if isinstance(value, (tuple, list)) and any(isinstance(item, (tuple, list)) for item in value):
        return ", ".join("(" + "; ".join(show(part) for part in item) + ")" for item in value)
    if isinstance(value, (tuple, list)):
        return "…".join(show(item) for item in value) if len(value) == 2 else ", ".join(show(item) for item in value)
    if isinstance(value, float):
        return f"{value:g}".replace(".", ",")
    return str(value)


def markdown():
    """The registry as a Markdown document (docs/constants.md)."""
    lines = [
        "# Реестр констант",
        "",
        "Все фиксированные числа программы: значение, единица, происхождение и смысл. Документ создан командой",
        "`python -m ir_line_synth constants --output docs/constants.md`; тесты проверяют, что он совпадает с кодом и что в коде",
        "нет числовых констант без описания.",
        "",
        "Происхождение: **измерено** — посчитано по реальным данным, источник назван; **подобрано** — выбрано",
        "вручную на эталонных данных и проверено просмотром и тестами; **внешнее** — взято из другой системы для",
        "согласованности; **ограничение** — предел объёма работы или размера; **определение** — задано форматом",
        "или смыслом параметра.",
        "",
        f"Значения в каналах и опросах подобраны на эталонном наборе ({REFERENCE}). Пороги видов объектов и "
        "диапазоны движения даны в строках и каналах эталонной линейки (строка 50 мс, шаг 9 мм); на другой "
        "линейке они пересчитываются в её строки и каналы с той же длительностью и тем же расстоянием "
        "(`motion.LineScale`; профиль линейки набора — ключ `sensor_profile` в data.yaml).",
        "",
    ]
    for module, items in groupby(REGISTRY, key=lambda item: item.module):
        lines += [f"## {module}.py", "", "| Константа | Значение | Единица | Происхождение | Смысл |",
                  "|---|---|---|---|---|"]
        lines += [f"| `{item.name}` | {show(item.value)} | {item.unit} | {item.origin} | {item.meaning} |"
                  for item in items]
        lines.append("")
    return "\n".join(lines)
