"""The model of a line of reflective IR distance sensors (docs/sensor_model.md).

A line is described by a *sensor profile*, a YAML file: the response of every channel
to distance and reflectance, the noise of the sensor, the range of its ADC, an optional
correction table of the firmware, how frames are formed and, optionally, the noise and
belt level measured in a working installation. Every step is a separate function:

    response.signal(d, rho)  ->  noise  ->  adc  ->  correct  ->  framing.frames_from_polls

A profile is made from calibration captures (calibration.fit_profile), from nominal
values with a spread between channels, or by hand.
"""

from dataclasses import dataclass, field
import math
from pathlib import Path

import numpy as np
import yaml

from .scene import integer, number


PROFILE_SCHEMA = 1
LAWS = ("power_offset",)
NORMALIZATIONS = ("tile_minmax", "none")
# Profiles shipped with the program, by name (profiles/<name>.yaml).
PROFILES = Path(__file__).with_name("profiles")
PROFILE_KEYS = {"profile", "name", "channels", "boards", "adc_max", "response", "noise", "operation",
                "correction", "frames", "source"}
CHANNEL_KEYS = ("gain", "power", "shift_mm", "offset")
# Bounds that keep a profile physical and its arithmetic finite.
MAX_ADC = 2 ** 24 - 1
POWER_RANGE = (0.05, 6.0)
SHIFT_RANGE = (-50.0, 200.0)


@dataclass(frozen=True)
class Response:
    """s_c(d, rho) = offset_c + rho * gain_c * (d + shift_c) ** -power_c, in codes (d in mm)."""

    gain: np.ndarray
    power: np.ndarray
    shift: np.ndarray
    offset: np.ndarray
    valid_mm: tuple = (0.0, math.inf)

    def reflected(self, distance, reflectance=1.0):
        """The reflected part of the signal, codes, for distances [..., channels]."""
        distance = np.asarray(distance, np.float64)
        base = distance + self.shift
        if np.any(base <= 0):
            raise ValueError("A distance is too short for the response law (d + shift <= 0)")
        return np.asarray(reflectance, np.float64) * self.gain * base ** -self.power

    def signal(self, distance, reflectance=1.0, offset=None):
        """The noise-free signal of every channel, codes (floating point)."""
        base = self.offset if offset is None else np.asarray(offset, np.float64)
        return base + self.reflected(distance, reflectance)

    def distance(self, signal, reflectance=1.0, offset=None):
        """The distance that gives `signal` (the inverse of `signal`), mm."""
        base = self.offset if offset is None else np.asarray(offset, np.float64)
        excess = (np.asarray(signal, np.float64) - base) / (np.asarray(reflectance, np.float64) * self.gain)
        if np.any(excess <= 0):
            raise ValueError("A signal at or below the offset of its channel has no distance")
        return excess ** (-1.0 / self.power) - self.shift


@dataclass(frozen=True)
class SensorNoise:
    """White noise of the sensor before rounding: variance scale_c * (a + k * signal), codes squared.

    `scale` is the variance factor of every channel (1 for all when the channels are alike).
    """

    a: float
    k: float
    scale: np.ndarray | None = None

    def sd(self, signal, channel=None):
        """The sd for signals [..., channels], or for signals of one `channel`."""
        variance = np.maximum(self.a + self.k * np.asarray(signal, np.float64), 0.0)
        if self.scale is not None:
            variance = variance * (self.scale if channel is None else self.scale[channel])
        return np.sqrt(variance)

    def sample(self, signal, rng):
        signal = np.asarray(signal, np.float64)
        return signal + self.sd(signal) * rng.standard_normal(signal.shape)


@dataclass(frozen=True)
class OperationNoise:
    """The bare belt of a working installation: level of every channel and the noise around it.

    The noise of channel c in poll t, standard deviation `sd` in all:
      sqrt(2 * periodic_share) * sd * sin(2 pi t / period + phase)          common to the line
    + sqrt(1 - periodic_share) * sd * (sqrt(board_share) * g[board(c), t] + sqrt(1 - board_share) * e[c, t]),
    where g and e are first-order autoregressive processes of unit variance with correlation
    `lag1` between consecutive polls, and the phase is random.
    """

    belt_level: np.ndarray
    sd: float
    lag1: float
    board_share: float
    period: float = 0.0
    periodic_share: float = 0.0

    def autocorrelation(self, lag):
        """The correlation of the noise of a channel `lag` polls apart."""
        periodic = self.periodic_share * math.cos(2 * math.pi * lag / self.period) if self.period else 0.0
        return periodic + (1 - self.periodic_share) * self.lag1 ** lag

    def sample(self, polls, boards, rng):
        """Noise [polls, channels] of a line whose channels form `boards` equal boards."""
        channels = len(self.belt_level)
        per_board = channels // boards
        series = np.empty((polls, boards + channels))
        innovation = math.sqrt(1 - self.lag1 ** 2)
        state = rng.standard_normal(boards + channels)
        for t in range(polls):
            state = self.lag1 * state + innovation * rng.standard_normal(boards + channels)
            series[t] = state
        common = np.repeat(series[:, :boards], per_board, axis=1)
        own = series[:, boards:]
        drift = math.sqrt(self.board_share) * common + math.sqrt(1 - self.board_share) * own
        noise = math.sqrt(1 - self.periodic_share) * drift
        if self.periodic_share and self.period:
            phase = rng.uniform(0, 2 * math.pi)
            wave = np.sin(2 * math.pi * np.arange(polls) / self.period + phase)
            noise = noise + math.sqrt(2 * self.periodic_share) * wave[:, None]
        return self.sd * noise


@dataclass(frozen=True)
class FrameSpec:
    """How frames are formed from polls (framing.py)."""

    pool: int          # polls averaged into one sample
    length: int        # samples in a frame
    normalize: str     # tile_minmax or none


@dataclass(frozen=True)
class SensorModel:
    name: str
    channels: int
    boards: int
    adc_max: int
    response: Response
    noise: SensorNoise
    frames: FrameSpec
    operation: OperationNoise | None = None
    correction: np.ndarray | None = None    # [adc_max + 1, channels] corrected code of every raw code
    source: dict = field(default_factory=dict)

    def adc(self, signal):
        """Integer codes: rounded and limited to the range of the ADC."""
        return np.clip(np.round(np.asarray(signal, np.float64)), 0, self.adc_max).astype(np.int64)

    def measure(self, distance, reflectance, rng):
        """Codes [..., channels] of surfaces at `distance` (mm) with `reflectance`, as on the calibration stand."""
        return self.adc(self.noise.sample(self.response.signal(distance, reflectance), rng))

    def empty_belt(self, polls, rng):
        """Codes [polls, channels] of the bare belt of the installation of the profile."""
        if self.operation is None:
            raise ValueError(f"The profile {self.name} has no operation section (bare belt of an installation)")
        return self.adc(self.operation.belt_level + self.operation.sample(polls, self.boards, rng))

    def correct(self, codes):
        """Codes after the correction table of the firmware (unchanged without one)."""
        codes = np.asarray(codes, np.int64)
        if self.correction is None:
            return codes
        return self.correction[np.clip(codes, 0, self.adc_max), np.arange(self.channels)]


# --------------------------------------------------------------------------- profile files

def keys(value, allowed, where):
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be a mapping")
    unknown = set(value) - set(allowed)
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}")
    return value


def channel_values(rows, channels):
    if not isinstance(rows, list) or len(rows) != channels:
        raise ValueError(f"response.channels must list {channels} channels")
    table = []
    for index, row in enumerate(rows):
        keys(row, CHANNEL_KEYS, f"response.channels[{index}]")
        table.append([number(row.get(key), f"channel {index} {key}", -1e12, 1e12) for key in CHANNEL_KEYS])
    return np.asarray(table, np.float64).T


def spread_values(nominal, spread, channels):
    """Channel parameters drawn around nominal values from the seed of the spread."""
    keys(nominal, CHANNEL_KEYS, "response.nominal")
    keys(spread, {"gain_log_sd", "power_sd", "offset_sd", "seed"}, "response.spread")
    base = [number(nominal.get(key), f"nominal {key}", -1e12, 1e12) for key in CHANNEL_KEYS]
    rng = np.random.default_rng(integer(spread.get("seed", 0), "spread seed", 0, 2**32 - 1))
    gain_sd = number(spread.get("gain_log_sd", 0.0), "gain_log_sd", 0, 2)
    power_sd = number(spread.get("power_sd", 0.0), "power_sd", 0, 2)
    offset_sd = number(spread.get("offset_sd", 0.0), "offset_sd", 0, 1e6)
    gain = base[0] * np.exp(gain_sd * rng.standard_normal(channels))
    power = base[1] + power_sd * rng.standard_normal(channels)
    shift = np.full(channels, base[2])
    offset = base[3] + offset_sd * rng.standard_normal(channels)
    return np.stack([gain, power, shift, offset])


def response_from(spec, channels):
    keys(spec, {"law", "valid_mm", "channels", "nominal", "spread"}, "response")
    if spec.get("law") not in LAWS:
        raise ValueError(f"response.law must be one of {', '.join(LAWS)}")
    if ("channels" in spec) == ("nominal" in spec):
        raise ValueError("response needs either channels (per channel) or nominal (with an optional spread)")
    if "channels" in spec:
        gain, power, shift, offset = channel_values(spec["channels"], channels)
    else:
        gain, power, shift, offset = spread_values(spec["nominal"], spec.get("spread", {}), channels)
    valid = spec.get("valid_mm", [0, math.inf])
    if not isinstance(valid, list) or len(valid) != 2:
        raise ValueError("response.valid_mm: [shortest, longest] in mm")
    low, high = (float(value) for value in valid)
    if not 0 <= low < high:
        raise ValueError("response.valid_mm: 0 <= shortest < longest")
    if np.any(gain <= 0):
        raise ValueError("Every channel needs a positive gain")
    if np.any(power < POWER_RANGE[0]) or np.any(power > POWER_RANGE[1]):
        raise ValueError(f"Channel powers must lie in {POWER_RANGE}")
    if np.any(shift < SHIFT_RANGE[0]) or np.any(shift > SHIFT_RANGE[1]) or np.any(low + shift <= 0):
        raise ValueError("Channel shifts must keep d + shift positive over valid_mm")
    return Response(gain, power, shift, offset, (low, high))


def correction_from(spec, channels, adc_max, folder):
    if spec in (None, "none"):
        return None
    keys(spec, {"table"}, "correction")
    path = Path(folder) / str(spec.get("table"))
    table = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    if table.shape != (adc_max + 1, channels + 1) or not np.array_equal(table[:, 0], np.arange(adc_max + 1)):
        raise ValueError(f"{path}: expected rows 0..{adc_max} (a raw code each) and {channels} channel columns")
    return table[:, 1:].astype(np.int64)


def operation_from(spec, channels):
    if spec is None:
        return None
    keys(spec, {"belt_level", "sd", "lag1", "board_share", "period", "periodic_share"}, "operation")
    level = spec.get("belt_level")
    if not isinstance(level, list) or len(level) != channels:
        raise ValueError(f"operation.belt_level must list {channels} levels")
    level = np.asarray([number(value, "belt level", 0, MAX_ADC) for value in level], np.float64)
    return OperationNoise(level, number(spec.get("sd"), "operation sd", 0, 1e6),
                          number(spec.get("lag1"), "operation lag1", 0, 0.9999),
                          number(spec.get("board_share"), "operation board_share", 0, 1),
                          number(spec.get("period", 0.0), "operation period", 0, 1e6),
                          number(spec.get("periodic_share", 0.0), "operation periodic_share", 0, 1))


def model_from(spec, folder=".", source_name="profile"):
    """A checked SensorModel from the content of a profile file (relative files from `folder`)."""
    keys(spec, PROFILE_KEYS, source_name)
    if spec.get("profile") != PROFILE_SCHEMA:
        raise ValueError(f"{source_name}: expected profile: {PROFILE_SCHEMA}")
    name = spec.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{source_name}: name must be a text")
    channels = integer(spec.get("channels"), "channels", 1, 1024)
    boards = integer(spec.get("boards", 1), "boards", 1, channels)
    if channels % boards:
        raise ValueError("channels must divide into equal boards")
    adc_max = integer(spec.get("adc_max"), "adc_max", 1, MAX_ADC)
    noise = keys(spec.get("noise", {}), {"variance", "channel_scale"}, "noise")
    variance = keys(noise.get("variance", {"a": 0, "k": 0}), {"a", "k"}, "noise.variance")
    scale = noise.get("channel_scale")
    if scale is not None:
        if not isinstance(scale, list) or len(scale) != channels:
            raise ValueError(f"noise.channel_scale must list {channels} factors")
        scale = np.asarray([number(value, "noise channel_scale", 0, 1e3) for value in scale], np.float64)
    frames = keys(spec.get("frames"), {"pool", "length", "normalize"}, "frames")
    if frames.get("normalize") not in NORMALIZATIONS:
        raise ValueError(f"frames.normalize must be one of {', '.join(NORMALIZATIONS)}")
    source = spec.get("source", {})
    if not isinstance(source, dict):
        raise ValueError("source must be a mapping")
    return SensorModel(
        name=name, channels=channels, boards=boards, adc_max=adc_max,
        response=response_from(spec.get("response"), channels),
        noise=SensorNoise(number(variance.get("a", 0), "noise a", 0, 1e9), number(variance.get("k", 0), "noise k", 0, 1e3),
                          scale),
        frames=FrameSpec(integer(frames.get("pool"), "frames.pool", 1, 10000),
                         integer(frames.get("length"), "frames.length", 1, 100000), frames["normalize"]),
        operation=operation_from(spec.get("operation"), channels),
        correction=correction_from(spec.get("correction", "none"), channels, adc_max, folder),
        source=source)


def load_profile(path_or_name):
    """A sensor profile from a YAML file, or one shipped with the program by its name."""
    path = Path(path_or_name)
    if not path.suffix and not path.exists():
        path = PROFILES / f"{path_or_name}.yaml"
    if not path.is_file():
        shipped = ", ".join(sorted(item.stem for item in PROFILES.glob("*.yaml")))
        raise ValueError(f"No sensor profile {path_or_name} (shipped profiles: {shipped})")
    try:
        spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"{path}: {error}") from error
    return model_from(spec, path.parent, str(path))


def rounded(value, digits=6):
    return float(f"{float(value):.{digits}g}")


def profile_dict(model):
    """The content of a profile file for a model with per-channel parameters."""
    response = model.response
    spec = dict(
        profile=PROFILE_SCHEMA, name=model.name, channels=model.channels, boards=model.boards,
        adc_max=model.adc_max,
        response=dict(law="power_offset", valid_mm=[rounded(v) for v in response.valid_mm], channels=[
            dict(gain=rounded(g), power=rounded(p), shift_mm=rounded(s), offset=rounded(o))
            for g, p, s, o in zip(response.gain, response.power, response.shift, response.offset)]),
        noise=dict(variance=dict(a=rounded(model.noise.a), k=rounded(model.noise.k))),
        correction="none",
        frames=dict(pool=model.frames.pool, length=model.frames.length, normalize=model.frames.normalize),
    )
    if model.operation is not None:
        operation = model.operation
        spec["operation"] = dict(belt_level=[rounded(v, 5) for v in operation.belt_level],
                                 sd=rounded(operation.sd), lag1=rounded(operation.lag1),
                                 board_share=rounded(operation.board_share), period=rounded(operation.period),
                                 periodic_share=rounded(operation.periodic_share))
    if model.noise.scale is not None:
        spec["noise"]["channel_scale"] = [rounded(v, 4) for v in model.noise.scale]
    if model.correction is not None:
        raise ValueError("A profile with a correction table is written by hand (it names the table file)")
    if model.source:
        spec["source"] = model.source
    return spec


def save_profile(model, path, header=""):
    # Flow style for the leaves: a channel, a list of levels or of files is one line.
    text = yaml.safe_dump(profile_dict(model), sort_keys=False, allow_unicode=True, width=120, default_flow_style=None)
    Path(path).write_text(header + text, encoding="utf-8", newline="\n")
