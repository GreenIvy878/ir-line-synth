"""A sensor profile from measurements: calibration captures and recordings of a bare belt.

Calibration captures are readings of every channel with a flat white target at known
distances. Two layouts are read:

- a folder of `data_<distance in mm>.csv`, one file per distance: a header, then a row per
  sample, the first column a sample number and then one column per channel;
- one CSV table with the columns `channel` (from 0), `distance_mm` and `value`, a row per sample.

The fit (docs/sensor_model.md):
- response: for every channel, s = offset + gain * (d + shift) ** -power by least squares
  on relative residuals over the nodes without saturation; power and shift by a grid that
  is refined twice, gain and offset exactly for every grid point;
- sensor noise: for every node, the standard deviation of the noise *before rounding* that
  gives the measured spread of the integer samples, then variance = scale_c * (a + k * signal)
  with a variance factor for every channel;
- operation (optional): the level of every channel on the bare belt of working recordings
  and the structure of the noise around it: its sd, a periodic part common to the line
  (period and share of variance) and a slow part (correlation of consecutive polls, share
  common to the channels of a board), from the autocorrelation of the belt noise.
"""

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re

import numpy as np

from .sensor_model import OperationNoise, Response, SensorModel, SensorNoise


# Grid of the response fit: power and shift ranges, and the steps of the three passes.
FIT_POWER = (0.3, 4.0)
FIT_SHIFT = (-4.5, 20.0)
FIT_STEPS = ((0.02, 0.25), (0.002, 0.025), (0.0002, 0.0025))
# Noise standard deviations searched when inverting the rounding (codes).
NOISE_SD_MAX = 20.0
# Bare belt: polls within this many codes of the belt level on every channel, in runs at least this long.
BELT_MARGIN = 12
BELT_MIN_RUN = 40
# The rhythm of the belt noise: lags of the autocorrelation fitted, the periods searched
# (polls), and the smallest share of variance for which a periodic part is kept.
RHYTHM_LAGS = 160
RHYTHM_PERIODS = (6.0, 200.0)
RHYTHM_MIN_SHARE = 0.02


@dataclass(frozen=True)
class Calibration:
    distances: np.ndarray       # [nodes] mm
    samples: tuple              # per node: [samples, channels] codes
    clip: int                   # a sample at this code or above is saturated
    files: tuple = ()           # the files read, for the source of a profile

    @property
    def channels(self):
        return self.samples[0].shape[1]

    @property
    def means(self):
        return np.stack([node.mean(0) for node in self.samples])

    @property
    def variances(self):
        return np.stack([node.var(0, ddof=1) for node in self.samples])

    @property
    def saturated(self):
        return np.stack([(node >= self.clip).any(0) for node in self.samples])


def read_captures(path, clip):
    """Calibration captures from a folder of data_<mm>.csv files or from one long table."""
    path = Path(path)
    if path.is_dir():
        distance_of = lambda item: float(item.stem.split("_", 1)[1])
        files = sorted(path.glob("data_*.csv"), key=distance_of)
        if not files:
            raise ValueError(f"{path}: no data_<mm>.csv files")
        nodes = [np.loadtxt(item, delimiter=",", skiprows=1, ndmin=2)[:, 1:] for item in files]
        distances = [distance_of(item) for item in files]
    else:
        files = [path]
        table = np.genfromtxt(path, delimiter=",", names=True)
        if set(table.dtype.names) != {"channel", "distance_mm", "value"}:
            raise ValueError(f"{path}: expected the columns channel, distance_mm, value")
        channels = int(table["channel"].max()) + 1
        distances, nodes = [], []
        for distance in np.unique(table["distance_mm"]):
            rows = table[table["distance_mm"] == distance]
            counts = np.bincount(rows["channel"].astype(int), minlength=channels)
            if counts.min() != counts.max():
                raise ValueError(f"{path}: distance {distance:g} mm has unequal samples per channel")
            order = np.lexsort((np.arange(len(rows)), rows["channel"]))
            nodes.append(rows["value"][order].reshape(channels, counts[0]).T)
            distances.append(float(distance))
    if len({node.shape[1] for node in nodes}) != 1 or any(len(node) < 2 for node in nodes):
        raise ValueError(f"{path}: every distance needs the same channels and at least two samples")
    return Calibration(np.asarray(distances), tuple(np.asarray(node, np.float64) for node in nodes), int(clip),
                       tuple(files))


# --------------------------------------------------------------------------- response

def fit_channel(distances, signal):
    """(gain, power, shift, offset) of one channel by relative least squares."""
    weight = 1.0 / signal

    def best(powers, shifts):
        X = (distances[None, None, :] + shifts[None, :, None]) ** -powers[:, None, None]
        w2 = weight ** 2
        s00, s01, s11 = w2.sum(), (w2 * X).sum(-1), (w2 * X * X).sum(-1)
        t0, t1 = (w2 * signal).sum(), (w2 * X * signal).sum(-1)
        det = s00 * s11 - s01 ** 2
        det = np.where(np.abs(det) > 1e-300, det, np.nan)
        offset = (t0 * s11 - s01 * t1) / det
        gain = (s00 * t1 - s01 * t0) / det
        error = (((offset[..., None] + gain[..., None] * X - signal) * weight) ** 2).sum(-1)
        error = np.where((gain > 0) & np.isfinite(error), error, np.inf)
        i, j = np.unravel_index(np.argmin(error), error.shape)
        return powers[i], shifts[j], gain[i, j], offset[i, j], error[i, j]

    low_shift = max(FIT_SHIFT[0], 0.5 - distances.min())
    power_range, shift_range = FIT_POWER, (low_shift, FIT_SHIFT[1])
    for power_step, shift_step in FIT_STEPS:
        powers = np.arange(power_range[0], power_range[1] + power_step / 2, power_step)
        shifts = np.arange(shift_range[0], shift_range[1] + shift_step / 2, shift_step)
        power, shift, gain, offset, _ = best(powers, shifts)
        power_range = (max(FIT_POWER[0], power - 5 * power_step), min(FIT_POWER[1], power + 5 * power_step))
        shift_range = (max(low_shift, shift - 5 * shift_step), min(FIT_SHIFT[1], shift + 5 * shift_step))
    return float(gain), float(power), float(shift), float(offset)


def fit_response(calibration, exclude=None):
    """The response of every channel; `exclude` (a node index) is left out of the fit."""
    means, saturated = calibration.means, calibration.saturated
    use = ~saturated
    if exclude is not None:
        use[exclude] = False
    parameters = []
    for channel in range(calibration.channels):
        nodes = use[:, channel]
        if nodes.sum() < 5:
            raise ValueError(f"Channel {channel} has fewer than 5 distances without saturation")
        parameters.append(fit_channel(calibration.distances[nodes], means[nodes, channel]))
    gain, power, shift, offset = (np.asarray(column) for column in zip(*parameters))
    valid = calibration.distances[~saturated.all(1)]
    return Response(gain, power, shift, offset, (float(valid.min()), float(valid.max())))


# --------------------------------------------------------------------------- sensor noise

def normal_cdf(values):
    return 0.5 * (1.0 + np.vectorize(math.erf)(np.asarray(values) / math.sqrt(2.0)))


def rounded_variance(mean, sd):
    """Variance of round(mean + sd * z), z standard normal: what the ADC makes of the noise."""
    if sd <= 0:
        return 0.0
    reach = int(math.ceil(8 * sd)) + 2
    codes = np.arange(math.floor(mean) - reach, math.ceil(mean) + reach + 1, dtype=np.float64)
    probability = normal_cdf((codes + 0.5 - mean) / sd) - normal_cdf((codes - 0.5 - mean) / sd)
    expected = (probability * codes).sum()
    return float((probability * (codes - expected) ** 2).sum())


def noise_before_rounding(mean, variance):
    """The sd of the noise before rounding whose rounded samples have `variance` (bisection)."""
    if variance <= 0:
        return 0.0
    low, high = 0.0, NOISE_SD_MAX
    for _ in range(50):
        middle = (low + high) / 2
        if rounded_variance(mean, middle) < variance:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def variance_law(x, y):
    """a, k >= 0 of y = a + k * x by least squares."""
    k, a = np.polyfit(x, y, 1)
    if k < 0:
        k, a = 0.0, float(y.mean())
    if a < 0:
        a, k = 0.0, float((x * y).sum() / (x * x).sum())
    return float(a), float(k)


def fit_noise(calibration):
    """SensorNoise and the sd before rounding of every node [nodes, channels].

    The law a + k * signal is fitted on all nodes, the factor of a channel is the mean of its
    variances over the mean of the law at its nodes (median factor 1), and the law is fitted
    again on the variances divided by the factors.
    """
    means, variances, saturated = calibration.means, calibration.variances, calibration.saturated
    sd = np.full(means.shape, np.nan)
    for node, channel in zip(*np.nonzero(~saturated)):
        sd[node, channel] = noise_before_rounding(means[node, channel], variances[node, channel])
    use = ~np.isnan(sd)
    a, k = variance_law(means[use], sd[use] ** 2)
    law = np.where(use, a + k * means, np.nan)
    scale = np.nansum(np.where(use, sd ** 2, np.nan), 0) / np.nansum(law, 0)
    scale = scale / np.median(scale)
    a, k = variance_law(means[use], (sd ** 2 / scale)[use])
    return SensorNoise(a, k, scale), sd


# --------------------------------------------------------------------------- operation

def numbered(folder, only=None):
    """CSV files of a folder in the order of the number at the end of their names, `only` those numbers."""
    files = []
    for path in Path(folder).glob("*.csv"):
        found = re.search(r"(\d+)$", path.stem)
        if found and (only is None or int(found.group(1)) in only):
            files.append((int(found.group(1)), path))
    if not files:
        raise ValueError(f"{folder}: no recordings" + (" with these numbers" if only else ""))
    return [path for _, path in sorted(files)]


def number_ranges(text):
    """A set of numbers from text like 2-22,25."""
    result = set()
    for part in str(text).split(","):
        if "-" in part:
            first, last = (int(value) for value in part.split("-", 1))
            result |= set(range(first, last + 1))
        elif part.strip():
            result.add(int(part))
    return result


def belt_runs(polls, level):
    """Runs of consecutive bare-belt polls (every channel within BELT_MARGIN of its level)."""
    bare = (np.abs(polls - level) <= BELT_MARGIN).all(1)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], bare.astype(int), [0]])))
    return [polls[start:end] for start, end in zip(edges[::2], edges[1::2]) if end - start >= BELT_MIN_RUN]


@dataclass(frozen=True)
class Belt:
    """The bare belt of a set of recordings."""
    levels: np.ndarray      # [recordings, channels] median code of every channel in every recording
    runs: tuple             # runs of bare-belt polls minus the level of their recording, [polls, channels]
    sd: float
    lag1: float
    board_share: float

    def autocorrelation(self, lag):
        """Correlation of the belt noise `lag` polls apart (over the runs longer than `lag`)."""
        pairs = [(run[lag:] * run[:-lag]).sum() for run in self.runs if len(run) > lag]
        powers = [(run[lag:] ** 2).sum() for run in self.runs if len(run) > lag]
        return float(sum(pairs) / sum(powers)) if powers else math.nan


def belt_statistics(recordings, boards):
    """The bare belt of recordings [polls, channels]: levels and the structure of the noise."""
    levels, runs = [], []
    for polls in recordings:
        level = np.median(polls, axis=0)
        levels.append(level)
        runs += [run - level for run in belt_runs(polls, level)]
    if not runs:
        raise ValueError("No bare belt found in the recordings")
    channels = runs[0].shape[1]
    values = np.concatenate(runs)
    variance = float((values ** 2).mean())
    lag1 = float(sum((run[1:] * run[:-1]).sum() for run in runs) / sum((run[1:] ** 2).sum() for run in runs))
    board_means = values.reshape(len(values), boards, channels // boards).mean(2)
    share = ((board_means ** 2).mean() / variance - boards / channels) / (1 - boards / channels)
    return Belt(np.stack(levels), tuple(runs), math.sqrt(variance), min(max(lag1, 0.0), 0.9999),
                min(max(float(share), 0.0), 1.0))


def fit_rhythm(belt):
    """(period, periodic share, lag1) of the autocorrelation s cos(2 pi l / T) + (1 - s) lag1 ** l.

    For every period T and slow correlation lag1 of a grid, the share s is exact by least
    squares; a share below RHYTHM_MIN_SHARE means no periodic part (period 0).
    """
    lags = np.arange(1, RHYTHM_LAGS + 1, dtype=np.float64)
    measured = np.asarray([belt.autocorrelation(int(lag)) for lag in lags])
    keep = ~np.isnan(measured)
    lags, measured = lags[keep], measured[keep]
    periods = np.arange(RHYTHM_PERIODS[0], RHYTHM_PERIODS[1] + 0.125, 0.25)
    correlations = 1 - np.geomspace(1e-4, 0.5, 300)
    slow = correlations[:, None] ** lags[None, :]                          # [lag1, lags]
    target = measured[None, :] - slow
    best = (math.inf, 0.0, 0.0, 0.0)
    for start in range(0, len(periods), 32):
        chunk = periods[start:start + 32]
        wave = np.cos(2 * np.pi * lags[None, :] / chunk[:, None])          # [periods, lags]
        difference = wave[:, None, :] - slow[None, :, :]                   # [periods, lag1, lags]
        share = np.clip((target[None] * difference).sum(-1) / np.maximum((difference ** 2).sum(-1), 1e-12), 0, 1)
        error = ((target[None] - share[..., None] * difference) ** 2).sum(-1)
        i, j = np.unravel_index(np.argmin(error), error.shape)
        if error[i, j] < best[0]:
            best = (float(error[i, j]), float(chunk[i]), float(share[i, j]), float(correlations[j]))
    _, period, share, lag1 = best
    if share < RHYTHM_MIN_SHARE:
        no_wave = ((measured[None, :] - slow) ** 2).sum(-1)
        return 0.0, 0.0, float(correlations[int(np.argmin(no_wave))])
    return period, share, lag1


def operation_parameters(belt):
    """The parameters of OperationNoise (without the belt level) from a Belt."""
    period, share, lag1 = fit_rhythm(belt)
    # The periodic part is common to the whole line, so it is common to every board too.
    board_share = min(max((belt.board_share - share) / (1 - share), 0.0), 1.0) if share < 1 else 0.0
    return dict(sd=belt.sd, lag1=lag1, board_share=board_share, period=period, periodic_share=share)


def fit_operation(recordings, boards):
    belt = belt_statistics(recordings, boards)
    return OperationNoise(np.median(belt.levels, axis=0), **operation_parameters(belt))


# --------------------------------------------------------------------------- profile

def digest(files):
    hasher = hashlib.sha256()
    for path in files:
        hasher.update(Path(path).read_bytes())
    return hasher.hexdigest()


def fit_profile(calibration, name, boards, adc_max, frames, recordings=None, recording_files=()):
    """A SensorModel fitted to calibration captures and, optionally, bare-belt recordings."""
    response = fit_response(calibration)
    noise, _ = fit_noise(calibration)
    source = dict(
        calibration=dict(files=[Path(path).name for path in calibration.files], sha256=digest(calibration.files),
                         distances_mm=[float(d) for d in calibration.distances],
                         samples_per_distance=int(len(calibration.samples[0])), saturation=int(calibration.clip)))
    operation = None
    if recordings:
        operation = fit_operation(recordings, boards)
        source["recordings"] = dict(files=[Path(path).name for path in recording_files],
                                    sha256=digest(recording_files), polls=int(sum(len(r) for r in recordings)))
    return SensorModel(name, calibration.channels, boards, int(adc_max), response, noise, frames, operation,
                       None, source)
