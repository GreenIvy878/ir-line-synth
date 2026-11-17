"""The sensor model: response, noise, ADC, correction, frames; the fit of a profile to
measurements recovers a known line; the check report and the commands."""

import contextlib
import copy
import csv
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import yaml

from ir_line_synth.__main__ import main
from ir_line_synth.calibration import (
    belt_statistics, fit_noise, fit_operation, fit_profile, fit_response, number_ranges, read_captures,
    rounded_variance,
)
from ir_line_synth.framing import frames_from_polls, normalize, read_recording, tiles
from ir_line_synth.sensor_model import (
    FrameSpec, OperationNoise, Response, SensorModel, SensorNoise, load_profile, model_from, profile_dict,
)
from ir_line_synth.sensor_report import check_frames, check_noise, report


DISTANCES = np.arange(5.0, 101.0, 5.0)


def line(channels=8, boards=2, seed=3):
    """A known line: channels differ in gain, power, shift, offset and noise."""
    rng = np.random.default_rng(seed)
    response = Response(gain=60000 * np.exp(0.2 * rng.standard_normal(channels)),
                        power=1.5 + 0.15 * rng.standard_normal(channels),
                        shift=4 + rng.uniform(-2, 2, channels), offset=20 + 8 * rng.standard_normal(channels))
    noise = SensorNoise(0.05, 1e-4, np.exp(0.3 * rng.standard_normal(channels)))
    return SensorModel("known", channels, boards, 2047, response, noise, FrameSpec(10, 40, "tile_minmax"))


def captures(model, samples=28, seed=4):
    """Samples [samples, channels] at every distance, as the stand would record them."""
    rng = np.random.default_rng(seed)
    return [model.measure(np.full((samples, model.channels), d), 1.0, rng) for d in DISTANCES]


def write_captures(folder, nodes):
    folder.mkdir(parents=True, exist_ok=True)
    for distance, node in zip(DISTANCES, nodes):
        rows = ["row," + ",".join(f"sensor_{c}" for c in range(node.shape[1]))]
        rows += [f"{i + 1}," + ",".join(str(int(v)) for v in row) for i, row in enumerate(node)]
        (folder / f"data_{int(distance)}.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return folder


def belt_recording(model, operation, polls, rng, objects=3):
    """Codes of a working line: the bare belt with a few objects passing."""
    codes = model.adc(operation.belt_level + operation.sample(polls, model.boards, rng)).astype(np.float64)
    for _ in range(objects):
        start, channel = int(rng.integers(0, polls - 60)), int(rng.integers(0, model.channels - 3))
        codes[start:start + 50, channel:channel + 3] += 300
    return codes


class ResponseTests(unittest.TestCase):
    def test_signal_and_distance_are_inverse(self):
        model = line()
        d = np.linspace(8, 95, 30)[:, None] * np.ones(model.channels)
        np.testing.assert_allclose(model.response.distance(model.response.signal(d)), d, rtol=1e-9)
        # Reflectance scales the reflected part only, not the offset.
        half = model.response.signal(d, 0.5) - model.response.offset
        np.testing.assert_allclose(half, 0.5 * (model.response.signal(d) - model.response.offset))
        np.testing.assert_allclose(model.response.distance(model.response.signal(d, 0.3), 0.3), d, rtol=1e-9)
        with self.assertRaisesRegex(ValueError, "too short"):
            model.response.signal(np.full(model.channels, -10.0))
        with self.assertRaisesRegex(ValueError, "no distance"):
            model.response.distance(model.response.offset)

    def test_adc_noise_and_correction(self):
        model = line()
        rng = np.random.default_rng(1)
        codes = model.measure(np.full((500, model.channels), 30.0), 1.0, rng)
        self.assertEqual(codes.dtype, np.int64)
        np.testing.assert_allclose(codes.mean(0), model.response.signal(np.full(model.channels, 30.0)), atol=0.2)
        np.testing.assert_array_equal(model.adc([-5.0, 0.4, 0.6, 1e9]), [0, 0, 1, model.adc_max])
        # A correction table maps every raw code of every channel.
        table = np.stack([np.arange(model.adc_max + 1)[::-1]] * model.channels, axis=1)
        corrected = SensorModel(**{**model.__dict__, "correction": table})
        np.testing.assert_array_equal(corrected.correct(codes), model.adc_max - codes)
        np.testing.assert_array_equal(model.correct(codes), codes)

    def test_rounding_of_noise(self):
        self.assertEqual(rounded_variance(10.0, 0.0), 0.0)
        self.assertAlmostEqual(rounded_variance(10.3, 5.0), 25 + 1 / 12, delta=0.02)
        # A small noise around a code is mostly rounded away: only beyond 2.5 sd does a sample move.
        self.assertAlmostEqual(rounded_variance(10.0, 0.2), 0.0124, delta=0.0005)


class ProfileTests(unittest.TestCase):
    def test_shipped_profiles(self):
        reference = load_profile("reference_line_40")
        self.assertEqual((reference.channels, reference.boards, reference.frames.pool), (40, 5, 10))
        self.assertIsNotNone(reference.operation)
        self.assertIn("sha256", reference.source["calibration"])
        example = load_profile("example_line_32")
        self.assertEqual(example.channels, 32)
        np.testing.assert_array_equal(example.response.gain, load_profile("example_line_32").response.gain)
        with self.assertRaisesRegex(ValueError, "shipped profiles"):
            load_profile("no_such_line")

    def test_spread_is_drawn_from_the_seed(self):
        spec = yaml.safe_load((Path(load_profile.__globals__["PROFILES"]) / "example_line_32.yaml").read_text())
        first = model_from(spec).response.gain
        other = copy.deepcopy(spec)
        other["response"]["spread"]["seed"] = 8
        self.assertFalse(np.allclose(first, model_from(other).response.gain))
        flat = copy.deepcopy(spec)
        flat["response"]["spread"] = {}
        np.testing.assert_array_equal(model_from(flat).response.gain, np.full(32, 30000.0))

    def test_a_written_profile_reads_back(self):
        model = SensorModel(**{**line().__dict__, "operation": OperationNoise(np.full(8, 100.0), 3.0, 0.99, 0.4, 42.5, 0.2)})
        back = model_from(yaml.safe_load(yaml.safe_dump(profile_dict(model))))
        for name in ("gain", "power", "shift", "offset"):
            np.testing.assert_allclose(getattr(back.response, name), getattr(model.response, name), rtol=1e-5)
        np.testing.assert_allclose(back.noise.scale, model.noise.scale, rtol=1e-3)
        self.assertEqual((back.operation.period, back.operation.periodic_share), (42.5, 0.2))

    def test_profiles_are_checked(self):
        good = profile_dict(line())
        cases = {
            "unknown key": lambda s: s.update(colour="red"),
            "schema": lambda s: s.update(profile=2),
            "channel count": lambda s: s["response"]["channels"].pop(),
            "negative gain": lambda s: s["response"]["channels"][0].update(gain=-1.0),
            "both forms": lambda s: s["response"].update(nominal=dict(gain=1, power=1, shift_mm=0, offset=0)),
            "normalize": lambda s: s["frames"].update(normalize="gamma"),
            "boards": lambda s: s.update(boards=3),
            "shift": lambda s: s["response"]["channels"][0].update(shift_mm=-6.0),
        }
        for name, damage in cases.items():
            spec = copy.deepcopy(good)
            damage(spec)
            with self.subTest(name), self.assertRaises(ValueError):
                model_from(spec)


class FramingTests(unittest.TestCase):
    def test_pooling_tiling_and_normalization(self):
        polls = np.arange((2 * 4 * 3 + 5) * 2, dtype=np.float64).reshape(-1, 2)  # 29 polls of 2 channels
        samples = tiles(polls, pool=3, length=4)
        self.assertEqual(samples.shape, (2, 4, 2))     # the last 5 polls make no whole tile
        np.testing.assert_array_equal(samples[0, 0], polls[:3].mean(0))
        frames, low, span = normalize(samples, "tile_minmax")
        for index in range(2):
            flat = samples[index].ravel()
            expected = np.floor((flat - flat.min()) / (flat.max() - flat.min()) * 255).astype(np.uint8)
            np.testing.assert_array_equal(frames[index].ravel(), expected)
        self.assertEqual((frames.min(), frames.max()), (0, 255))
        flat_tile, _, _ = normalize(np.ones((1, 4, 2)), "tile_minmax")
        self.assertFalse(flat_tile.any())
        raw, low, _ = normalize(samples, "none")
        self.assertIs(raw, samples)
        self.assertIsNone(low)

    def test_recordings_are_read_like_the_line_writes_them(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "DATA_7.csv"
            path.write_text("﻿1,2,3,\n\n4,5,6,9\n", encoding="utf-8")
            np.testing.assert_array_equal(read_recording(path, 3), [[1, 2, 3], [4, 5, 6]])
            with self.assertRaisesRegex(ValueError, "expected 4"):
                read_recording(path, 4)
        self.assertEqual(number_ranges("2-4,7"), {2, 3, 4, 7})


class FitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.model = line()
        cls.nodes = captures(cls.model)
        write_captures(cls.root / "captures", cls.nodes)
        cls.calibration = read_captures(cls.root / "captures", 2047)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_both_layouts_of_captures_read_alike(self):
        with open(self.root / "long.csv", "w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["channel", "distance_mm", "value"])
            for distance, node in zip(DISTANCES, self.nodes):
                for channel in range(node.shape[1]):
                    writer.writerows([channel, distance, value] for value in node[:, channel])
        table = read_captures(self.root / "long.csv", 2047)
        np.testing.assert_array_equal(table.distances, self.calibration.distances)
        np.testing.assert_array_equal(table.means, self.calibration.means)

    def test_the_fit_recovers_the_response(self):
        fitted = fit_response(self.calibration)
        grid = np.linspace(10, 95, 18)[:, None] * np.ones(self.model.channels)
        error = fitted.signal(grid) / self.model.response.signal(grid) - 1
        self.assertLess(np.abs(error).max(), 0.01)
        # A node left out still fits nearly as well.
        held = fit_response(self.calibration, exclude=8)
        point = np.full(self.model.channels, DISTANCES[8])
        self.assertLess(np.abs(held.signal(point) / self.model.response.signal(point) - 1).max(), 0.01)

    def test_the_fitted_noise_gives_the_measured_spread(self):
        # Noise of a fraction of a code is mostly rounded away, so its parameters are poorly
        # determined; what the model must give is the spread of the rounded samples.
        noise, _ = fit_noise(self.calibration)
        fitted = SensorModel(**{**self.model.__dict__, "noise": noise})
        found = check_noise(self.calibration, fitted)
        self.assertGreater(found["inside_5_95"], 0.8)
        self.assertLess(abs(found["measured_median"] - found["simulated_median"]), 0.05)
        self.assertGreater(np.corrcoef(noise.scale, self.model.noise.scale)[0, 1], 0.7)

    def test_the_fit_recovers_the_operation_noise(self):
        truth = OperationNoise(np.linspace(90, 160, self.model.channels).round(), 3.0, 0.995, 0.4, 40.0, 0.25)
        rng = np.random.default_rng(9)
        recordings = [belt_recording(self.model, truth, 3000, rng) for _ in range(16)]
        fitted = fit_operation(recordings, self.model.boards)
        np.testing.assert_allclose(fitted.belt_level, truth.belt_level, atol=1.5)
        self.assertAlmostEqual(fitted.period, 40.0, delta=1.0)
        self.assertAlmostEqual(fitted.periodic_share, 0.25, delta=0.08)
        self.assertAlmostEqual(fitted.sd, 3.0, delta=0.6)
        self.assertAlmostEqual(fitted.board_share, 0.4, delta=0.15)
        self.assertGreater(len(belt_statistics(recordings, self.model.boards).runs), 16)
        # A belt without a rhythm gets no periodic part.
        plain = OperationNoise(truth.belt_level, 3.0, 0.9, 0.4)
        flat = fit_operation([belt_recording(self.model, plain, 3000, rng, 0) for _ in range(6)], self.model.boards)
        self.assertEqual((flat.period, flat.periodic_share), (0.0, 0.0))

    def test_profile_report_and_commands(self):
        recordings = self.root / "recordings"
        recordings.mkdir()
        operation = OperationNoise(np.full(self.model.channels, 120.0), 2.0, 0.99, 0.3, 30.0, 0.2)
        rng = np.random.default_rng(11)
        for number in range(1, 5):
            polls = belt_recording(self.model, operation, 1200, rng)
            np.savetxt(recordings / f"DATA_{number}.csv", polls, fmt="%d", delimiter=",")
        # A dataset of frames made from the recordings, with its manifest.
        dataset = self.root / "frames"
        (dataset / "train" / "images").mkdir(parents=True)
        (dataset / "data.yaml").write_text("data:\n  train: train\n  val: train\n", encoding="utf-8")
        rows = []
        for number in (1, 2):
            frames = frames_from_polls(read_recording(recordings / f"DATA_{number}.csv", self.model.channels),
                                       self.model.frames)
            for tile in range(len(frames)):
                name = f"scan_{number}_{tile}"
                Image.fromarray(frames[tile]).save(dataset / "train" / "images" / f"{name}.png")
                rows.append(dict(name=name, split="train", recording=f"DATA_{number}.csv", tile=tile))
        with open(dataset / "manifest.csv", "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["name", "split", "recording", "tile"])
            writer.writeheader()
            writer.writerows(rows)
        self.assertEqual(check_frames(dataset, recordings, self.model.frames, self.model.channels)["differing"], 0)
        image = dataset / "train" / "images" / f"{rows[0]['name']}.png"
        pixels = np.asarray(Image.open(image)).copy()
        pixels[0, 0] ^= 1
        Image.fromarray(pixels).save(image)
        found = check_frames(dataset, recordings, self.model.frames, self.model.channels)
        self.assertEqual((found["differing"], found["pixels"]), (1, 1))

        out = io.StringIO()
        profile = self.root / "fitted.yaml"
        with contextlib.redirect_stdout(out):
            self.assertEqual(main(["fit-sensor", "--calibration", str(self.root / "captures"), "--recordings",
                                   str(recordings), "--only", "1-4", "--name", "fitted", "--boards", "2",
                                   "--adc-max", "2047", "--pool", "10", "--length", "40", "--normalize",
                                   "tile_minmax", "--output", str(profile)]), 0)
            self.assertEqual(main(["sensor-report", "--profile", str(profile), "--calibration",
                                   str(self.root / "captures"), "--recordings", str(recordings),
                                   "--dataset", str(dataset), "--output", str(self.root / "report.md")]), 0)
        fitted = load_profile(profile)
        self.assertEqual((fitted.channels, fitted.boards, fitted.operation.period > 0), (8, 2, True))
        text = (self.root / "report.md").read_text(encoding="utf-8")
        for section in ("П1", "П2", "П3", "П4", "П5"):
            self.assertIn(f"## {section}.", text)
        self.assertIn("| Кадров с отличиями | 1 |", text)
        # The same report without the dataset leaves out the frame check.
        model = fit_profile(self.calibration, "x", 2, 2047, self.model.frames)
        self.assertNotIn("## П3.", report(model, self.calibration, [], [], []))


if __name__ == "__main__":
    unittest.main()
