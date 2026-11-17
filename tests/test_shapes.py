"""Objects described by shape: the height of known figures and the spot they leave in the signal of the line."""

import math
import unittest

import numpy as np

from ir_line_synth import motion
from ir_line_synth.sensor_model import FrameSpec, OperationNoise, Response, SensorModel, SensorNoise
from ir_line_synth.shapes import (
    Installation, Pose, SmoothField, Spots, Superquadric, Surface, belt_base, belt_points, geometric_box, height,
    label_box, line_codes, line_signal, natural_rows, passage, surface,
)


def line(channels=16, operation=True):
    """A line of alike channels over a measured belt (or a calibration-stand line without one)."""
    response = Response(gain=np.full(channels, 60000.0), power=np.full(channels, 1.6),
                        shift=np.full(channels, 2.0), offset=np.full(channels, 20.0), valid_mm=(5.0, 100.0))
    belt = OperationNoise(np.linspace(90, 150, channels), 3.6, 0.95, 0.4, 42.75, 0.2) if operation else None
    return SensorModel("alike", channels, 4, 4095, response, SensorNoise(0.05, 1e-4),
                       FrameSpec(10, 40, "tile_minmax"), belt)


SETUP = Installation(height_mm=60.0, pitch_mm=8.0, poll_hz=200.0, belt_reflectance=0.3)


class HeightTests(unittest.TestCase):
    def test_sphere(self):
        sphere = Superquadric(40, 40, 40)
        r = np.linspace(0, 30, 61)
        expected = np.where(r <= 20, 20 + np.sqrt(np.maximum(400 - r ** 2, 0)), 0)
        np.testing.assert_allclose(height(sphere, Pose(), r, 0), expected, atol=1e-9)
        np.testing.assert_allclose(height(sphere, Pose(), 0, r), expected, atol=1e-9)
        self.assertEqual(height(sphere, Pose(), 0, 0), 40)

    def test_ellipsoid_outline_and_top(self):
        egg = Superquadric(57, 44, 44)
        self.assertAlmostEqual(float(height(egg, Pose(), 0, 0)), 44)
        self.assertGreater(height(egg, Pose(), 28.4, 0), 0)
        self.assertEqual(height(egg, Pose(), 28.6, 0), 0)
        self.assertGreater(height(egg, Pose(), 0, 21.9), 0)
        self.assertEqual(height(egg, Pose(), 0, 22.1), 0)

    def test_box_is_flat_with_a_rectangular_outline(self):
        box = Superquadric(100, 40, 30, vertical=0.1, horizontal=0.1)
        u, v = np.meshgrid(np.linspace(-45, 45, 19), np.linspace(-18, 18, 13))
        np.testing.assert_allclose(height(box, Pose(), u, v), 30, rtol=0.03)
        self.assertEqual(height(box, Pose(), 50.5, 0), 0)
        self.assertEqual(height(box, Pose(), 0, 20.5), 0)

    def test_standing_cylinder_has_a_round_outline_and_a_flat_top(self):
        can = Superquadric(30, 30, 50, vertical=0.1, horizontal=1.0)
        self.assertAlmostEqual(float(height(can, Pose(), 0, 0)), 50, delta=1e-9)
        self.assertGreater(height(can, Pose(), 10, 10), 48)
        self.assertEqual(height(can, Pose(), 11, 11), 0)   # inside the square, outside the circle

    def test_pose_moves_and_turns_the_object(self):
        bar = Superquadric(100, 20, 10, 0.2, 0.2)
        self.assertGreater(height(bar, Pose(), 45, 0), 0)
        self.assertEqual(height(bar, Pose(yaw=90), 45, 0), 0)
        self.assertGreater(height(bar, Pose(yaw=90), 0, 45), 0)
        self.assertGreater(height(bar, Pose(x=200, y=-30), 245, -30), 0)
        self.assertEqual(height(bar, Pose(x=200, y=-30), 45, 0), 0)

    def test_parameters_are_checked(self):
        for args in ((0, 10, 10), (10, 10, 2000), (10, 10, 10, 0.01), (10, 10, 10, 1, 3)):
            with self.assertRaises(ValueError):
                Superquadric(*args)
        with self.assertRaises(ValueError):
            Installation(60, 8, 0, 0.3)


class SignalTests(unittest.TestCase):
    def setUp(self):
        self.model = line()
        self.sphere = Superquadric(38, 38, 38)
        self.speed = 100.0                     # mm/s: 0.5 mm per poll at 200 Hz
        self.x, self.y = belt_points(SETUP, self.model.channels, 400, self.speed)
        self.heights = height(self.sphere, Pose(x=100.0, y=0.0), self.x, self.y)

    def test_belt_points(self):
        self.assertEqual(self.x.shape, (400, 16))
        np.testing.assert_allclose(self.x[:3, 0], [0, 0.5, 1.0])
        np.testing.assert_allclose(self.y[0, [0, 7, 8, 15]], [-60, -4, 4, 60])

    def test_spot_has_the_size_of_the_object(self):
        covered = self.heights > 0
        channels = np.flatnonzero(covered.any(axis=0))
        # Channels at |y| < 19 mm: y = -12, -4, 4, 12.
        np.testing.assert_array_equal(channels, [6, 7, 8, 9])
        polls = np.flatnonzero(covered[:, 7])
        # Under the channel at y = 4 mm the outline spans 2 * sqrt(19^2 - 4^2) mm, 0.5 mm per poll.
        self.assertAlmostEqual(len(polls) * 0.5, 2 * math.sqrt(361 - 16), delta=1.0)
        self.assertAlmostEqual((polls[0] + polls[-1]) / 2 * 0.5, 100, delta=0.5)

    def test_signal_is_the_belt_level_off_the_object_and_the_law_on_it(self):
        signal = line_signal(self.model, SETUP, self.heights, 1.0)
        off = self.heights == 0
        np.testing.assert_allclose(signal[off], np.broadcast_to(self.model.operation.belt_level, signal.shape)[off])
        peak = 200, 7                           # x = 100 mm, y = -4 mm
        d = 60 - self.heights[peak]
        expected = belt_base(self.model, SETUP)[7] + 60000 * (d + 2) ** -1.6
        self.assertAlmostEqual(signal[peak], expected, places=9)
        self.assertGreater(signal[peak], signal[peak[0], 6])      # the top is closer than the side

    def test_signal_above_the_base_is_proportional_to_reflectance(self):
        base = belt_base(self.model, SETUP)
        on = self.heights > 0
        bright = line_signal(self.model, SETUP, self.heights, 1.0) - base
        dark = line_signal(self.model, SETUP, self.heights, 0.25) - base
        np.testing.assert_allclose(dark[on], 0.25 * bright[on])

    def test_without_a_measured_belt_the_base_is_the_calibration_offset(self):
        model = line(operation=False)
        signal = line_signal(model, SETUP, np.zeros((5, 16)), 1.0)
        np.testing.assert_allclose(signal, 20 + 0.3 * 60000 * 62.0 ** -1.6)

    def test_object_too_close_to_the_line_is_refused(self):
        tall = height(Superquadric(40, 40, 58), Pose(x=100), self.x, self.y)
        with self.assertRaisesRegex(ValueError, "valid from 5 mm"):
            line_signal(self.model, SETUP, tall, 1.0)

    def test_codes_are_integers_around_the_signal_and_repeat_with_the_seed(self):
        signal = line_signal(self.model, SETUP, self.heights, 1.0)
        codes = line_codes(self.model, signal, np.random.default_rng(5))
        self.assertEqual(codes.dtype, np.int64)
        self.assertTrue(((codes >= 0) & (codes <= 4095)).all())
        np.testing.assert_array_equal(codes, line_codes(self.model, signal, np.random.default_rng(5)))
        self.assertLess(abs(float(np.mean(codes - signal))), 2.0)
        self.assertLess(float(np.std(codes - signal)), 6.0)
        quiet = line_codes(line(operation=False), signal, np.random.default_rng(5))
        self.assertLess(float(np.abs(quiet - signal).max()), 3.0)


class SurfaceTests(unittest.TestCase):
    def setUp(self):
        self.box = Superquadric(200, 200, 40, 0.1, 0.1)
        grid = np.arange(-90, 90, 1.0)
        self.x, self.y = np.meshgrid(grid, grid)

    def test_field_has_unit_variance_and_the_stated_correlation(self):
        values, near, far = [], [], []
        for seed in range(40):
            field = SmoothField(5.0, np.random.default_rng(seed))
            points = np.random.default_rng(100 + seed).uniform(-500, 500, (200, 2))
            here = field(points[:, 0], points[:, 1])
            values.append(here)
            near.append(np.corrcoef(here, field(points[:, 0] + 2.0, points[:, 1]))[0, 1])
            far.append(np.corrcoef(here, field(points[:, 0], points[:, 1] + 20.0))[0, 1])
        self.assertAlmostEqual(float(np.var(np.concatenate(values))), 1.0, delta=0.1)
        self.assertAlmostEqual(float(np.mean(near)), math.exp(-4 / 50), delta=0.05)
        self.assertAlmostEqual(float(np.mean(far)), 0.0, delta=0.05)

    def test_a_plain_surface_changes_nothing(self):
        heights, rho = surface(self.box, Pose(), Surface(reflectance=0.7), self.x, self.y)
        np.testing.assert_array_equal(heights, height(self.box, Pose(), self.x, self.y))
        np.testing.assert_array_equal(rho, 0.7)

    def test_roughness(self):
        plain = height(self.box, Pose(), self.x, self.y)
        rough, _ = surface(self.box, Pose(), Surface(roughness_mm=1.5, roughness_size_mm=4.0, seed=3), self.x, self.y)
        on = plain > 0
        np.testing.assert_array_equal(rough > 0, on)
        relief = (rough - plain)[on]
        self.assertAlmostEqual(float(relief.std()), 1.5, delta=0.4)
        self.assertLess(float(np.abs(relief).max()), 1.5 * 5 + 1e-9)
        with self.assertRaisesRegex(ValueError, "Roughness"):
            surface(self.box, Pose(), Surface(roughness_mm=2.5, roughness_size_mm=4.0), self.x, self.y)
        with self.assertRaisesRegex(ValueError, "needs roughness_size_mm"):
            Surface(roughness_mm=1.0)

    def test_reflectance_spread_is_lognormal_around_the_median(self):
        _, rho = surface(self.box, Pose(), Surface(0.8, 0.3, 6.0, seed=4), self.x, self.y)
        logs = np.log(rho[height(self.box, Pose(), self.x, self.y) > 0])
        self.assertAlmostEqual(float(np.median(logs)), math.log(0.8), delta=0.1)
        self.assertAlmostEqual(float(logs.std()), 0.3, delta=0.08)

    def test_spots_cover_the_stated_share_with_the_stated_factor(self):
        shares = []
        for seed in range(20):
            finish = Surface(0.9, spots=(Spots(0.2, 0.0, 3.0), Spots(0.05, 4.0, 2.0)), seed=seed)
            _, rho = surface(self.box, Pose(), finish, self.x, self.y)
            self.assertTrue(set(np.round(np.unique(rho), 9)) <= {0.0, 0.9, 3.6})
            shares.append(((rho == 0).mean(), (rho > 1).mean()))
        dark, glint = np.mean(shares, axis=0)
        self.assertAlmostEqual(dark, 0.2, delta=0.03)
        self.assertAlmostEqual(glint, 0.05 * 0.8, delta=0.015)   # glints inside dropouts stay dark

    def test_the_surface_moves_and_turns_with_the_object(self):
        finish = Surface(0.9, 0.2, 5.0, 1.0, 4.0, (Spots(0.3, 0.2, 3.0),), seed=8)
        egg = Superquadric(57, 44, 44)
        u, v = np.meshgrid(np.linspace(-25, 25, 11), np.linspace(-18, 18, 9))
        still = surface(egg, Pose(), finish, u, v)
        angle = math.radians(30)
        x = 100 + math.cos(angle) * u - math.sin(angle) * v
        y = -7 + math.sin(angle) * u + math.cos(angle) * v
        moved = surface(egg, Pose(100, -7, 30), finish, x, y)
        np.testing.assert_allclose(moved[0], still[0], atol=1e-9)
        np.testing.assert_allclose(moved[1], still[1], atol=1e-9)
        other = surface(egg, Pose(), Surface(0.9, 0.2, 5.0, 1.0, 4.0, (Spots(0.3, 0.2, 3.0),), seed=9), u, v)
        self.assertFalse(np.allclose(other[1], still[1]))

    def test_adding_spots_keeps_the_relief_and_the_spread(self):
        base = Surface(0.9, 0.2, 5.0, 1.0, 4.0, seed=8)
        spotted = Surface(0.9, 0.2, 5.0, 1.0, 4.0, (Spots(0.3, 0.0, 3.0),), seed=8)
        plain, spotty = surface(self.box, Pose(), base, self.x, self.y), surface(self.box, Pose(), spotted, self.x, self.y)
        np.testing.assert_array_equal(plain[0], spotty[0])
        lit = spotty[1] > 0
        np.testing.assert_array_equal(plain[1][lit], spotty[1][lit])

    def test_reflectance_falls_with_the_tilt_of_the_surface(self):
        sphere = Superquadric(40, 40, 40)
        r = np.array([0.0, 5.0, 10.0, 15.0, 18.0])
        _, plain = surface(sphere, Pose(), Surface(0.8), r, np.zeros_like(r))
        for power in (1.0, 2.0):
            _, tilted = surface(sphere, Pose(), Surface(0.8, tilt_power=power), r, np.zeros_like(r))
            # On a sphere the cosine of the tilt at r from the top is sqrt(R^2 - r^2) / R.
            np.testing.assert_allclose(tilted, 0.8 * (np.sqrt(400 - r ** 2) / 20) ** power, rtol=1e-3)
        np.testing.assert_array_equal(plain, 0.8)
        # A flat top keeps its reflectance; the box falls only at its edge.
        box = Superquadric(100, 60, 20, 0.1, 0.1)
        _, top = surface(box, Pose(), Surface(0.8, tilt_power=2.0), np.array([0.0, 30.0]), np.array([0.0, 10.0]))
        np.testing.assert_allclose(top, 0.8, rtol=1e-6)
        with self.assertRaises(ValueError):
            Surface(tilt_power=-1)

    def test_dropouts_leave_only_the_base_in_the_signal(self):
        model = line()
        x, y = belt_points(SETUP, 16, 300, 100.0)
        heights, rho = surface(Superquadric(57, 44, 44), Pose(x=75), Surface(spots=(Spots(0.5, 0.0, 4.0),), seed=2), x, y)
        signal = line_signal(model, SETUP, heights, rho)
        dropped = (heights > 0) & (rho == 0)
        self.assertTrue(dropped.any())
        np.testing.assert_allclose(signal[dropped], np.broadcast_to(belt_base(model, SETUP), signal.shape)[dropped])


class PassageTests(unittest.TestCase):
    def setUp(self):
        self.model = line()                     # 10 polls per row
        self.egg = Superquadric(57, 44, 44)
        self.speed = 100.0                      # 0.5 mm per poll, 5 mm per row
        self.rows = natural_rows(self.egg, self.model, SETUP, self.speed)

    def steady(self, speed=1.0, lateral=0.0, gain=1.0, flip=False, rows=None):
        times = motion.output_times(0.0, rows or self.rows, np.full(8, speed))
        n = len(times)
        return dict(kind="blob", flip=flip, speed=[speed] * 8, lateral=[lateral] * n, gain=[gain] * n)

    def test_natural_rows(self):
        self.assertAlmostEqual(self.rows, 57 / 5)
        self.assertAlmostEqual(natural_rows(self.egg, self.model, SETUP, self.speed, yaw=90), 44 / 5)

    def test_a_steady_passage_is_the_object_carried_by_the_belt(self):
        heights, rho = passage(self.egg, Surface(0.8), self.steady(), self.model, SETUP, self.speed, y=3.0)
        x, y = belt_points(SETUP, 16, len(heights), self.speed)
        # Output row 0 shows row -1 of the natural passage: half a row (2.5 mm) before the front reaches the line.
        carried = height(self.egg, Pose(x=2.5 + 57 / 2, y=3.0), x, y)
        self.assertGreater(float(np.isclose(heights, carried, atol=1e-3).mean()), 0.998)
        np.testing.assert_array_equal(rho, 0.8)

    def test_speed_stretches_and_lateral_shifts_the_trace(self):
        calm, _ = passage(self.egg, Surface(), self.steady(), self.model, SETUP, self.speed)
        slow, _ = passage(self.egg, Surface(), self.steady(speed=0.5), self.model, SETUP, self.speed)
        aside, _ = passage(self.egg, Surface(), self.steady(lateral=1.0), self.model, SETUP, self.speed)
        polls = lambda h: int((h > 0).any(axis=1).sum())
        self.assertAlmostEqual(polls(slow) / polls(calm), 2.0, delta=0.05)
        np.testing.assert_allclose(aside[:, 1:], calm[:, :-1], atol=1e-9)

    def test_contact_gain_multiplies_the_reflectance(self):
        move = self.steady()
        move["gain"][5:8] = [0.0, 0.0, 0.0]
        heights, rho = passage(self.egg, Surface(0.9), move, self.model, SETUP, self.speed)
        np.testing.assert_array_equal(rho[50:71], 0.0)
        np.testing.assert_allclose(rho[80:], 0.9)
        self.assertTrue((heights[50:71] > 0).any())

    def test_flip_mirrors_across_the_belt(self):
        move = self.steady(lateral=0.7, rows=natural_rows(self.egg, self.model, SETUP, self.speed, yaw=25))
        straight, _ = passage(self.egg, Surface(), move, self.model, SETUP, self.speed, y=2.0, yaw=25)
        mirrored, _ = passage(self.egg, Surface(), dict(move, flip=True), self.model, SETUP, self.speed, y=2.0, yaw=25)
        np.testing.assert_allclose(mirrored, straight[:, ::-1], atol=1e-9)

    def test_a_motion_for_another_passage_is_refused(self):
        with self.assertRaisesRegex(ValueError, "another passage"):
            passage(Superquadric(90, 44, 44), Surface(), self.steady(), self.model, SETUP, self.speed)

    def test_every_kind_of_motion_moves_an_egg_under_the_reference_setup(self):
        setup = Installation(55.0, 10.0, 200.0, 0.3)
        lengths = {}
        for kind in motion.KINDS:
            rng = np.random.default_rng(11)
            covered = []
            for _ in range(12):
                rows = natural_rows(self.egg, self.model, setup, 160.0)
                move = motion.sample(kind, rows, (0.0, rows), rng)
                self.assertEqual(move["own"], 1.0) if kind in ("rolling", "swinging") else None
                heights, rho = passage(self.egg, Surface(), move, self.model, setup, 160.0)
                signal = line_signal(self.model, setup, heights, rho)
                self.assertTrue(np.isfinite(signal).all())
                covered.append((heights > 0).any(axis=1).sum() / self.model.frames.pool)
            lengths[kind] = np.median(covered)
        self.assertGreater(lengths["long"], lengths["blob"])
        self.assertGreater(lengths["blob"], lengths["short"])
        self.assertLess(lengths["short"], 5)
        self.assertGreater(lengths["long"], 10)


class LabelTests(unittest.TestCase):
    def setUp(self):
        self.model = line()                     # 16 channels, 10 polls per row, pitch 8 mm
        self.speed = 100.0                      # 0.5 mm per poll, 5 mm per row

    def test_box_of_a_known_object(self):
        # A box 40 mm long and 20 mm wide at y = 0: channels at y = -4 and 4 (7 and 8) only. It lies over
        # x = 80 ... 120 mm: polls 161 ... 239 (polls 160 and 240 fall exactly on its edge, outside), rows 16 ... 23.
        x, y = belt_points(SETUP, 16, 400, self.speed)
        heights = height(Superquadric(40, 20, 10, 0.1, 0.1), Pose(x=100.0), x, y)
        np.testing.assert_array_equal(geometric_box(heights, 10), [7, 16, 9, 24])
        aside = height(Superquadric(40, 20, 10, 0.1, 0.1), Pose(x=100.0, y=-6.0), x, y)
        np.testing.assert_array_equal(geometric_box(aside, 10)[[0, 2]], [6, 9])

    def test_every_cell_at_the_edge_of_the_box_holds_the_object(self):
        egg = Superquadric(57, 44, 44)
        rows = natural_rows(egg, self.model, SETUP, self.speed)
        rng = np.random.default_rng(4)
        for kind in motion.KINDS:
            move = motion.sample(kind, rows, (0.0, rows), rng)
            heights, rho = passage(egg, Surface(spots=(Spots(0.3, 0.0, 3.0),)), move, self.model, SETUP, self.speed)
            covered = (heights > 0).reshape(-1, 10, 16).any(axis=1)
            x1, y1, x2, y2 = geometric_box(heights, 10).astype(int)
            self.assertFalse(covered[:y1].any() or covered[y2:].any() or covered[:, :x1].any() or covered[:, x2:].any())
            self.assertTrue(covered[y1].any() and covered[y2 - 1].any() and covered[:, x1].any() and covered[:, x2 - 1].any())

    def test_margins_widen_the_box_inside_the_frame(self):
        box = np.asarray([7.0, 16.0, 9.0, 24.0])
        np.testing.assert_allclose(label_box(box, [0, 0, 0, 0], (40, 16)), box)
        np.testing.assert_allclose(label_box(box, [-0.243, -0.041, 0.194, -0.037], (40, 16)),
                                   [6.757, 15.959, 9.194, 23.963])
        np.testing.assert_allclose(label_box(box, [-10, -20, 10, 20], (40, 16)), [0, 0, 16, 40])
        with self.assertRaisesRegex(ValueError, "empty"):
            label_box(box, [3, 0, -3, 0], (40, 16))
        with self.assertRaises(ValueError):
            label_box(box, [0, 0, 0], (40, 16))

    def test_no_box_when_the_object_is_under_no_channel(self):
        self.assertIsNone(geometric_box(np.zeros((40, 16)), 10))
        with self.assertRaisesRegex(ValueError, "whole rows"):
            geometric_box(np.zeros((45, 16)), 10)


if __name__ == "__main__":
    unittest.main()
