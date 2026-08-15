import unittest

import numpy as np

from BalloonPoppingGymEnv.agents.gnc.controller import Controller


class _Vehicle:
    gimbal_rate_limit = 60.0


class TestControllerTuning(unittest.TestCase):
    def setUp(self):
        self.controller = Controller.__new__(Controller)
        self.controller.dt = 0.01
        self.controller.vehicle = _Vehicle()
        self.controller.direction_rate_feedforward = 1.0
        self.controller.previous_desired_dir_world = None
        self.controller.commanded_tvc = np.zeros(2)
        self.controller.integral_error = np.ones(2)

    def test_direction_rate_tracks_rotation_of_guidance_command(self):
        np.testing.assert_array_equal(
            self.controller._desired_direction_rate(np.array([1.0, 0.0, 0.0])),
            np.zeros(3),
        )

        rate = self.controller._desired_direction_rate(np.array([0.0, 1.0, 0.0]))

        np.testing.assert_allclose(rate, [0.0, 0.0, 50.0 * np.pi])

    def test_slew_limiter_matches_actuator_per_axis_limit(self):
        first = self.controller._limit_gimbal_slew(np.array([10.0, -10.0]))
        second = self.controller._limit_gimbal_slew(np.array([10.0, -10.0]))

        np.testing.assert_allclose(first, [0.6, -0.6])
        np.testing.assert_allclose(second, [1.2, -1.2])
        self.assertLess(self.controller.integral_error[0], 1.0)
