import logging
import unittest

import numpy as np

from BalloonPoppingGymEnv.agents.gnc.navigator import Navigator


class _Vehicle:
    planning_speed = 15.0


class TestNavigatorLookahead(unittest.TestCase):
    def setUp(self):
        self.navigator = Navigator.__new__(Navigator)
        self.navigator.logger = logging.getLogger(__name__)
        self.navigator.vehicle = _Vehicle()

    def test_zero_weight_preserves_position_only_command(self):
        original = np.array([1.0, 2.0, 3.0])

        result = self.navigator._blend_lookahead_acceleration(
            original,
            zem=np.array([3.0, 0.0, 0.0]),
            target_state=np.zeros(6),
            next_target_state=np.array([0.0, 10.0, 0.0, 0.0, 0.0, 0.0]),
            rocket_vel=np.array([5.0, 0.0, 0.0]),
            t_go=2.0,
            weight=0.0,
        )

        np.testing.assert_array_equal(result, original)

    def test_following_target_shapes_terminal_velocity_without_changing_zem(self):
        result = self.navigator._blend_lookahead_acceleration(
            position_accel=np.zeros(3),
            zem=np.zeros(3),
            target_state=np.zeros(6),
            next_target_state=np.array([0.0, 10.0, 0.0, 0.0, 0.0, 0.0]),
            rocket_vel=np.array([15.0, 0.0, 0.0]),
            t_go=2.0,
            weight=1.0,
        )

        # With the intercept already solved, the extra command turns velocity
        # from +x towards the following target on +y.
        np.testing.assert_allclose(result, [15.0, -15.0, 0.0])

    def test_coincident_targets_do_not_create_an_arbitrary_exit_direction(self):
        original = np.array([1.0, 2.0, 3.0])

        result = self.navigator._blend_lookahead_acceleration(
            original,
            zem=np.zeros(3),
            target_state=np.zeros(6),
            next_target_state=np.zeros(6),
            rocket_vel=np.zeros(3),
            t_go=1.0,
            weight=0.5,
        )

        np.testing.assert_array_equal(result, original)

    def test_corridor_blends_towards_unbraked_flyby_zem(self):
        self.navigator.t_go_min = 0.3
        self.navigator.miss_tolerance = 1.5
        self.navigator.nav_constant = 3.0

        result = self.navigator._blend_corridor_acceleration(
            committed_accel=np.zeros(3),
            corridor_target_state=np.array([20.0, 5.0, 0.0, 0.0, 0.0, 0.0]),
            rocket_pos=np.zeros(3),
            rocket_vel=np.array([10.0, 0.0, 0.0]),
            committed_t_go=3.0,
            weight=1.0,
            horizon=0.0,
        )

        np.testing.assert_allclose(result, [0.0, 3.75, 0.0])

    def test_corridor_ignores_a_balloon_after_its_closest_pass(self):
        self.navigator.t_go_min = 0.3
        self.navigator.miss_tolerance = 1.5

        committed = np.array([1.0, 2.0, 3.0])
        result = self.navigator._blend_corridor_acceleration(
            committed_accel=committed,
            corridor_target_state=np.array([-5.0, 5.0, 0.0, 0.0, 0.0, 0.0]),
            rocket_pos=np.zeros(3),
            rocket_vel=np.array([10.0, 0.0, 0.0]),
            committed_t_go=3.0,
            weight=1.0,
            horizon=0.0,
        )

        np.testing.assert_array_equal(result, committed)

    def test_corridor_waits_until_the_configured_horizon(self):
        self.navigator.t_go_min = 0.3
        self.navigator.miss_tolerance = 1.5
        self.navigator.nav_constant = 3.0
        committed = np.array([1.0, 2.0, 3.0])

        result = self.navigator._blend_corridor_acceleration(
            committed_accel=committed,
            corridor_target_state=np.array([20.0, 5.0, 0.0, 0.0, 0.0, 0.0]),
            rocket_pos=np.zeros(3),
            rocket_vel=np.array([10.0, 0.0, 0.0]),
            committed_t_go=3.0,
            weight=1.0,
            horizon=1.5,
        )

        np.testing.assert_array_equal(result, committed)
