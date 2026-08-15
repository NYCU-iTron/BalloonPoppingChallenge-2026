import logging
import unittest

import numpy as np

from BalloonPoppingGymEnv.agents.gnc.selector import Selector


class TestSelectorTurnCost(unittest.TestCase):
    def test_close_straight_leg_is_preferred_to_a_turn(self):
        selector = Selector.__new__(Selector)
        selector.logger = logging.getLogger(__name__)
        selector.vehicle = _ConstantSpeedVehicle()
        selector.time_budget_fraction = 1.0
        selector.pad_origin = np.zeros(3)
        selector.time_weight = 1.0
        selector.angle_weight = 20.0
        selector.target_reward = 3_000.0
        selector.short_turn_recovery_factor = 2.0
        selector.max_segment_dist = 1_000.0
        selector.too_far_weight = 0.0
        selector.beam_width = 20
        selector.max_chain_length = 2
        states = np.array(
            [
                [10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [15.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [20.0, 20.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        self.assertEqual(selector.select_targets_beam(states), [0, 1])

    def test_ranked_beam_candidates_keep_alternatives_without_changing_best(self):
        selector = Selector.__new__(Selector)
        selector.logger = logging.getLogger(__name__)
        selector.vehicle = _ConstantSpeedVehicle()
        selector.time_budget_fraction = 1.0
        selector.pad_origin = np.zeros(3)
        selector.time_weight = 1.0
        selector.angle_weight = 1.0
        selector.target_reward = 3_000.0
        selector.short_turn_recovery_factor = 2.0
        selector.max_segment_dist = 1_000.0
        selector.too_far_weight = 0.0
        selector.beam_width = 20
        selector.max_chain_length = 2
        states = np.array(
            [
                [10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [12.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [15.0, -1.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        ranked = selector.rank_target_chains_beam(states, limit=3)

        self.assertEqual(ranked[0][1], selector.select_targets_beam(states))
        self.assertEqual(len(ranked), 3)
        self.assertEqual(len({tuple(chain) for _utility, chain in ranked}), 3)

    def test_turn_time_is_used_when_predicting_moving_target_position(self):
        selector = Selector.__new__(Selector)
        selector.vehicle = _MovingTargetVehicle()
        selector.short_turn_recovery_factor = 2.0
        balloon = {
            "pos": np.array([0.0, 100.0, 0.0]),
            "vel": np.array([2.0, 0.0, 0.0]),
        }

        leg, _, leg_time, turn = selector._predict_leg_from_heading(
            balloon,
            from_pos=np.zeros(3),
            chain_depart=0.0,
            t_since_launch=0.0,
            incoming_dir=np.array([1.0, 0.0, 0.0]),
        )

        self.assertGreater(turn, 1.0)
        self.assertGreater(leg_time, 15.0)
        self.assertGreater(leg[0], 30.0)

    def test_current_bearing_is_not_hidden_by_future_target_drift(self):
        selector = Selector.__new__(Selector)
        selector.vehicle = _MovingTargetVehicle()
        selector.short_turn_recovery_factor = 0.0
        balloon = {
            "pos": np.array([0.0, 10.0, 0.0]),
            "vel": np.array([10.0, 0.0, 0.0]),
        }

        _, _, _, turn = selector._predict_leg_from_heading(
            balloon,
            from_pos=np.zeros(3),
            chain_depart=0.0,
            t_since_launch=0.0,
            incoming_dir=np.array([1.0, 0.0, 0.0]),
        )

        self.assertGreaterEqual(turn, np.pi / 2.0)

    def test_expensive_extra_target_does_not_always_beat_smoothness(self):
        selector = Selector.__new__(Selector)
        selector.logger = logging.getLogger(__name__)
        selector.vehicle = _ConstantSpeedVehicle()
        selector.time_budget_fraction = 1.0
        selector.pad_origin = np.zeros(3)
        selector.time_weight = 1.0
        selector.angle_weight = 20.0
        selector.target_reward = 100.0
        selector.short_turn_recovery_factor = 2.0
        selector.max_segment_dist = 1_000.0
        selector.too_far_weight = 0.0
        selector.beam_width = 20
        selector.max_chain_length = 2
        states = np.array(
            [
                [10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [-10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        self.assertEqual(len(selector.select_targets_beam(states)), 1)

    def test_conservative_route_wins_when_it_is_not_shorter(self):
        selector = Selector.__new__(Selector)
        selector.use_beam_search = True
        selector.short_turn_recovery_factor = 2.0
        selector.conservative_short_turn_recovery_factor = 12.0
        selector.select_targets_beam = lambda _states, **_kwargs: (
            [7, 8] if selector.short_turn_recovery_factor == 12.0 else [1, 2]
        )

        self.assertEqual(selector.plan_chain(np.empty((0, 6))), [7, 8])
        self.assertEqual(selector.short_turn_recovery_factor, 2.0)

    def test_longer_permissive_route_survives_conservative_model(self):
        selector = Selector.__new__(Selector)
        selector.use_beam_search = True
        selector.short_turn_recovery_factor = 2.0
        selector.conservative_short_turn_recovery_factor = 12.0
        selector.select_targets_beam = lambda _states, **_kwargs: (
            [7] if selector.short_turn_recovery_factor == 12.0 else [1, 2]
        )

        self.assertEqual(selector.plan_chain(np.empty((0, 6))), [1, 2])
        self.assertEqual(selector.short_turn_recovery_factor, 2.0)

    def test_measured_speed_is_carried_through_a_straight_leg(self):
        selector = Selector.__new__(Selector)
        selector.vehicle = _ConstantSpeedVehicle()
        selector.short_turn_recovery_factor = 2.0
        balloon = {
            "pos": np.array([20.0, 0.0, 0.0]),
            "vel": np.zeros(3),
        }

        _, _, leg_time, turn, exit_velocity = selector._predict_leg_from_velocity(
            balloon,
            from_pos=np.zeros(3),
            chain_depart=0.0,
            t_since_launch=0.0,
            incoming_velocity=np.array([20.0, 0.0, 0.0]),
        )

        self.assertEqual(turn, 0.0)
        self.assertAlmostEqual(leg_time, 1.0)
        np.testing.assert_allclose(exit_velocity, [20.0, 0.0, 0.0])

    def test_short_leg_retains_unfinished_turn_in_exit_direction(self):
        selector = Selector.__new__(Selector)
        selector.vehicle = _ConstantSpeedVehicle()
        selector.short_turn_recovery_factor = 0.0
        selector.turn_geometry_speed = 20.0
        balloon = {
            "pos": np.array([0.0, 10.0, 0.0]),
            "vel": np.zeros(3),
        }

        _, _, _, turn, exit_velocity = selector._predict_leg_from_velocity(
            balloon,
            from_pos=np.zeros(3),
            chain_depart=0.0,
            t_since_launch=0.0,
            incoming_velocity=np.array([20.0, 0.0, 0.0]),
        )

        exit_heading = np.arctan2(exit_velocity[1], exit_velocity[0])
        self.assertAlmostEqual(turn, np.pi / 2.0)
        self.assertAlmostEqual(exit_heading, 0.25)
        self.assertLess(exit_heading, turn)


class _MovingTargetVehicle:
    planning_speed = 10.0
    min_transit_accel = 1.0

    @staticmethod
    def max_lateral_accel(_time):
        return 10.0

    @staticmethod
    def transit_time(length, turn_angle, _time, _climb):
        return length / 10.0 + 5.0 * turn_angle


class _ConstantSpeedVehicle:
    burn_time = 100.0
    planning_speed = 10.0
    min_transit_accel = 1.0

    @staticmethod
    def max_lateral_accel(_time):
        return 10.0

    @staticmethod
    def transit_time(length, turn_angle, _time, _climb):
        return length / 10.0 + turn_angle

    @staticmethod
    def transit_state(length, turn_angle, _time, _climb, incoming_speed):
        speed = 10.0 if incoming_speed is None else incoming_speed
        return length / speed + turn_angle, speed


if __name__ == "__main__":
    unittest.main()
