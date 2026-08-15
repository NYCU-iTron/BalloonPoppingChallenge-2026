import logging
import unittest

import numpy as np

from BalloonPoppingGymEnv.agents.gnc.selector import Selector
from BalloonPoppingGymEnv.agents.itron_agent import ITronAgent


class TestITronTargetCommitment(unittest.TestCase):
    def test_fixed_route_waits_for_launch_time_and_every_target_release(self):
        agent = ITronAgent.__new__(ITronAgent)
        agent.fixed_target_list = (1, 2)
        agent.fixed_launch_time = 10.0

        too_early = {
            "simulation_time": 9.99,
            "balloon_status": np.array([[1], [1], [1]]),
        }
        unreleased = {
            "simulation_time": 10.0,
            "balloon_status": np.array([[1], [1], [0]]),
        }
        ready = {
            "simulation_time": 10.0,
            "balloon_status": np.array([[1], [1], [1]]),
        }

        self.assertFalse(agent._fixed_route_is_ready(too_early))
        self.assertFalse(agent._fixed_route_is_ready(unreleased))
        self.assertTrue(agent._fixed_route_is_ready(ready))

    def test_fixed_route_rejects_an_out_of_range_target(self):
        agent = ITronAgent.__new__(ITronAgent)
        agent.fixed_target_list = (3,)
        agent.fixed_launch_time = 0.0
        observation = {
            "simulation_time": 0.0,
            "balloon_status": np.ones((3, 1), dtype=int),
        }

        with self.assertRaisesRegex(ValueError, "outside"):
            agent._fixed_route_is_ready(observation)

    def test_active_target_is_never_skipped(self):
        agent = ITronAgent.__new__(ITronAgent)
        agent.selector = Selector.__new__(Selector)
        agent.target_idx_list = [1, 2]
        agent.current_target_idx = 0
        observation = {"balloon_status": np.array([[0], [1], [1]])}

        self.assertEqual(agent._current_target(observation), 1)
        self.assertEqual(agent.current_target_idx, 0)

    def test_next_target_becomes_selectable_only_after_a_pop(self):
        agent = ITronAgent.__new__(ITronAgent)
        agent.selector = Selector.__new__(Selector)
        agent.target_idx_list = [1, 2]
        agent.current_target_idx = 0
        observation = {"balloon_status": np.array([[0], [2], [1]])}

        self.assertEqual(agent._current_target(observation), 2)
        self.assertEqual(agent.current_target_idx, 1)

    def test_completed_leg_records_prediction_error(self):
        agent = ITronAgent.__new__(ITronAgent)
        agent.logger = logging.getLogger(__name__)
        agent.leg_diagnostics = []
        agent._engagement_prediction = {
            "target": 1,
            "start_time": 10.0,
            "duration": 2.5,
            "position": np.array([12.0, 0.0, 0.0]),
            "velocity": np.array([20.0, 0.0, 0.0]),
        }
        observation = {"balloon_status": np.array([[1], [2]])}
        rocket_state = np.zeros(16)
        rocket_state[0:3] = [11.0, 0.0, 0.0]
        rocket_state[3:6] = [15.0, 0.0, 0.0]

        agent._record_completed_engagement(observation, rocket_state, 13.0)

        self.assertIsNone(agent._engagement_prediction)
        self.assertEqual(len(agent.leg_diagnostics), 1)
        diagnostic = agent.leg_diagnostics[0]
        self.assertEqual(diagnostic["target"], 1)
        self.assertEqual(diagnostic["actual_duration"], 3.0)
        self.assertEqual(diagnostic["direction_error_deg"], 0.0)
        self.assertEqual(diagnostic["position_error"], 1.0)
        self.assertTrue(np.isnan(diagnostic["distance"]))


if __name__ == "__main__":
    unittest.main()
