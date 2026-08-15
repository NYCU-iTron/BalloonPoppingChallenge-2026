import unittest

import numpy as np

from BalloonPoppingGymEnv.envs.pool_env import PoolEnv


class TestPoolEnv(unittest.TestCase):
    def setUp(self):
        self.env = PoolEnv.__new__(PoolEnv)
        self.env.balloon_parameters = {"num": 2}
        self.env.simulation_parameters = {"max_time": 0.4, "time_step": 0.1}
        self.env._balloon_release_at_step = np.array([0, 2])
        self.env._raw_source_trajectories = None

    def test_pool_tracks_follow_each_balloons_release_time(self):
        tracks = np.zeros((2, 6, 4))
        tracks[0, 0] = [0.0, 1.0, 2.0, 3.0]
        tracks[1, 0] = [10.0, 11.0, 12.0, 13.0]

        self.env.update_source_trajectories(tracks)
        self.env._BalloonPoppingEnv__generate_balloon_flights()

        np.testing.assert_array_equal(
            self.env._balloon_flights[0, 0], [0.0, 1.0, 2.0, 3.0]
        )
        np.testing.assert_array_equal(
            self.env._balloon_flights[1, 0], [10.0, 10.0, 10.0, 11.0]
        )

    def test_invalid_track_shape_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Trajectory shape mismatch"):
            self.env.update_source_trajectories(np.zeros((3, 6, 4)))

    def test_reset_generation_requires_injected_tracks(self):
        with self.assertRaisesRegex(RuntimeError, "update_source_trajectories"):
            self.env._BalloonPoppingEnv__generate_balloon_flights()


if __name__ == "__main__":
    unittest.main()
