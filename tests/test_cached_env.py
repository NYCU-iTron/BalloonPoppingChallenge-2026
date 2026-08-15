import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
from BalloonPoppingGymEnv.envs.cached_env import (
    CachedBalloonEnv,
    scenario_parameter_fingerprint,
)


_PRODUCTION_GENERATOR = "_BalloonPoppingEnv__generate_balloon_flights"


class TestCachedBalloonEnv(unittest.TestCase):
    def _bare_env(self, cache_dir: Path, *, release_steps=(0, 2)):
        env = CachedBalloonEnv.__new__(CachedBalloonEnv)
        env.trajectory_cache_dir = cache_dir
        env.parameter_fingerprint = scenario_parameter_fingerprint(
            {
                "scenario": {"number": 1, "random_seed": 0},
                "simulation": {"max_time": 0.4, "time_step": 0.1},
                "balloon": {"num": 2},
            }
        )
        env.last_trajectory_cache_path = None
        env.last_trajectory_cache_hit = False
        env.scenario_parameters = {"number": 1}
        env.simulation_parameters = {"max_time": 0.4, "time_step": 0.1}
        env.balloon_parameters = {"num": 2}
        env._balloon_release_at_step = np.asarray(release_steps, dtype=int)
        env._np_random_seed = 7
        return env

    def test_first_generation_is_saved_and_second_run_is_exactly_replayed(self):
        generated = np.arange(48, dtype=float).reshape(2, 6, 4)

        def fake_generation(env):
            env._balloon_flights = generated.copy()

        with tempfile.TemporaryDirectory() as directory:
            first = self._bare_env(Path(directory))
            with patch.object(
                BalloonPoppingEnv,
                _PRODUCTION_GENERATOR,
                side_effect=fake_generation,
            ) as production:
                first._BalloonPoppingEnv__generate_balloon_flights()

            self.assertEqual(production.call_count, 1)
            self.assertFalse(first.last_trajectory_cache_hit)
            self.assertTrue(first.last_trajectory_cache_path.is_file())

            second = self._bare_env(Path(directory))
            with patch.object(
                BalloonPoppingEnv,
                _PRODUCTION_GENERATOR,
                side_effect=AssertionError("Monte Carlo should not run"),
            ):
                second._BalloonPoppingEnv__generate_balloon_flights()

            self.assertTrue(second.last_trajectory_cache_hit)
            np.testing.assert_array_equal(second._balloon_flights, generated)

    def test_release_schedule_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            first = self._bare_env(Path(directory))
            first._balloon_flights = np.zeros((2, 6, 4))
            cache_path = first.trajectory_cache_path(7)
            first._write_trajectory_cache(cache_path, 7)

            second = self._bare_env(Path(directory), release_steps=(2, 0))
            with self.assertRaisesRegex(ValueError, "release schedule"):
                second._BalloonPoppingEnv__generate_balloon_flights()

    def test_configured_seed_is_not_part_of_the_physics_fingerprint(self):
        first = {"scenario": {"number": 1, "random_seed": 1}, "value": 2}
        second = {"scenario": {"number": 1, "random_seed": 99}, "value": 2}

        self.assertEqual(
            scenario_parameter_fingerprint(first),
            scenario_parameter_fingerprint(second),
        )

    def test_physics_change_gets_a_different_fingerprint(self):
        first = {"scenario": {"number": 1}, "simulation": {"time_step": 0.1}}
        second = {"scenario": {"number": 1}, "simulation": {"time_step": 0.2}}

        self.assertNotEqual(
            scenario_parameter_fingerprint(first),
            scenario_parameter_fingerprint(second),
        )


if __name__ == "__main__":
    unittest.main()
