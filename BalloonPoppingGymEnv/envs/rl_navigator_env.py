from pathlib import Path
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from BalloonPoppingGymEnv.utils.schema import Schema
from BalloonPoppingGymEnv.agents.gnc.estimator import Estimator
from BalloonPoppingGymEnv.agents.gnc.selector import Selector
from BalloonPoppingGymEnv.agents.gnc.controller import Controller
from BalloonPoppingGymEnv.utils.reward_calculator import RewardCalculator
from BalloonPoppingGymEnv.utils.rl_utils import (
    compute_rl_observation,
    scale_rl_action,
    RL_FRAME_SKIP,
)

class RLNavigatorEnv(gym.Wrapper):
    def __init__(self, env: gym.Env, given_parameters, pool_path_list: list[Path]):
        super().__init__(env)

        self.rocket_state = None
        self.launch_inclination_heading = None

        self.estimator = Estimator(given_parameters)
        self.selector = Selector(given_parameters)
        self.controller = Controller(given_parameters)
        self.reward_calculator = RewardCalculator(given_parameters)

        # Balloon trajectory pool
        self.pools = [
            np.load(pool_path, mmap_mode="r") for pool_path in pool_path_list
        ]
        self.pool_idx = 0
        self.num_balloons = self.env.unwrapped.balloon_parameters["num"]
        self.sample_rng = np.random.default_rng()

        # ax, ay, az
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(3,),
            dtype=np.float32
        )

        # rel_pos (3)
        # rel_vel (3)
        # rocket_vel (3)
        # altitude (2)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(11,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        # Sample tracks from pool
        indices = self.sample_rng.choice(
            self.pools[self.pool_idx].shape[0], size=self.num_balloons, replace=False
        )
        tracks = np.asarray(self.pools[self.pool_idx][indices])

        # Rotate tracks
        theta = self.sample_rng.uniform(0.0, 2.0 * np.pi)
        c, s = np.cos(theta), np.sin(theta)
        rotation = np.array([[c, -s], [s, c]], dtype=tracks.dtype)
        tracks[:, 0:2, :] = np.einsum("ij,njt->nit", rotation, tracks[:, 0:2, :])
        tracks[:, 3:5, :] = np.einsum("ij,njt->nit", rotation, tracks[:, 3:5, :])

        self.estimator.reset()
        self.selector.reset()
        self.controller.reset()
        self.reward_calculator.reset()

        self.rocket_state = None
        self.launch_inclination_heading = None

        self.env.unwrapped.set_wind_rotation(theta)
        self.env.unwrapped.update_source_trajectories(tracks)
        observation, info = self.env.reset(seed=seed, options=options)

        # Fast forward to the launch time
        while not self.selector.should_launch(observation):
            action = {
                "launch": False,
                "launch_inclination_heading": np.array([90.0, 0.0]),
                "tvc": np.zeros(2),
                "roll": 0.0,
                "throttle": 0.0
            }
            observation, reward, terminated, truncated, info = self.env.step(action)

            if terminated or truncated:
                break

        self.launch_inclination_heading = self.selector.get_launch_heading(observation)

        # Update states
        self.rocket_state = self.estimator.estimate_rocket(observation)
        simulation_time = observation[Schema.Observation.SIMULATION_TIME]
        self.controller.update(
            rocket_state=self.rocket_state,
            simulation_time=simulation_time,
        )

        # Select target
        pred_balloon_states = self.estimator.predict_balloons(observation)
        target_idx = self.selector.select_target(
            balloon_states=pred_balloon_states,
            rocket_state=self.rocket_state,
        )

        # Get target state
        raw_balloon_states = observation[Schema.Observation.BALLOON_STATES]
        target_state = self.selector.get_target_state(
            balloon_states=raw_balloon_states,
            target_idx=target_idx,
        )

        rl_obs = compute_rl_observation(
            rocket_state=self.rocket_state,
            target_state=target_state,
        )

        return rl_obs, info

    def step(self, rl_action):
        rl_reward = 0.0
        pop_count = 0.0
        desired_acc = scale_rl_action(rl_action)

        for _ in range(RL_FRAME_SKIP):
            tvc, roll, throttle = self.controller.compute(
                desired_acc=desired_acc,
            )

            action = {
                "launch": True,
                "launch_inclination_heading": self.launch_inclination_heading,
                "tvc": tvc,
                "roll": roll,
                "throttle": throttle,
            }

            observation, reward, terminated, truncated, info = self.env.step(action)
            pop_count += reward

            # Update states
            self.rocket_state = self.estimator.estimate_rocket(observation)
            simulation_time = observation[Schema.Observation.SIMULATION_TIME]
            self.controller.update(
                rocket_state=self.rocket_state,
                simulation_time=simulation_time,
            )

            # Check truncated
            if self.controller.burnout and not (terminated or truncated):
                burn_elapsed = simulation_time - self.controller.ignition_time
                if burn_elapsed >= self.controller.burn_time + 5.0:
                    truncated = True

            if not (terminated or truncated) and info.get("popped_count", 0) == self.num_balloons:
                truncated = True

            if terminated or truncated:
                break

        # Select target
        pred_balloon_states = self.estimator.predict_balloons(observation)
        target_idx = self.selector.select_target(
            balloon_states=pred_balloon_states,
            rocket_state=self.rocket_state,
        )

        # Get target state
        raw_balloon_states = observation[Schema.Observation.BALLOON_STATES]
        target_state = self.selector.get_target_state(
            balloon_states=raw_balloon_states,
            target_idx=target_idx,
        )

        rl_obs = compute_rl_observation(
            rocket_state=self.rocket_state,
            target_state=target_state,
        )

        rl_reward, rl_reward_dict = self.reward_calculator.compute(
            observation=observation,
            pop_count=pop_count,
            rocket_state=self.rocket_state,
            target_idx=target_idx,
            target_state=target_state,
            desired_acc=desired_acc,
            terminated=terminated,
            info=info,
        )

        return rl_obs, rl_reward, terminated, truncated, info
