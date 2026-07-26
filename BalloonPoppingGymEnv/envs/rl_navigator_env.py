from pathlib import Path
import gymnasium as gym
from gymnasium import spaces
import numpy as np

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
        self.given_parameters = given_parameters

        self.estimator = Estimator(given_parameters)
        self.selector = Selector(given_parameters)
        self.controller = Controller(given_parameters)
        self.reward_calculator = RewardCalculator()

        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(3,),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(10,), # [rel_pos(3), rel_vel(3), rocket_vel(3), altitude(1)]
            dtype=np.float32
        )

        self.launch_inclination_heading = None

        # Balloon trajectory pool
        self.pools = [
            np.load(pool_path, mmap_mode="r") for pool_path in pool_path_list
        ]
        self.pool_idx = 0
        self.num_balloons = self.env.unwrapped.balloon_parameters["num"]
        self.sample_rng = np.random.default_rng()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

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

        self.env.update_source_trajectories(tracks)
        observation, info = self.env.reset()

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

        self.rocket_state = self.estimator.estimate_rocket(observation)
        balloon_states = self.estimator.predict_balloons(observation)

        target_idx = self.selector.select_target(
            balloon_states=balloon_states,
            rocket_state=self.rocket_state,
        )

        target_state = self.estimator.predict_target(
            observation=observation,
            target_idx=target_idx,
        )

        rl_obs = compute_rl_observation(
            rocket_state=self.rocket_state,
            target_state=target_state,
        )

        return rl_obs, info

    def step(self, rl_action):
        rl_reward = 0.0
        popped = 0.0
        desired_acc = scale_rl_action(rl_action)

        for _ in range(RL_FRAME_SKIP):
            tvc, roll, throttle = self.controller.compute(
                rocket_state=self.rocket_state,
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
            popped += reward
            self.rocket_state = self.estimator.estimate_rocket(observation)

            if terminated or truncated:
                break

        balloon_states = self.estimator.predict_balloons(observation)

        target_idx = self.selector.select_target(
            balloon_states=balloon_states,
            rocket_state=self.rocket_state,
        )

        target_state = self.estimator.predict_target(
            observation=observation,
            target_idx=target_idx,
        )

        rl_obs = compute_rl_observation(
            rocket_state=self.rocket_state,
            target_state=target_state,
        )

        rl_reward, rl_reward_dict = self.reward_calculator.compute(
            observation=observation,
            popped=popped,
            rocket_state=self.rocket_state,
            target_idx=target_idx,
            target_state=target_state,
            desired_acc=desired_acc,
            terminated=terminated,
            info=info,
        )

        return rl_obs, rl_reward, terminated, truncated, info
