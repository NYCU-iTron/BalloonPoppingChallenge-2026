import gymnasium as gym
from gymnasium import spaces
import numpy as np

from BalloonPoppingGymEnv.agents.gnc.estimator import Estimator
from BalloonPoppingGymEnv.agents.gnc.selector import Selector
from BalloonPoppingGymEnv.agents.gnc.controller import Controller
from BalloonPoppingGymEnv.utils.rl_utils import compute_rl_observation, compute_rl_reward

class RLNavigatorEnv(gym.Wrapper):
    def __init__(self, env, given_parameters):
        super().__init__(env)
        self.given_parameters = given_parameters

        self.estimator = Estimator(given_parameters)
        self.selector = Selector(given_parameters)
        self.controller = Controller(given_parameters)

        # Action:
        #   3D acceleration command (3)
        #   Throttle command (1)
        self.action_space = spaces.Box(
            low=np.array([-30.0, -30.0, -30.0, 0.0], dtype=np.float32),
            high=np.array([30.0, 30.0, 30.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # Observation:
        #   Relative position (3)
        #   Relative velocity (3)
        #   Rocket velocity (3)
        #   Rocket altitude (1)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )

    def reset(self, **kwargs):
        self.observation, info = self.env.reset(**kwargs)

        self.estimator.reset()
        self.selector.reset()
        self.controller.reset()

        # Fast forward to the launch time
        while not self.selector.should_launch(self.observation):
            idle_action = {
                "launch": False,  # Keep clamp engaged on the launchpad
                "launch_inclination_heading": np.array([90.0, 0.0]),
                "tvc": np.zeros(2),
                "roll": 0.0,
                "throttle": self.controller.throttle_min
            }
            self.observation, reward, terminated, truncated, info = self.env.step(idle_action)

            if terminated or truncated:
                self.observation, info = self.env.reset(**kwargs)
                break

        # Update states
        self.rocket_state = self.estimator.estimate_rocket(self.observation)
        balloon_states = self.estimator.predict_balloons(self.observation)

        # Update target
        target_idx = self.selector.select_target(balloon_states, self.rocket_state)
        target_state = self.estimator.predict_target(self.observation, target_idx)

        self.prev_action = np.zeros(4, dtype=np.float32)
        rl_obs = compute_rl_observation(self.rocket_state, target_state)

        return rl_obs, info

    def step(self, rl_action):
        a_cmd_world = rl_action[0:3]
        desired_throttle = float(rl_action[3])

        tvc, roll, throttle = self.controller.compute(self.rocket_state, a_cmd_world, desired_throttle)

        is_launched = self.selector.should_launch(self.observation)
        launch_inclination_heading = self.selector.get_launch_heading(self.observation)

        action = {
            "launch": is_launched,
            "launch_inclination_heading": launch_inclination_heading,
            "tvc": tvc,
            "roll": roll,
            "throttle": throttle,
        }

        self.observation, reward, terminated, truncated, info = self.env.step(action)

        # Update states
        self.rocket_state = self.estimator.estimate_rocket(self.observation)
        balloon_states = self.estimator.predict_balloons(self.observation)

        # Update target
        target_idx = self.selector.select_target(balloon_states, self.rocket_state)
        target_state = self.estimator.predict_target(self.observation, target_idx)

        action_delta = rl_action - self.prev_action
        self.prev_action = rl_action.copy()

        rl_obs = compute_rl_observation(self.rocket_state, target_state)
        rl_reward = compute_rl_reward(self.observation, info, self.rocket_state, target_state,
                                      reward, terminated, action_delta)

        return rl_obs, rl_reward, terminated, truncated, info
