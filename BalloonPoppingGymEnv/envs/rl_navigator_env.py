import gymnasium as gym
from gymnasium import spaces
import numpy as np

from BalloonPoppingGymEnv.agents.gnc.estimator import Estimator
from BalloonPoppingGymEnv.agents.gnc.selector import Selector
from BalloonPoppingGymEnv.agents.gnc.controller import Controller
from BalloonPoppingGymEnv.utils.schema import Schema


class RLNavigatorEnv(gym.Wrapper):
    def __init__(self, env, given_parameters):
        super().__init__(env)
        self.given_parameters = given_parameters

        self.estimator = Estimator(given_parameters)
        self.selector = Selector(given_parameters)
        self.controller = Controller(given_parameters)

        self.action_space = spaces.Box(
            low=np.array([-30.0, -30.0, -30.0, 0.0], dtype=np.float32),
            high=np.array([30.0, 30.0, 30.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)

        self.estimator.reset()
        self.selector.reset()
        self.controller.reset()

        # TODO: Fast forward to launch time
        t = observation[Schema.Observation.SIMULATION_TIME]
        is_launched = t >= self.selector.get_launch_time(observation)


        return self.get_rl_observation(observation), info

    def step(self, rl_action):
        a_cmd_world = rl_action[0:3]
        desired_throttle = float(rl_action[3])

        tvc, roll, throttle = self.controller.compute(rocket_state, a_cmd_world, desired_throttle)

        is_launched = self.selector.should_launch(self.observation)
        launch_inclination_heading = self.selector.get_launch_heading(self.observation)

        action = {
            "launch": is_launched,
            "launch_inclination_heading": launch_inclination_heading,
            "tvc": tvc,
            "roll": roll,
            "throttle": throttle,
        }

        observation, reward, terminated, truncated, info = self.env.step(action)

        # Update states
        rocket_state = self.estimator.estimate_rocket(observation)
        balloon_states = self.estimator.predict_balloons(observation)

        # Update target
        target_idx = self.selector.select_target(balloon_states, rocket_state)
        target_state = self.estimator.predict_target(observation, target_idx)

        rl_obs = self.get_rl_observation(rocket_state, target_state)


        # --------------------------------- RL Reward -------------------------------- #
        if info.get("crashed", False):
            rl_reward = -100.0
        if info.get("popped_count", 0) > 0:
            rl_reward = 200.0 * info["popped_count"]
        else:
            rl_reward = -1.0

        return rl_obs, rl_reward, terminated, truncated, info


    def get_rl_observation(self, rocket_state, target_state):

        rel_pos = target_state[0:3] - rocket_state[0:3]
        rel_vel = (target_state[3:6] if target_state.size >= 6 else np.zeros(3)) - rocket_state[3:6]
        rocket_vel = rocket_state[3:6]
        z = rocket_state[2]

        rl_obs = np.concatenate([rel_pos, rel_vel, rocket_vel, z]).astype(np.float32)

        return rl_obs
