import gymnasium as gym
from gymnasium import spaces
import numpy as np

from BalloonPoppingGymEnv.agents.gnc.estimator import Estimator
from BalloonPoppingGymEnv.agents.gnc.selector import Selector
from BalloonPoppingGymEnv.agents.gnc.controller import Controller


class RLTrainingEnv(gym.Wrapper):
    def __init__(self, env, given_parameters):
        super().__init__(env)
        self.given_parameters = given_parameters

        # State tracking cache for observations
        self.current_obs = None

        self.estimator = Estimator(given_parameters)
        self.selector = Selector(given_parameters)
        self.controller = Controller(given_parameters)

        # Cast tracking limits to float32 explicitly to suppress Gymnasium precision warnings
        self.action_space = spaces.Box(
            low=np.array([-30.0, -30.0, -30.0, 0.0], dtype=np.float32),
            high=np.array([30.0, 30.0, 30.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # Cache the initial observation
        self.current_obs = obs

        self.estimator.reset()
        self.selector.reset()
        self.controller.reset()
        return self._get_rl_obs(obs, info), info

    def step(self, rl_action):
        # 1. Parse RL action into Guidance acceleration commands
        a_cmd_world = rl_action[0:3]
        desired_throttle = float(rl_action[3])

        # 2. Use the CACHED observation from the current state to run GNC pipeline
        rocket_state = self.estimator.estimate_rocket(self.current_obs)
        balloon_states = self.estimator.predict_balloons(self.current_obs)
        target_idx = self.selector.select_target(balloon_states, rocket_state)

        # 3. Use Controller baseline to bridge Guidance to Actuators
        tvc, roll, throttle = self.controller.compute(rocket_state, a_cmd_world, desired_throttle)

        # 4. Construct raw action package for the physics engine
        raw_action = {
            "launch": True,
            "launch_inclination_heading": np.array([90.0, 0.0]),
            "tvc": tvc,
            "roll": roll,
            "throttle": throttle,
        }

        # 5. Step physics and extract the NEW observation
        next_obs, raw_reward, terminated, truncated, info = self.env.step(raw_action)

        # 6. Update the cache for the next time step loop
        self.current_obs = next_obs

        # 7. Format outputs for the RL brain
        rl_obs = self._get_rl_obs(next_obs, info)
        rl_reward = self._compute_shaped_reward(rocket_state, info, terminated)

        return rl_obs, rl_reward, terminated, truncated, info

    def _get_rl_obs(self, observation, info):
        # Build relative kinematics features
        rocket_state = self.estimator.estimate_rocket(observation)
        balloon_states = self.estimator.predict_balloons(observation)
        target_idx = self.selector.select_target(balloon_states, rocket_state)
        target_state = self.estimator.predict_target(observation, target_idx)

        if target_state is None or np.isnan(target_state).any():
            return np.zeros(9, dtype=np.float32)

        rel_pos = target_state[0:3] - rocket_state[0:3]
        rel_vel = (target_state[3:6] if target_state.size >= 6 else np.zeros(3)) - rocket_state[3:6]
        rocket_vel = rocket_state[3:6]

        return np.concatenate([rel_pos, rel_vel, rocket_vel]).astype(np.float32)

    def _compute_shaped_reward(self, rocket_state, info, terminated):
        if info.get("crashed", False):
            return -100.0
        if info.get("popped_count", 0) > 0:
            return 200.0 * info["popped_count"]
        return -0.1
