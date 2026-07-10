import gymnasium as gym
from gymnasium import spaces
import numpy as np

from BalloonPoppingGymEnv.agents.gnc.estimator import Estimator
from BalloonPoppingGymEnv.agents.gnc.selector import Selector
from BalloonPoppingGymEnv.agents.gnc.controller import Controller


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

        rl_obs = self.get_rl_observation(self.rocket_state, target_state)

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

        rl_obs = self.get_rl_observation(self.rocket_state, target_state)
        rl_reward = self.compute_reward(self.rocket_state, target_state, info, rl_action)

        return rl_obs, rl_reward, terminated, truncated, info

    def get_rl_observation(self, rocket_state, target_state):
        rel_pos = target_state[0:3] - rocket_state[0:3]
        rel_vel = (target_state[3:6] if target_state.size >= 6 else np.zeros(3)) - rocket_state[3:6]

        rocket_vel = rocket_state[3:6]
        z = rocket_state[2]

        rl_obs = np.concatenate([rel_pos, rel_vel, rocket_vel, [z]]).astype(np.float32)

        return rl_obs

    def compute_reward(self, rocket_state, target_state, info, rl_action):
        # 1. Sparse Terminal Events (Critical Gates)
        if info.get("crashed", False):
            return -100.0

        if info.get("popped_count", 0) > 0:
            # Scale reward dynamically if multiple balloons are popped
            return 200.0 * info["popped_count"]

        # 2. Non-Linear Geometrical Tracking (Bounded Distance Penalty)
        # Replaced linear penalty with a hyperbolic function to prevent scaling explosion over far distances.
        # This keeps the maximum step penalty bounded between [0.0, -0.5]
        rel_pos = target_state[0:3] - rocket_state[0:3]
        distance = np.linalg.norm(rel_pos)
        distance_penalty = -0.5 * (distance / (distance + 100.0))

        # 3. Kinematic Alignment (Closing Velocity Reward)
        # Scaled down coefficient to prevent random high-speed drift exploitation.
        alignment_reward = 0.0
        if distance > 1e-3:
            rocket_vel = rocket_state[3:6]
            unit_rel_pos = rel_pos / distance
            closing_speed = np.dot(rocket_vel, unit_rel_pos)
            alignment_reward = 0.02 * closing_speed

        # 4. Actuator Regularization (Smooth Action Delta Penalty)
        # Tracks and penalizes high-frequency control chatter (jitter) instead of absolute thrust effort.
        if not hasattr(self, "prev_action") or self.prev_action is None:
            self.prev_action = np.zeros(4, dtype=np.float32)

        action_delta = rl_action - self.prev_action
        smoothness_penalty = -0.05 * np.linalg.norm(action_delta)

        # Cache current action as historical state reference for the next cycle step
        self.prev_action = rl_action.copy()

        # 5. Temporal Efficiency (Balanced Time Penalty)
        # Lowered to -0.02. At 20Hz, a full 10-second flight costs -4.0 total points.
        time_penalty = -0.02

        # Aggregate final reshaped scalar feedback
        total_reward = distance_penalty + alignment_reward + smoothness_penalty + time_penalty
        return float(total_reward)
