import gymnasium as gym
import numpy as np

from BalloonPoppingGymEnv.utils.schema import Schema
from BalloonPoppingGymEnv.utils.static_reward_calculator import StaticRewardCalculator
from BalloonPoppingGymEnv.utils.reference_trajectory import generate_reference_trajectory
from BalloonPoppingGymEnv.utils.e2e_utils import (
    action_space,
    observation_space,
    RLObservator,
    launch_schedule,
    scale_rl_action,
)

TARGET_AGL_RANGE = (60.0, 150.0)
TARGET_RADIUS_MIN = 5.0
TARGET_RADIUS_FRAC = 0.5

class E2EStaticEnv(gym.Wrapper):
    def __init__(self, env: gym.Env, given_parameters):
        super().__init__(env)

        self.env = env
        self.num_balloons = 1
        self.rl_observator = RLObservator(given_parameters)
        self.reward_calculator = StaticRewardCalculator(given_parameters)
        self.ground_elevation = float(
            given_parameters[Schema.Given.Section.ENVIRONMENT][Schema.Given.Environment.ELEVATION]
        )

        self._target_rng = np.random.default_rng()

        self.action_space = action_space
        self.observation_space = observation_space

    def _sample_target_position(self) -> np.ndarray:
        target_agl = self._target_rng.uniform(*TARGET_AGL_RANGE)
        target_z = self.ground_elevation + target_agl
        radius = self._target_rng.uniform(TARGET_RADIUS_MIN, TARGET_RADIUS_FRAC * target_agl)
        bearing = self._target_rng.uniform(0.0, 2.0 * np.pi)
        return np.array([radius * np.cos(bearing), radius * np.sin(bearing), target_z])

    def reset(self, seed=None, options=None):
        self.reward_calculator.reset()
        if seed is not None:
            self._target_rng = np.random.default_rng(seed)

        target_position = self._sample_target_position()
        self.env.update_balloons(target_position)
        self.env.reset(seed=seed, options=options)

        observation, info = launch_schedule(self.env, self.rl_observator)

        reference_trajectory = generate_reference_trajectory(
            start_pos=self.rl_observator.rocket_pos.copy(),
            start_vel=self.rl_observator.rocket_vel.copy(),
            target_pos=target_position,
            ground_elevation=self.ground_elevation,
        )
        self.reward_calculator.set_reference_trajectory(reference_trajectory)

        target_state = observation[Schema.Observation.BALLOON_STATES][0]
        rl_obs = self.rl_observator.get_rl_obs(target_state=target_state)

        return rl_obs, info

    def step(self, rl_action):
        pop_count = 0.0
        tvc, roll, throttle = scale_rl_action(rl_action)

        action = {
            "launch": True,
            "launch_inclination_heading": [0, 0],
            "tvc": tvc,
            "roll": roll,
            "throttle": throttle,
        }

        observation, reward, terminated, truncated, info = self.env.step(action)
        pop_count += reward

        # Update states
        self.rl_observator.update_state(observation)

        # Check truncated -- burnout means zero thrust, so TVC has no authority
        # left to act on regardless of what the policy commands from here.
        if self.rl_observator.burnout and not (terminated or truncated):
            truncated = True

        if not (terminated or truncated) and info.get("popped_count", 0) == self.num_balloons:
            truncated = True

        # Get target state
        target_state = observation[Schema.Observation.BALLOON_STATES][0]

        # Get rl observation
        rl_obs = self.rl_observator.get_rl_obs(
            target_state=target_state,
            action=action,
        )

        rocket_sensors = observation[Schema.Observation.ROCKET_SENSORS]
        rocket_pos = rocket_sensors[6:9]
        rocket_vel = rocket_sensors[9:12]
        rocket_state = np.concatenate([rocket_pos, rocket_vel])
        target_idx = 0

        rl_reward, rl_reward_dict = self.reward_calculator.compute(
            pop_count=pop_count,
            rocket_state=rocket_state,
            target_idx=target_idx,
            target_state=target_state,
            terminated=terminated,
            info=info,
        )

        return rl_obs, rl_reward, terminated, truncated, info
