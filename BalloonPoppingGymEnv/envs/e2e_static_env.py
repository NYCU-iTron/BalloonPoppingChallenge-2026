import gymnasium as gym

from BalloonPoppingGymEnv.utils.schema import Schema
from BalloonPoppingGymEnv.agents.gnc.selector import Selector
from BalloonPoppingGymEnv.agents.gnc.estimator import Estimator
from BalloonPoppingGymEnv.utils.reward_calculator import RewardCalculator
from BalloonPoppingGymEnv.utils.e2e_utils import (
    action_space,
    observation_space,
    compute_rl_observation,
)

class E2EStaticEnv(gym.Wrapper):
    def __init__(self, env: gym.Env, given_parameters):
        super().__init__(env)

        self.selector = Selector(given_parameters)
        self.estimator = Estimator(given_parameters)
        self.reward_calculator = RewardCalculator(given_parameters)

        self.action_space = action_space
        self.observation_space = observation_space

    def reset(self, seed=None, options=None):
        self.estimator.reset()
        self.selector.reset()
        self.reward_calculator.reset()

        self.rocket_state = None
        self.launch_inclination_heading = None

        observation, info = self.env.reset(seed=seed, options=options)

        self.launch_inclination_heading = self.selector.get_launch_heading(observation)

        # Update states
        simulation_time = observation[Schema.Observation.SIMULATION_TIME]

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
