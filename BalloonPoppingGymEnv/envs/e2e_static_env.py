import gymnasium as gym

from BalloonPoppingGymEnv.utils.schema import Schema
from BalloonPoppingGymEnv.utils.reward_calculator import RewardCalculator
from BalloonPoppingGymEnv.utils.e2e_utils import (
    action_space,
    observation_space,
    launch_schedule,
    scale_rl_action,
)

class E2EStaticEnv(gym.Wrapper):
    def __init__(self, env: gym.Env, given_parameters):
        super().__init__(env)

        self.env = env
        self.num_balloons = 1
        self.reward_calculator = RewardCalculator(given_parameters)

        self.action_space = action_space
        self.observation_space = observation_space

    def reset(self, seed=None, options=None):
        # Update balloon pos
        # self.env.update_balloons

        self.reward_calculator.reset()
        self.env.reset(seed=seed, options=options)
        rl_obs, info = launch_schedule(self.env)

        return rl_obs, info

    def step(self, rl_action):
        rl_reward = 0.0
        pop_count = 0.0
        tvc, roll, throttle = scale_rl_action(rl_action)

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
        simulation_time = observation[Schema.Observation.SIMULATION_TIME]

        # Check truncated
        if self.controller.burnout and not (terminated or truncated):
            burn_elapsed = simulation_time - self.controller.ignition_time
            if burn_elapsed >= self.controller.burn_time + 5.0:
                truncated = True

        if not (terminated or truncated) and info.get("popped_count", 0) == self.num_balloons:
            truncated = True

        # Select target

        # Get target state
        raw_balloon_states = observation[Schema.Observation.BALLOON_STATES]

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
