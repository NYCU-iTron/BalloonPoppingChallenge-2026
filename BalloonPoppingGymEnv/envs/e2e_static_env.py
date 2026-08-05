import gymnasium as gym

from BalloonPoppingGymEnv.utils.schema import Schema
from BalloonPoppingGymEnv.utils.static_reward_calculator import StaticRewardCalculator
from BalloonPoppingGymEnv.utils.e2e_utils import (
    action_space,
    observation_space,
    RLObservator,
    launch_schedule,
    scale_rl_action,
)

class E2EStaticEnv(gym.Wrapper):
    def __init__(self, env: gym.Env, given_parameters):
        super().__init__(env)

        self.env = env
        self.num_balloons = 1
        self.rl_observator = RLObservator(given_parameters)
        self.reward_calculator = StaticRewardCalculator(given_parameters)

        self.action_space = action_space
        self.observation_space = observation_space

    def reset(self, seed=None, options=None):
        # Update balloon pos
        # self.env.update_balloons

        self.reward_calculator.reset()
        self.env.reset(seed=seed, options=options)
        observation, info = launch_schedule(self.env, self.rl_observator)

        target_state = observation[Schema.Observation.BALLOON_STATES][0]
        rl_obs = self.rl_observator.get_rl_obs(target_state=target_state)

        return rl_obs, info

    def step(self, rl_action):
        rl_reward = 0.0
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

        # Check truncated
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
