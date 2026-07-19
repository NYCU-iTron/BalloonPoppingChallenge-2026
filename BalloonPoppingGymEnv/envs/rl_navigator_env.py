import gymnasium as gym
from gymnasium import spaces
import numpy as np

from BalloonPoppingGymEnv.agents.gnc.estimator import Estimator
from BalloonPoppingGymEnv.agents.gnc.selector import Selector
from BalloonPoppingGymEnv.agents.gnc.navigator import Navigator
from BalloonPoppingGymEnv.agents.gnc.controller import Controller
from BalloonPoppingGymEnv.utils.rl_utils import (
    compute_rl_observation,
    compute_rl_reward,
    compute_target_distance,
    failure_penalty,
    RL_FRAME_SKIP,
    RL_RESIDUAL_ACCEL_LIMIT,
    RL_RESIDUAL_THROTTLE_LIMIT,
)

class RLNavigatorEnv(gym.Wrapper):
    # Miss detection: the episode ends once the rocket has receded more than
    # MISS_MARGIN meters beyond its closest approach to the tracked target for
    # MISS_PATIENCE consecutive policy decisions. The margin absorbs transient
    # geometry during maneuvers; the patience filters momentary recessions.
    MISS_MARGIN = 50.0
    MISS_PATIENCE = 5
    def __init__(self, env, given_parameters, pool_path):
        super().__init__(env)
        self.given_parameters = given_parameters

        self.estimator = Estimator(given_parameters)
        self.selector = Selector(given_parameters)
        self.navigator = Navigator(given_parameters)
        self.controller = Controller(given_parameters)

        # Balloon trajectory pool
        self._pool = np.load(pool_path, mmap_mode="r")
        self._pool_capacity = self._pool.shape[0]
        self._num_balloons = self.env.unwrapped.balloon_parameters["num"]
        self._sample_rng = np.random.default_rng()

        # Action (RESIDUAL on the PN guidance law, see rl_utils):
        #   3D acceleration correction added to the PN command (3)
        #   Throttle correction added to the PN throttle (1)
        # Zero action = pure proportional navigation.
        self.action_space = spaces.Box(
            low=np.array([-RL_RESIDUAL_ACCEL_LIMIT] * 3 + [-RL_RESIDUAL_THROTTLE_LIMIT], dtype=np.float32),
            high=np.array([RL_RESIDUAL_ACCEL_LIMIT] * 3 + [RL_RESIDUAL_THROTTLE_LIMIT], dtype=np.float32),
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

    def _resample_and_inject(self):
        """Sample a fresh balloon subset from the pool and hand it to the inner env."""
        indices = self._sample_rng.choice(
            self._pool_capacity, size=self._num_balloons, replace=False
        )

        tracks = np.asarray(self._pool[indices])
        self.env.unwrapped.update_source_trajectories(tracks)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._sample_rng = np.random.default_rng(seed)

        self._resample_and_inject()
        self.observation, info = self.env.reset(seed=seed, options=options)

        self.estimator.reset()
        self.selector.reset()
        self.navigator.reset()
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
                self._resample_and_inject()
                self.observation, info = self.env.reset(options=options)
                break

        # Update states
        self.rocket_state = self.estimator.estimate_rocket(self.observation)
        balloon_states = self.estimator.predict_balloons(self.observation)

        # Update target
        target_idx = self.selector.select_target(balloon_states, self.rocket_state)
        target_state = self.estimator.predict_target(self.observation, target_idx)
        self.target_state = target_state

        self._reset_target_tracking(target_idx, target_state)

        self.prev_action = np.zeros(4, dtype=np.float32)
        rl_obs = compute_rl_observation(self.rocket_state, target_state)

        return rl_obs, info

    def _reset_target_tracking(self, target_idx, target_state):
        """Restart distance bookkeeping for a (new) tracked target.

        Called on reset and whenever the selector switches targets, so neither
        the shaping delta nor the miss detector ever sees the distance jump
        between two different balloons.
        """
        self._tracked_target_idx = target_idx
        self._prev_target_distance = compute_target_distance(self.rocket_state, target_state)
        self._closest_target_distance = self._prev_target_distance
        self._recede_decisions = 0

    def step(self, rl_action):
        # The policy emits a residual, held for the frame skip; the PN baseline
        # underneath is recomputed every sim step so the guidance stays reactive
        # between policy decisions.
        residual_accel = rl_action[0:3]
        residual_throttle = float(rl_action[3])

        # Smoothness penalty
        action_delta = rl_action - self.prev_action
        self.prev_action = rl_action.copy()

        total_rl_reward = 0.0
        terminated = truncated = False
        info = {}
        target_state = self.target_state

        for i in range(RL_FRAME_SKIP):
            a_pn, throttle_pn = self.navigator.compute(self.target_state, self.rocket_state)
            if a_pn is None:
                # No valid target geometry: defer to the controller's safe mode.
                a_cmd_world, desired_throttle = None, None
            else:
                a_cmd_world = a_pn + residual_accel
                desired_throttle = float(np.clip(throttle_pn + residual_throttle, 0.0, 1.0))

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
            self.target_state = target_state

            if target_idx != self._tracked_target_idx:
                self._reset_target_tracking(target_idx, target_state)

            # Update closest approach BEFORE scoring so the crash penalty (inside
            # compute_rl_reward) is graded by the closest distance, consistent
            # with the miss penalty below.
            distance = compute_target_distance(self.rocket_state, target_state)
            self._closest_target_distance = min(self._closest_target_distance, distance)

            step_delta = action_delta if i == 0 else np.zeros_like(action_delta)
            total_rl_reward += compute_rl_reward(self.observation, info, self.rocket_state, target_state,
                                                 reward, terminated, step_delta,
                                                 self._prev_target_distance, self._closest_target_distance)

            self._prev_target_distance = distance

            # All balloons popped: the objective is complete, so end the episode
            # instead of burning the rest of the flight as junk experience. The
            # pop rewards for this sub-step were already banked above.
            if info.get("popped_count", 0) >= self._num_balloons:
                terminated = True
                info["all_popped"] = True
                break

            if terminated or truncated:
                break

        # Miss detection (once per policy decision): receding beyond the closest
        # approach means the pass failed and the rocket cannot realistically turn
        # back, so the post-miss flight would only pollute the rollout buffer.
        # The penalty is graded by closest approach: a near miss costs least,
        # preserving the gradient toward actually popping.
        if not (terminated or truncated):
            receding = self._prev_target_distance > self._closest_target_distance + self.MISS_MARGIN
            self._recede_decisions = self._recede_decisions + 1 if receding else 0
            if self._recede_decisions >= self.MISS_PATIENCE:
                terminated = True
                info["missed_target"] = True
                total_rl_reward += failure_penalty(self._closest_target_distance)

        rl_obs = compute_rl_observation(self.rocket_state, target_state)

        return rl_obs, total_rl_reward, terminated, truncated, info
