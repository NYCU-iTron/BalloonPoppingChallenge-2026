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
    # End the episode once the rocket has receded MISS_MARGIN m beyond its
    # closest approach for MISS_PATIENCE consecutive decisions. The margin
    # absorbs transient geometry during maneuvers; the patience filters
    # momentary recessions.
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

        # Residual on PN, normalized to [-1, 1]^4 and scaled to physical limits
        # in step(); zero action = pure PN. The uniform per-dim range keeps
        # PPO's shared log_std meaningful (the raw accel/throttle limits differ
        # by 30x, which made throttle exploration pure clipping).
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # [rel_pos(3), rel_vel(3), rocket_vel(3), altitude(1)]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )

    def resample_and_inject(self):
        indices = self._sample_rng.choice(
            self._pool_capacity, size=self._num_balloons, replace=False
        )
        tracks = np.asarray(self._pool[indices])

        theta = self._sample_rng.uniform(0.0, 2.0 * np.pi)
        c, s = np.cos(theta), np.sin(theta)
        rotation = np.array([[c, -s], [s, c]], dtype=tracks.dtype)
        tracks[:, 0:2, :] = np.einsum("ij,njt->nit", rotation, tracks[:, 0:2, :])
        tracks[:, 3:5, :] = np.einsum("ij,njt->nit", rotation, tracks[:, 3:5, :])

        # Rotate the rocket's wind field by the SAME angle: balloons and rocket
        # share one sky, so the wind that shaped the injected drift must be the
        # wind the rocket flies in.
        self.env.unwrapped.set_wind_rotation(theta)
        self.env.unwrapped.update_source_trajectories(tracks)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._sample_rng = np.random.default_rng(seed)

        self.resample_and_inject()
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
                self.resample_and_inject()
                self.observation, info = self.env.reset(options=options)
                break

        self.rocket_state = self.estimator.estimate_rocket(self.observation)
        balloon_states = self.estimator.predict_balloons(self.observation)

        target_idx = self.selector.select_target(balloon_states, self.rocket_state)
        target_state = self.estimator.predict_target(self.observation, target_idx)
        self.target_state = target_state

        self._reset_target_tracking(target_idx, target_state)

        # Never resets on a target switch: grades the terminal reward and is the
        # leading training indicator (pops start once it crosses the radius).
        self._episode_closest_approach = self._true_target_distance(self._tracked_target_idx)

        self.prev_action = np.zeros(4, dtype=np.float32)
        rl_obs = compute_rl_observation(self.rocket_state, target_state)

        return rl_obs, info

    def _true_target_distance(self, target_idx):
        """Rocket distance to the ACTUAL tracked balloon, not the aim point.

        The guidance frame (observation, PN command) is relative to the
        estimator's lead-predicted point, which sits meters from the balloon
        itself; only the real separation decides a pop. Returns inf when there
        is no live target to measure against.
        """
        if target_idx is None or not np.isfinite(self.rocket_state[0]):
            return np.inf
        inner = self.env.unwrapped
        balloon_states = inner._balloon_states
        if target_idx >= len(balloon_states) or inner._balloon_status[target_idx, 0] != 1:
            return np.inf
        distance = float(np.linalg.norm(balloon_states[target_idx, :3] - self.rocket_state[0:3]))
        return distance if np.isfinite(distance) else np.inf

    def _reset_target_tracking(self, target_idx, target_state):
        """Restart per-target bookkeeping so neither the shaping delta nor the
        miss detector sees the distance jump between two different balloons."""
        self._tracked_target_idx = target_idx
        self._prev_target_distance = compute_target_distance(self.rocket_state, target_state)
        self._closest_target_distance = self._prev_target_distance
        self._recede_decisions = 0

    def step(self, rl_action):
        # The residual is held for the frame skip; the PN baseline underneath is
        # recomputed every sim step so guidance stays reactive between decisions.
        rl_action = np.clip(np.asarray(rl_action, dtype=np.float32), -1.0, 1.0)
        residual_accel = rl_action[0:3] * RL_RESIDUAL_ACCEL_LIMIT
        residual_throttle = float(rl_action[3]) * RL_RESIDUAL_THROTTLE_LIMIT

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

            action = {
                "launch": self.selector.should_launch(self.observation),
                "launch_inclination_heading": self.selector.get_launch_heading(self.observation),
                "tvc": tvc,
                "roll": roll,
                "throttle": throttle,
            }

            self.observation, reward, terminated, truncated, info = self.env.step(action)

            self.rocket_state = self.estimator.estimate_rocket(self.observation)
            balloon_states = self.estimator.predict_balloons(self.observation)

            target_idx = self.selector.select_target(balloon_states, self.rocket_state)
            target_state = self.estimator.predict_target(self.observation, target_idx)
            self.target_state = target_state

            if target_idx != self._tracked_target_idx:
                self._reset_target_tracking(target_idx, target_state)

            # Two distances: the aim point drives shaping and miss detection (it
            # is the guidance frame the policy sees), while the TRUE balloon
            # separation grades the terminal reward -- only it decides a pop.
            # Both are updated before scoring so the crash penalty inside
            # compute_rl_reward matches the miss penalty below.
            distance = compute_target_distance(self.rocket_state, target_state)
            self._closest_target_distance = min(self._closest_target_distance, distance)
            self._episode_closest_approach = min(self._episode_closest_approach,
                                                 self._true_target_distance(target_idx))

            # The terminal grade uses the EPISODE-WIDE best approach: the
            # selector chains targets, so grading only the final engagement
            # would erase a near-pop achieved earlier in the flight.
            step_delta = action_delta if i == 0 else np.zeros_like(action_delta)
            total_rl_reward += compute_rl_reward(self.observation, info, self.rocket_state, target_state,
                                                 reward, terminated, step_delta,
                                                 self._prev_target_distance, self._episode_closest_approach)

            self._prev_target_distance = distance

            # Objective complete -- end rather than burn the rest of the flight
            # as junk experience. The pop reward was already banked above.
            if info.get("popped_count", 0) >= self._num_balloons:
                terminated = True
                info["all_popped"] = True
                break

            if terminated or truncated:
                break

        # Miss detection, once per policy decision: receding past the closest
        # approach means the pass failed and the rocket cannot turn back, so the
        # remaining flight would only pollute the rollout buffer.
        if not (terminated or truncated):
            receding = self._prev_target_distance > self._closest_target_distance + self.MISS_MARGIN
            self._recede_decisions = self._recede_decisions + 1 if receding else 0
            if self._recede_decisions >= self.MISS_PATIENCE:
                terminated = True
                info["missed_target"] = True
                total_rl_reward += failure_penalty(self._episode_closest_approach)

        if (terminated or truncated) and np.isfinite(self._episode_closest_approach):
            info["closest_approach"] = float(self._episode_closest_approach)

        rl_obs = compute_rl_observation(self.rocket_state, target_state)

        return rl_obs, total_rl_reward, terminated, truncated, info
