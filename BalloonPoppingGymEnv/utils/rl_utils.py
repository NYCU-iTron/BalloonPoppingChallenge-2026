import numpy as np

# The guidance policy emits one command every RL_FRAME_SKIP simulation steps;
# the inner attitude controller still runs every step. Must be identical in the
# training env (RLNavigatorEnv) and the deployment path (RLAgent).
RL_FRAME_SKIP = 5

# The RL action is a CORRECTION on top of PN, not an absolute command, so zero
# action = pure PN and PPO explores around a working intercept. Shared by the
# training action space and the deployment clipping to keep semantics identical.
RL_RESIDUAL_ACCEL_LIMIT = 10.0     # (m/s^2) lateral correction authority
RL_RESIDUAL_THROTTLE_LIMIT = 0.3   # throttle correction authority

# Terminal near-miss ramp. A purely linear failure cost is nearly flat over the
# meters that decide a pop (9.2 -> 1.5 m was worth only +7.7 against a +1000
# pop), leaving a cliff with no gradient. This ramp makes that stretch worth
# ~+270 while staying under the pop reward, so popping is always strictly better.
NEAR_MISS_BONUS = 300.0
NEAR_MISS_SIGMA = 5.0  # (m) ramp width

# Log-distance shaping potential: closing 1 m at 2 m out is worth ~5x closing
# 1 m at 8 m out, concentrating the signal where precision decides the outcome.
SHAPING_GAIN = 2.0
SHAPING_CLIP = 1.0        # per-step bound; also caps target-switch jumps
BALLOON_RADIUS = 1.5      # (m) potential floor -- inside this the pop reward takes over

def compute_target_distance(rocket_state, target_state):
    """Rocket-to-target distance, with invalid geometry (pre-launch / crashed /
    no target) mapped to 300 m so consumers never see NaN."""
    distance = float(np.linalg.norm(target_state[0:3] - rocket_state[0:3]))
    if not np.isfinite(distance):
        distance = 300.0
    return distance

def failure_penalty(closest_distance):
    """Terminal cost for a FAILED episode (crash or fly-by miss), graded by the
    closest approach to the ACTUAL balloon achieved this episode.

    Grading by closest rather than the distance at death credits a run that
    reaches the balloon and then overshoots; otherwise the reward would teach
    the agent to avoid approaching whenever it risks overshooting. Both failure
    modes (crash in compute_rl_reward, miss in RLNavigatorEnv) share this.

    Yields -99 at 9.2 m, +54 at 4 m, +172 at 1.5 m.
    """
    distance = min(closest_distance, 300.0)
    ramp = NEAR_MISS_BONUS * np.exp(-((distance / NEAR_MISS_SIGMA) ** 2))
    return float(-100.0 - distance + ramp)

def compute_rl_observation(rocket_state, target_state):
    rel_pos = target_state[0:3] - rocket_state[0:3]
    rel_vel = (target_state[3:6] if target_state.size >= 6 else np.zeros(3)) - rocket_state[3:6]

    rocket_vel = rocket_state[3:6]
    z = rocket_state[2]

    rl_obs = np.concatenate([rel_pos, rel_vel, rocket_vel, [z]]).astype(np.float32)

    # The estimator emits NaN/inf on invalid states (pre-launch, post-crash, no
    # target); a single NaN observation produces NaN network outputs.
    rl_obs = np.nan_to_num(rl_obs, nan=0.0, posinf=1e4, neginf=-1e4)

    return rl_obs

def compute_rl_reward(
    observation: dict,
    info: dict,
    rocket_state: np.ndarray,
    target_state: np.ndarray,
    reward: int,
    terminated: bool,
    action_delta: np.ndarray,
    prev_distance: float | None = None,
    closest_distance: float | None = None
) -> float:
    z = rocket_state[2]
    distance = compute_target_distance(rocket_state, target_state)

    # Pop: checked before termination so clearing the last balloon is a pure
    # success, never a "death". Single reward channel for pops.
    if reward > 0:
        return 1000.0 * reward

    # Episode ended without a pop. Pops were banked above, so this only grades
    # how it ended. Callers that track it pass the episode's closest approach.
    if terminated:
        fail_distance = closest_distance if closest_distance is not None else distance
        return failure_penalty(fail_distance)

    # Progress shaping: k * (log d_prev - log d_curr) telescopes to
    # k * log(d_start / d_end) regardless of pacing, so it cannot be farmed by
    # loitering, while the per-meter gradient grows as the target nears.
    alignment_reward = 0.0
    if prev_distance is not None and np.isfinite(prev_distance):
        log_delta = np.log(max(prev_distance, BALLOON_RADIUS)) - np.log(max(distance, BALLOON_RADIUS))
        alignment_reward = float(np.clip(SHAPING_GAIN * log_delta, -SHAPING_CLIP, SHAPING_CLIP))

    # Smoothness only -- no flat time penalty. action_delta is driven by
    # EXPLORATION NOISE (E|delta| ~= 0.96 at std 0.36), so this coefficient acts
    # as a per-step tax the policy can only escape by ending episodes early. At
    # 0.15 it cost -59/episode and taught the agent to chop throttle and die
    # sooner rather than improve the intercept. Keep the episode total to a few
    # reward so it can never rival the terminal signal.
    smoothness_penalty = -0.01 * float(np.linalg.norm(action_delta))

    # Low-altitude attitude guard: cos_tilt is the world-up component of the
    # body up-axis (1 = upright, 0 = horizontal).
    tilt_penalty = 0.0
    quat = rocket_state[9:13]
    if not np.isnan(quat).any():
        qn = float(np.linalg.norm(quat))
        if qn > 1e-9:
            _, qx, qy, _ = quat / qn
            cos_tilt = 1.0 - 2.0 * (qx * qx + qy * qy)
            if z < 150.0 and cos_tilt < 0.707:  # tilted more than ~45 deg
                tilt_penalty = -0.2 * (0.707 - cos_tilt)

    total_reward = alignment_reward + smoothness_penalty + tilt_penalty
    if not np.isfinite(total_reward):
        total_reward = 0.0
    return float(total_reward)

def linear_schedule(initial_value, final_value):
    def schedule(progress_remaining):
        return final_value + progress_remaining * (initial_value - final_value)
    return schedule
