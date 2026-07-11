import numpy as np

def compute_rl_observation(rocket_state, target_state):
    rel_pos = target_state[0:3] - rocket_state[0:3]
    rel_vel = (target_state[3:6] if target_state.size >= 6 else np.zeros(3)) - rocket_state[3:6]

    rocket_vel = rocket_state[3:6]
    z = rocket_state[2]

    rl_obs = np.concatenate([rel_pos, rel_vel, rocket_vel, [z]]).astype(np.float32)

    return rl_obs

def compute_rl_reward(
    observation: dict,
    info: dict,
    rocket_state: np.ndarray,
    target_state: np.ndarray,
    reward: int,
    terminated: bool,
    action_delta: np.ndarray
) -> float:
    # =========================================================================
    # 1. Sparse Terminal Events & Absolute Objective Overlords
    # =========================================================================
    # Priority A: Instantaneous balloon popping success event
    if reward > 0:
        return 1000.0 * reward

    # Priority B: Episode termination handling (Out of fuel / Crashed / Timed out)
    if terminated:
        total_popped = info.get("popped_count", 0)
        # Static baseline penalty for dying. No survival_bonus to eliminate hover exploits.
        # Heavily amplified tactical bonus to ensure winning strictly dominates the policy.
        return float(-100.0 + total_popped * 2000.0)

    # Extract kinematic states safely for continuous shaping rewards
    rocket_pos = rocket_state[0:3]
    rocket_vel = rocket_state[3:6]
    z = rocket_state[2]

    # Target-relative tactical geometry calculation
    target_z = target_state[2]
    is_target_above = target_z > z
    rel_pos = target_state[0:3] - rocket_pos
    distance = np.linalg.norm(rel_pos)

    # =========================================================================
    # 2. Continuous Tracking Hub (Fractional & Soft-Clipped Strategy)
    # =========================================================================
    # Bounded fractional distance shaping. Strictly maps between 0.0 and -1.0.
    # Eliminates spatial penalty explosions regardless of how far the rocket drifts.
    distance_penalty = -1.0 * (distance / (distance + 100.0))

    # Kinematic Alignment via Vector Dot Product (Handles both UP and DOWN targets)
    alignment_reward = 0.0
    if distance > 1e-3:
        unit_rel_pos = rel_pos / distance
        # closing_speed > 0 means chasing towards target; < 0 means escaping from target
        closing_speed = np.dot(rocket_vel, unit_rel_pos)

        if closing_speed > 0:
            alignment_reward = 0.2 * closing_speed
        else:
            # Soft-clipped drift penalty to maintain mild guidance gradients without shattering PPO
            alignment_reward = max(0.2 * closing_speed, -0.5)

    # =========================================================================
    # 3. Regularizations & Direct Time Efficiency Drains
    # =========================================================================
    smoothness_penalty = -0.05 * np.linalg.norm(action_delta)
    time_penalty = -0.05  # Constant pressure forces the rocket to intercept ASAP

    # =========================================================================
    # 4. Dynamic Physics Protection (Relative Log Strategy)
    # =========================================================================
    stability_reward = 0.0
    v_z = rocket_vel[2]

    if v_z < 0.0:
        if is_target_above:
            # Unintentional falling: Log-compressed to shield gradients during terminal dives
            falling_speed = abs(v_z)
            stability_reward = -0.3 * float(np.log1p(falling_speed))
        else:
            # Intentional diving: Rewarded for actively pursuing lower altitude objectives
            stability_reward = 0.02 * abs(v_z)
    else:
        if is_target_above:
            # Intentional climbing: Rewarded for pursuing higher altitude objectives
            stability_reward = 0.05 * v_z
        else:
            # Unintentional climbing away from lower targets
            stability_reward = -0.1 * v_z

    # =========================================================================
    # 5. Low-Altitude Ground Proximity Guard
    # =========================================================================
    tilt_penalty = 0.0
    horizontal_speed = np.linalg.norm(rocket_vel[0:2])
    if z < 150.0 and horizontal_speed > v_z:
        tilt_penalty = -0.2

    # Sum total unified continuous shaping feedback surface
    total_reward = (
        distance_penalty
        + alignment_reward
        + smoothness_penalty
        + time_penalty
        + stability_reward
        + tilt_penalty
    )
    return float(total_reward)
