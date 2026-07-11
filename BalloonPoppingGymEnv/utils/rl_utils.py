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
    # 1. Sparse Terminal Events & Dynamic Fate Evaluation
    # Priority A: Fresh hit detected in this current step
    if reward > 0:
        return 1000.0 * reward

    # Priority B: Episode naturally terminated (Out of fuel / Crashed)
    if terminated:
        # Safely extract simulation_time from the observation dictionary matrix
        survival_time = observation.get("simulation_time", 0.0)
        total_popped = info.get("popped_count", 0)

        # Fixed base penalty for ending the flight session
        terminal_base = -100.0

        # PRIORITIZED SURVIVAL: Linearly scales with flight duration to mitigate the penalty.
        # Safely reward the agent for staying airborne longer even if it hasn't popped balloons yet.
        survival_bonus = survival_time * 2.0

        # Tactical reward for mission completion objectives
        tactical_bonus = total_popped * 50.0

        return float(terminal_base + survival_bonus + tactical_bonus)

    # Extract kinematic states safely for continuous shaping rewards
    rocket_pos = rocket_state[0:3]
    rocket_vel = rocket_state[3:6]
    z = rocket_state[2]
    v_x, v_y, v_z = rocket_vel[0], rocket_vel[1], rocket_vel[2]

    # 2. Non-Linear Geometrical Tracking (Bounded Distance Penalty)
    rel_pos = target_state[0:3] - rocket_pos
    distance = np.linalg.norm(rel_pos)
    distance_penalty = -0.5 * (distance / (distance + 100.0))

    # 3. Kinematic Alignment (Closing Velocity Reward)
    alignment_reward = 0.0
    if distance > 1e-3:
        unit_rel_pos = rel_pos / distance
        closing_speed = np.dot(rocket_vel, unit_rel_pos)
        alignment_reward = 0.02 * closing_speed

    # 4. Actuator Regularization (Smooth Action Delta Penalty)
    smoothness_penalty = -0.05 * np.linalg.norm(action_delta)

    # 5. Temporal Efficiency (Balanced Time Penalty)
    time_penalty = -0.02

    # 6. Launch Phase & Attitude Protection (Anti-Gravity-Turn Crash Shield)
    if v_z < 0.0:
        falling_speed = abs(v_z)
        stability_reward = -0.3 * float(np.log1p(falling_speed))
    else:
        stability_reward = 0.05 * v_z if z < target_state[2] else 0.0

    tilt_penalty = 0.0
    horizontal_speed = np.linalg.norm([v_x, v_y])
    if z < 150.0 and horizontal_speed > v_z:
        tilt_penalty = -0.2

    # Sum total shaped scalar feedback
    total_reward = (
        distance_penalty
        + alignment_reward
        + smoothness_penalty
        + time_penalty
        + stability_reward
        + tilt_penalty
    )
    return float(total_reward)
