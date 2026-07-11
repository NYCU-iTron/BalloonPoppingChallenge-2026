import numpy as np

def compute_rl_observation(rocket_state, target_state):
    rel_pos = target_state[0:3] - rocket_state[0:3]
    rel_vel = (target_state[3:6] if target_state.size >= 6 else np.zeros(3)) - rocket_state[3:6]

    rocket_vel = rocket_state[3:6]
    z = rocket_state[2]

    rl_obs = np.concatenate([rel_pos, rel_vel, rocket_vel, [z]]).astype(np.float32)

    return rl_obs

def compute_rl_reward(rocket_state: np.ndarray, target_state: np.ndarray, info: dict, action_delta: np.ndarray) -> float:
    if info.get("crashed", False):
        return -100.0

    if info.get("popped_count", 0) > 0:
        return 200.0 * info["popped_count"]

    # Distance tracking penalty
    rel_pos = target_state[0:3] - rocket_state[0:3]
    distance = np.linalg.norm(rel_pos)
    distance_penalty = -0.5 * (distance / (distance + 100.0))

    # Kinematic closing velocity alignment
    alignment_reward = 0.0
    if distance > 1e-3:
        rocket_vel = rocket_state[3:6]
        unit_rel_pos = rel_pos / distance
        closing_speed = np.dot(rocket_vel, unit_rel_pos)
        alignment_reward = 0.02 * closing_speed

    # Actuator smoothness regulation
    smoothness_penalty = -0.05 * np.linalg.norm(action_delta)

    # Constant time decay penalty
    time_penalty = -0.02

    total_reward = distance_penalty + alignment_reward + smoothness_penalty + time_penalty
    return float(total_reward)
