import numpy as np

# Action-repeat / frame-skip: the RL guidance policy emits one command every
# RL_FRAME_SKIP simulation steps; the inner attitude controller still runs every
# step. With time_step=0.01s this makes each RL decision cover 0.1s, extending
# the effective planning horizon by 10x for the same gamma. Must be identical in
# the training env (RLNavigatorEnv) and the deployment path (RLAgent).
RL_FRAME_SKIP = 15

def compute_rl_observation(rocket_state, target_state):
    rel_pos = target_state[0:3] - rocket_state[0:3]
    rel_vel = (target_state[3:6] if target_state.size >= 6 else np.zeros(3)) - rocket_state[3:6]

    rocket_vel = rocket_state[3:6]
    z = rocket_state[2]

    rl_obs = np.concatenate([rel_pos, rel_vel, rocket_vel, [z]]).astype(np.float32)

    # The estimator can emit NaN/inf on invalid states (pre-launch, post-crash,
    # no valid target). Never feed those to the policy: a single NaN observation
    # produces NaN network outputs. The deployment path guards this in
    # navigator.compute; the env must guard it here for train/deploy symmetry.
    rl_obs = np.nan_to_num(rl_obs, nan=0.0, posinf=1e4, neginf=-1e4)

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
    rocket_pos = rocket_state[0:3]
    rocket_vel = rocket_state[3:6]
    z = rocket_state[2]
    rel_pos = target_state[0:3] - rocket_pos
    distance = float(np.linalg.norm(rel_pos))
    # Invalid geometry (NaN from a terminal/crashed state) must never yield a NaN
    # reward: a single NaN in the rollout buffer corrupts the PPO update and turns
    # the whole policy's outputs into NaN. Treat it as "far away".
    if not np.isfinite(distance):
        distance = 300.0

    # =========================================================================
    # 1. Sparse terminal events (dominant objective signals)
    # =========================================================================
    # Priority A: balloon popped this step. Returns before the termination check
    # so clearing the last balloon is a pure success, never a "death". This is
    # the single reward channel for pops (no duplicate terminal bonus), so credit
    # assignment is unambiguous.
    if reward > 0:
        return 1000.0 * reward

    # Priority B: episode ended without a pop (fuel out / crash / timeout).
    # Pops were already banked immediately above, so we only shape *how* it ended:
    # a base cost plus a term that rewards having gotten close to a target before
    # dying. There is NO per-step cost of living (see below), so ending early only
    # forfeits future pop opportunities -- the agent is never better off dying.
    if terminated:
        return float(-100.0 - 1.0 * min(distance, 300.0))

    # =========================================================================
    # 2. Progress shaping (potential-based via closing speed)
    # =========================================================================
    # closing_speed = -d(distance)/dt: >0 approaching, <0 receding. Because it is
    # the time-derivative of distance, integrating it telescopes to the net change
    # in distance, so it cannot be farmed by loitering (unlike an absolute-distance
    # penalty, which taxed every step and made suicide optimal). Symmetric clip
    # keeps PPO gradients bounded. Handles targets above and below identically.
    alignment_reward = 0.0
    if distance > 1e-3:
        unit_rel_pos = rel_pos / distance
        closing_speed = float(np.dot(rocket_vel, unit_rel_pos))
        alignment_reward = float(np.clip(0.15 * closing_speed, -0.5, 0.5))

    # =========================================================================
    # 3. Light regularizers (no unconditional per-step penalty)
    # =========================================================================
    # Action smoothness only; urgency comes from gamma discounting, not a flat
    # time penalty (which reintroduces the cost-of-living / suicide incentive).
    smoothness_penalty = -0.02 * float(np.linalg.norm(action_delta))

    # =========================================================================
    # 4. Low-altitude attitude guard (true tilt from the quaternion)
    # =========================================================================
    # Penalize the rocket lying over near the ground. cos_tilt is the world-up
    # component of the body up-axis (1 = upright, 0 = horizontal). Fixes the old
    # guard that compared a speed magnitude against signed v_z.
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
