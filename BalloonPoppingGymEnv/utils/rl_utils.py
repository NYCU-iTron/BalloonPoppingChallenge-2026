import numpy as np

# Action-repeat / frame-skip: the RL guidance policy emits one command every
# RL_FRAME_SKIP simulation steps; the inner attitude controller still runs every
# step. With time_step=0.01s this makes each RL decision cover 0.1s, extending
# the effective planning horizon by 10x for the same gamma. Must be identical in
# the training env (RLNavigatorEnv) and the deployment path (RLAgent).
RL_FRAME_SKIP = 5

# Residual guidance: the RL action is a CORRECTION on top of the PN guidance
# law, not an absolute command. Zero action = pure PN (a proven intercept
# baseline), so an untrained policy already flies sensibly and PPO explores
# around a working behavior instead of around free fall. Bounds are shared by
# the training env (action space) and the deployment path (clipping) so the
# semantics stay identical on both sides.
RL_RESIDUAL_ACCEL_LIMIT = 10.0     # (m/s^2) lateral correction authority
RL_RESIDUAL_THROTTLE_LIMIT = 0.3   # throttle correction authority

def compute_target_distance(rocket_state, target_state):
    """Rocket-to-target distance with the same finite guard as the reward.

    Invalid geometry (pre-launch / crashed / no target -> NaN states) maps to a
    constant 300 m so consumers never see NaN and shaping deltas vanish there.
    """
    distance = float(np.linalg.norm(target_state[0:3] - rocket_state[0:3]))
    if not np.isfinite(distance):
        distance = 300.0
    return distance

def failure_penalty(closest_distance):
    """Terminal cost for a FAILED episode (rocket crash or fly-by miss).

    Graded by the closest approach achieved this episode, not the distance at the
    moment of death: a run that reaches the balloon and then overshoots into a
    crash must be credited for getting close, otherwise the reward would teach the
    agent to avoid approaching whenever it risks overshooting. Both failure modes
    (crash in compute_rl_reward, miss in RLNavigatorEnv) share this one formula.
    """
    return float(-100.0 - min(closest_distance, 300.0))

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
    action_delta: np.ndarray,
    prev_distance: float | None = None,
    closest_distance: float | None = None
) -> float:
    z = rocket_state[2]
    # NaN-guarded distance (see compute_target_distance): a single NaN in the
    # rollout buffer corrupts the PPO update and turns the whole policy's
    # outputs into NaN.
    distance = compute_target_distance(rocket_state, target_state)

    # =========================================================================
    # 1. Sparse terminal events (dominant objective signals)
    # =========================================================================
    # Priority A: balloon popped this step. Returns before the termination check
    # so clearing the last balloon is a pure success, never a "death". This is
    # the single reward channel for pops (no duplicate terminal bonus), so credit
    # assignment is unambiguous.
    if reward > 0:
        return 1000.0 * reward

    # Priority B: episode ended without a pop (fuel out / crash). Pops were
    # already banked immediately above, so we only shape *how* it ended: a base
    # cost plus a term that rewards having gotten close before dying. Uses the
    # episode's closest approach when the caller tracks it (RLNavigatorEnv),
    # otherwise the current distance. There is NO per-step cost of living, so
    # ending early only forfeits future pops -- the agent is never better off dying.
    if terminated:
        fail_distance = closest_distance if closest_distance is not None else distance
        return failure_penalty(fail_distance)

    # =========================================================================
    # 2. Progress shaping (potential-based via distance delta)
    # =========================================================================
    # r = k * (d_prev - d_curr): the per-step *change* in distance, so the sum
    # telescopes to the net distance closed regardless of how many steps it took.
    # This is speed-invariant -- unlike a clipped closing-speed bonus, which paid
    # a fixed amount per step spent approaching and therefore paid MORE for
    # approaching slowly (a loiter-farming exploit). The clip is in distance
    # units (max plausible per-step displacement), which also caps the spurious
    # jump when the tracked target switches. Closing 300 m earns ~= +60 total,
    # keeping shaping a signpost far below the +1000 pop reward.
    alignment_reward = 0.0
    if prev_distance is not None and np.isfinite(prev_distance):
        distance_delta = float(np.clip(prev_distance - distance, -3.0, 3.0))
        alignment_reward = 0.2 * distance_delta

    # =========================================================================
    # 3. Light regularizers (no unconditional per-step penalty)
    # =========================================================================
    # Action smoothness only; urgency comes from gamma discounting, not a flat
    # time penalty (which reintroduces the cost-of-living / suicide incentive).
    # action_delta is in NORMALIZED action units ([-1,1]^4 residuals), so a
    # full swing has |delta| = 4; the coefficient keeps the max penalty ~-0.6
    # per decision, well below the shaping scale.
    smoothness_penalty = -0.15 * float(np.linalg.norm(action_delta))

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
