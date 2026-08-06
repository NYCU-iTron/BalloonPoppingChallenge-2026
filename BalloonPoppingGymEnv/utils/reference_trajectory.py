import numpy as np

# Duplicated from agents/gnc/selector.py::Selector -- same measured hardware
# envelope fact (lateral authority vs. altitude), kept independent here to
# avoid importing agents/* from utils/* (this repo's import direction is
# agents/* -> utils/* only). Update both copies together if these change.
LATERAL_BUDGET_LOW = 2.5     # (m/s^2) at/below BUDGET_LOW_ALT
LATERAL_BUDGET_HIGH = 8.0    # (m/s^2) at/above BUDGET_HIGH_ALT
BUDGET_LOW_ALT = 50.0        # (m AGL)
BUDGET_HIGH_ALT = 150.0      # (m AGL)
MAX_BRAKE_ACCEL = 9.81       # (m/s^2) engine off: gravity is the only brake


def _lateral_budget(altitude_agl: float) -> float:
    blend = np.clip((altitude_agl - BUDGET_LOW_ALT) / (BUDGET_HIGH_ALT - BUDGET_LOW_ALT), 0.0, 1.0)
    return LATERAL_BUDGET_LOW + blend * (LATERAL_BUDGET_HIGH - LATERAL_BUDGET_LOW)


def generate_reference_trajectory(
    start_pos: np.ndarray,
    start_vel: np.ndarray,
    target_pos: np.ndarray,
    ground_elevation: float,
    dt: float = 0.02,
    max_duration: float = 30.0,
    arrival_radius: float = 2.0,
    max_closing_speed: float = 25.0,
    closing_gain: float = 0.6,
    steering_gain: float = 4.0,
    min_steering_speed: float = 3.0,
) -> np.ndarray:
    """Point-mass forward integration from the real launch-handoff state to
    target_pos: a kinematic guidance ideal for Stage-0 reward shaping to
    track, not a flyable open-loop command (no gravity/thrust/attitude here).

    Steers by nulling the line-of-sight rotation rate (accelerate
    perpendicular to LOS, proportional to closing speed x LOS rate) rather
    than just pointing at the target's current position -- pure "point and
    go" pursuit was tried first and does not reliably converge: once
    misaligned and closing fast, it cannot turn tightly enough and orbits the
    target indefinitely instead of arriving. LOS-rate nulling is what
    actually prevents that (it is the textbook reason proportional
    navigation exists, not a vehicle-specific guidance choice), so it is
    reused here as basic kinematics -- with a fresh gain and no coupling to
    Navigator's own (separately, extensively debugged) closing-speed
    machinery. Turn radius (r=v^2/a) still falls out fresh each instant from
    the live speed against the altitude-scheduled budget, never a
    fixed-radius arc.

    Returns an (N, 8) array: columns [t, x, y, z, vx, vy, vz, arc_length_s].
    """
    pos = np.array(start_pos, dtype=float)
    vel = np.array(start_vel, dtype=float)
    t = 0.0
    s = 0.0
    samples = [np.concatenate(([t], pos, vel, [s]))]

    while True:
        rel = target_pos - pos
        dist = float(np.linalg.norm(rel))
        if dist <= arrival_radius or t >= max_duration:
            break
        los_hat = rel / dist

        v_rel = -vel  # target is stationary
        v_closing = float(np.dot(vel, los_hat))  # > 0 while approaching

        budget = _lateral_budget(pos[2] - ground_elevation)

        # --- Lateral: null the LOS rotation rate (proportional navigation) --- #
        omega_los = np.cross(rel, v_rel) / max(dist * dist, 1e-9)
        a_lateral = steering_gain * max(v_closing, min_steering_speed) * np.cross(omega_los, los_hat)
        a_lateral_norm = float(np.linalg.norm(a_lateral))
        if a_lateral_norm > budget:
            a_lateral *= budget / a_lateral_norm

        # --- Along-LOS closing speed: braking-distance-bounded P control --- #
        # Also capped by how much turn the current LOS rate demands: a fast,
        # tight turn already eats most of the lateral budget, so don't also
        # ask for a high closing speed on top of it (same idea as capping
        # instantaneous demand against available authority, just expressed
        # via the LOS rate actually being nulled here instead of a bearing
        # error).
        omega_mag = float(np.linalg.norm(omega_los))
        v_turn_cap = budget / (steering_gain * omega_mag) if omega_mag > 1e-6 else np.inf
        v_close_brakeable = float(np.sqrt(max(2.0 * MAX_BRAKE_ACCEL * dist, 0.0)))
        v_close_target = min(v_close_brakeable, max_closing_speed, v_turn_cap)
        a_along = np.clip(
            closing_gain * (v_close_target - v_closing), -MAX_BRAKE_ACCEL, MAX_BRAKE_ACCEL
        ) * los_hat

        a_total = a_lateral + a_along

        # Semi-implicit Euler: velocity first, then position with updated vel.
        vel = vel + a_total * dt
        prev_pos = pos.copy()
        pos = pos + vel * dt
        s += float(np.linalg.norm(pos - prev_pos))
        t += dt

        samples.append(np.concatenate(([t], pos, vel, [s])))

    return np.array(samples)
