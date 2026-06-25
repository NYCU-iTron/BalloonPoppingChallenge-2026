import logging
import numpy as np


class Navigator:
    def __init__(self, given_parameters):
        self.logger = logging.getLogger(__name__)
        self.given_parameters = given_parameters

        # --- Tunable guidance gains ---------------------------------------
        self.guidance_gain = 4.5
        self.idle_throttle = 0.15
        self.min_guidance_throttle = 0.35  # Lowered to allow tighter turns
        self.terminal_distance = 20.0

    def reset(self):
        pass

    def compute(self, target_pos: np.ndarray | None, rocket_state: np.ndarray) -> tuple[None, None] | tuple[np.ndarray, float]:
        """
        Computes the desired angular rates and throttle command.
        """
        if target_pos is None or np.isnan(target_pos).any():
            return None, None

        rocket_pos = rocket_state[0:3]
        rocket_vel = rocket_state[3:6]
        rocket_acc = rocket_state[6:9]
        rocket_quat = rocket_state[9:13]
        rocket_gyro = rocket_state[13:16]

        quat_norm = np.linalg.norm(rocket_quat)
        if quat_norm > 1e-9:
            rocket_quat = rocket_quat / quat_norm

        target_pos = np.asarray(target_pos, dtype=float).reshape(-1)[0:3]

        # --- Line of sight to the PREDICTED intercept point ----------------
        los = target_pos - rocket_pos
        distance = np.linalg.norm(los)
        if distance < 1e-3:
            return np.array([0.0, 0.0, 0.0]), 1.0
        los_hat = los / distance

        # --- Momentum Compensation (Drift Rejection) ----------------------
        # Calculate component of velocity perpendicular to the line of sight
        v_perp = rocket_vel - np.dot(rocket_vel, los_hat) * los_hat

        # Guide the thrust vector slightly into the drift to counteract momentum lag
        # k_drift modifies how aggressively we fight the existing velocity vector
        k_drift = 0.12
        desired_dir_world = los_hat - k_drift * v_perp

        dir_norm = np.linalg.norm(desired_dir_world)
        if dir_norm > 1e-9:
            desired_dir_world /= dir_norm
        else:
            desired_dir_world = los_hat

        # --- Inline World-to-Body Quaternion Rotation ---------------------
        qw, qx, qy, qz = rocket_quat
        q_vec = np.array([qx, qy, qz])
        t = 2.0 * np.cross(-q_vec, desired_dir_world)
        desired_dir_body = desired_dir_world + qw * t + np.cross(-q_vec, t)

        # --- Attitude error in body frame ---------------------------------
        d = desired_dir_body
        rot_axis = np.array([-d[1], d[0], 0.0])
        sin_mag = np.linalg.norm(rot_axis)
        angle = np.arctan2(sin_mag, d[2])
        if sin_mag > 1e-9:
            rot_axis = rot_axis / sin_mag

        desired_rates = np.zeros(3)
        desired_rates[0] = self.guidance_gain * angle * rot_axis[0]  # pitch (wx)
        desired_rates[1] = self.guidance_gain * angle * rot_axis[1]  # yaw   (wy)
        desired_rates[2] = 0.0                                       # roll

        # --- Advanced Throttle Scheduling ---------------------------------
        # 1. Nose alignment component
        alignment = np.clip(d[2], 0.0, 1.0)

        # 2. Velocity vector alignment component (Crucial for high inclination)
        v_norm = np.linalg.norm(rocket_vel)
        if v_norm > 0.5:
            v_hat = rocket_vel / v_norm
            vel_alignment = np.clip(np.dot(v_hat, los_hat), 0.0, 1.0)
        else:
            vel_alignment = 1.0

        # Combined throttle mapping: If velocity is drifting sideways, force throttle DOWN
        # This acts as a kinematic brake to drastically sharpen the turning radius
        effective_alignment = alignment * vel_alignment
        throttle = self.min_guidance_throttle + (1.0 - self.min_guidance_throttle) * effective_alignment

        # Terminal Phase Damping
        if distance < self.terminal_distance:
            cross_range_speed = np.linalg.norm(v_perp)
            closing_speed = np.dot(rocket_vel, los_hat)
            if cross_range_speed > 1.0 and closing_speed > 0.0:
                ease = cross_range_speed / (cross_range_speed + abs(closing_speed) + 1e-6)
                throttle *= np.clip(1.0 - 0.4 * ease, 0.5, 1.0)

        throttle = float(np.clip(throttle, 0.0, 1.0))

        return desired_rates, throttle
