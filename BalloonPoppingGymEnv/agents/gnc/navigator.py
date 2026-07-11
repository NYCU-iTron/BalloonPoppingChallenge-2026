import logging
import numpy as np


class Navigator:
    """Guidance law.

    Produces a *world-frame lateral acceleration command* via proportional
    navigation (PN) plus an axial throttle command for energy management. It
    deliberately does NOT touch attitude, body-frame rotation, or gravity
    compensation -- those belong to the autopilot/inner loop in Controller.
    """

    def __init__(self, given_parameters):
        self.logger = logging.getLogger(__name__)
        self.given_parameters = given_parameters

        # --- Tunable guidance parameters ----------------------------------
        self.nav_constant = 3.0          # PN navigation gain N (typ. 3-5)
        self.max_lateral_accel = 30.0    # (m/s^2) cap on PN command (~3 g)
        self.cruise_throttle = 0.9       # baseline climb throttle
        self.terminal_distance = 20.0    # (m) range to start terminal braking

    def reset(self):
        pass

    def compute(self, target_state: np.ndarray | None, rocket_state: np.ndarray) -> tuple[None, None] | tuple[np.ndarray, float]:
        """
        Compute the world-frame lateral acceleration command and throttle.

        Parameters
        ----------
        target_state : np.ndarray | None
            Predicted target state [pos(3), vel(3)] from the estimator, or None.
        rocket_state : np.ndarray
            Estimated state [pos(3), vel(3), acc(3), quat(4), gyro(3)].

        Returns
        -------
        a_cmd_world : np.ndarray | None
            Desired lateral acceleration (perpendicular to the line of sight)
            in the world frame, shape (3,). None when there is no valid target.
        throttle : float | None
            Energy-management throttle in [0, 1]. None when no valid target.
        """
        if target_state is None or np.isnan(target_state).any():
            return None, None

        rocket_pos = rocket_state[0:3]
        rocket_vel = rocket_state[3:6]

        target_state = np.asarray(target_state, dtype=float).reshape(-1)
        target_pos = target_state[0:3]
        target_vel = target_state[3:6] if target_state.size >= 6 else np.zeros(3)

        # --- Line of sight -------------------------------------------------
        los = target_pos - rocket_pos
        distance = np.linalg.norm(los)
        if distance < 1e-3:
            return np.zeros(3), 1.0
        los_hat = los / distance

        # --- Proportional navigation --------------------------------------
        # Relative velocity and closing speed (positive when approaching).
        v_rel = target_vel - rocket_vel
        v_closing = -np.dot(v_rel, los_hat)

        # LOS angular rate vector: omega = (r x v_rel) / (r . r)
        omega_los = np.cross(los, v_rel) / np.dot(los, los)

        # True PN: lateral acceleration command perpendicular to the LOS.
        # a_cmd = N * Vc * (omega x los_hat)
        a_cmd = self.nav_constant * max(v_closing, 0.0) * np.cross(omega_los, los_hat)

        # Saturate the lateral command so terminal geometry cannot blow it up.
        a_norm = np.linalg.norm(a_cmd)
        if a_norm > self.max_lateral_accel:
            a_cmd = a_cmd * (self.max_lateral_accel / a_norm)

        # --- Throttle: energy management ----------------------------------
        # Hold a high climb throttle; brake near the target if drifting across
        # the line of sight to tighten the terminal turn.
        throttle = self.cruise_throttle
        if distance < self.terminal_distance:
            v_perp = v_rel - np.dot(v_rel, los_hat) * los_hat
            cross_range_speed = np.linalg.norm(v_perp)
            if cross_range_speed > 1.0 and v_closing > 0.0:
                ease = cross_range_speed / (cross_range_speed + abs(v_closing) + 1e-6)
                throttle *= np.clip(1.0 - 0.4 * ease, 0.5, 1.0)

        throttle = float(np.clip(throttle, 0.0, 1.0))

        return a_cmd, throttle
