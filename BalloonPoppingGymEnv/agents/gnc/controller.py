import logging
import numpy as np
from BalloonPoppingGymEnv.utils.schema import Schema


class Controller:
    def __init__(self, given_parameters):
        self.logger = logging.getLogger(__name__)
        self.given_parameters = given_parameters

        control_cfg = given_parameters[Schema.Given.Section.ROCKET][Schema.Given.Rocket.CONTROL]
        self.max_gimbal = control_cfg[Schema.Given.Control.GIMBAL_RANGE]
        self.max_roll = control_cfg[Schema.Given.Control.MAX_ROLL_TORQUE]
        self.throttle_min = control_cfg[Schema.Given.Control.THROTTLE_RANGE][0]
        self.throttle_max = control_cfg[Schema.Given.Control.THROTTLE_RANGE][1]

        # Time step
        self.sampling_rate = given_parameters[Schema.Given.Section.ROCKET][Schema.Given.Rocket.SENSORS][Schema.Given.Sensors.SAMPLING_RATE]
        self.dt = 1.0 / self.sampling_rate

        # PI integral memory (pitch, yaw); reset between episodes.
        self.integral_error = np.zeros(2)

    def reset(self):
        self.integral_error = np.zeros(2)

    def compute(self, rocket_state: np.ndarray, target_rates: np.ndarray | None, desired_throttle: float) -> tuple[np.ndarray, float, float]:
        """
        Returns (tvc [x, y], roll, throttle) clipped within actuator limits.
        Features thrust-compensated gain scheduling and anti-windup PI control.

        Parameters
        ----------
        rocket_state : np.ndarray
            Estimated state [pos(3), vel(3), acc(3), quat(4), gyro(3)]. The body
            angular rates are read directly from the gyro channel (indices 13:16).
        target_rates : np.ndarray | None
            Desired body angular rates [wx, wy, wz] from the navigator, or None.
        desired_throttle : float
            Desired throttle from the navigator.
        """
        # 1. Safe Guard: Handle missing or invalid guidance targets gracefully
        if target_rates is None or np.isnan(target_rates).any():
            self.integral_error = np.zeros(2)  # Clear tracking memory
            return np.zeros(2), 0.0, self.throttle_min

        target_rates = np.asarray(target_rates, dtype=float).reshape(-1)

        # Body angular rates straight from the gyro channel of the estimated
        # state: [pos(3), vel(3), acc(3), quat(4), gyro(3)] -> gyro at [13:16].
        actual_rates = np.asarray(rocket_state, dtype=float).reshape(-1)[13:16]

        # Sensor safety guard before launch (gyro is NaN until liftoff).
        if np.isnan(actual_rates).any():
            actual_rates = np.zeros(3)

        error = target_rates[0:3] - actual_rates

        # 2. Schedule Throttle Boundary First
        throttle = np.clip(desired_throttle, self.throttle_min, self.throttle_max)

        # 3. Control Gains Configuration
        kp_gimbal = 2.0
        ki_gimbal = 0.5
        kp_roll = 1.0

        # 4. TVC Control Authority Compensation (Gain Scheduling)
        # Scale proportional gain inversely with throttle to keep uniform angular acceleration
        dynamic_kp = kp_gimbal / max(throttle, 0.15)

        # 5. Integral Accumulation with Clamping Anti-Windup
        self.integral_error += error[0:2] * self.dt
        self.integral_error = np.clip(self.integral_error, -0.05, 0.05)

        # 6. Compute Raw Actuator Commands
        raw_pitch_gimbal = (dynamic_kp * error[0]) + (ki_gimbal * self.integral_error[0])
        raw_yaw_gimbal = (dynamic_kp * error[1]) + (ki_gimbal * self.integral_error[1])
        raw_roll_torque = kp_roll * error[2]

        # 7. Actuator Saturation Clamping
        raw_tvc = np.array([raw_pitch_gimbal, raw_yaw_gimbal])
        tvc = np.clip(raw_tvc, -self.max_gimbal, self.max_gimbal)
        roll = np.clip(raw_roll_torque, -self.max_roll, self.max_roll)

        return tvc, roll, throttle
