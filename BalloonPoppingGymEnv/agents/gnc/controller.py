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

        # Attitude (outer) loop gain: maps pointing error to desired body rate.
        self.attitude_gain = 4.5

        # PI integral memory (pitch, yaw); reset between episodes.
        self.integral_error = np.zeros(2)

    def reset(self):
        self.integral_error = np.zeros(2)

    def compute(self, rocket_state: np.ndarray, a_cmd_world: np.ndarray | None, desired_throttle: float | None) -> tuple[np.ndarray, float, float]:
        """
        Autopilot + inner rate loop.

        Converts a world-frame lateral acceleration command into a thrust
        direction (with gravity compensation), turns the pointing error into a
        desired body rate, and closes an anti-windup PI loop on body rates to
        produce TVC gimbal commands.

        Parameters
        ----------
        rocket_state : np.ndarray
            Estimated state [pos(3), vel(3), acc(3), quat(4), gyro(3)].
        a_cmd_world : np.ndarray | None
            Desired world-frame lateral acceleration from the navigator, or None.
        desired_throttle : float | None
            Energy-management throttle from the navigator, or None.

        Returns
        -------
        (tvc [x, y], roll, throttle) clipped within actuator limits.
        """
        # 1. Safe guard: handle missing or invalid guidance commands gracefully.
        if a_cmd_world is None or np.isnan(a_cmd_world).any():
            self.integral_error = np.zeros(2)  # Clear tracking memory
            return np.zeros(2), 0.0, self.throttle_min

        rocket_state = np.asarray(rocket_state, dtype=float).reshape(-1)
        rocket_quat = rocket_state[9:13]
        actual_rates = rocket_state[13:16]

        # Sensor safety guard before launch (gyro is NaN until liftoff).
        if np.isnan(actual_rates).any():
            actual_rates = np.zeros(3)

        quat_norm = np.linalg.norm(rocket_quat)
        if quat_norm > 1e-9:
            rocket_quat = rocket_quat / quat_norm

        # 2. Autopilot: acceleration command -> thrust direction.
        # Thrust must produce the lateral command AND cancel gravity, so the
        # required thrust acceleration is a_cmd minus the gravity vector.
        a_cmd_world = np.asarray(a_cmd_world, dtype=float).reshape(-1)[0:3]
        a_thrust = a_cmd_world - np.array([0.0, 0.0, -9.81])
        thrust_norm = np.linalg.norm(a_thrust)
        if thrust_norm > 1e-9:
            desired_dir_world = a_thrust / thrust_norm
        else:
            desired_dir_world = np.array([0.0, 0.0, 1.0])

        # 3. World-to-body quaternion rotation of the desired thrust direction.
        qw, qx, qy, qz = rocket_quat
        q_vec = np.array([qx, qy, qz])
        t = 2.0 * np.cross(-q_vec, desired_dir_world)
        desired_dir_body = desired_dir_world + qw * t + np.cross(-q_vec, t)

        # 4. Attitude error -> desired body rates (outer P loop).
        # Body z is the thrust axis; rotate it onto desired_dir_body.
        d = desired_dir_body
        rot_axis = np.array([-d[1], d[0], 0.0])
        sin_mag = np.linalg.norm(rot_axis)
        angle = np.arctan2(sin_mag, d[2])
        if sin_mag > 1e-9:
            rot_axis = rot_axis / sin_mag

        desired_rates = np.zeros(3)
        desired_rates[0] = self.attitude_gain * angle * rot_axis[0]  # pitch (wx)
        desired_rates[1] = self.attitude_gain * angle * rot_axis[1]  # yaw   (wy)
        desired_rates[2] = 0.0                                       # roll

        # 5. Throttle boundary (pass-through energy command).
        throttle = float(np.clip(desired_throttle, self.throttle_min, self.throttle_max))

        # 6. Inner rate loop: anti-windup PI on body rates.
        error = desired_rates - actual_rates

        kp_gimbal = 2.0
        ki_gimbal = 0.5
        kp_roll = 1.0

        # TVC authority compensation: scale gain inversely with throttle so the
        # angular acceleration stays uniform as thrust changes.
        dynamic_kp = kp_gimbal / max(throttle, 0.15)

        self.integral_error += error[0:2] * self.dt
        self.integral_error = np.clip(self.integral_error, -0.05, 0.05)

        raw_pitch_gimbal = (dynamic_kp * error[0]) + (ki_gimbal * self.integral_error[0])
        raw_yaw_gimbal = (dynamic_kp * error[1]) + (ki_gimbal * self.integral_error[1])
        raw_roll_torque = kp_roll * error[2]

        # 7. Actuator saturation clamping.
        raw_tvc = np.array([raw_pitch_gimbal, raw_yaw_gimbal])
        tvc = np.clip(raw_tvc, -self.max_gimbal, self.max_gimbal)
        roll = np.clip(raw_roll_torque, -self.max_roll, self.max_roll)

        return tvc, roll, throttle
