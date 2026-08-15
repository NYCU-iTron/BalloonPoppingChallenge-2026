import logging
import numpy as np

from BalloonPoppingGymEnv.agents.gnc.vehicle import Vehicle
from BalloonPoppingGymEnv.utils.schema import Schema


class Controller:
    """Attitude autopilot: points the thrust axis where guidance asked.

    Its only job is tracking a thrust direction. It does not compensate gravity,
    does not decide how hard to push, and does not second-guess the magnitude of
    the command -- those belong to the guidance law, which owns the whole
    acceleration vector. Previously they were split across both modules and the
    achieved acceleration was never reconciled with the commanded one.

    Two nested loops: an outer proportional loop turning pointing error into a
    body rate, and an inner proportional-integral loop turning rate error into
    gimbal deflection. Both are sized from the vehicle's measured control
    authority rather than from hand-picked constants.
    """

    def __init__(
        self,
        given_parameters,
        *,
        loop_separation: float = 2.0,
        rate_command_margin: float = 0.8,
        integral_gain: float = 2.0,
        direction_rate_feedforward: float = 0.0,
        slew_aware: bool = False,
    ):
        self.logger = logging.getLogger(__name__)
        self.given_parameters = given_parameters
        self.vehicle = Vehicle(given_parameters)

        self.sampling_rate = given_parameters[Schema.Given.Section.ROCKET][
            Schema.Given.Rocket.SENSORS
        ][Schema.Given.Sensors.SAMPLING_RATE]
        self.dt = 1.0 / self.sampling_rate

        # --- Loop shaping -------------------------------------------------
        # Both time constants come from the actuator rather than being picked:
        # the gimbal needs this long to cross its own range, so a rate loop
        # tuned faster than that only saturates. The outer loop is then kept
        # slower for cascade separation; Scenario-specific evaluation may tune
        # the ratio without changing the controller's general default.
        self.rate_time_constant = self.vehicle.gimbal_slew_time()
        self.loop_separation = float(loop_separation)
        if self.loop_separation <= 0.0:
            raise ValueError("loop_separation must be positive")
        self.attitude_time_constant = self.loop_separation * self.rate_time_constant
        self.roll_time_constant = 0.50  # (s) roll rate damping

        # The rate command is capped at what the gimbal can actually build up
        # within one rate time constant, with headroom so the inner loop stays
        # linear. Previously a fixed 1.5 rad/s ceiling sat *above* the gimbal's
        # own saturation threshold of 1.28 rad/s, so every large attitude change
        # drove the actuator straight onto its stop and the loop lost control
        # authority exactly when it was turning hardest.
        self.rate_command_margin = float(rate_command_margin)
        if self.rate_command_margin <= 0.0:
            raise ValueError("rate_command_margin must be positive")

        # Integral trim on the rate loop, in gimbal degrees. Sized as a fraction
        # of real authority so it can actually correct a standing bias -- the
        # previous limits let it contribute 0.025 of the 15 available degrees.
        self.integral_gain = float(integral_gain)
        if self.integral_gain < 0.0:
            raise ValueError("integral_gain must not be negative")
        self.max_integral_gimbal = 0.3 * self.vehicle.max_gimbal

        # Feed forward motion of the guidance command itself. The original
        # loop reacts only after pointing error appears, which is costly when
        # corridor guidance rotates the thrust direction through two corners
        # in a few seconds. Zero preserves the established controller exactly.
        self.direction_rate_feedforward = float(direction_rate_feedforward)
        if not 0.0 <= self.direction_rate_feedforward <= 1.0:
            raise ValueError("direction_rate_feedforward must be between 0 and 1")

        # When enabled, issue only commands the actuator can realise on this
        # timestep. The simulator applies the same per-axis rate limit; doing it
        # here makes the controller's remembered command equal the real gimbal.
        self.slew_aware = bool(slew_aware)

        self.integral_error = np.zeros(2)
        self.previous_desired_dir_world = None
        self.commanded_tvc = np.zeros(2)

    def reset(self):
        self.integral_error = np.zeros(2)
        self.previous_desired_dir_world = None
        self.commanded_tvc = np.zeros(2)

    def compute(
        self,
        rocket_state: np.ndarray,
        thrust_dir_world: np.ndarray | None,
        throttle: float | None,
        t_since_launch: float,
    ) -> tuple[np.ndarray, float, float]:
        """
        Parameters
        ----------
        rocket_state : np.ndarray
            Estimated state [pos(3), vel(3), acc(3), quat(4), gyro(3)].
        thrust_dir_world : np.ndarray | None
            Unit vector from guidance that the thrust axis should follow.
        throttle : float | None
            Throttle from guidance, passed straight through to the actuator.
        t_since_launch : float
            Seconds since liftoff, used to size the loop gains.

        Returns
        -------
        (tvc [x, y] in degrees, roll torque, throttle), all within actuator limits.
        """
        # Without thrust the gimbal produces no moment: commanding it just winds
        # up the integrator against a plant that cannot respond.
        authority = self.vehicle.pitch_accel_per_gimbal_deg(t_since_launch)
        if (
            thrust_dir_world is None
            or throttle is None
            or np.isnan(thrust_dir_world).any()
            or authority <= 0.0
        ):
            self.integral_error = np.zeros(2)
            return np.zeros(2), 0.0, self.vehicle.throttle_min

        rocket_state = np.asarray(rocket_state, dtype=float).reshape(-1)
        rocket_quat = rocket_state[9:13]
        actual_rates = rocket_state[13:16]

        # Gyro reads NaN until liftoff.
        if np.isnan(actual_rates).any():
            actual_rates = np.zeros(3)

        quat_norm = np.linalg.norm(rocket_quat)
        if quat_norm > 1e-9:
            rocket_quat = rocket_quat / quat_norm

        # 1. Rotate the desired thrust direction into the body frame.
        desired_dir_world = np.asarray(thrust_dir_world, dtype=float).reshape(-1)[0:3]
        norm = np.linalg.norm(desired_dir_world)
        if norm < 1e-9:
            self.integral_error = np.zeros(2)
            return np.zeros(2), 0.0, self.vehicle.throttle_min
        desired_dir_world = desired_dir_world / norm

        desired_direction_rate_world = self._desired_direction_rate(desired_dir_world)

        qw, qx, qy, qz = rocket_quat
        q_vec = np.array([qx, qy, qz])
        t = 2.0 * np.cross(-q_vec, desired_dir_world)
        desired_dir_body = desired_dir_world + qw * t + np.cross(-q_vec, t)
        rate_t = 2.0 * np.cross(-q_vec, desired_direction_rate_world)
        desired_direction_rate_body = (
            desired_direction_rate_world + qw * rate_t + np.cross(-q_vec, rate_t)
        )

        # 2. Pointing error: the rotation carrying body z onto the desired
        #    direction. Body z is the thrust axis.
        d = desired_dir_body
        rot_axis = np.array([-d[1], d[0], 0.0])  # z_body x d
        sin_mag = float(np.linalg.norm(rot_axis))
        angle = float(np.arctan2(sin_mag, d[2]))
        if sin_mag > 1e-9:
            rot_axis = rot_axis / sin_mag

        # 3. Outer loop: pointing error -> body rate command, capped at a rate
        #    the gimbal can build within one rate time constant so the inner
        #    loop below never has to ask for more deflection than it owns.
        max_body_rate = (
            self.rate_command_margin
            * self.vehicle.max_angular_accel(t_since_launch)
            * self.rate_time_constant
        )
        rate_magnitude = min(angle / self.attitude_time_constant, max_body_rate)
        desired_rates = np.array(
            [
                rate_magnitude * rot_axis[0],  # pitch (wx)
                rate_magnitude * rot_axis[1],  # yaw   (wy)
            ]
        )
        desired_rates += (
            self.direction_rate_feedforward * desired_direction_rate_body[:2]
        )
        desired_rate_norm = float(np.linalg.norm(desired_rates))
        if desired_rate_norm > max_body_rate:
            desired_rates *= max_body_rate / desired_rate_norm

        # 4. Inner loop: rate error -> gimbal. The proportional gain is whatever
        #    it takes to correct the error within one rate time constant given
        #    the current authority, so it schedules itself as propellant burns.
        error = desired_rates - actual_rates[0:2]
        kp = 1.0 / (self.rate_time_constant * authority)

        self.integral_error += error * self.dt
        integral_limit = self.max_integral_gimbal / max(self.integral_gain, 1e-9)
        self.integral_error = np.clip(
            self.integral_error, -integral_limit, integral_limit
        )

        raw_tvc = kp * error + self.integral_gain * self.integral_error

        # 5. Saturate on the *combined* deflection. Clipping each axis on its own
        #    lets the pair reach 15*sqrt(2) = 21.2 degrees, past what the nozzle
        #    can physically do, and skews the thrust direction while doing it.
        tvc_norm = float(np.linalg.norm(raw_tvc))
        if tvc_norm > self.vehicle.max_gimbal:
            scale = self.vehicle.max_gimbal / tvc_norm
            raw_tvc = raw_tvc * scale
            # Anti-windup: stop integrating once the actuator is on its stop.
            self.integral_error *= scale

        if self.slew_aware:
            raw_tvc = self._limit_gimbal_slew(raw_tvc)
        else:
            self.commanded_tvc = raw_tvc.copy()

        # 6. Roll: no roll command is needed, so damp the rate to zero.
        roll_gain = self.vehicle.roll_inertia / self.roll_time_constant
        roll = float(
            np.clip(
                -roll_gain * actual_rates[2],
                -self.vehicle.max_roll_torque,
                self.vehicle.max_roll_torque,
            )
        )

        throttle = float(
            np.clip(throttle, self.vehicle.throttle_min, self.vehicle.throttle_max)
        )

        return raw_tvc, roll, throttle

    def _desired_direction_rate(self, desired_dir_world: np.ndarray) -> np.ndarray:
        """Angular velocity carrying the previous guidance direction here."""
        previous = self.previous_desired_dir_world
        self.previous_desired_dir_world = desired_dir_world.copy()
        if previous is None or self.direction_rate_feedforward <= 0.0:
            return np.zeros(3)

        cross = np.cross(previous, desired_dir_world)
        sin_angle = float(np.linalg.norm(cross))
        cos_angle = float(np.clip(np.dot(previous, desired_dir_world), -1.0, 1.0))
        if sin_angle < 1e-12:
            return np.zeros(3)
        angle = float(np.arctan2(sin_angle, cos_angle))
        return cross * (angle / (sin_angle * self.dt))

    def _limit_gimbal_slew(self, requested_tvc: np.ndarray) -> np.ndarray:
        """Mirror the simulator's independent per-axis gimbal rate limit."""
        max_change = self.vehicle.gimbal_rate_limit * self.dt
        change = np.clip(
            np.asarray(requested_tvc, dtype=float) - self.commanded_tvc,
            -max_change,
            max_change,
        )
        limited = self.commanded_tvc + change
        if not np.allclose(limited, requested_tvc):
            # The actuator cannot realise the integral contribution yet.
            self.integral_error *= 0.95
        self.commanded_tvc = limited
        return limited
