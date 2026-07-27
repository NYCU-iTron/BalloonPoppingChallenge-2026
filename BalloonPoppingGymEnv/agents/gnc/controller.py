import logging
import numpy as np
from BalloonPoppingGymEnv.utils.schema import Schema


class Controller:
    # Altitude-scheduled tilt limit on the commanded thrust direction. Near the
    # ground the thrust must keep enough vertical component to hover, otherwise
    # a lateral command converts directly into altitude loss and ground impact.
    # The allowance opens up with altitude so guidance regains full authority
    # once there is room to recover.
    TILT_LIMIT_LOW_DEG = 10.0    # (deg) allowed tilt at/below TILT_LOW_ALT
    TILT_LIMIT_HIGH_DEG = 70.0   # (deg) allowed tilt at/above TILT_HIGH_ALT
    TILT_LOW_ALT = 50.0          # (m AGL)
    TILT_HIGH_ALT = 250.0        # (m AGL)

    # TVC torque is proportional to thrust, so at zero throttle the attitude
    # loop has no authority at all and the vehicle is left to the fins. A small
    # floor keeps the gimbal effective (0.05 still yields ~0.5 rad/s^2) while
    # contributing only ~0.6 m/s^2 of unwanted acceleration.
    MIN_CONTROL_THROTTLE = 0.05

    GRAVITY = np.array([0.0, 0.0, -9.81])

    def __init__(self, given_parameters):
        self.logger = logging.getLogger(__name__)
        self.given_parameters = given_parameters

        env_cfg = given_parameters[Schema.Given.Section.ENVIRONMENT]
        self.ground_elevation = float(env_cfg[Schema.Given.Environment.ELEVATION])

        rocket_cfg = given_parameters[Schema.Given.Section.ROCKET]
        control_cfg = rocket_cfg[Schema.Given.Rocket.CONTROL]
        self.max_gimbal = control_cfg[Schema.Given.Control.GIMBAL_RANGE]
        self.max_roll = control_cfg[Schema.Given.Control.MAX_ROLL_TORQUE]
        self.throttle_min = control_cfg[Schema.Given.Control.THROTTLE_RANGE][0]
        self.throttle_max = control_cfg[Schema.Given.Control.THROTTLE_RANGE][1]

        self.sampling_rate = rocket_cfg[Schema.Given.Rocket.SENSORS][Schema.Given.Sensors.SAMPLING_RATE]
        self.dt = 1.0 / self.sampling_rate

        # Mass schedule for converting an acceleration magnitude into throttle.
        # Propellant drains at a fixed rate regardless of throttle, so mass is a
        # known function of time alone.
        self.max_thrust, self.initial_mass, self.propellant_mass, self.burn_time = \
            self._mass_model(rocket_cfg)

        # Attitude (outer) loop gain: maps pointing error to desired body rate.
        self.attitude_gain = 4.5

        self.reset()

    def reset(self):
        self.integral_error = np.zeros(2)   # PI memory (pitch, yaw)
        self.elapsed_time = 0.0             # drives the mass schedule

    def compute(self, rocket_state: np.ndarray, desired_acc: np.ndarray) -> tuple[np.ndarray, float, float]:
        """
        Autopilot + inner rate loop.

        Takes a single world-frame acceleration command and derives both
        outputs from it: the thrust direction (after gravity compensation) sets
        the attitude, and its magnitude sets the throttle. Deriving both from
        one vector means the commanded tilt and the thrust magnitude can never
        contradict each other, which they could when throttle arrived
        separately.

        Parameters
        ----------
        rocket_state : np.ndarray
            Estimated state [pos(3), vel(3), acc(3), quat(4), gyro(3)].
        desired_acc : np.ndarray | None
            Desired world-frame acceleration from the navigator, or None.

        Returns
        -------
        (tvc [x, y], roll, throttle) clipped within actuator limits.
        """
        self.elapsed_time += self.dt

        # 1. Missing or invalid guidance: hold attitude neutral and idle.
        if desired_acc is None or not np.all(np.isfinite(np.asarray(desired_acc, dtype=float))):
            self.integral_error = np.zeros(2)
            return np.zeros(2), 0.0, self.throttle_min

        rocket_state = np.asarray(rocket_state, dtype=float).reshape(-1)
        rocket_quat = self._unit_quat(rocket_state[9:13])
        actual_rates = rocket_state[13:16]
        if not np.all(np.isfinite(actual_rates)):
            actual_rates = np.zeros(3)      # gyro is NaN until liftoff

        # 2. Thrust must supply the command and cancel gravity.
        a_thrust = np.asarray(desired_acc, dtype=float).reshape(-1)[0:3] - self.GRAVITY
        thrust_accel = float(np.linalg.norm(a_thrust))
        desired_dir_world = (a_thrust / thrust_accel if thrust_accel > 1e-9
                             else np.array([0.0, 0.0, 1.0]))

        # 2b. Altitude-scheduled tilt limiter (survivability guard): rotate the
        # commanded direction toward vertical when low, keeping its bearing.
        desired_dir_world = self._limit_tilt(desired_dir_world, rocket_state[2])

        # 3. Throttle from the commanded magnitude: thrust = m * |a_thrust|.
        throttle = self._throttle_for(thrust_accel)

        # 4. World-to-body rotation of the desired thrust direction.
        qw, qx, qy, qz = rocket_quat
        q_vec = np.array([qx, qy, qz])
        t = 2.0 * np.cross(-q_vec, desired_dir_world)
        desired_dir_body = desired_dir_world + qw * t + np.cross(-q_vec, t)

        # 5. Attitude error -> desired body rates (outer P loop).
        # Body z is the thrust axis; rotate it onto desired_dir_body.
        d = desired_dir_body
        rot_axis = np.array([-d[1], d[0], 0.0])
        sin_mag = float(np.linalg.norm(rot_axis))
        angle = float(np.arctan2(sin_mag, d[2]))
        if sin_mag > 1e-9:
            rot_axis = rot_axis / sin_mag

        desired_rates = np.array([
            self.attitude_gain * angle * rot_axis[0],   # pitch (wx)
            self.attitude_gain * angle * rot_axis[1],   # yaw   (wy)
            0.0,                                        # roll
        ])

        # 6. Inner rate loop: anti-windup PI on body rates.
        error = desired_rates - actual_rates

        kp_gimbal, ki_gimbal, kp_roll = 2.0, 0.5, 1.0

        # TVC authority scales with thrust, so raise the gain as throttle drops
        # to keep the angular response roughly uniform.
        dynamic_kp = kp_gimbal / max(throttle, 0.15)

        # Integrate only on finite error: a single NaN would otherwise latch the
        # integral permanently (np.clip does not filter NaN) and disable the
        # attitude loop for the rest of the episode.
        if np.all(np.isfinite(error[0:2])):
            self.integral_error = np.clip(self.integral_error + error[0:2] * self.dt,
                                          -0.05, 0.05)

        raw_tvc = dynamic_kp * error[0:2] + ki_gimbal * self.integral_error
        raw_roll = kp_roll * error[2]

        # 7. Actuator saturation.
        tvc = np.clip(np.nan_to_num(raw_tvc), -self.max_gimbal, self.max_gimbal)
        roll = float(np.clip(np.nan_to_num(raw_roll), -self.max_roll, self.max_roll))

        return tvc, roll, throttle

    # ------------------------------------------------------------------ #

    def _throttle_for(self, thrust_accel: float) -> float:
        """Throttle that delivers the requested thrust acceleration."""
        mass = self.initial_mass - self.propellant_mass * min(
            self.elapsed_time / self.burn_time, 1.0)
        throttle = mass * thrust_accel / self.max_thrust
        floor = max(self.throttle_min, self.MIN_CONTROL_THROTTLE)
        return float(np.clip(throttle, floor, self.throttle_max))

    def _limit_tilt(self, direction: np.ndarray, altitude_msl: float) -> np.ndarray:
        altitude_agl = altitude_msl - self.ground_elevation
        if not np.isfinite(altitude_agl):
            return direction

        blend = np.clip((altitude_agl - self.TILT_LOW_ALT)
                        / (self.TILT_HIGH_ALT - self.TILT_LOW_ALT), 0.0, 1.0)
        tilt_limit = np.radians(self.TILT_LIMIT_LOW_DEG
                                + blend * (self.TILT_LIMIT_HIGH_DEG - self.TILT_LIMIT_LOW_DEG))

        horizontal = np.hypot(direction[0], direction[1])
        tilt = np.arctan2(horizontal, direction[2])
        if tilt <= tilt_limit:
            return direction

        # A straight-down command has no horizontal bearing to preserve, so pick
        # an arbitrary one rather than leaving it unclamped.
        bearing = (direction[0:2] / horizontal if horizontal > 1e-9
                   else np.array([1.0, 0.0]))
        return np.array([bearing[0] * np.sin(tilt_limit),
                         bearing[1] * np.sin(tilt_limit),
                         np.cos(tilt_limit)])

    @staticmethod
    def _unit_quat(quat: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(quat)
        if not np.isfinite(norm) or norm < 1e-9:
            return np.array([1.0, 0.0, 0.0, 0.0])
        return quat / norm

    @staticmethod
    def _mass_model(rocket_cfg) -> tuple[float, float, float, float]:
        motor = rocket_cfg[Schema.Given.Rocket.MOTOR]
        tank = rocket_cfg[Schema.Given.Rocket.TANK]
        body = rocket_cfg[Schema.Given.Rocket.ROCKET_BODY]

        max_thrust = float(motor[Schema.Given.Motor.THRUST_SOURCE])
        burn_time = float(motor[Schema.Given.Motor.BURN_TIME])

        grain_mass = (float(motor[Schema.Given.Motor.GRAIN_DENSITY])
                      * float(motor[Schema.Given.Motor.GRAIN_NUMBER])
                      * np.pi
                      * (float(motor[Schema.Given.Motor.GRAIN_OUTER_RADIUS]) ** 2
                         - float(motor[Schema.Given.Motor.GRAIN_INITIAL_INNER_RADIUS]) ** 2)
                      * float(motor[Schema.Given.Motor.GRAIN_INITIAL_HEIGHT]))

        liquid = float(tank[Schema.Given.Tank.INITIAL_LIQUID_MASS])
        gas = float(tank[Schema.Given.Tank.INITIAL_GAS_MASS])
        flow_rate = float(tank[Schema.Given.Tank.LIQUID_MASS_FLOW_RATE_OUT])

        initial_mass = (float(body[Schema.Given.RocketBody.MASS])
                        + float(motor[Schema.Given.Motor.DRY_MASS])
                        + grain_mass + liquid + gas)
        # Oxidiser drains at a fixed rate independent of throttle; the grain is
        # consumed over the same burn.
        propellant_mass = min(flow_rate * burn_time, liquid) + grain_mass

        return max_thrust, initial_mass, propellant_mass, burn_time
