import logging
import numpy as np
from BalloonPoppingGymEnv.utils.schema import Schema


class Controller:
    ATTITUDE_GAIN = 2.0        # (1/s) attitude error -> body rate
    MAX_BODY_RATE = 1.2        # (rad/s)
    RATE_KP = 6.0              # (1/s) rate loop bandwidth after plant inversion
    RATE_KI = 6.0              # (1/s^2)
    INTEGRAL_LIMIT = 1.5       # (rad) enough for the integral to reach full gimbal
    ROLL_KP = 2.0              # (1/s)
    MIN_THROTTLE = 0.05        # keeps some TVC torque available while braking

    TILT_LIMIT_LOW_DEG = 10.0
    TILT_LIMIT_HIGH_DEG = 70.0
    TILT_LOW_ALT = 50.0
    TILT_HIGH_ALT = 250.0

    GRAVITY = np.array([0.0, 0.0, -9.81])

    def __init__(self, given_parameters):
        self.logger = logging.getLogger(__name__)
        self.given_parameters = given_parameters

        env_cfg = given_parameters[Schema.Given.Section.ENVIRONMENT]
        self.ground_elevation = float(env_cfg[Schema.Given.Environment.ELEVATION])

        rocket_cfg = given_parameters[Schema.Given.Section.ROCKET]
        control_cfg = rocket_cfg[Schema.Given.Rocket.CONTROL]

        self.max_gimbal = control_cfg[Schema.Given.Control.GIMBAL_RANGE]
        self.gimbal_rate_limit = control_cfg[Schema.Given.Control.GIMBAL_RATE_LIMIT]
        self.max_roll = control_cfg[Schema.Given.Control.MAX_ROLL_TORQUE]
        self.throttle_min = control_cfg[Schema.Given.Control.THROTTLE_RANGE][0]
        self.throttle_max = control_cfg[Schema.Given.Control.THROTTLE_RANGE][1]
        self.throttle_rate_limit = control_cfg[Schema.Given.Control.THROTTLE_RATE_LIMIT]

        self.sampling_rate = rocket_cfg[Schema.Given.Rocket.SENSORS][Schema.Given.Sensors.SAMPLING_RATE]
        self.dt = 1.0 / self.sampling_rate

        motor_cfg = rocket_cfg[Schema.Given.Rocket.MOTOR]
        tank_cfg = rocket_cfg[Schema.Given.Rocket.TANK]
        body_cfg = rocket_cfg[Schema.Given.Rocket.ROCKET_BODY]

        self.max_thrust = float(motor_cfg[Schema.Given.Motor.THRUST_SOURCE])
        self.burn_time = float(motor_cfg[Schema.Given.Motor.BURN_TIME])

        motor_z = float(motor_cfg[Schema.Given.Motor.MOTOR_POSITION])
        nozzle_z = motor_z + float(motor_cfg[Schema.Given.Motor.NOZZLE_POSITION])
        grain_z = motor_z + float(motor_cfg[Schema.Given.Motor.GRAINS_CENTER_OF_MASS_POSITION])
        tank_z = motor_z + float(tank_cfg[Schema.Given.Tank.TANK_POSITION])
        body_z = float(body_cfg[Schema.Given.RocketBody.CENTER_OF_MASS_WITHOUT_MOTOR])

        grain_mass = (float(motor_cfg[Schema.Given.Motor.GRAIN_DENSITY])
                      * float(motor_cfg[Schema.Given.Motor.GRAIN_NUMBER]) * np.pi
                      * (float(motor_cfg[Schema.Given.Motor.GRAIN_OUTER_RADIUS]) ** 2
                         - float(motor_cfg[Schema.Given.Motor.GRAIN_INITIAL_INNER_RADIUS]) ** 2)
                      * float(motor_cfg[Schema.Given.Motor.GRAIN_INITIAL_HEIGHT]))
        grain_mass += float(motor_cfg[Schema.Given.Motor.DRY_MASS])

        liquid = float(tank_cfg[Schema.Given.Tank.INITIAL_LIQUID_MASS])
        gas = float(tank_cfg[Schema.Given.Tank.INITIAL_GAS_MASS])
        flow_rate = float(tank_cfg[Schema.Given.Tank.LIQUID_MASS_FLOW_RATE_OUT])
        liquid_end = max(liquid - flow_rate * self.burn_time, 0.0)

        body_mass = float(body_cfg[Schema.Given.RocketBody.MASS])
        body_inertia = float(body_cfg[Schema.Given.RocketBody.INERTIA][0])
        self.roll_inertia = float(body_cfg[Schema.Given.RocketBody.INERTIA][2])

        # Mass, pitch inertia and nozzle moment arm at ignition and at burnout;
        # everything in between is interpolated on burn fraction.
        schedule = []
        for grain_m, liquid_m in ((grain_mass, liquid), (0.0, liquid_end)):
            tank_m = liquid_m + gas
            total = body_mass + grain_m + tank_m
            com = (body_mass * body_z + grain_m * grain_z + tank_m * tank_z) / total
            inertia = (body_inertia + body_mass * (body_z - com) ** 2
                       + grain_m * (grain_z - com) ** 2 + tank_m * (tank_z - com) ** 2)
            schedule.append((total, inertia, abs(com - nozzle_z)))

        (self.initial_mass, self.initial_inertia, self.initial_arm), \
            (self.final_mass, self.final_inertia, self.final_arm) = schedule
        self.propellant_mass = self.initial_mass - self.final_mass

        self.reset()

    def reset(self):
        self.integral = np.zeros(2)
        self.ignition_time = None
        self.burnout = False
        self.mass = self.initial_mass
        self.inertia = self.initial_inertia
        self.moment_arm = self.initial_arm
        self.max_thrust_accel = self.max_thrust / self.initial_mass
        self.quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.body_rates = np.zeros(3)
        self.altitude_agl = 0.0
        self.prev_tvc = np.zeros(2)
        self.prev_throttle = self.throttle_min

    def update(self, rocket_state: np.ndarray, simulation_time: float) -> None:
        # First call is the ignition step: both agent paths idle before launch.
        if self.ignition_time is None:
            self.ignition_time = simulation_time
        burn_elapsed = max(simulation_time - self.ignition_time, 0.0)
        self.burnout = burn_elapsed >= self.burn_time

        fraction = min(burn_elapsed / self.burn_time, 1.0)
        self.mass = self.initial_mass - self.propellant_mass * fraction
        self.inertia = self.initial_inertia + (self.final_inertia - self.initial_inertia) * fraction
        self.moment_arm = self.initial_arm + (self.final_arm - self.initial_arm) * fraction
        self.max_thrust_accel = self.max_thrust / self.mass

        state = np.asarray(rocket_state, dtype=float).reshape(-1)
        quat_norm = np.linalg.norm(state[9:13])
        self.quat = (state[9:13] / quat_norm if np.isfinite(quat_norm) and quat_norm > 1e-9
                     else np.array([1.0, 0.0, 0.0, 0.0]))
        self.body_rates = state[13:16] if np.all(np.isfinite(state[13:16])) else np.zeros(3)
        self.altitude_agl = state[2] - self.ground_elevation

    def compute(self, desired_acc: np.ndarray) -> tuple[np.ndarray, float, float]:
        if desired_acc is None or not np.all(np.isfinite(np.asarray(desired_acc, dtype=float))):
            self.integral = np.zeros(2)
            self.prev_tvc = np.zeros(2)
            self.prev_throttle = self.throttle_min
            return np.zeros(2), 0.0, self.throttle_min

        # Thrust supplies the command and cancels gravity; clamp to what the
        # engine can actually deliver, keeping the direction.
        a_thrust = np.asarray(desired_acc, dtype=float).reshape(-1)[0:3] - self.GRAVITY
        thrust_accel = float(np.linalg.norm(a_thrust))
        if thrust_accel > self.max_thrust_accel:
            a_thrust *= self.max_thrust_accel / thrust_accel
            thrust_accel = self.max_thrust_accel
        direction = a_thrust / thrust_accel if thrust_accel > 1e-9 else np.array([0.0, 0.0, 1.0])

        # Altitude-scheduled tilt limit: near the ground the thrust must keep
        # enough vertical component to hold altitude.
        blend = (np.clip((self.altitude_agl - self.TILT_LOW_ALT)
                         / (self.TILT_HIGH_ALT - self.TILT_LOW_ALT), 0.0, 1.0)
                 if np.isfinite(self.altitude_agl) else 1.0)
        tilt_limit = np.radians(self.TILT_LIMIT_LOW_DEG
                                + blend * (self.TILT_LIMIT_HIGH_DEG - self.TILT_LIMIT_LOW_DEG))
        horizontal = float(np.hypot(direction[0], direction[1]))
        if np.arctan2(horizontal, direction[2]) > tilt_limit:
            bearing = direction[0:2] / horizontal if horizontal > 1e-9 else np.array([1.0, 0.0])
            direction = np.array([bearing[0] * np.sin(tilt_limit),
                                  bearing[1] * np.sin(tilt_limit),
                                  np.cos(tilt_limit)])

        throttle = np.clip(self.mass * thrust_accel / self.max_thrust,
                           max(self.throttle_min, self.MIN_THROTTLE), self.throttle_max)
        throttle_step = self.throttle_rate_limit * self.dt
        throttle = float(self.prev_throttle
                         + np.clip(throttle - self.prev_throttle, -throttle_step, throttle_step))
        self.prev_throttle = throttle

        qw, qx, qy, qz = self.quat
        q_vec = np.array([qx, qy, qz])
        t = 2.0 * np.cross(-q_vec, direction)
        direction_body = direction + qw * t + np.cross(-q_vec, t)

        # Shortest rotation taking body z onto the commanded direction.
        axis = np.array([-direction_body[1], direction_body[0], 0.0])
        sin_mag = float(np.linalg.norm(axis))
        angle = float(np.arctan2(sin_mag, direction_body[2]))
        if sin_mag > 1e-9:
            axis /= sin_mag

        rate_cmd = self.ATTITUDE_GAIN * angle * axis[0:2]
        rate_norm = float(np.linalg.norm(rate_cmd))
        if rate_norm > self.MAX_BODY_RATE:
            rate_cmd *= self.MAX_BODY_RATE / rate_norm

        rate_error = rate_cmd - self.body_rates[0:2]
        torque = self.inertia * (self.RATE_KP * rate_error + self.RATE_KI * self.integral)

        # Allocation: torque -> gimbal angle, which divides out thrust and arm
        # so the loop gains stay valid across throttle and burn state.
        thrust = self.max_thrust * throttle
        ratio = (torque / (thrust * self.moment_arm) if thrust > 1e-6
                 else np.full(2, np.inf))
        tvc = np.degrees(np.arcsin(np.clip(ratio, -1.0, 1.0)))
        tvc = np.clip(tvc, -self.max_gimbal, self.max_gimbal)
        gimbal_step = self.gimbal_rate_limit * self.dt
        tvc = self.prev_tvc + np.clip(tvc - self.prev_tvc, -gimbal_step, gimbal_step)
        self.prev_tvc = tvc

        saturated = (np.any(np.abs(ratio) >= 1.0)
                     or np.any(np.abs(tvc) >= self.max_gimbal - 1e-9))
        if not saturated and not self.burnout and np.all(np.isfinite(rate_error)):
            self.integral = np.clip(self.integral + rate_error * self.dt,
                                    -self.INTEGRAL_LIMIT, self.INTEGRAL_LIMIT)

        roll = float(np.clip(-self.ROLL_KP * self.roll_inertia * self.body_rates[2],
                             -self.max_roll, self.max_roll))
        return tvc, roll, throttle
