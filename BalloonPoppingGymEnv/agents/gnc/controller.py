import logging
import numpy as np
from BalloonPoppingGymEnv.utils.schema import Schema


class Controller:
    """Turns one world-frame acceleration command into TVC, roll and throttle.

    Direction sets the attitude, magnitude sets the throttle, so the two can
    never contradict each other. Measured envelope: ~7 m/s^2 of lateral
    acceleration above the tilt limiter, 0.4-0.9 s to reach a step, under
    2 m/s^2 of steady tracking error.
    """

    # Speed comes from the rate loop; raising ATTITUDE_GAIN instead buys little
    # rise time for a lot of overshoot.
    ATTITUDE_GAIN = 2.0        # (1/s) attitude error -> body rate
    MAX_BODY_RATE = 1.2        # (rad/s)
    RATE_KP = 8.0              # (1/s) rate loop bandwidth after plant inversion
    RATE_KI = 8.0              # (1/s^2)
    INTEGRAL_LIMIT = 1.5       # (rad) enough to drive the gimbal to full travel
    ROLL_KP = 2.0              # (1/s)
    MIN_THROTTLE = 0.05        # TVC torque scales with thrust; keep some in hand

    # Net upward acceleration is only ~3 m/s^2, so arresting a sink is
    # expensive: 20 m/s costs ~80 m, 30 m/s costs ~170 m. What has to be
    # limited is therefore a sink rate the remaining altitude can still absorb.
    # An altitude-only rule is both too strict while climbing and too
    # permissive while already dropping.
    GROUND_BUFFER = 20.0       # (m AGL) held in reserve
    SINK_TAU = 1.0             # (s) how briskly vz is pulled back

    # No altitude buffer to fall back on right off the pad: guarantee net
    # climb there instead of just bounding how hard the command may sink.
    # Short window on purpose -- just past the rail, not the whole GROUND_BUFFER.
    MIN_CLIMB_ACCEL = 2.0     # (m/s^2) guaranteed net climb near the ground
    MIN_CLIMB_ALT = 5.0       # (m AGL) above which guidance regains full authority

    # Below this vertical speed, lateral authority is throttled back to avoid AoA stall
    MIN_LATERAL_SPEED = 15.0  # (m/s) for full lateral authority

    # Backstop: a large tilt near the ground is slow to undo whatever the sink
    # rate says. Scheduled gently, since the envelope above carries the load.
    TILT_LIMIT_LOW_DEG = 20.0
    TILT_LIMIT_HIGH_DEG = 70.0
    TILT_LOW_ALT = 30.0
    TILT_HIGH_ALT = 150.0

    # Thrust and gravity alone do not deliver the command: a sustained lateral
    # command swings the velocity vector round faster than the body, so the
    # angle of attack reverses and the fins push back with several m/s^2. The
    # trim measures that rather than modelling it, which also covers mass,
    # thrust and wind error. It must settle slower than the attitude loop.
    ACC_TRIM_TAU = 2.0         # (s)
    ACC_TRIM_LIMIT = 8.0       # (m/s^2) measured peak demand is ~5
    ACC_FILTER_TAU = 0.15      # (s) smoothing on the differentiated velocity

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
        # everything between is interpolated on burn fraction.
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
        self.vertical_velocity = 0.0
        self.velocity = np.zeros(3)
        self.achieved_acc = np.zeros(3)
        self.acc_trim = np.zeros(3)
        self.prev_velocity = None
        self.prev_tvc = np.zeros(2)
        self.prev_throttle = self.throttle_min

    def update(self, rocket_state: np.ndarray, simulation_time: float) -> None:
        """Refresh the vehicle model and state cache. Once per environment step."""
        # Both agent paths idle before launch, so the first call is ignition.
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
        self.vertical_velocity = state[5] if np.isfinite(state[5]) else 0.0
        self.velocity = state[3:6] if np.all(np.isfinite(state[3:6])) else np.zeros(3)

        # Differentiating velocity rather than reading the accelerometer: the
        # sensor's gravity convention is not guaranteed, and getting it wrong
        # biases the trim by a full g. Smoothed, since differentiation is noisy.
        if self.prev_velocity is not None:
            raw = (self.velocity - self.prev_velocity) / self.dt
            blend = min(self.dt / self.ACC_FILTER_TAU, 1.0)
            self.achieved_acc = self.achieved_acc + (raw - self.achieved_acc) * blend
        self.prev_velocity = self.velocity.copy()

    def compute(self, desired_acc: np.ndarray) -> tuple[np.ndarray, float, float]:
        """Command -> (tvc [deg], roll torque, throttle), within actuator limits."""
        if desired_acc is None or not np.all(np.isfinite(np.asarray(desired_acc, dtype=float))):
            self.integral = np.zeros(2)
            self.prev_tvc = np.zeros(2)
            self.prev_throttle = self.throttle_min
            return np.zeros(2), 0.0, self.throttle_min

        command = np.asarray(desired_acc, dtype=float).reshape(-1)[0:3].copy()

        # Floor the vertical command at the sink this altitude can still arrest.
        # Inactive while climbing or dropping slowly, so it costs no agility.
        if np.isfinite(self.altitude_agl):
            recover_accel = max(self.max_thrust_accel + self.GRAVITY[2], 0.1)
            allowed_sink = np.sqrt(2.0 * recover_accel
                                   * max(self.altitude_agl - self.GROUND_BUFFER, 0.0))
            command[2] = max(command[2],
                             (-allowed_sink - self.vertical_velocity) / self.SINK_TAU)

            climb_floor = self.MIN_CLIMB_ACCEL * np.clip(
                1.0 - self.altitude_agl / self.MIN_CLIMB_ALT, 0.0, 1.0)
            command[2] = max(command[2], climb_floor)

        lateral_gate = np.clip(self.vertical_velocity / self.MIN_LATERAL_SPEED, 0.0, 1.0)
        command[0:2] *= lateral_gate

        # Thrust supplies the command, cancels gravity and carries the trim.
        a_thrust = command - self.GRAVITY + self.acc_trim
        thrust_accel = float(np.linalg.norm(a_thrust))
        thrust_limited = thrust_accel > self.max_thrust_accel
        if thrust_limited:
            a_thrust *= self.max_thrust_accel / thrust_accel
            thrust_accel = self.max_thrust_accel
        direction = a_thrust / thrust_accel if thrust_accel > 1e-9 else np.array([0.0, 0.0, 1.0])

        blend = (np.clip((self.altitude_agl - self.TILT_LOW_ALT)
                         / (self.TILT_HIGH_ALT - self.TILT_LOW_ALT), 0.0, 1.0)
                 if np.isfinite(self.altitude_agl) else 1.0)
        tilt_limit = np.radians(self.TILT_LIMIT_LOW_DEG
                                + blend * (self.TILT_LIMIT_HIGH_DEG - self.TILT_LIMIT_LOW_DEG))
        horizontal = float(np.hypot(direction[0], direction[1]))
        tilt_limited = np.arctan2(horizontal, direction[2]) > tilt_limit
        if tilt_limited:
            # A straight-down command has no bearing to keep, so pick one.
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

        # Torque -> gimbal angle. Dividing out thrust and arm keeps the loop
        # gains valid across throttle and burn state, so no gain scheduling.
        thrust = self.max_thrust * throttle
        ratio = (torque / (thrust * self.moment_arm) if thrust > 1e-6
                 else np.full(2, np.inf))
        tvc = np.degrees(np.arcsin(np.clip(ratio, -1.0, 1.0)))
        tvc = np.clip(tvc, -self.max_gimbal, self.max_gimbal)
        gimbal_step = self.gimbal_rate_limit * self.dt
        tvc = self.prev_tvc + np.clip(tvc - self.prev_tvc, -gimbal_step, gimbal_step)
        self.prev_tvc = tvc

        # Integrate only while the command is actually being followed: a limited
        # or saturated command leaves an error on purpose, and chasing it would
        # wind the loop up against a guard that is doing its job.
        saturated = (np.any(np.abs(ratio) >= 1.0)
                     or np.any(np.abs(tvc) >= self.max_gimbal - 1e-9))
        if not saturated and not self.burnout and np.all(np.isfinite(rate_error)):
            self.integral = np.clip(self.integral + rate_error * self.dt,
                                    -self.INTEGRAL_LIMIT, self.INTEGRAL_LIMIT)

        if not (thrust_limited or tilt_limited or self.burnout
                or self.prev_velocity is None):
            error = command - self.achieved_acc
            self.acc_trim = np.clip(self.acc_trim + error * self.dt / self.ACC_TRIM_TAU,
                                    -self.ACC_TRIM_LIMIT, self.ACC_TRIM_LIMIT)

        roll = float(np.clip(-self.ROLL_KP * self.roll_inertia * self.body_rates[2],
                             -self.max_roll, self.max_roll))
        return tvc, roll, throttle
