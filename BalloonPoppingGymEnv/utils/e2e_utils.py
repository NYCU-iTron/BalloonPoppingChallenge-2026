import numpy as np
from gymnasium import spaces
import gymnasium as gym

from BalloonPoppingGymEnv.utils.schema import Schema


# roll (1)
# tvc (2)
# throttle (1)
action_space = spaces.Box(
    low=-1,
    high=1,
    shape=(4,),
    dtype=np.float32
)

# aim angle (1)
# relative dist (1)
# relative body pos (3)
# relative body vel (3)
# rocket z (1)
# rocket body vx (1)
# rocket body vy (1)
# rocket body vz (1)
# rocket world vz (1)
# rocket acc (3)
# rocket quat (4)
# rocket body rates (3)
# sin alpha (1)
# sin beta (1)
# prev tvc (2)
# prev roll (1)
# prev throttle (1)
observation_space = spaces.Box(
    low=-np.inf,
    high=np.inf,
    shape=(29,),
    dtype=np.float32
)

def make_custom_env(scenario_params, given_params):
    def _init():
        from BalloonPoppingGymEnv.envs.static_balloon_world import BalloonPoppingEnv
        from BalloonPoppingGymEnv.envs.e2e_static_env import E2EStaticEnv

        raw_env = BalloonPoppingEnv(render_mode=None, parameters=scenario_params)
        return E2EStaticEnv(raw_env, given_params)
    return _init

def linear_schedule(initial_value, final_value):
    def schedule(progress_remaining):
        return final_value + progress_remaining * (initial_value - final_value)
    return schedule

def scale_rl_action(normalized_action: np.ndarray) -> tuple[np.ndarray, float, float]:
    max_roll_torque = 10
    gimbal_range = 15
    throttle_range = [0.0, 1.0]

    roll = normalized_action[0] * max_roll_torque
    tvc = normalized_action[1:3] * gimbal_range
    # tvc/roll's plain `value * max` scaling only works because those ranges
    # are symmetric about 0; throttle_range=[0,1] is not, so the same
    # pattern would let a negative action produce negative throttle
    # (unclamped downstream). Proper [-1,1] -> [lo,hi] rescale instead.
    throttle = throttle_range[0] + (normalized_action[3] + 1.0) * 0.5 * (
        throttle_range[1] - throttle_range[0]
    )

    return tvc, roll, throttle

class RLObservator:
    def __init__(self, given_parameters):
        self.sampling_rate = given_parameters[Schema.Given.Section.ROCKET][Schema.Given.Rocket.SENSORS][Schema.Given.Sensors.SAMPLING_RATE]
        self.dt = 1.0 / self.sampling_rate
        self.burn_time = float(given_parameters[Schema.Given.Section.ROCKET][Schema.Given.Rocket.MOTOR][Schema.Given.Motor.BURN_TIME])

        self.reset()

    def reset(self) -> None:
        self.rocket_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.rocket_pos = np.zeros(3)
        self.rocket_vel = np.zeros(3)
        self.rocket_acc = np.zeros(3)
        self.rocket_gyro = np.zeros(3)
        self.sin_alpha = 0.0
        self.sin_beta = 0.0
        self.prev_tvc = np.zeros(2)
        self.prev_roll = 0.0
        self.prev_throttle = 0.0

        self.simulation_time = 0.0
        self.launch_schedule_end_time = None
        self.ignition_time = None
        self.burnout = False

    def mark_launch_complete(self) -> None:
        self.launch_schedule_end_time = self.simulation_time

    def update_state(self, observation: dict) -> None:
        self.simulation_time = float(observation[Schema.Observation.SIMULATION_TIME])
        rocket_sensors = observation[Schema.Observation.ROCKET_SENSORS]

        # gyroscopes are NaN before ignition -- nothing to integrate yet
        if not np.isnan(rocket_sensors[:3]).any():
            if self.ignition_time is None:
                self.ignition_time = self.simulation_time
            self.burnout = (self.simulation_time - self.ignition_time) >= self.burn_time

            # Parse sensor data
            self.rocket_gyro = rocket_sensors[0:3] # gyroscopes
            self.rocket_acc = rocket_sensors[3:6] # accelerometers
            self.rocket_pos = rocket_sensors[6:9] # GNSS position
            self.rocket_vel = rocket_sensors[9:12] # GNSS velocity

            delta_theta = self.rocket_gyro * self.dt
            theta_mag = np.linalg.norm(delta_theta)

            if theta_mag > 1e-8:
                # Generate delta rotation quaternion
                qw_d = np.cos(theta_mag / 2.0)
                qxyz_d = (delta_theta / theta_mag) * np.sin(theta_mag / 2.0)
                q_delta = np.array([qw_d, qxyz_d[0], qxyz_d[1], qxyz_d[2]])

                # Perform quaternion multiplication
                # quat = quat x q_delta
                qw, qx, qy, qz = self.rocket_quat
                dw, dx, dy, dz = q_delta

                new_qw = qw * dw - qx * dx - qy * dy - qz * dz
                new_qx = qw * dx + qx * dw + qy * dz - qz * dy
                new_qy = qw * dy - qx * dz + qy * dw + qz * dx
                new_qz = qw * dz + qx * dy - qy * dx + qz * dw

                self.rocket_quat = np.array([new_qw, new_qx, new_qy, new_qz])

                # Normalize to eliminate compounding numerical drift errors
                self.rocket_quat /= np.linalg.norm(self.rocket_quat)

    def get_rl_obs(self, target_state: np.ndarray, action: dict | None = None) -> np.ndarray:
        if action is not None:
            self.prev_tvc = np.asarray(action["tvc"], dtype=float)
            self.prev_roll = float(action["roll"])
            self.prev_throttle = float(action["throttle"])

        target_pos = target_state[0:3]
        target_vel = target_state[3:6]

        rel_pos = target_pos - self.rocket_pos
        rel_vel = target_vel - self.rocket_vel
        dist = float(np.linalg.norm(rel_pos))

        qw, qx, qy, qz = self.rocket_quat
        body_z = np.array([
            2 * (qx * qz + qw * qy),
            2 * (qy * qz - qw * qx),
            1 - 2 * (qx * qx + qy * qy),
        ])
        los_hat = rel_pos / dist if dist > 1e-6 else body_z
        aim_angle = float(np.arccos(np.clip(np.dot(los_hat, body_z), -1.0, 1.0)))

        rel_body_pos = self._to_body_frame(rel_pos)
        rel_body_vel = self._to_body_frame(rel_vel)
        rocket_body_vel = self._to_body_frame(self.rocket_vel)

        speed = float(np.linalg.norm(self.rocket_vel))
        min_speed = 1.0
        if speed > min_speed:
            self.sin_alpha = float(np.clip(rocket_body_vel[0] / speed, -1.0, 1.0))
            self.sin_beta = float(np.clip(rocket_body_vel[1] / speed, -1.0, 1.0))
        else:
            self.sin_alpha = 0.0
            self.sin_beta = 0.0

        return np.concatenate([
            [aim_angle, dist],
            rel_body_pos,
            rel_body_vel,
            [self.rocket_pos[2]],
            [rocket_body_vel[0], rocket_body_vel[1], rocket_body_vel[2]],
            [self.rocket_vel[2]],
            self.rocket_acc,
            self.rocket_quat,
            self.rocket_gyro,
            [self.sin_alpha, self.sin_beta],
            self.prev_tvc,
            [self.prev_roll, self.prev_throttle],
        ]).astype(np.float32)

    def _to_body_frame(self, vec: np.ndarray) -> np.ndarray:
        qw, qx, qy, qz = self.rocket_quat
        q_vec = np.array([qx, qy, qz])
        t = 2.0 * np.cross(-q_vec, vec)
        return vec + qw * t + np.cross(-q_vec, t)

def launch_schedule(env: gym.Env, rl_observator: RLObservator) -> tuple[dict, dict]:
    target_altitude = 40
    sampling_rate = rl_observator.sampling_rate
    launch_inclination_heading = np.array([90.0, 0.0])
    rate_targets = np.zeros(3)
    KP = np.array([100.0, 100.0, 100.0])
    KI = np.array([0.0, 0.0, 5.0])
    KD = np.array([0.0, 0.0, 0.0])

    rate_errors = np.zeros((3, 1))
    rocket_sensors = None
    action = None
    info = None

    while True:
        if rocket_sensors is not None and not np.isnan(rocket_sensors[:3]).any():
            rate_errors = np.append(rate_errors, (rate_targets - rocket_sensors[:3]).reshape(-1, 1), axis=1)
            error_integral = np.sum(rate_errors, axis=1) / sampling_rate
            error_derivative = (
                (rate_errors[:, -1] - rate_errors[:, -2]) * sampling_rate
                if rate_errors.shape[1] > 1 else np.zeros(3)
            )
            torque_cmd = KP * rate_errors[:, -1] + KI * error_integral + KD * error_derivative
        else:
            torque_cmd = np.zeros(3)

        action = {
            "launch": True,
            "launch_inclination_heading": launch_inclination_heading,
            "tvc": torque_cmd[0:2],
            "roll": torque_cmd[2],
            "throttle": 1.0,
        }

        observation, _, terminated, truncated, info = env.step(action)
        rl_observator.update_state(observation)

        rocket_sensors = observation[Schema.Observation.ROCKET_SENSORS]
        rocket_pos = rocket_sensors[6:9]
        reached_altitude = np.all(np.isfinite(rocket_pos)) and rocket_pos[2] >= target_altitude
        if reached_altitude or terminated or truncated:
            break

    rl_observator.mark_launch_complete()
    return observation, info
