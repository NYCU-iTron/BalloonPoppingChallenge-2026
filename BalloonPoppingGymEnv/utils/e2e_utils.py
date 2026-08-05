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

def scale_rl_action(normalized_action: np.ndarray) -> tuple[np.ndarray, float, float]:
    max_roll_torque = 10
    gimbal_range = 15
    throttle_range = [0.0, 1.0]

    roll = normalized_action[0] * max_roll_torque
    tvc = normalized_action[1:3] * gimbal_range

    return tvc, roll, throttle

def launch_schedule(env: gym.Env, rl_observator: RLObservator):
    # while ...:
    #     action = {
    #         "launch": True,
    #         "launch_inclination_heading": launch_inclination_heading,
    #         "tvc": tvc,
    #         "roll": roll,
    #         "throttle": throttle,
    #     }

    #     observation, _, _, _, info = env.step(action)
    #     rl_observator.update(observation)

    # rl_obs = rl_observator.get_observation()
    # return rl_obs, info
    ...

class RLObservator:
    def __init__(self, given_parameters):
        # Time step
        self.sampling_rate = given_parameters[Schema.Given.Section.ROCKET][Schema.Given.Rocket.SENSORS][Schema.Given.Sensors.SAMPLING_RATE]
        self.dt = 1.0 / self.sampling_rate


    def update(self, observation: dict) -> None:
        rocket_sensors = observation[Schema.Observation.ROCKET_SENSORS]

        # Parse sensor data
        rocket_gyro = rocket_sensors[0:3] # gyroscopes
        rocket_acc = rocket_sensors[3:6] # accelerometers
        rocket_pos = rocket_sensors[6:9] # GNSS position
        rocket_vel = rocket_sensors[9:12] # GNSS velocity

        delta_theta = rocket_gyro * self.dt
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

    def get_observation(self) -> np.ndarray:
        ...
