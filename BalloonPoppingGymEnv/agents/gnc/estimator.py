import logging
import numpy as np
from collections import deque
from BalloonPoppingGymEnv.utils.schema import Schema

class Estimator:
    def __init__(self, given_parameters):
        self.logger = logging.getLogger(__name__)
        self.given_parameters = given_parameters

        # Time step
        self.sampling_rate = given_parameters[Schema.Given.Section.ROCKET][Schema.Given.Rocket.SENSORS][Schema.Given.Sensors.SAMPLING_RATE]
        self.dt = 1.0 / self.sampling_rate

        # Rocket state
        self.rocket_pos = np.zeros(3)
        self.rocket_vel = np.zeros(3)
        self.rocket_acc = np.zeros(3)
        self.rocket_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.rocket_gyro = np.zeros(3)
        self.rocket_state = np.concatenate([self.rocket_pos, self.rocket_vel, self.rocket_acc, self.rocket_quat, self.rocket_gyro])

        # Target prediction parameters
        self.num_balloons = given_parameters[Schema.Given.Section.BALLOON][Schema.Given.Balloon.NUM]
        self.logger.info(f"Estimator initialized with {self.num_balloons} balloons.")
        self.vel_history_len = 100
        self.error_buffer_len = 200
        self.max_error_dist = given_parameters[Schema.Given.Section.BALLOON][Schema.Given.Balloon.RADIUS] * 1.5
        self.max_prediction_horizon = 4.0 # seconds

        # Target prediction state
        self.tracks = {}
        self.error_buffer = deque(maxlen=self.error_buffer_len)

        self.logger.info("Estimator initialized.")

    def reset(self):
        """
        Resets estimator internal storage states.
        """
        # Reset rocket state
        self.rocket_gyro = np.zeros(3)
        self.rocket_acc = np.zeros(3)
        self.rocket_pos = np.zeros(3)
        self.rocket_vel = np.zeros(3)
        self.rocket_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.rocket_state = np.concatenate([self.rocket_pos, self.rocket_vel, self.rocket_acc, self.rocket_quat, self.rocket_gyro])

        # Reset target prediction state
        self.error_buffer.clear()
        self.tracks = {}

    def update_rocket_state(self, observation: dict) -> np.ndarray:
        """
        Estimate rocket state from IMU and GNSS measurements.

        Returns
        -------
        state : np.ndarray
            The estimated 16-dimensional state vector structured as [pos(3), vel(3), acc(3), quat(4), gyro(3)]:
            - pos : Position coordinates (x, y, z).
            - vel : Velocity components (vx, vy, vz).
            - acc : Body-frame specific force from the accelerometer (ax, ay, az);
                    includes gravity reaction, not world-frame acceleration.
            - quat : Orientation quaternion (qw, qx, qy, qz).
            - gyro : Angular velocity components (gx, gy, gz).
        """
        rocket_sensors = observation[Schema.Observation.ROCKET_SENSORS]

        # gyroscopes will be NaN before launch
        if np.isnan(rocket_sensors[:3]).any():
            return self.rocket_state

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

        self.rocket_state = np.concatenate([self.rocket_pos, self.rocket_vel, self.rocket_acc, self.rocket_quat, self.rocket_gyro])
        return self.rocket_state

    def predict_balloons(self, observation: dict) -> np.ndarray:
        balloon_status = np.array(observation[Schema.Observation.BALLOON_STATUS], dtype=int).flatten()
        balloon_states = np.array(observation[Schema.Observation.BALLOON_STATES], dtype=float)

        pred_horizon = 1.0 # seconds
        pred = balloon_states.copy()
        pred[:, :3] += balloon_states[:, 3:6] * pred_horizon

        inactive_mask = (balloon_status == 0) | (balloon_status == 2)
        pred[inactive_mask, :3] = np.nan
        pred[inactive_mask, 3:6] = 0.0

        return pred

    def predict_target(self, observation: dict, target_idx: int) -> np.ndarray:
        """
        Predict selected target balloon position with adaptive TOF.

        Compares past predictions against the current actual position to build
        a running error estimate, then uses that error to constrain the
        prediction horizon.

        Parameters
        ----------
        observation : dict
            Current observation (contains simulation_time and balloon_states).
        target_idx : int
            Index of the selected balloon; resets history on target switch.

        Returns
        -------
        predicted_pos : np.ndarray
            Shape (3,): [predicted_pos(3)].
        """
        # Check if target exists
        if target_idx is None:
            return np.full(3, np.nan)

        if target_idx not in self.tracks:
            self.tracks[target_idx] = {
                "short_pred_pos": None,
                "short_expire_time": 0.0,
                "vel_history": deque(maxlen=self.vel_history_len)
            }

        # Get time variables
        current_time = float(observation[Schema.Observation.SIMULATION_TIME])
        current_step = int(round(current_time * self.sampling_rate))

        # Get target balloon state
        balloon_states = np.array(observation[Schema.Observation.BALLOON_STATES], dtype=float)
        balloon_state = balloon_states[target_idx]
        balloon_pos = balloon_state[:3]
        balloon_vel = balloon_state[3:6]

        # Update velocity history
        track = self.tracks[target_idx]
        track["vel_history"].append(balloon_vel)

        # Update prediction error buffer
        if track["short_pred_pos"] is None:
            track["short_pred_pos"] = balloon_pos + balloon_vel * 0.1
            track["short_expire_time"] = current_time + 0.1

        elif current_time >= track["short_expire_time"]:
            # Calculate exact elapsed time interval for the short-term window
            short_dt = current_time - (track["short_expire_time"] - 0.1)
            if short_dt > 0:
                # Measure deviation between the historical prediction and reality
                error = np.linalg.norm(track["short_pred_pos"] - balloon_pos)
                error_rate = error / short_dt
                self.error_buffer.append(error_rate)

            track["short_pred_pos"] = balloon_pos + balloon_vel * 0.1
            track["short_expire_time"] = current_time + 0.1

        # --------------------------------- Geometry --------------------------------- #
        # Unit displacement vector from rocket to balloon
        displacement = balloon_pos - self.rocket_pos
        dist = np.linalg.norm(displacement)
        unit_disp = displacement / max(dist, 1e-6)

        # Relative velocity
        v_rel = self.rocket_vel - balloon_vel

        # Project relative velocity onto the target displacement vector
        v_closing = np.dot(v_rel, unit_disp)

        # Geometric prediction horizon based on closing kinematics
        t_geo = dist / max(v_closing, 1.0)

        # ----------------------------------- Risk ----------------------------------- #
        # Compute error rate from error buffer
        mean_error_rate = float(np.mean(self.error_buffer)) if self.error_buffer else 0.0

        # Compute velocity std
        if len(track["vel_history"]) > 1:
            vel_matrix = np.array(track["vel_history"])
            vel_std = float(np.mean(np.std(vel_matrix, axis=0)))
        else:
            vel_std = 0.0

        w1 = 0.6  # Weight for historical prediction model error
        w2 = 0.4  # Weight for atmospheric wind gust instability

        sigma = w1 * mean_error_rate + w2 * vel_std

        t_risk = self.max_error_dist / max(sigma, 1e-6)

        # ---------------------------------- Fusion ---------------------------------- #
        # Soft Fusion via Dynamic Damping to prevent harsh switching jitter
        pred_horizon = t_geo * np.exp(-t_geo / t_risk)

        # Bound the final prediction horizon safely between physical constraints
        pred_horizon = min(max(pred_horizon, self.dt), self.max_prediction_horizon)

        # Generate state prediction and map into grid storage
        predicted_pos = balloon_pos + balloon_vel * pred_horizon

        return predicted_pos
