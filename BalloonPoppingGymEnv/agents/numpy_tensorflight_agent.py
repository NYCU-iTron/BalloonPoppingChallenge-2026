"""NumPy-only deployment agent for a Phase 4 ``deployment.npz`` artifact."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np

from BalloonPoppingGymEnv.agents.base_agent import BaseAgent


# ``Phase4Trainer.export_self_contained_agent`` replaces this marker in a
# generated submission source. Keeping the repository template empty makes the
# ordinary local path use the separate, inspectable deployment.npz artifact.
EMBEDDED_DEPLOYMENT_BASE64 = ""


def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.asarray(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        )
    )


def _rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    q0, q1, q2, q3 = quaternion / max(np.linalg.norm(quaternion), 1e-15)
    return np.asarray(
        (
            (
                1 - 2 * (q2 * q2 + q3 * q3),
                2 * (q1 * q2 - q0 * q3),
                2 * (q1 * q3 + q0 * q2),
            ),
            (
                2 * (q1 * q2 + q0 * q3),
                1 - 2 * (q1 * q1 + q3 * q3),
                2 * (q2 * q3 - q0 * q1),
            ),
            (
                2 * (q1 * q3 - q0 * q2),
                2 * (q2 * q3 + q0 * q1),
                1 - 2 * (q1 * q1 + q2 * q2),
            ),
        )
    )


class NumpyTensorFlightAgent(BaseAgent):
    """Official-observation policy with launch, bootstrap, and PPO boundaries."""

    def __init__(
        self,
        given_parameters,
        *,
        artifact_path: str | Path | None = None,
    ) -> None:
        super().__init__(given_parameters)
        if artifact_path is None and EMBEDDED_DEPLOYMENT_BASE64:
            artifact_source = io.BytesIO(
                base64.b64decode(EMBEDDED_DEPLOYMENT_BASE64, validate=True)
            )
        else:
            artifact_source = (
                Path(__file__).with_name("deployment.npz")
                if artifact_path is None
                else Path(artifact_path)
            )
        with np.load(artifact_source, allow_pickle=False) as artifact:
            if int(artifact["schema_version"]) != 1:
                raise ValueError("unsupported TensorFlight deployment schema")
            self.weights = [
                np.asarray(artifact[f"actor_weight_{index}"], dtype=np.float64)
                for index in range(3)
            ]
            self.biases = [
                np.asarray(artifact[f"actor_bias_{index}"], dtype=np.float64)
                for index in range(3)
            ]
            self.normalizer_mean = np.asarray(
                artifact["normalizer_mean"], dtype=np.float64
            )
            self.normalizer_variance = np.asarray(
                artifact["normalizer_variance"], dtype=np.float64
            )
            self.normalizer_epsilon = float(artifact["normalizer_epsilon"])
            self.normalizer_clip = float(artifact["normalizer_clip"])
            self.launch_attitude = np.asarray(
                artifact["launch_attitude"], dtype=np.float64
            )
            self.launch_time = float(artifact["launch_time"])
            self.handoff_altitude_agl = float(artifact["handoff_altitude_agl"])
            self.max_roll_torque = float(artifact["max_roll_torque"])
            self.max_gimbal_angle = float(artifact["max_gimbal_angle"])
            self.throttle_range = np.asarray(
                artifact["throttle_range"], dtype=np.float64
            )
        self.sampling_rate = float(
            given_parameters["rocket"]["sensors"]["sampling_rate"]
        )
        self.ground_elevation = float(given_parameters["environment"]["elevation"])
        self.reset()

    def reset(self) -> None:
        self.position = np.asarray((0.0, 0.0, self.ground_elevation))
        self.velocity = np.zeros(3)
        self.specific_force = np.zeros(3)
        self.angular_rate = np.zeros(3)
        self.attitude = np.asarray((1.0, 0.0, 0.0, 0.0))
        self.integral = np.zeros(3)
        self.previous_action = np.asarray((0.0, 0.0, 0.0, 1.0))
        self.controller_active = False
        self.last_time = -np.inf

    def _update_estimate(self, sensors: np.ndarray) -> bool:
        finite = bool(np.isfinite(sensors).all())
        if not finite:
            return False
        self.angular_rate = sensors[0:3].copy()
        self.specific_force = sensors[3:6].copy()
        self.position = sensors[6:9].copy()
        self.velocity = sensors[9:12].copy()
        delta_theta = self.angular_rate / self.sampling_rate
        magnitude = float(np.linalg.norm(delta_theta))
        if magnitude > 1e-8:
            delta = np.concatenate(
                (
                    [np.cos(magnitude / 2)],
                    delta_theta / magnitude * np.sin(magnitude / 2),
                )
            )
            self.attitude = _quaternion_multiply(self.attitude, delta)
            self.attitude /= np.linalg.norm(self.attitude)
        return True

    def _target(self, observation: dict) -> np.ndarray | None:
        states = np.asarray(observation["balloon_states"], dtype=np.float64)
        status = np.asarray(observation["balloon_status"]).reshape(-1)
        candidates = (status == 1) & np.isfinite(states).all(axis=1)
        if not candidates.any():
            return None
        distance = np.linalg.norm(states[:, :3] - self.position, axis=1)
        return states[int(np.argmin(np.where(candidates, distance, np.inf)))].copy()

    def _observation(self, target: np.ndarray) -> np.ndarray:
        relative_position = target[:3] - self.position
        relative_velocity = target[3:6] - self.velocity
        distance = float(np.linalg.norm(relative_position))
        rotation = _rotation_matrix(self.attitude)
        body_z = rotation[:, 2]
        line_of_sight = relative_position / distance if distance > 1e-6 else body_z
        aim_angle = float(np.arccos(np.clip(np.dot(line_of_sight, body_z), -1.0, 1.0)))
        relative_body_position = rotation.T @ relative_position
        relative_body_velocity = rotation.T @ relative_velocity
        rocket_body_velocity = rotation.T @ self.velocity
        speed = float(np.linalg.norm(self.velocity))
        sin_alpha = np.clip(rocket_body_velocity[0] / speed, -1, 1) if speed > 1 else 0
        sin_beta = np.clip(rocket_body_velocity[1] / speed, -1, 1) if speed > 1 else 0
        return np.concatenate(
            (
                [aim_angle, distance],
                relative_body_position,
                relative_body_velocity,
                [self.position[2]],
                rocket_body_velocity,
                [self.velocity[2]],
                self.specific_force,
                self.attitude,
                self.angular_rate,
                [sin_alpha, sin_beta],
                self.previous_action[1:3],
                self.previous_action[0:1],
                self.previous_action[3:4],
            )
        )

    def _policy(self, observation: np.ndarray) -> np.ndarray:
        value = (observation - self.normalizer_mean) / np.sqrt(
            self.normalizer_variance + self.normalizer_epsilon
        )
        value = np.clip(value, -self.normalizer_clip, self.normalizer_clip)
        value = np.tanh(self.weights[0] @ value + self.biases[0])
        value = np.tanh(self.weights[1] @ value + self.biases[1])
        return np.tanh(self.weights[2] @ value + self.biases[2])

    def _scale(self, normalized: np.ndarray) -> np.ndarray:
        result = np.empty(4, dtype=np.float64)
        result[0] = normalized[0] * self.max_roll_torque
        result[1:3] = normalized[1:3] * self.max_gimbal_angle
        low, high = self.throttle_range
        result[3] = low + (normalized[3] + 1) * 0.5 * (high - low)
        return result

    def _bootstrap(self, sensors_finite: bool) -> np.ndarray:
        if sensors_finite:
            error = -self.angular_rate
            self.integral += error / self.sampling_rate
            command = np.asarray((100.0, 100.0, 100.0)) * error
            command += np.asarray((0.0, 0.0, 5.0)) * self.integral
        else:
            command = np.zeros(3)
        return np.asarray((command[2], command[0], command[1], 1.0))

    def get_action(self, observation):
        simulation_time = float(observation["simulation_time"])
        if simulation_time < self.last_time:
            self.reset()
        self.last_time = simulation_time
        sensors = np.asarray(observation["rocket_sensors"], dtype=np.float64)
        sensors_finite = self._update_estimate(sensors)
        if sensors_finite:
            altitude_agl = self.position[2] - self.ground_elevation
            self.controller_active |= altitude_agl >= self.handoff_altitude_agl
        target = self._target(observation)
        if self.controller_active and target is not None:
            control = self._scale(self._policy(self._observation(target)))
        else:
            control = self._bootstrap(sensors_finite)
        self.previous_action = control.copy()
        return {
            "launch": simulation_time >= self.launch_time,
            "launch_inclination_heading": self.launch_attitude.copy(),
            "roll": float(control[0]),
            "tvc": control[1:3].copy(),
            "throttle": float(control[3]),
        }
