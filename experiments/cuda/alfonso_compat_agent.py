"""Non-release GNC compatibility fixture adapted from ``origin/alfonso``.

The original branch diverged before the v0.1.1 observation schema and its RL
entry points were later removed.  This module keeps the useful estimator,
target-ordering, proportional-navigation, and inner-loop ideas in one
dependency-free evaluation fixture.  It deliberately never reads ``info`` or
the oracle 13-state. Phase 5 showed that it runs on v0.1.1, but it does not pass
closed-loop TensorFlight transfer and must not be presented as a release agent.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from BalloonPoppingGymEnv.agents.base_agent import BaseAgent


def _safe_action(
    *, launch: bool, launch_attitude: np.ndarray, throttle: float
) -> dict[str, object]:
    return {
        "launch": launch,
        "launch_inclination_heading": launch_attitude.copy(),
        "tvc": np.zeros(2),
        "roll": 0.0,
        "throttle": float(throttle),
    }


class _SensorEstimator:
    def __init__(self, given_parameters: dict) -> None:
        sensors = given_parameters["rocket"]["sensors"]
        self.dt = 1.0 / float(sensors["sampling_rate"])
        self.balloon_radius = float(given_parameters["balloon"]["radius"])
        self.ground_elevation = float(given_parameters["environment"]["elevation"])
        self.error_buffer: deque[float] = deque(maxlen=200)
        self.reset()

    def reset(self) -> None:
        self.position = np.asarray((0.0, 0.0, self.ground_elevation))
        self.velocity = np.zeros(3)
        self.specific_force = np.zeros(3)
        self.attitude = np.asarray((1.0, 0.0, 0.0, 0.0))
        self.angular_rate = np.zeros(3)
        self.tracks: dict[int, dict[str, object]] = {}
        self.error_buffer.clear()

    @property
    def state(self) -> np.ndarray:
        return np.concatenate(
            (
                self.position,
                self.velocity,
                self.specific_force,
                self.attitude,
                self.angular_rate,
            )
        )

    def update(self, observation: dict) -> np.ndarray:
        sensors = np.asarray(observation["rocket_sensors"], dtype=np.float64)
        if not np.isfinite(sensors).all():
            return self.state
        self.angular_rate = sensors[0:3].copy()
        self.specific_force = sensors[3:6].copy()
        self.position = sensors[6:9].copy()
        self.velocity = sensors[9:12].copy()
        delta_theta = self.angular_rate * self.dt
        magnitude = float(np.linalg.norm(delta_theta))
        if magnitude > 1e-8:
            vector = delta_theta / magnitude * np.sin(magnitude / 2)
            delta = np.concatenate(([np.cos(magnitude / 2)], vector))
            w, x, y, z = self.attitude
            dw, dx, dy, dz = delta
            self.attitude = np.asarray(
                (
                    w * dw - x * dx - y * dy - z * dz,
                    w * dx + x * dw + y * dz - z * dy,
                    w * dy - x * dz + y * dw + z * dx,
                    w * dz + x * dy - y * dx + z * dw,
                )
            )
            self.attitude /= max(np.linalg.norm(self.attitude), 1e-15)
        return self.state

    def predict_target(self, observation: dict, target_index: int) -> np.ndarray:
        time_value = float(observation["simulation_time"])
        state = np.asarray(observation["balloon_states"], dtype=np.float64)[
            target_index
        ]
        position = state[:3]
        velocity = state[3:6]
        track = self.tracks.setdefault(
            target_index,
            {
                "short_position": None,
                "expiry": 0.0,
                "velocity": deque(maxlen=100),
            },
        )
        velocity_history = track["velocity"]
        assert isinstance(velocity_history, deque)
        velocity_history.append(velocity.copy())
        short_position = track["short_position"]
        expiry = float(track["expiry"])
        if short_position is None:
            track["short_position"] = position + velocity * 0.1
            track["expiry"] = time_value + 0.1
        elif time_value >= expiry:
            elapsed = time_value - (expiry - 0.1)
            if elapsed > 0:
                self.error_buffer.append(
                    float(
                        np.linalg.norm(np.asarray(short_position) - position) / elapsed
                    )
                )
            track["short_position"] = position + velocity * 0.1
            track["expiry"] = time_value + 0.1

        displacement = position - self.position
        distance = float(np.linalg.norm(displacement))
        direction = displacement / max(distance, 1e-6)
        closing = float(np.dot(self.velocity - velocity, direction))
        geometric_horizon = distance / max(closing, 1.0)
        mean_error_rate = (
            float(np.mean(self.error_buffer)) if self.error_buffer else 0.0
        )
        velocity_std = (
            float(np.mean(np.std(np.asarray(velocity_history), axis=0)))
            if len(velocity_history) > 1
            else 0.0
        )
        uncertainty = 0.6 * mean_error_rate + 0.4 * velocity_std
        risk_horizon = self.balloon_radius * 1.5 / max(uncertainty, 1e-6)
        horizon = geometric_horizon * np.exp(-geometric_horizon / risk_horizon)
        horizon = float(np.clip(horizon, self.dt, 4.0))
        return np.concatenate((position + velocity * horizon, velocity))


class _TargetPlanner:
    def __init__(self, *, launch_time: float = 70.0, target_count: int = 10) -> None:
        self.launch_time = float(launch_time)
        self.target_count = int(target_count)

    def should_launch(self, observation: dict) -> bool:
        return float(observation["simulation_time"]) >= self.launch_time

    @staticmethod
    def launch_attitude(observation: dict) -> np.ndarray:
        states = np.asarray(observation["balloon_states"], dtype=np.float64)
        status = np.asarray(observation["balloon_status"]).reshape(-1)
        valid = (status == 1) & np.isfinite(states).all(axis=1)
        velocity = states[valid, 3:5]
        mean_velocity = (
            velocity.mean(axis=0) if velocity.size else np.asarray((1.0, 0.0))
        )
        norm = float(np.linalg.norm(mean_velocity))
        direction = mean_velocity / norm if norm > 1e-5 else np.asarray((1.0, 0.0))
        heading = float(np.degrees(np.arctan2(direction[1], direction[0])) % 360)
        return np.asarray((90.0, heading))

    def select_targets(self, observation: dict) -> list[int]:
        states = np.asarray(observation["balloon_states"], dtype=np.float64)
        status = np.asarray(observation["balloon_status"]).reshape(-1)
        valid_indices = np.flatnonzero((status == 1) & np.isfinite(states).all(axis=1))
        if valid_indices.size == 0:
            return []
        positions = states[valid_indices, :3]
        velocities = states[valid_indices, 3:5]
        horizontal = velocities.mean(axis=0)
        horizontal /= max(float(np.linalg.norm(horizontal)), 1e-6)
        radius = np.linalg.norm(positions[:, :2], axis=1)
        denominator = float(np.dot(radius, radius))
        slope = (
            float(np.dot(radius, positions[:, 2]) / denominator)
            if denominator > 1e-6
            else 1.0
        )
        axis = np.asarray((horizontal[0], horizontal[1], slope))
        axis /= max(float(np.linalg.norm(axis)), 1e-6)
        progress = positions @ axis
        cross_track = np.linalg.norm(positions - progress[:, None] * axis, axis=1)
        forward = progress > 0
        candidates = valid_indices[forward]
        if candidates.size == 0:
            candidates = valid_indices
            candidate_progress = progress
            candidate_cross_track = cross_track
        else:
            candidate_progress = progress[forward]
            candidate_cross_track = cross_track[forward]
        # The branch's dynamic-programming selector strongly preferred forward
        # progress and low cross-track error.  This stable lexicographic form
        # preserves that intent while supporting fewer than ten valid balloons.
        order = np.lexsort((candidate_cross_track, candidate_progress))
        count = min(self.target_count, candidates.size)
        return candidates[order[:count]].astype(int).tolist()


class _ProportionalNavigator:
    navigation_constant = 3.0
    maximum_acceleration = 30.0

    @classmethod
    def compute(
        cls, target_state: np.ndarray, rocket_state: np.ndarray
    ) -> tuple[np.ndarray, float]:
        line_of_sight = target_state[:3] - rocket_state[:3]
        distance = float(np.linalg.norm(line_of_sight))
        if not np.isfinite(distance) or distance < 1e-3:
            return np.zeros(3), 1.0
        direction = line_of_sight / distance
        relative_velocity = target_state[3:6] - rocket_state[3:6]
        closing = -float(np.dot(relative_velocity, direction))
        los_rate = np.cross(line_of_sight, relative_velocity) / max(
            float(np.dot(line_of_sight, line_of_sight)), 1e-9
        )
        acceleration = (
            cls.navigation_constant * max(closing, 0.0) * np.cross(los_rate, direction)
        )
        magnitude = float(np.linalg.norm(acceleration))
        if magnitude > cls.maximum_acceleration:
            acceleration *= cls.maximum_acceleration / magnitude
        throttle = 0.9
        if distance < 20.0:
            transverse = (
                relative_velocity - np.dot(relative_velocity, direction) * direction
            )
            transverse_speed = float(np.linalg.norm(transverse))
            if transverse_speed > 1.0 and closing > 0:
                ease = transverse_speed / (transverse_speed + abs(closing) + 1e-6)
                throttle *= float(np.clip(1.0 - 0.4 * ease, 0.5, 1.0))
        return acceleration, float(np.clip(throttle, 0.0, 1.0))


class _InnerController:
    def __init__(self, given_parameters: dict) -> None:
        control = given_parameters["rocket"]["control"]
        self.maximum_gimbal = float(control["max_gimbal_angle"])
        self.maximum_roll = float(control["max_roll_torque"])
        self.throttle_range = tuple(float(item) for item in control["throttle_range"])
        sampling_rate = float(given_parameters["rocket"]["sensors"]["sampling_rate"])
        self.dt = 1.0 / sampling_rate
        self.integral = np.zeros(2)

    def reset(self) -> None:
        self.integral[:] = 0

    def compute(
        self, rocket_state: np.ndarray, acceleration: np.ndarray, throttle: float
    ) -> tuple[np.ndarray, float, float]:
        quaternion = rocket_state[9:13].copy()
        quaternion /= max(float(np.linalg.norm(quaternion)), 1e-15)
        angular_rate = np.nan_to_num(rocket_state[13:16])
        thrust_acceleration = acceleration - np.asarray((0.0, 0.0, -9.81))
        desired_world = thrust_acceleration / max(
            float(np.linalg.norm(thrust_acceleration)), 1e-15
        )
        scalar = quaternion[0]
        vector = quaternion[1:4]
        intermediate = 2.0 * np.cross(-vector, desired_world)
        desired_body = (
            desired_world + scalar * intermediate + np.cross(-vector, intermediate)
        )
        rotation_axis = np.asarray((-desired_body[1], desired_body[0], 0.0))
        sine = float(np.linalg.norm(rotation_axis))
        angle = float(np.arctan2(sine, desired_body[2]))
        if sine > 1e-9:
            rotation_axis /= sine
        desired_rate = 4.5 * angle * rotation_axis
        desired_rate[2] = 0
        low, high = self.throttle_range
        throttle = float(np.clip(throttle, low, high))
        error = desired_rate - angular_rate
        self.integral += error[:2] * self.dt
        self.integral = np.clip(self.integral, -0.05, 0.05)
        proportional_gain = 2.0 / max(throttle, 0.15)
        tvc = proportional_gain * error[:2] + 0.5 * self.integral
        return (
            np.clip(tvc, -self.maximum_gimbal, self.maximum_gimbal),
            float(np.clip(error[2], -self.maximum_roll, self.maximum_roll)),
            throttle,
        )


class AlfonsoCompatibilityAgent(BaseAgent):
    """A repaired v0.1.1 compatibility port of the alfonso GNC baseline."""

    def __init__(
        self,
        given_parameters: dict,
        *,
        launch_time: float = 70.0,
        target_count: int = 10,
    ) -> None:
        super().__init__(given_parameters)
        self.estimator = _SensorEstimator(given_parameters)
        self.planner = _TargetPlanner(
            launch_time=launch_time, target_count=target_count
        )
        self.controller = _InnerController(given_parameters)
        self.reset()

    def reset(self) -> None:
        self.estimator.reset()
        self.controller.reset()
        self.launched = False
        self.launch_attitude = np.asarray((90.0, 0.0))
        self.targets: list[int] = []
        self.target_cursor = 0
        self.last_time = -np.inf

    def get_action(self, observation: dict) -> dict[str, object]:
        simulation_time = float(observation["simulation_time"])
        if simulation_time < self.last_time:
            self.reset()
        self.last_time = simulation_time
        if not self.launched:
            self.launched = self.planner.should_launch(observation)
            if not self.launched:
                return _safe_action(
                    launch=False,
                    launch_attitude=self.launch_attitude,
                    throttle=self.controller.throttle_range[0],
                )
            self.launch_attitude = self.planner.launch_attitude(observation)
            self.targets = self.planner.select_targets(observation)
            self.target_cursor = 0

        rocket_state = self.estimator.update(observation)
        status = np.asarray(observation["balloon_status"]).reshape(-1)
        while (
            self.target_cursor < len(self.targets)
            and status[self.targets[self.target_cursor]] == 2
        ):
            self.target_cursor += 1
        if self.target_cursor >= len(self.targets):
            return _safe_action(
                launch=True,
                launch_attitude=self.launch_attitude,
                throttle=self.controller.throttle_range[0],
            )
        target_state = self.estimator.predict_target(
            observation, self.targets[self.target_cursor]
        )
        acceleration, desired_throttle = _ProportionalNavigator.compute(
            target_state, rocket_state
        )
        tvc, roll, throttle = self.controller.compute(
            rocket_state, acceleration, desired_throttle
        )
        return {
            "launch": True,
            "launch_inclination_heading": self.launch_attitude.copy(),
            "tvc": tvc,
            "roll": roll,
            "throttle": throttle,
        }
