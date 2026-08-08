"""Agent-visible contracts for the CUDA training experiment.

The simulator is allowed to own a canonical 13-state vector, but competition
agents are not allowed to observe it.  This module therefore keeps oracle data
in a separate type and defines every launch, handoff, selector, and controller
interface only in terms of the observation fields exposed by the official
environment.

The historical E2E observation builder is included as a compatibility fixture.
It reconstructs its rocket features from the 12 sensor values; it does not read
``info["rocket_states"]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol, runtime_checkable

import numpy as np


POST_LAUNCH_ACTION_FIELDS = ("roll", "tvc_x", "tvc_y", "throttle")
HISTORICAL_E2E_OBSERVATION_SIZE = 29


def _array(
    value: object,
    shape: tuple[int, ...],
    name: str,
    *,
    finite: bool,
    dtype: np.dtype | type = np.float64,
) -> np.ndarray:
    result = np.asarray(value, dtype=dtype)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
    if finite and not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result.copy()


@dataclass(frozen=True)
class OracleRocketState:
    """Canonical 13-state used only by diagnostics and fidelity reports."""

    values: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "values",
            _array(self.values, (13,), "oracle rocket state", finite=False),
        )


@dataclass(frozen=True)
class AgentObservation:
    """The complete observation that an official competition agent may see."""

    simulation_time: float
    balloon_status: np.ndarray
    balloon_states: np.ndarray
    rocket_sensors: np.ndarray

    def __post_init__(self) -> None:
        simulation_time = float(self.simulation_time)
        if not np.isfinite(simulation_time):
            raise ValueError("simulation_time must be finite")

        balloon_states = np.asarray(self.balloon_states, dtype=np.float64)
        if balloon_states.ndim != 2 or balloon_states.shape[1] != 6:
            raise ValueError(
                f"balloon_states must have shape [N, 6], got {balloon_states.shape}"
            )
        balloon_status = np.asarray(self.balloon_status, dtype=np.int64).reshape(-1)
        if balloon_status.shape != (balloon_states.shape[0],):
            raise ValueError(
                "balloon_status must have one entry per balloon, got "
                f"{balloon_status.shape}"
            )
        if not np.isin(balloon_status, (0, 1, 2)).all():
            raise ValueError("balloon_status values must be 0, 1, or 2")

        object.__setattr__(self, "simulation_time", simulation_time)
        object.__setattr__(self, "balloon_status", balloon_status.copy())
        object.__setattr__(self, "balloon_states", balloon_states.copy())
        object.__setattr__(
            self,
            "rocket_sensors",
            _array(
                self.rocket_sensors,
                (12,),
                "rocket_sensors",
                finite=False,
            ),
        )

    @classmethod
    def from_official(cls, observation: Mapping[str, object]) -> AgentObservation:
        """Copy the four fields returned to ``BaseAgent.get_action``."""

        return cls(
            simulation_time=float(observation["simulation_time"]),
            balloon_status=np.asarray(observation["balloon_status"]),
            balloon_states=np.asarray(observation["balloon_states"]),
            rocket_sensors=np.asarray(observation["rocket_sensors"]),
        )

    @property
    def sensors_finite(self) -> bool:
        return bool(np.isfinite(self.rocket_sensors).all())


@dataclass(frozen=True)
class EstimatedRocketFeatures:
    """Sensor-derived 16-state used by agent-side components."""

    position: np.ndarray
    velocity: np.ndarray
    specific_force: np.ndarray
    attitude_quaternion: np.ndarray
    angular_rate: np.ndarray

    def __post_init__(self) -> None:
        for name, size in (
            ("position", 3),
            ("velocity", 3),
            ("specific_force", 3),
            ("attitude_quaternion", 4),
            ("angular_rate", 3),
        ):
            object.__setattr__(
                self,
                name,
                _array(getattr(self, name), (size,), name, finite=True),
            )

    def as_vector(self) -> np.ndarray:
        return np.concatenate(
            (
                self.position,
                self.velocity,
                self.specific_force,
                self.attitude_quaternion,
                self.angular_rate,
            )
        )


@dataclass(frozen=True)
class LaunchDecision:
    """The only outputs owned by a launch planner."""

    launch: bool
    inclination_heading: np.ndarray

    def __post_init__(self) -> None:
        attitude = _array(
            self.inclination_heading,
            (2,),
            "inclination_heading",
            finite=True,
        )
        if not 0.0 <= attitude[0] <= 90.0:
            raise ValueError("launch inclination must be in [0, 90] degrees")
        if not 0.0 <= attitude[1] <= 360.0:
            raise ValueError("launch heading must be in [0, 360] degrees")
        object.__setattr__(self, "launch", bool(self.launch))
        object.__setattr__(self, "inclination_heading", attitude)


@dataclass(frozen=True)
class PostLaunchAction:
    """Physical post-launch controls, separate from launch planning."""

    roll: float
    tvc: np.ndarray
    throttle: float

    def __post_init__(self) -> None:
        roll = float(self.roll)
        throttle = float(self.throttle)
        if not np.isfinite(roll) or not np.isfinite(throttle):
            raise ValueError("roll and throttle must be finite")
        object.__setattr__(self, "roll", roll)
        object.__setattr__(self, "throttle", throttle)
        object.__setattr__(
            self,
            "tvc",
            _array(self.tvc, (2,), "tvc", finite=True),
        )

    def as_vector(self) -> np.ndarray:
        """Return ``[roll, tvc_x, tvc_y, throttle]``."""

        return np.array(
            [self.roll, self.tvc[0], self.tvc[1], self.throttle],
            dtype=np.float64,
        )

    def as_official_action(self, launch_attitude: np.ndarray) -> dict[str, object]:
        """Build the official action mapping after the launch has occurred."""

        return {
            "launch": True,
            "launch_inclination_heading": _array(
                launch_attitude,
                (2,),
                "launch_attitude",
                finite=True,
            ),
            "tvc": self.tvc.copy(),
            "roll": self.roll,
            "throttle": self.throttle,
        }


@runtime_checkable
class LaunchPlannerProtocol(Protocol):
    def plan(self, observation: AgentObservation) -> LaunchDecision: ...


@runtime_checkable
class BootstrapControllerProtocol(Protocol):
    def compute(
        self,
        observation: AgentObservation,
        estimate: EstimatedRocketFeatures,
    ) -> PostLaunchAction: ...


@runtime_checkable
class ControlHandoffProtocol(Protocol):
    def controller_active(
        self,
        observation: AgentObservation,
        estimate: EstimatedRocketFeatures,
    ) -> bool: ...


@runtime_checkable
class TargetSelectorProtocol(Protocol):
    def select_target(
        self,
        observation: AgentObservation,
        estimate: EstimatedRocketFeatures,
    ) -> int | None: ...


@runtime_checkable
class PostLaunchControllerProtocol(Protocol):
    def compute(
        self,
        observation: AgentObservation,
        estimate: EstimatedRocketFeatures,
        target_index: int | None,
    ) -> PostLaunchAction: ...


def initial_attitude_quaternion(inclination: float, heading: float) -> np.ndarray:
    """Match the official launch inclination/heading quaternion convention."""

    psi = np.radians(-float(heading))
    theta = np.radians(float(inclination) - 90.0)
    phi = 0.0
    return np.array(
        [
            np.cos(phi / 2) * np.cos(theta / 2) * np.cos(psi / 2)
            + np.sin(phi / 2) * np.sin(theta / 2) * np.sin(psi / 2),
            np.sin(phi / 2) * np.cos(theta / 2) * np.cos(psi / 2)
            - np.cos(phi / 2) * np.sin(theta / 2) * np.sin(psi / 2),
            np.cos(phi / 2) * np.sin(theta / 2) * np.cos(psi / 2)
            + np.sin(phi / 2) * np.cos(theta / 2) * np.sin(psi / 2),
            np.cos(phi / 2) * np.cos(theta / 2) * np.sin(psi / 2)
            - np.sin(phi / 2) * np.sin(theta / 2) * np.cos(psi / 2),
        ],
        dtype=np.float64,
    )


class HistoricalSensorEstimator:
    """Port of the historical E2E sensor-only state estimator."""

    def __init__(self, *, sampling_rate: float, ground_elevation: float) -> None:
        if sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive")
        self.dt = 1.0 / float(sampling_rate)
        self.ground_elevation = float(ground_elevation)
        self.reset()

    def reset(self, launch_attitude: np.ndarray | None = None) -> None:
        attitude = (
            np.array([90.0, 0.0])
            if launch_attitude is None
            else _array(launch_attitude, (2,), "launch_attitude", finite=True)
        )
        self._position = np.array([0.0, 0.0, self.ground_elevation])
        self._velocity = np.zeros(3)
        self._specific_force = np.zeros(3)
        self._attitude = initial_attitude_quaternion(attitude[0], attitude[1])
        self._angular_rate = np.zeros(3)

    @property
    def estimate(self) -> EstimatedRocketFeatures:
        return EstimatedRocketFeatures(
            position=self._position,
            velocity=self._velocity,
            specific_force=self._specific_force,
            attitude_quaternion=self._attitude,
            angular_rate=self._angular_rate,
        )

    def update(self, observation: AgentObservation) -> EstimatedRocketFeatures:
        sensors = observation.rocket_sensors
        if not np.isfinite(sensors).all():
            return self.estimate

        self._angular_rate = sensors[0:3].copy()
        self._specific_force = sensors[3:6].copy()
        self._position = sensors[6:9].copy()
        self._velocity = sensors[9:12].copy()

        delta_theta = self._angular_rate * self.dt
        magnitude = float(np.linalg.norm(delta_theta))
        if magnitude > 1e-8:
            delta = np.concatenate(
                (
                    [np.cos(magnitude / 2.0)],
                    delta_theta / magnitude * np.sin(magnitude / 2.0),
                )
            )
            self._attitude = _quaternion_multiply(self._attitude, delta)
            self._attitude /= np.linalg.norm(self._attitude)
        return self.estimate


class TimedLaunchPlanner:
    """Deterministic launch-only fixture; it owns no post-launch handoff."""

    def __init__(
        self,
        *,
        launch_time: float = 0.01,
        inclination_heading: np.ndarray | None = None,
    ) -> None:
        self.launch_time = float(launch_time)
        self.inclination_heading = (
            np.array([90.0, 0.0])
            if inclination_heading is None
            else _array(
                inclination_heading,
                (2,),
                "inclination_heading",
                finite=True,
            )
        )

    def plan(self, observation: AgentObservation) -> LaunchDecision:
        return LaunchDecision(
            launch=observation.simulation_time >= self.launch_time,
            inclination_heading=self.inclination_heading,
        )


class HistoricalBootstrapController:
    """Historical rate-hold control used between launch and PPO handoff."""

    def __init__(self, *, sampling_rate: float) -> None:
        self.sampling_rate = float(sampling_rate)
        self.kp = np.array([100.0, 100.0, 100.0])
        self.ki = np.array([0.0, 0.0, 5.0])
        self.kd = np.zeros(3)
        self.reset()

    def reset(self) -> None:
        self._integral = np.zeros(3)
        self._previous_error = np.zeros(3)
        self._has_previous = False

    def compute(
        self,
        observation: AgentObservation,
        estimate: EstimatedRocketFeatures,
    ) -> PostLaunchAction:
        if not isinstance(estimate, EstimatedRocketFeatures):
            raise TypeError("bootstrap controller requires EstimatedRocketFeatures")
        if not observation.sensors_finite:
            return PostLaunchAction(roll=0.0, tvc=np.zeros(2), throttle=1.0)

        error = -estimate.angular_rate
        self._integral += error / self.sampling_rate
        derivative = (
            (error - self._previous_error) * self.sampling_rate
            if self._has_previous
            else np.zeros(3)
        )
        command = self.kp * error + self.ki * self._integral + self.kd * derivative
        self._previous_error = error
        self._has_previous = True
        return PostLaunchAction(roll=command[2], tvc=command[:2], throttle=1.0)


class EstimatedAltitudeHandoff:
    """Latch PPO activation using GNSS-derived altitude, never oracle altitude."""

    def __init__(self, *, ground_elevation: float, altitude_agl: float = 40.0) -> None:
        self.ground_elevation = float(ground_elevation)
        self.altitude_agl = float(altitude_agl)
        self.reset()

    def reset(self) -> None:
        self._active = False

    def controller_active(
        self,
        observation: AgentObservation,
        estimate: EstimatedRocketFeatures,
    ) -> bool:
        if not isinstance(estimate, EstimatedRocketFeatures):
            raise TypeError("handoff requires EstimatedRocketFeatures")
        if observation.sensors_finite:
            altitude_agl = float(estimate.position[2] - self.ground_elevation)
            self._active = self._active or altitude_agl >= self.altitude_agl
        return self._active


class NearestEstimatedTargetSelector:
    """Minimal selector fixture that only uses allowed estimated features."""

    def select_target(
        self,
        observation: AgentObservation,
        estimate: EstimatedRocketFeatures,
    ) -> int | None:
        if not isinstance(estimate, EstimatedRocketFeatures):
            raise TypeError("target selector requires EstimatedRocketFeatures")
        active = observation.balloon_status == 1
        finite = np.isfinite(observation.balloon_states).all(axis=1)
        candidates = active & finite
        if not candidates.any():
            return None
        distance = np.linalg.norm(
            observation.balloon_states[:, :3] - estimate.position,
            axis=1,
        )
        return int(np.argmin(np.where(candidates, distance, np.inf)))


class HistoricalE2EObservationBuilder:
    """Build the historical 29-feature E2E input from estimated state."""

    size = HISTORICAL_E2E_OBSERVATION_SIZE

    def build(
        self,
        estimate: EstimatedRocketFeatures,
        target_state: np.ndarray,
        previous_action: PostLaunchAction | None = None,
    ) -> np.ndarray:
        if not isinstance(estimate, EstimatedRocketFeatures):
            raise TypeError("E2E builder requires EstimatedRocketFeatures")
        target = _array(target_state, (6,), "target_state", finite=True)
        previous = (
            PostLaunchAction(roll=0.0, tvc=np.zeros(2), throttle=0.0)
            if previous_action is None
            else previous_action
        )

        relative_position = target[:3] - estimate.position
        relative_velocity = target[3:6] - estimate.velocity
        distance = float(np.linalg.norm(relative_position))
        body_z = _quaternion_rotate(
            estimate.attitude_quaternion,
            np.array([0.0, 0.0, 1.0]),
        )
        line_of_sight = relative_position / distance if distance > 1e-6 else body_z
        aim_angle = float(np.arccos(np.clip(np.dot(line_of_sight, body_z), -1.0, 1.0)))
        relative_body_position = _quaternion_rotate_inverse(
            estimate.attitude_quaternion,
            relative_position,
        )
        relative_body_velocity = _quaternion_rotate_inverse(
            estimate.attitude_quaternion,
            relative_velocity,
        )
        rocket_body_velocity = _quaternion_rotate_inverse(
            estimate.attitude_quaternion,
            estimate.velocity,
        )
        speed = float(np.linalg.norm(estimate.velocity))
        if speed > 1.0:
            sin_alpha = float(np.clip(rocket_body_velocity[0] / speed, -1.0, 1.0))
            sin_beta = float(np.clip(rocket_body_velocity[1] / speed, -1.0, 1.0))
        else:
            sin_alpha = 0.0
            sin_beta = 0.0

        result = np.concatenate(
            (
                [aim_angle, distance],
                relative_body_position,
                relative_body_velocity,
                [estimate.position[2]],
                rocket_body_velocity,
                [estimate.velocity[2]],
                estimate.specific_force,
                estimate.attitude_quaternion,
                estimate.angular_rate,
                [sin_alpha, sin_beta],
                previous.tvc,
                [previous.roll, previous.throttle],
            )
        ).astype(np.float32)
        if result.shape != (self.size,) or not np.isfinite(result).all():
            raise ValueError("historical E2E observation must be finite and 29-D")
        return result


@dataclass
class MaskedRunningNormalizer:
    """Running mean/variance that ignores inactive or non-finite rows."""

    size: int = HISTORICAL_E2E_OBSERVATION_SIZE
    epsilon: float = 1e-8
    count: int = 0
    mean: np.ndarray = field(init=False)
    m2: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.mean = np.zeros(self.size, dtype=np.float64)
        self.m2 = np.zeros(self.size, dtype=np.float64)

    @property
    def variance(self) -> np.ndarray:
        if self.count < 2:
            return np.ones(self.size, dtype=np.float64)
        return self.m2 / self.count

    def update(
        self,
        batch: np.ndarray,
        controller_active: np.ndarray,
        sensors_finite: np.ndarray,
    ) -> np.ndarray:
        values = np.asarray(batch, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.size:
            raise ValueError(f"batch must have shape [B, {self.size}]")
        active = np.asarray(controller_active, dtype=bool)
        if active.shape != (values.shape[0],):
            raise ValueError("controller_active must have shape [B]")
        finite_sensors = np.asarray(sensors_finite, dtype=bool)
        if finite_sensors.shape != (values.shape[0],):
            raise ValueError("sensors_finite must have shape [B]")
        accepted = active & finite_sensors & np.isfinite(values).all(axis=1)
        selected = values[accepted]
        if selected.size == 0:
            return accepted

        batch_count = selected.shape[0]
        batch_mean = selected.mean(axis=0)
        batch_m2 = ((selected - batch_mean) ** 2).sum(axis=0)
        if self.count == 0:
            self.mean = batch_mean
            self.m2 = batch_m2
            self.count = batch_count
            return accepted

        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean += delta * (batch_count / total)
        self.m2 += batch_m2 + delta * delta * self.count * batch_count / total
        self.count = total
        return accepted

    def normalize(self, batch: np.ndarray, *, clip: float = 10.0) -> np.ndarray:
        values = np.asarray(batch, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.size:
            raise ValueError(f"batch must have shape [B, {self.size}]")
        safe = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        normalized = (safe - self.mean) / np.sqrt(self.variance + self.epsilon)
        return np.clip(normalized, -clip, clip).astype(np.float32)


def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ]
    )


def _quaternion_rotate(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    q_vector = quaternion[1:4]
    intermediate = 2.0 * np.cross(q_vector, vector)
    return vector + quaternion[0] * intermediate + np.cross(q_vector, intermediate)


def _quaternion_rotate_inverse(
    quaternion: np.ndarray, vector: np.ndarray
) -> np.ndarray:
    conjugate = quaternion.copy()
    conjugate[1:4] *= -1.0
    return _quaternion_rotate(conjugate, vector)
