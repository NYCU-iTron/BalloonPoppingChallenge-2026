"""Parameter-driven gust and sensor effects for TensorFlight."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from BalloonPoppingGymEnv.tensorflight.balloons import quaternion_to_matrix


Tensor = torch.Tensor


def _triple(value: float | Sequence[float]) -> tuple[float, float, float]:
    if isinstance(value, (int, float)):
        scalar = float(value)
        return scalar, scalar, scalar
    if len(value) != 3:
        raise ValueError("sensor vector parameters must contain three values")
    return tuple(float(item) for item in value)  # type: ignore[return-value]


@dataclass(frozen=True)
class SensorEffectConfig:
    sampling_rate: float = 100.0
    gyro_position: float = 0.0
    accelerometer_position: float = 0.0
    gnss_position: float = 0.0
    accelerometer_consider_gravity: bool = True
    gyro_noise_density: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gyro_random_walk_density: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gyro_constant_bias: tuple[float, float, float] = (0.0, 0.0, 0.0)
    accelerometer_noise_density: tuple[float, float, float] = (0.0, 0.0, 0.0)
    accelerometer_random_walk_density: tuple[float, float, float] = (0.0, 0.0, 0.0)
    accelerometer_constant_bias: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gnss_position_accuracy: float = 0.0
    gnss_altitude_accuracy: float = 0.0
    gnss_velocity_accuracy: float = 0.0

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> SensorEffectConfig:
        return cls(
            sampling_rate=float(values.get("sampling_rate", 100.0)),
            gyro_position=float(values.get("gyro_position", 0.0)),
            accelerometer_position=float(values.get("accelerometer_position", 0.0)),
            gnss_position=float(values.get("gnss_position", 0.0)),
            gyro_noise_density=_triple(values.get("gyro_noise_density", 0.0)),
            gyro_random_walk_density=_triple(
                values.get("gyro_random_walk_density", 0.0)
            ),
            gyro_constant_bias=_triple(values.get("gyro_constant_bias", 0.0)),
            accelerometer_noise_density=_triple(
                values.get("accelerometer_noise_density", 0.0)
            ),
            accelerometer_random_walk_density=_triple(
                values.get("accelerometer_random_walk_density", 0.0)
            ),
            accelerometer_constant_bias=_triple(
                values.get("accelerometer_constant_bias", 0.0)
            ),
            gnss_position_accuracy=float(values.get("gnss_position_accuracy", 0.0)),
            gnss_altitude_accuracy=float(values.get("gnss_altitude_accuracy", 0.0)),
            gnss_velocity_accuracy=float(values.get("gnss_velocity_accuracy", 0.0)),
        )


class LinearGustProfile:
    """Per-environment altitude gust nodes with tensor linear interpolation."""

    def __init__(
        self,
        x_nodes: Tensor,
        y_nodes: Tensor,
        *,
        altitude_spacing: float,
    ) -> None:
        if x_nodes.ndim != 2 or x_nodes.shape != y_nodes.shape:
            raise ValueError("gust nodes must have matching [B, K] shapes")
        if x_nodes.shape[1] < 2 or altitude_spacing <= 0:
            raise ValueError("gust profiles require at least two positive-spaced nodes")
        self.x_nodes = x_nodes
        self.y_nodes = y_nodes
        self.altitude_spacing = float(altitude_spacing)

    @classmethod
    def sample(
        cls,
        num_envs: int,
        *,
        max_height: float,
        altitude_spacing: float,
        max_gust_speed: float,
        decay_height: float,
        generator: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
    ) -> LinearGustProfile:
        if decay_height <= 0 or max_height <= 0:
            raise ValueError("gust max_height and decay_height must be positive")
        count = int(torch.ceil(torch.tensor(max_height / altitude_spacing)).item()) + 1
        altitude = torch.arange(count, device=device, dtype=dtype) * altitude_spacing
        decay = torch.exp(-altitude / decay_height)
        uniform = torch.rand(
            (num_envs, count, 2), generator=generator, device=device, dtype=dtype
        )
        values = (2 * uniform - 1) * max_gust_speed * decay[None, :, None]
        return cls(values[..., 0], values[..., 1], altitude_spacing=altitude_spacing)

    def __call__(self, altitude: Tensor) -> Tensor:
        original_shape = altitude.shape
        if original_shape[0] != self.x_nodes.shape[0]:
            raise ValueError("gust altitude first dimension must match num_envs")
        flat = altitude.reshape(altitude.shape[0], -1)
        scaled = flat / self.altitude_spacing
        lower = torch.floor(scaled).to(torch.int64).clamp(0, self.x_nodes.shape[1] - 2)
        fraction = (scaled - lower.to(scaled.dtype)).clamp(0, 1)
        upper = lower + 1
        x0 = torch.gather(self.x_nodes, 1, lower)
        x1 = torch.gather(self.x_nodes, 1, upper)
        y0 = torch.gather(self.y_nodes, 1, lower)
        y1 = torch.gather(self.y_nodes, 1, upper)
        result = torch.stack(
            (x0 + fraction * (x1 - x0), y0 + fraction * (y1 - y0)), dim=-1
        )
        return result.reshape(*original_shape, 2)


class TensorSensorSuite:
    """Batched official-shaped gyro/accelerometer/GNSS observation model."""

    def __init__(
        self,
        num_envs: int,
        config: SensorEffectConfig,
        *,
        device: str | torch.device,
        dtype: torch.dtype,
        seed: int = 0,
    ) -> None:
        if config.sampling_rate <= 0:
            raise ValueError("sensor sampling_rate must be positive")
        self.num_envs = num_envs
        self.config = config
        self.device = torch.device(device)
        self.dtype = dtype
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.gyro_drift = torch.zeros((num_envs, 3), device=self.device, dtype=dtype)
        self.accel_drift = torch.zeros_like(self.gyro_drift)
        self.gyro_noise = torch.tensor(
            config.gyro_noise_density, device=self.device, dtype=dtype
        )
        self.gyro_walk = torch.tensor(
            config.gyro_random_walk_density, device=self.device, dtype=dtype
        )
        self.gyro_bias = torch.tensor(
            config.gyro_constant_bias, device=self.device, dtype=dtype
        )
        self.accel_noise = torch.tensor(
            config.accelerometer_noise_density, device=self.device, dtype=dtype
        )
        self.accel_walk = torch.tensor(
            config.accelerometer_random_walk_density,
            device=self.device,
            dtype=dtype,
        )
        self.accel_bias = torch.tensor(
            config.accelerometer_constant_bias, device=self.device, dtype=dtype
        )

    def reset(self, mask: Tensor | None = None) -> None:
        if mask is None:
            self.gyro_drift.zero_()
            self.accel_drift.zero_()
            return
        reset_mask = torch.as_tensor(mask, device=self.device, dtype=torch.bool).clone()
        self.gyro_drift[reset_mask] = 0
        self.accel_drift[reset_mask] = 0

    def _inertial_measurement(self, ideal: Tensor, *, gyro: bool) -> Tensor:
        rate = self.config.sampling_rate
        noise = self.gyro_noise if gyro else self.accel_noise
        walk = self.gyro_walk if gyro else self.accel_walk
        bias = self.gyro_bias if gyro else self.accel_bias
        drift = self.gyro_drift if gyro else self.accel_drift
        white = (
            torch.randn(
                ideal.shape,
                generator=self.generator,
                device=self.device,
                dtype=self.dtype,
            )
            * noise
            * rate**0.5
        )
        drift.add_(
            torch.randn(
                ideal.shape,
                generator=self.generator,
                device=self.device,
                dtype=self.dtype,
            )
            * walk
            / rate**0.5
        )
        return ideal + white + drift + bias

    def measure(self, state: Tensor, rhs: Tensor, *, gravity: Tensor) -> Tensor:
        """Measure the official gyro/accelerometer/GNSS 12-vector.

        ``gravity`` is the positive local gravitational acceleration.  The
        official Scenario 1 accelerometer uses ``consider_gravity=True``, so
        the downward gravity vector is removed from inertial acceleration.
        """
        gravity = torch.as_tensor(gravity, device=self.device, dtype=self.dtype)
        if gravity.shape != (self.num_envs,):
            raise ValueError(f"gravity must have shape ({self.num_envs},)")
        rotation = quaternion_to_matrix(state[..., 6:10])
        gyro = self._inertial_measurement(state[..., 10:13], gyro=True)
        inertial_acceleration = rhs[..., 3:6]
        if self.config.accelerometer_consider_gravity:
            gravity_vector = torch.stack(
                (torch.zeros_like(gravity), torch.zeros_like(gravity), -gravity),
                dim=-1,
            )
            inertial_acceleration = inertial_acceleration + gravity_vector
        accel_offset = torch.zeros_like(inertial_acceleration)
        accel_offset[..., 2] = self.config.accelerometer_position
        angular_acceleration = rhs[..., 10:13]
        inertial_acceleration = (
            inertial_acceleration
            + torch.linalg.cross(angular_acceleration, accel_offset, dim=-1)
            + torch.linalg.cross(
                state[..., 10:13],
                torch.linalg.cross(state[..., 10:13], accel_offset, dim=-1),
                dim=-1,
            )
        )
        accel_body = torch.matmul(
            rotation.transpose(-1, -2), inertial_acceleration.unsqueeze(-1)
        ).squeeze(-1)
        accel = self._inertial_measurement(accel_body, gyro=False)
        position_sigma = torch.tensor(
            (
                self.config.gnss_position_accuracy,
                self.config.gnss_position_accuracy,
                self.config.gnss_altitude_accuracy,
            ),
            device=self.device,
            dtype=self.dtype,
        )
        gnss_offset = torch.zeros_like(state[..., :3])
        gnss_offset[..., 2] = self.config.gnss_position
        gnss_world_offset = torch.matmul(rotation, gnss_offset.unsqueeze(-1)).squeeze(
            -1
        )
        position = (
            state[..., :3]
            + gnss_world_offset
            + torch.randn(
                (self.num_envs, 3),
                generator=self.generator,
                device=self.device,
                dtype=self.dtype,
            )
            * position_sigma
        )
        gnss_rotational_velocity = torch.matmul(
            rotation,
            torch.linalg.cross(state[..., 10:13], gnss_offset, dim=-1).unsqueeze(-1),
        ).squeeze(-1)
        velocity = (
            state[..., 3:6]
            + gnss_rotational_velocity
            + torch.randn(
                (self.num_envs, 3),
                generator=self.generator,
                device=self.device,
                dtype=self.dtype,
            )
            * self.config.gnss_velocity_accuracy
        )
        return torch.cat((gyro, accel, position, velocity), dim=-1)
