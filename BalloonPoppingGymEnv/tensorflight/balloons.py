"""Online batched Scenario 1 balloon dynamics.

The official environment precomputes every balloon trajectory.  This module
stores only the previous/current/next public six-state plus per-balloon
parameters and advances released balloons online.  It is intentionally
competition-specific; it is not a generic RocketPy replacement.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Callable, Mapping

import torch

from BalloonPoppingGymEnv.tensorflight.rocket import LinearTable


Tensor = torch.Tensor
WindProvider = Callable[[Tensor], Tensor]
BalloonResetSampler = Callable[[int], "BalloonBatchParameters"]


def _tensor(value: object, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    return torch.as_tensor(value, device=device, dtype=dtype)


def _positive_normal(
    mean: float,
    standard_deviation: float,
    shape: tuple[int, ...],
    *,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Sample a normal conditioned on values accepted by RocketPy objects."""
    result = mean + standard_deviation * torch.randn(
        shape, generator=generator, device=device, dtype=dtype
    )
    invalid = result <= 0
    while bool(invalid.any()):
        replacement = mean + standard_deviation * torch.randn(
            (int(invalid.sum()),),
            generator=generator,
            device=device,
            dtype=dtype,
        )
        result[invalid] = replacement
        invalid = result <= 0
    return result


def quaternion_to_matrix(quaternion: Tensor) -> Tensor:
    """Return body-to-world rotation matrices for scalar-first quaternions."""
    quaternion = quaternion / torch.linalg.vector_norm(
        quaternion, dim=-1, keepdim=True
    ).clamp_min(torch.finfo(quaternion.dtype).tiny)
    q0, q1, q2, q3 = quaternion.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (q2.square() + q3.square()),
            2 * (q1 * q2 - q0 * q3),
            2 * (q1 * q3 + q0 * q2),
            2 * (q1 * q2 + q0 * q3),
            1 - 2 * (q1.square() + q3.square()),
            2 * (q2 * q3 - q0 * q1),
            2 * (q1 * q3 - q0 * q2),
            2 * (q2 * q3 + q0 * q1),
            1 - 2 * (q1.square() + q2.square()),
        ),
        dim=-1,
    ).reshape(*quaternion.shape[:-1], 3, 3)


@dataclass(frozen=True)
class BalloonBatchParameters:
    """Realized stochastic parameters for a ``[B, N]`` balloon batch."""

    initial_state: Tensor
    origin_offset: Tensor
    quaternion: Tensor
    dry_mass: Tensor
    volume: Tensor
    inertia: Tensor
    release_step: Tensor
    rail_exit_time: Tensor

    def to(self, *, device: torch.device, dtype: torch.dtype) -> BalloonBatchParameters:
        converted: dict[str, Tensor] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name == "release_step":
                converted[field.name] = value.to(device=device, dtype=torch.int64)
            else:
                converted[field.name] = value.to(device=device, dtype=dtype)
        return BalloonBatchParameters(**converted)

    @property
    def batch_shape(self) -> tuple[int, int]:
        return tuple(self.dry_mass.shape)  # type: ignore[return-value]

    def validate(self) -> None:
        batch_shape = self.dry_mass.shape
        expected = {
            "initial_state": (*batch_shape, 6),
            "origin_offset": (*batch_shape, 3),
            "quaternion": (*batch_shape, 4),
            "volume": batch_shape,
            "inertia": (*batch_shape, 3),
            "release_step": batch_shape,
            "rail_exit_time": batch_shape,
        }
        for name, shape in expected.items():
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} must have shape {tuple(shape)}")
        if self.dry_mass.ndim != 2:
            raise ValueError("balloon parameters must use [B, N] batch dimensions")


def sample_scenario1_balloon_parameters(
    num_envs: int,
    num_balloons: int,
    *,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    release_interval: float = 0.5,
    dt: float = 0.01,
    elevation: float = 20.0,
    latitude: float = 22.1749259,
    radius: float = 1.5,
    mass: float = 0.8,
    mass_std: float = 0.2,
    volume_std: float = 0.5,
    inertia_std: float = 0.1,
    latitude_std: float = 0.001,
    longitude_std: float = 0.001,
    propellant_initial_mass: float = 0.03534291735288518,
    rail_length: float = 0.1,
    thrust: float = 50.0,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> BalloonBatchParameters:
    """Sample the published Scenario 1 distributions directly as tensors.

    A supplied canonical fixture remains the authority for numerical fidelity;
    this sampler provides independent training worlds without a RocketPy reset.
    """
    if num_envs < 1 or num_balloons < 1:
        raise ValueError("num_envs and num_balloons must be positive")
    target_device = torch.device(device)
    if generator is None:
        if seed is None:
            raise ValueError("seed is required when generator is not supplied")
        generator = torch.Generator(device=target_device).manual_seed(seed)
    elif seed is not None:
        raise ValueError("pass either seed or generator, not both")
    shape = (num_envs, num_balloons)
    dry_mass = _positive_normal(
        mass,
        mass_std,
        shape,
        generator=generator,
        device=target_device,
        dtype=dtype,
    )
    nominal_volume = 4 / 3 * torch.pi * radius**3
    volume = _positive_normal(
        float(nominal_volume),
        volume_std,
        shape,
        generator=generator,
        device=target_device,
        dtype=dtype,
    )
    inertia = _positive_normal(
        1.0,
        inertia_std,
        (*shape, 3),
        generator=generator,
        device=target_device,
        dtype=dtype,
    )

    latitude_delta = (
        2 * torch.rand(shape, generator=generator, device=target_device, dtype=dtype)
        - 1
    ) * latitude_std
    longitude_delta = (
        2 * torch.rand(shape, generator=generator, device=target_device, dtype=dtype)
        - 1
    ) * longitude_std
    latitude_rad = torch.tensor(latitude, device=target_device, dtype=dtype).deg2rad()
    eccentricity_squared = 6.69437999014e-3
    semimajor_axis = 6_378_137.0
    denominator = torch.sqrt(1 - eccentricity_squared * torch.sin(latitude_rad) ** 2)
    prime_vertical = semimajor_axis / denominator
    meridional = semimajor_axis * (1 - eccentricity_squared) / denominator**3
    east = (
        longitude_delta.deg2rad()
        * (prime_vertical + elevation)
        * torch.cos(latitude_rad)
    )
    north = latitude_delta.deg2rad() * (meridional + elevation)
    up = -(east.square() + north.square()) / (2 * semimajor_axis)
    origin_offset = torch.stack((east, north, up), dim=-1)
    initial_state = torch.zeros((*shape, 6), device=target_device, dtype=dtype)
    initial_state[..., :3] = origin_offset
    initial_state[..., 2] += elevation

    inclination = 90 + 5 * torch.randn(
        shape, generator=generator, device=target_device, dtype=dtype
    )
    heading = 180 + 90 * torch.randn(
        shape, generator=generator, device=target_device, dtype=dtype
    )
    theta = (inclination - 90).deg2rad()
    psi = -heading.deg2rad()
    quaternion = torch.stack(
        (
            torch.cos(theta / 2) * torch.cos(psi / 2),
            -torch.sin(theta / 2) * torch.sin(psi / 2),
            torch.sin(theta / 2) * torch.cos(psi / 2),
            torch.cos(theta / 2) * torch.sin(psi / 2),
        ),
        dim=-1,
    )
    body_z = quaternion_to_matrix(quaternion)[..., :, 2]

    spacing = int(round(release_interval / dt))
    schedule = (
        torch.arange(num_balloons, device=target_device, dtype=torch.int64) * spacing
    )
    random_keys = torch.rand(
        shape, generator=generator, device=target_device, dtype=dtype
    )
    permutation = torch.argsort(random_keys, dim=1)
    release_step = schedule.expand(num_envs, -1).gather(1, permutation)

    initial_mass = dry_mass + propellant_initial_mass
    axial_acceleration = torch.clamp_min(
        thrust / initial_mass - body_z[..., 2] * 9.80665, 1e-6
    )
    rail_exit_time = torch.sqrt(2 * rail_length / axial_acceleration).clamp(0.03, 0.15)
    return BalloonBatchParameters(
        initial_state=initial_state,
        origin_offset=origin_offset,
        quaternion=quaternion,
        dry_mass=dry_mass,
        volume=volume,
        inertia=inertia,
        release_step=release_step,
        rail_exit_time=rail_exit_time,
    )


class Scenario1BalloonSampler:
    """Persistent device RNG for independent stochastic episode resets."""

    def __init__(
        self,
        num_balloons: int,
        *,
        seed: int,
        release_interval: float = 0.5,
        dt: float = 0.01,
        elevation: float = 20.0,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if num_balloons < 1:
            raise ValueError("num_balloons must be positive")
        self.num_balloons = num_balloons
        self.release_interval = release_interval
        self.dt = dt
        self.elevation = elevation
        self.device = torch.device(device)
        self.dtype = dtype
        self.generator = torch.Generator(device=self.device).manual_seed(seed)

    def __call__(self, num_envs: int) -> BalloonBatchParameters:
        return sample_scenario1_balloon_parameters(
            num_envs,
            self.num_balloons,
            generator=self.generator,
            release_interval=self.release_interval,
            dt=self.dt,
            elevation=self.elevation,
            device=self.device,
            dtype=self.dtype,
        )


class TensorBalloonDynamics:
    """Six-state translation model exactly specialized to Scenario 1 balloons."""

    def __init__(
        self,
        model_data: Mapping[str, object],
        parameters: BalloonBatchParameters,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
        wind_provider: WindProvider | None = None,
    ) -> None:
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("balloon dynamics supports float32 or float64")
        self.device = torch.device(device)
        self.dtype = dtype
        self.parameters = parameters.to(device=self.device, dtype=dtype)
        self.parameters.validate()
        self.rotation = quaternion_to_matrix(self.parameters.quaternion)
        self.body_z = self.rotation[..., :, 2]
        self.wind_provider = wind_provider

        atmosphere = model_data["atmosphere"]
        constants = model_data["constants"]
        if not isinstance(atmosphere, Mapping) or not isinstance(constants, Mapping):
            raise TypeError("balloon model data must contain atmosphere and constants")
        self.tables: dict[str, LinearTable] = {}
        for name in ("density", "wind_velocity_x", "wind_velocity_y", "gravity"):
            table = atmosphere[name]
            if not isinstance(table, Mapping):
                raise TypeError(f"atmosphere table {name} must be a mapping")
            self.tables[name] = LinearTable(
                _tensor(table["x"], device=self.device, dtype=dtype),
                _tensor(table["y"], device=self.device, dtype=dtype),
            )
        self.earth_rotation = _tensor(
            constants["earth_rotation"], device=self.device, dtype=dtype
        )
        self.drag_area = float(constants["drag_area"])
        self.drag_coefficient = float(constants["drag_coefficient"])
        self.propellant_initial_mass = float(constants["propellant_initial_mass"])
        self.burn_duration = float(constants["burn_duration"])
        self.thrust = float(constants["thrust"])

    def replace_parameters(
        self, parameters: BalloonBatchParameters, mask: Tensor
    ) -> None:
        incoming = parameters.to(device=self.device, dtype=self.dtype)
        incoming.validate()
        if incoming.batch_shape != self.parameters.batch_shape:
            raise ValueError("replacement balloon parameters must keep [B, N] shape")
        for field in fields(self.parameters):
            current = getattr(self.parameters, field.name)
            current[mask] = getattr(incoming, field.name)[mask]
        rotation = quaternion_to_matrix(self.parameters.quaternion[mask])
        self.rotation[mask] = rotation
        self.body_z[mask] = rotation[..., :, 2]

    def _atmosphere(self, state: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        local_altitude = state[..., 2] - self.parameters.origin_offset[..., 2]
        density = self.tables["density"](local_altitude)
        gravity = self.tables["gravity"](local_altitude)
        wind = torch.stack(
            (
                self.tables["wind_velocity_x"](local_altitude),
                self.tables["wind_velocity_y"](local_altitude),
            ),
            dim=-1,
        )
        if self.wind_provider is not None:
            wind = wind + self.wind_provider(local_altitude)
        return density, gravity, wind

    def total_mass(self, elapsed: Tensor) -> Tensor:
        remaining = torch.clamp(1 - elapsed / self.burn_duration, 0, 1)
        return self.parameters.dry_mass + self.propellant_initial_mass * remaining

    def rail_rhs(self, elapsed: Tensor, state: Tensor) -> Tensor:
        _, gravity, _ = self._atmosphere(state)
        mass = self.total_mass(elapsed)
        axial = torch.clamp_min(self.thrust / mass - self.body_z[..., 2] * gravity, 0)
        acceleration = self.body_z * axial.unsqueeze(-1)
        return torch.cat((state[..., 3:6], acceleration), dim=-1)

    def free_rhs(self, elapsed: Tensor, state: Tensor) -> Tensor:
        density, gravity, wind_xy = self._atmosphere(state)
        velocity = state[..., 3:6]
        zero = torch.zeros_like(density)
        wind = torch.cat((wind_xy, zero.unsqueeze(-1)), dim=-1)
        stream_body = torch.matmul(
            self.rotation.transpose(-1, -2),
            (wind - velocity).unsqueeze(-1),
        ).squeeze(-1)
        aerodynamic_velocity = -stream_body
        speed = torch.linalg.vector_norm(aerodynamic_velocity, dim=-1)
        alpha = torch.atan2(aerodynamic_velocity[..., 1], aerodynamic_velocity[..., 2])
        beta = torch.atan2(aerodynamic_velocity[..., 0], aerodynamic_velocity[..., 2])
        drag = 0.5 * density * speed.square() * self.drag_area * self.drag_coefficient
        sin_alpha, cos_alpha = torch.sin(alpha), torch.cos(alpha)
        sin_beta, cos_beta = torch.sin(beta), torch.cos(beta)
        force_body = torch.stack(
            (
                -drag * sin_beta,
                -drag * sin_alpha * cos_beta,
                -drag * cos_alpha * cos_beta,
            ),
            dim=-1,
        )
        aerodynamic_force = torch.matmul(
            self.rotation, force_body.unsqueeze(-1)
        ).squeeze(-1)

        mass = self.total_mass(elapsed)
        burning = (elapsed > 0) & (elapsed < self.burn_duration)
        thrust_force = self.body_z * torch.where(
            burning, torch.full_like(mass, self.thrust), zero
        ).unsqueeze(-1)
        weight_buoyancy = torch.stack(
            (zero, zero, (-mass + density * self.parameters.volume) * gravity),
            dim=-1,
        )
        coriolis = -2 * torch.linalg.cross(
            self.earth_rotation.expand_as(velocity), velocity, dim=-1
        )
        acceleration = (
            aerodynamic_force + thrust_force + weight_buoyancy
        ) / mass.unsqueeze(-1) + coriolis
        return torch.cat((velocity, acceleration), dim=-1)

    def rhs(self, elapsed: Tensor, state: Tensor) -> Tensor:
        on_rail = elapsed < self.parameters.rail_exit_time
        return torch.where(
            on_rail.unsqueeze(-1),
            self.rail_rhs(elapsed, state),
            self.free_rhs(elapsed, state),
        )

    def rk4_step(
        self,
        elapsed: Tensor,
        state: Tensor,
        *,
        dt: float,
        substeps: int,
    ) -> Tensor:
        if dt <= 0 or substeps < 1:
            raise ValueError("dt and substeps must be positive")
        result = state
        time = elapsed
        step = dt / substeps
        for _ in range(substeps):
            k1 = self.rhs(time, result)
            k2 = self.rhs(time + step / 2, result + step * k1 / 2)
            k3 = self.rhs(time + step / 2, result + step * k2 / 2)
            k4 = self.rhs(time + step, result + step * k3)
            result = result + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            time = time + step
        return result


class TensorBalloonWorld:
    """Online ``[B, N]`` balloon world with no full-horizon trajectory tensor."""

    def __init__(
        self,
        model_data: Mapping[str, object],
        parameters: BalloonBatchParameters,
        *,
        dt: float = 0.01,
        max_time: float = 150.0,
        substeps: int = 2,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
        wind_provider: WindProvider | None = None,
        reset_sampler: BalloonResetSampler | None = None,
    ) -> None:
        if dt <= 0 or max_time <= 0:
            raise ValueError("dt and max_time must be positive")
        self.device = torch.device(device)
        self.dtype = dtype
        self.dt = float(dt)
        self.max_steps = int(round(max_time / dt))
        self.substeps = int(substeps)
        self.dynamics = TensorBalloonDynamics(
            model_data,
            parameters,
            device=self.device,
            dtype=dtype,
            wind_provider=wind_provider,
        )
        self.parameters = self.dynamics.parameters
        self.reset_sampler = reset_sampler
        self.num_envs, self.num_balloons = self.parameters.batch_shape
        self.previous_state = self.parameters.initial_state.clone()
        self.current_state = self.parameters.initial_state.clone()
        self.next_state = self.parameters.initial_state.clone()
        self.status = torch.zeros(
            (self.num_envs, self.num_balloons),
            device=self.device,
            dtype=torch.int8,
        )
        self.current_step = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.int64
        )

    def reset(self, mask: Tensor | None = None) -> None:
        if mask is None:
            reset_mask = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        else:
            reset_mask = torch.as_tensor(
                mask, device=self.device, dtype=torch.bool
            ).clone()
            if reset_mask.shape != (self.num_envs,):
                raise ValueError(f"reset mask must have shape ({self.num_envs},)")
        if self.reset_sampler is not None and bool(reset_mask.any()):
            self.dynamics.replace_parameters(
                self.reset_sampler(self.num_envs), reset_mask
            )
        self.previous_state[reset_mask] = self.parameters.initial_state[reset_mask]
        self.current_state[reset_mask] = self.parameters.initial_state[reset_mask]
        self.next_state[reset_mask] = self.parameters.initial_state[reset_mask]
        self.status[reset_mask] = 0
        self.current_step[reset_mask] = 0

    def step(self, active_envs: Tensor | None = None) -> Tensor:
        if active_envs is None:
            active = self.current_step < self.max_steps - 1
        else:
            active = torch.as_tensor(
                active_envs, device=self.device, dtype=torch.bool
            ) & (self.current_step < self.max_steps - 1)
            if active.shape != (self.num_envs,):
                raise ValueError(f"active_envs must have shape ({self.num_envs},)")
        step_before = self.current_step[:, None]
        released_before = step_before >= self.parameters.release_step
        elapsed = (
            torch.clamp(step_before - self.parameters.release_step, min=0).to(
                self.dtype
            )
            * self.dt
        )
        integrated = self.dynamics.rk4_step(
            elapsed,
            self.current_state,
            dt=self.dt,
            substeps=self.substeps,
        )
        advance = active[:, None] & released_before
        self.next_state.copy_(
            torch.where(advance.unsqueeze(-1), integrated, self.current_state)
        )
        self.previous_state.copy_(self.current_state)
        self.current_state.copy_(self.next_state)
        self.current_step.add_(active.to(torch.int64))

        released_now = self.current_step[:, None] >= self.parameters.release_step
        self.status.copy_(
            torch.where(
                (self.status == 0) & released_now,
                torch.ones_like(self.status),
                self.status,
            )
        )
        return active

    @property
    def state_storage_elements(self) -> int:
        return sum(
            tensor.numel()
            for tensor in (self.previous_state, self.current_state, self.next_state)
        )
