"""GPU-native post-launch rocket plus 100-balloon Phase 3 environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from experiments.cuda.phase2_rocket import (
    ActuatorSpec,
    Scenario1ActuatorBank,
    Scenario1TensorRocket,
    dopri5_step,
    rk4_step,
)
from experiments.cuda.phase3_balloon import (
    BalloonBatchParameters,
    BalloonResetSampler,
    TensorBalloonWorld,
    WindProvider,
)
from experiments.cuda.phase3_effects import SensorEffectConfig, TensorSensorSuite


Tensor = torch.Tensor


@dataclass(frozen=True)
class TensorAgentObservation:
    """Only the four fields exposed by the official agent observation."""

    simulation_time: Tensor
    balloon_status: Tensor
    balloon_states: Tensor
    rocket_sensors: Tensor


@dataclass(frozen=True)
class TensorEnvironmentStep:
    observation: TensorAgentObservation
    reward: Tensor
    terminated: Tensor
    truncated: Tensor


@dataclass(frozen=True)
class TensorOracleState:
    """Private diagnostic state, deliberately separate from policy output."""

    rocket_state: Tensor
    actuator_output: Tensor
    closest_distance: Tensor
    hit_mask: Tensor


@dataclass(frozen=True)
class TensorFlightEnvironmentConfig:
    dt: float = 0.01
    max_time: float = 150.0
    elevation: float = 20.0
    balloon_radius: float = 1.5
    rocket_substeps: int = 1
    balloon_substeps: int = 1
    integrator: str = "rk4"
    critical_distance_float64: bool = False


def segment_distance(
    rocket_start: Tensor,
    rocket_end: Tensor,
    balloon_start: Tensor,
    balloon_end: Tensor,
) -> Tensor:
    """Official independent-parameter swept segment distance for ``[B,N]``."""
    direction_a = rocket_end - rocket_start
    direction_b = balloon_end - balloon_start
    offset = rocket_start - balloon_start
    a = (direction_a * direction_a).sum(-1)
    b = (direction_a * direction_b).sum(-1)
    c = (direction_a * offset).sum(-1)
    e = (direction_b * direction_b).sum(-1)
    f = (direction_b * offset).sum(-1)
    epsilon = 1e-12
    safe_a = a.clamp_min(epsilon)
    safe_e = e.clamp_min(epsilon)
    zero = torch.zeros_like(a)
    one = torch.ones_like(a)
    s0 = (-c / safe_a).clamp(0, 1)
    s1 = ((b - c) / safe_a).clamp(0, 1)
    t0 = (f / safe_e).clamp(0, 1)
    t1 = ((b + f) / safe_e).clamp(0, 1)
    normal = torch.linalg.cross(direction_a, direction_b, dim=-1)
    denominator = (normal * normal).sum(-1)
    denominator_safe = denominator.clamp_min(torch.finfo(offset.dtype).tiny)
    si = (torch.linalg.cross(direction_b, offset, dim=-1) * normal).sum(
        -1
    ) / denominator_safe
    ti = (torch.linalg.cross(direction_a, offset, dim=-1) * normal).sum(
        -1
    ) / denominator_safe
    solvable = denominator > 0
    si = torch.where(solvable, si.clamp(0, 1), zero)
    ti = torch.where(solvable, ti.clamp(0, 1), zero)
    s = torch.stack((zero, one, s0, s1, si), dim=-1)
    t = torch.stack((t0, t1, zero, one, ti), dim=-1)
    separation = (
        offset.unsqueeze(-2)
        + s.unsqueeze(-1) * direction_a.unsqueeze(-2)
        - t.unsqueeze(-1) * direction_b.unsqueeze(-2)
    )
    return torch.sqrt((separation * separation).sum(-1).amin(-1).clamp_min(0))


class TensorFlightEnvironment:
    """Post-launch batched environment whose public step contains no true state."""

    action_size = 4

    def __init__(
        self,
        rocket_model_data: Mapping[str, object],
        balloon_model_data: Mapping[str, object],
        balloon_parameters: BalloonBatchParameters,
        initial_rocket_state: Tensor,
        actuator_specs: Sequence[ActuatorSpec],
        *,
        actuator_demand_rate: float,
        rocket_elapsed: Tensor | float = 0.01,
        start_step: int = 0,
        initial_cached_rhs: Tensor | None = None,
        sensor_config: SensorEffectConfig | None = None,
        wind_provider: WindProvider | None = None,
        balloon_reset_sampler: BalloonResetSampler | None = None,
        config: TensorFlightEnvironmentConfig | None = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        seed: int = 0,
    ) -> None:
        self.config = config or TensorFlightEnvironmentConfig()
        if self.config.integrator not in {"rk4", "official_fsal"}:
            raise ValueError("integrator must be 'rk4' or 'official_fsal'")
        self.device = torch.device(device)
        self.dtype = dtype
        self.initial_rocket_state = torch.as_tensor(
            initial_rocket_state, device=self.device, dtype=dtype
        ).clone()
        if (
            self.initial_rocket_state.ndim != 2
            or self.initial_rocket_state.shape[1] != 13
        ):
            raise ValueError("initial_rocket_state must have shape [B, 13]")
        self.num_envs = self.initial_rocket_state.shape[0]
        if balloon_parameters.batch_shape[0] != self.num_envs:
            raise ValueError("rocket and balloon batch sizes must match")
        self.rocket_model = Scenario1TensorRocket(
            rocket_model_data,
            device=self.device,
            dtype=dtype,
            wind_provider=wind_provider,
        )
        self.balloons = TensorBalloonWorld(
            balloon_model_data,
            balloon_parameters,
            dt=self.config.dt,
            max_time=self.config.max_time,
            substeps=self.config.balloon_substeps,
            device=self.device,
            dtype=dtype,
            wind_provider=wind_provider,
            reset_sampler=balloon_reset_sampler,
        )
        self.actuators = Scenario1ActuatorBank(
            self.num_envs,
            actuator_specs,
            demand_rate=actuator_demand_rate,
            device=self.device,
            dtype=dtype,
        )
        elapsed = torch.as_tensor(rocket_elapsed, device=self.device, dtype=dtype)
        self.initial_rocket_elapsed = torch.broadcast_to(
            elapsed, (self.num_envs,)
        ).clone()
        self.rocket_elapsed = self.initial_rocket_elapsed.clone()
        self.rocket_state = self.initial_rocket_state.clone()
        self.terminated = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.truncated = torch.zeros_like(self.terminated)
        # ActiveRocketPy records an impact root, then needs two subsequent
        # ``step_simulation`` calls to advance into and mark its terminal phase.
        # Preserve that evaluator-visible lifecycle while freezing the rocket.
        self.impact_countdown = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.int8
        )
        self.reward = torch.zeros(self.num_envs, device=self.device, dtype=dtype)
        self.closest_distance = torch.full(
            (self.num_envs, self.balloons.num_balloons),
            torch.inf,
            device=self.device,
            dtype=dtype,
        )
        self.hit_mask = torch.zeros_like(self.closest_distance, dtype=torch.bool)
        self.sensor_suite = TensorSensorSuite(
            self.num_envs,
            sensor_config or SensorEffectConfig(),
            device=self.device,
            dtype=dtype,
            seed=seed,
        )
        if initial_cached_rhs is None:
            self.initial_cached_rhs = self.rocket_model.rhs(
                self.rocket_elapsed,
                self.rocket_state,
                self.actuators.output,
            ).clone()
        else:
            self.initial_cached_rhs = torch.as_tensor(
                initial_cached_rhs, device=self.device, dtype=dtype
            ).clone()
        self.cached_rhs = self.initial_cached_rhs.clone()
        self.rocket_sensors = self._measure_sensors(
            self.rocket_state,
            self.rocket_model.rhs(
                self.rocket_elapsed, self.rocket_state, self.actuators.output
            ),
        )
        self.start_step = int(start_step)
        if self.start_step < 0:
            raise ValueError("start_step must be non-negative")
        for _ in range(self.start_step):
            self.balloons.step()

    def _observation(self) -> TensorAgentObservation:
        return TensorAgentObservation(
            simulation_time=self.balloons.current_step.to(self.dtype) * self.config.dt,
            balloon_status=self.balloons.status,
            balloon_states=self.balloons.current_state,
            rocket_sensors=self.rocket_sensors,
        )

    def _measure_sensors(self, state: Tensor, rhs: Tensor) -> Tensor:
        # ActiveRocketPy 473447d's Accelerometer.measure queries gravity with
        # ``u[3]`` (x velocity), not altitude. Preserve that pinned oracle
        # behavior here; a future upstream correction must deliberately update
        # both the oracle fixture and this compatibility path.
        gravity = self.rocket_model.atmosphere_tables["gravity"](state[..., 3])
        return self.sensor_suite.measure(state, rhs, gravity=gravity)

    def observation(self) -> TensorAgentObservation:
        return self._observation()

    def oracle_state(self) -> TensorOracleState:
        return TensorOracleState(
            rocket_state=self.rocket_state,
            actuator_output=self.actuators.output,
            closest_distance=self.closest_distance,
            hit_mask=self.hit_mask,
        )

    def reset(self, mask: Tensor | None = None) -> TensorAgentObservation:
        if mask is None:
            reset_mask = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        else:
            reset_mask = torch.as_tensor(
                mask, device=self.device, dtype=torch.bool
            ).clone()
            if reset_mask.shape != (self.num_envs,):
                raise ValueError(f"reset mask must have shape ({self.num_envs},)")
        self.rocket_state[reset_mask] = self.initial_rocket_state[reset_mask]
        self.rocket_elapsed[reset_mask] = self.initial_rocket_elapsed[reset_mask]
        self.cached_rhs[reset_mask] = self.initial_cached_rhs[reset_mask]
        self.terminated[reset_mask] = False
        self.truncated[reset_mask] = False
        self.impact_countdown[reset_mask] = 0
        self.reward[reset_mask] = 0
        self.closest_distance[reset_mask] = torch.inf
        self.hit_mask[reset_mask] = False
        self.actuators.reset(reset_mask)
        self.sensor_suite.reset(reset_mask)
        self.balloons.reset(reset_mask)
        for _ in range(self.start_step):
            self.balloons.step(reset_mask)
        endpoint_rhs = self.rocket_model.rhs(
            self.rocket_elapsed, self.rocket_state, self.actuators.output
        )
        measured = self._measure_sensors(self.rocket_state, endpoint_rhs)
        self.rocket_sensors[reset_mask] = measured[reset_mask]
        return self._observation()

    def step(self, command: Tensor) -> TensorEnvironmentStep:
        command = torch.as_tensor(command, device=self.device, dtype=self.dtype)
        if command.shape != (self.num_envs, self.action_size):
            raise ValueError(
                f"command must have shape ({self.num_envs}, {self.action_size})"
            )
        active = ~(self.terminated | self.truncated)
        impact_pending = self.impact_countdown > 0
        flying = active & ~impact_pending
        finite = torch.isfinite(command).all(-1)
        usable = torch.where(
            (active & finite).unsqueeze(-1), command, self.actuators.output
        )
        actuator_output = self.actuators.update(usable, validate_finite=False)
        previous_rocket = self.rocket_state
        if self.config.integrator == "official_fsal":
            integrated, endpoint_rhs = dopri5_step(
                self.rocket_model,
                self.rocket_elapsed,
                previous_rocket,
                actuator_output,
                dt=self.config.dt,
                initial_rhs=self.cached_rhs,
            )
            self.cached_rhs.copy_(
                torch.where(flying.unsqueeze(-1), endpoint_rhs, self.cached_rhs)
            )
        else:
            integrated = rk4_step(
                self.rocket_model,
                self.rocket_elapsed,
                previous_rocket,
                actuator_output,
                dt=self.config.dt,
                substeps=self.config.rocket_substeps,
            )
            endpoint_rhs = self.rocket_model.rhs(
                self.rocket_elapsed + self.config.dt,
                integrated,
                actuator_output,
            )
            self.cached_rhs.copy_(
                torch.where(flying.unsqueeze(-1), endpoint_rhs, self.cached_rhs)
            )
        next_rocket = torch.where(flying.unsqueeze(-1), integrated, previous_rocket)
        new_impact = flying & (integrated[:, 2] < self.config.elevation)
        altitude_span = previous_rocket[:, 2] - integrated[:, 2]
        impact_fraction = torch.where(
            altitude_span.abs() > torch.finfo(self.dtype).eps,
            (previous_rocket[:, 2] - self.config.elevation) / altitude_span,
            torch.ones_like(altitude_span),
        ).clamp(0, 1)
        impact_state = previous_rocket + impact_fraction.unsqueeze(-1) * (
            integrated - previous_rocket
        )
        impact_state[:, 2] = self.config.elevation
        impact_state[:, 6:10] = impact_state[:, 6:10] / torch.linalg.vector_norm(
            impact_state[:, 6:10], dim=-1, keepdim=True
        ).clamp_min(torch.finfo(self.dtype).tiny)
        next_rocket = torch.where(new_impact.unsqueeze(-1), impact_state, next_rocket)
        self.balloons.step(active)

        rocket_start = previous_rocket[:, None, :3].expand(
            -1, self.balloons.num_balloons, -1
        )
        rocket_end = next_rocket[:, None, :3].expand_as(rocket_start)
        if self.config.critical_distance_float64 and self.dtype == torch.float32:
            distance = segment_distance(
                rocket_start.to(torch.float64),
                rocket_end.to(torch.float64),
                self.balloons.previous_state[..., :3].to(torch.float64),
                self.balloons.current_state[..., :3].to(torch.float64),
            ).to(self.dtype)
        else:
            distance = segment_distance(
                rocket_start,
                rocket_end,
                self.balloons.previous_state[..., :3],
                self.balloons.current_state[..., :3],
            )
        released = self.balloons.status == 1
        hits = released & active[:, None] & (distance <= self.config.balloon_radius)
        self.balloons.status.masked_fill_(hits, 2)
        self.closest_distance.copy_(distance)
        self.hit_mask.copy_(hits)
        self.reward.copy_(hits.sum(-1).to(self.dtype))

        self.rocket_state.copy_(next_rocket)
        elapsed_fraction = torch.where(
            new_impact, impact_fraction, torch.ones_like(impact_fraction)
        )
        self.rocket_elapsed.add_(
            flying.to(self.dtype) * elapsed_fraction * self.config.dt
        )
        terminal_now = active & impact_pending & (self.impact_countdown == 1)
        self.impact_countdown.copy_(
            torch.where(
                new_impact,
                torch.full_like(self.impact_countdown, 2),
                torch.where(
                    active & impact_pending,
                    self.impact_countdown - 1,
                    self.impact_countdown,
                ),
            )
        )
        self.terminated |= terminal_now
        self.truncated |= self.balloons.current_step >= self.balloons.max_steps - 1
        measurement_rhs = self.rocket_model.rhs(
            self.rocket_elapsed, self.rocket_state, actuator_output
        )
        measured = self._measure_sensors(self.rocket_state, measurement_rhs)
        self.rocket_sensors.copy_(
            torch.where(flying.unsqueeze(-1), measured, self.rocket_sensors)
        )
        return TensorEnvironmentStep(
            observation=self._observation(),
            reward=self.reward,
            terminated=self.terminated,
            truncated=self.truncated,
        )
