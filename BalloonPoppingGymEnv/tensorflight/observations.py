"""Device-native observation, handoff, selector, and shaping components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from BalloonPoppingGymEnv.tensorflight.balloons import quaternion_to_matrix
from BalloonPoppingGymEnv.tensorflight.environment import TensorAgentObservation


Tensor = torch.Tensor
OBSERVATION_SIZE = 29
ACTION_SIZE = 4


def _quaternion_multiply(left: Tensor, right: Tensor) -> Tensor:
    lw, lx, ly, lz = left.unbind(-1)
    rw, rx, ry, rz = right.unbind(-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


@dataclass(frozen=True)
class TensorEstimatedRocketFeatures:
    """Sensor-derived features; no canonical rocket state is accepted here."""

    position: Tensor
    velocity: Tensor
    specific_force: Tensor
    attitude_quaternion: Tensor
    angular_rate: Tensor


@dataclass(frozen=True)
class TensorPreparedObservation:
    raw: Tensor
    normalized: Tensor
    controller_active: Tensor
    sensors_finite: Tensor
    target_index: Tensor
    target_available: Tensor
    target_state: Tensor
    sin_alpha: Tensor
    sin_beta: Tensor


class TensorSensorEstimator:
    """Batched port of the historical E2E estimator."""

    def __init__(
        self,
        num_envs: int,
        *,
        sampling_rate: float,
        ground_elevation: float,
        device: str | torch.device,
        dtype: torch.dtype,
    ) -> None:
        if num_envs < 1 or sampling_rate <= 0:
            raise ValueError("num_envs and sampling_rate must be positive")
        self.num_envs = num_envs
        self.dt = 1.0 / float(sampling_rate)
        self.ground_elevation = float(ground_elevation)
        self.device = torch.device(device)
        self.dtype = dtype
        self.position = torch.zeros((num_envs, 3), device=self.device, dtype=dtype)
        self.position[:, 2] = ground_elevation
        self.velocity = torch.zeros_like(self.position)
        self.specific_force = torch.zeros_like(self.position)
        self.angular_rate = torch.zeros_like(self.position)
        self.attitude_quaternion = torch.zeros(
            (num_envs, 4), device=self.device, dtype=dtype
        )
        self.attitude_quaternion[:, 0] = 1

    @property
    def features(self) -> TensorEstimatedRocketFeatures:
        return TensorEstimatedRocketFeatures(
            position=self.position,
            velocity=self.velocity,
            specific_force=self.specific_force,
            attitude_quaternion=self.attitude_quaternion,
            angular_rate=self.angular_rate,
        )

    def reset(self, mask: Tensor | None = None) -> None:
        reset_mask = (
            torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
            if mask is None
            else torch.as_tensor(mask, device=self.device, dtype=torch.bool).clone()
        )
        if reset_mask.shape != (self.num_envs,):
            raise ValueError(f"reset mask must have shape ({self.num_envs},)")
        self.position[reset_mask] = 0
        self.position[reset_mask, 2] = self.ground_elevation
        self.velocity[reset_mask] = 0
        self.specific_force[reset_mask] = 0
        self.angular_rate[reset_mask] = 0
        self.attitude_quaternion[reset_mask] = 0
        self.attitude_quaternion[reset_mask, 0] = 1

    def update(self, observation: TensorAgentObservation) -> Tensor:
        sensors = observation.rocket_sensors
        if sensors.shape != (self.num_envs, 12):
            raise ValueError(f"rocket_sensors must have shape ({self.num_envs}, 12)")
        finite = torch.isfinite(sensors).all(-1)
        angular_rate = torch.nan_to_num(sensors[:, 0:3])
        delta_theta = angular_rate * self.dt
        magnitude = torch.linalg.vector_norm(delta_theta, dim=-1)
        safe_magnitude = magnitude.clamp_min(torch.finfo(self.dtype).tiny)
        vector = (
            delta_theta
            / safe_magnitude.unsqueeze(-1)
            * torch.sin(magnitude / 2).unsqueeze(-1)
        )
        vector = torch.where(
            (magnitude > 1e-8).unsqueeze(-1), vector, torch.zeros_like(vector)
        )
        delta = torch.cat((torch.cos(magnitude / 2).unsqueeze(-1), vector), dim=-1)
        attitude = _quaternion_multiply(self.attitude_quaternion, delta)
        attitude = attitude / torch.linalg.vector_norm(
            attitude, dim=-1, keepdim=True
        ).clamp_min(torch.finfo(self.dtype).tiny)
        self.position.copy_(
            torch.where(finite.unsqueeze(-1), sensors[:, 6:9], self.position)
        )
        self.velocity.copy_(
            torch.where(finite.unsqueeze(-1), sensors[:, 9:12], self.velocity)
        )
        self.specific_force.copy_(
            torch.where(finite.unsqueeze(-1), sensors[:, 3:6], self.specific_force)
        )
        self.angular_rate.copy_(
            torch.where(finite.unsqueeze(-1), angular_rate, self.angular_rate)
        )
        self.attitude_quaternion.copy_(
            torch.where(finite.unsqueeze(-1), attitude, self.attitude_quaternion)
        )
        return finite

    def state_dict(self) -> dict[str, Tensor]:
        return {
            "position": self.position.clone(),
            "velocity": self.velocity.clone(),
            "specific_force": self.specific_force.clone(),
            "angular_rate": self.angular_rate.clone(),
            "attitude_quaternion": self.attitude_quaternion.clone(),
        }

    def load_state_dict(self, state: Mapping[str, Tensor]) -> None:
        for name in self.state_dict():
            getattr(self, name).copy_(
                torch.as_tensor(state[name], device=self.device, dtype=self.dtype)
            )


class TensorAltitudeHandoff:
    """Latch controller activation from estimated GNSS altitude only."""

    def __init__(
        self,
        num_envs: int,
        *,
        ground_elevation: float,
        altitude_agl: float,
        device: str | torch.device,
    ) -> None:
        self.ground_elevation = float(ground_elevation)
        self.altitude_agl = float(altitude_agl)
        self.active = torch.zeros(num_envs, device=device, dtype=torch.bool)

    def update(
        self, features: TensorEstimatedRocketFeatures, sensors_finite: Tensor
    ) -> Tensor:
        reached = features.position[:, 2] - self.ground_elevation >= self.altitude_agl
        self.active |= sensors_finite & reached
        return self.active

    def reset(self, mask: Tensor) -> None:
        self.active[mask] = False


class TensorBootstrapController:
    """Historical angular-rate PID used before the sensor-derived handoff."""

    def __init__(
        self,
        num_envs: int,
        *,
        sampling_rate: float,
        device: str | torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.sampling_rate = float(sampling_rate)
        self.integral = torch.zeros((num_envs, 3), device=device, dtype=dtype)
        self.previous_error = torch.zeros_like(self.integral)
        self.has_previous = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.kp = torch.tensor((100.0, 100.0, 100.0), device=device, dtype=dtype)
        self.ki = torch.tensor((0.0, 0.0, 5.0), device=device, dtype=dtype)

    def reset(self, mask: Tensor) -> None:
        self.integral[mask] = 0
        self.previous_error[mask] = 0
        self.has_previous[mask] = False

    def compute(
        self,
        features: TensorEstimatedRocketFeatures,
        sensors_finite: Tensor,
        controller_active: Tensor,
    ) -> Tensor:
        update = sensors_finite & ~controller_active
        error = -features.angular_rate
        candidate_integral = self.integral + error / self.sampling_rate
        self.integral.copy_(
            torch.where(update.unsqueeze(-1), candidate_integral, self.integral)
        )
        command = self.kp * error + self.ki * self.integral
        self.previous_error.copy_(
            torch.where(update.unsqueeze(-1), error, self.previous_error)
        )
        self.has_previous |= update
        result = torch.zeros(
            (features.position.shape[0], ACTION_SIZE),
            device=features.position.device,
            dtype=features.position.dtype,
        )
        result[:, 0] = command[:, 2]
        result[:, 1:3] = command[:, 0:2]
        result[:, 3] = 1.0
        return result


def select_nearest_target(
    observation: TensorAgentObservation,
    features: TensorEstimatedRocketFeatures,
) -> tuple[Tensor, Tensor, Tensor]:
    """Select from released finite balloons using estimated position."""
    finite = torch.isfinite(observation.balloon_states).all(-1)
    candidates = (observation.balloon_status == 1) & finite
    distance_squared = (
        (observation.balloon_states[..., :3] - features.position[:, None, :])
        .square()
        .sum(-1)
    )
    distance_squared = torch.where(
        candidates, distance_squared, torch.full_like(distance_squared, torch.inf)
    )
    target_index = distance_squared.argmin(-1)
    available = candidates.any(-1)
    gather_index = target_index[:, None, None].expand(-1, 1, 6)
    target = observation.balloon_states.gather(1, gather_index).squeeze(1)
    fallback = torch.cat((features.position, features.velocity), dim=-1)
    target = torch.where(available.unsqueeze(-1), target, fallback)
    target_index = torch.where(
        available, target_index, torch.full_like(target_index, -1)
    )
    return target_index, available, target


def build_historical_observation(
    features: TensorEstimatedRocketFeatures,
    target_state: Tensor,
    previous_action: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Build the historical 29-vector entirely with tensor operations."""
    relative_position = target_state[:, :3] - features.position
    relative_velocity = target_state[:, 3:6] - features.velocity
    distance = torch.linalg.vector_norm(relative_position, dim=-1)
    rotation = quaternion_to_matrix(features.attitude_quaternion)
    body_z = rotation[..., :, 2]
    line_of_sight = torch.where(
        (distance > 1e-6).unsqueeze(-1),
        relative_position / distance.clamp_min(1e-6).unsqueeze(-1),
        body_z,
    )
    aim_angle = torch.acos((line_of_sight * body_z).sum(-1).clamp(-1, 1))
    rotation_t = rotation.transpose(-1, -2)
    relative_body_position = torch.matmul(
        rotation_t, relative_position.unsqueeze(-1)
    ).squeeze(-1)
    relative_body_velocity = torch.matmul(
        rotation_t, relative_velocity.unsqueeze(-1)
    ).squeeze(-1)
    rocket_body_velocity = torch.matmul(
        rotation_t, features.velocity.unsqueeze(-1)
    ).squeeze(-1)
    speed = torch.linalg.vector_norm(features.velocity, dim=-1)
    sin_alpha = torch.where(
        speed > 1.0, (rocket_body_velocity[:, 0] / speed).clamp(-1, 1), 0
    )
    sin_beta = torch.where(
        speed > 1.0, (rocket_body_velocity[:, 1] / speed).clamp(-1, 1), 0
    )
    result = torch.cat(
        (
            aim_angle.unsqueeze(-1),
            distance.unsqueeze(-1),
            relative_body_position,
            relative_body_velocity,
            features.position[:, 2:3],
            rocket_body_velocity,
            features.velocity[:, 2:3],
            features.specific_force,
            features.attitude_quaternion,
            features.angular_rate,
            sin_alpha.unsqueeze(-1),
            sin_beta.unsqueeze(-1),
            previous_action[:, 1:3],
            previous_action[:, 0:1],
            previous_action[:, 3:4],
        ),
        dim=-1,
    )
    if result.shape[-1] != OBSERVATION_SIZE:
        raise RuntimeError("historical observation shape changed")
    return result, sin_alpha, sin_beta


class TensorRunningNormalizer:
    """Masked parallel running statistics that stay on the training device."""

    def __init__(
        self,
        size: int,
        *,
        device: str | torch.device,
        epsilon: float = 1e-8,
        clip: float = 10.0,
    ) -> None:
        self.size = size
        self.device = torch.device(device)
        self.epsilon = float(epsilon)
        self.clip = float(clip)
        self.count = torch.zeros((), device=self.device, dtype=torch.float64)
        self.mean = torch.zeros(size, device=self.device, dtype=torch.float64)
        self.m2 = torch.zeros_like(self.mean)

    @property
    def variance(self) -> Tensor:
        return torch.where(self.count >= 2, self.m2 / self.count.clamp_min(1), 1.0)

    def update(self, values: Tensor, accepted: Tensor) -> None:
        values64 = torch.nan_to_num(values).to(torch.float64)
        accepted = accepted & torch.isfinite(values).all(-1)
        weight = accepted.to(torch.float64).unsqueeze(-1)
        batch_count = weight.sum()
        batch_mean = (values64 * weight).sum(0) / batch_count.clamp_min(1)
        batch_m2 = ((values64 - batch_mean).square() * weight).sum(0)
        total = self.count + batch_count
        delta = batch_mean - self.mean
        updated_mean = self.mean + delta * batch_count / total.clamp_min(1)
        updated_m2 = (
            self.m2
            + batch_m2
            + delta.square() * self.count * batch_count / total.clamp_min(1)
        )
        has_batch = batch_count > 0
        self.mean.copy_(torch.where(has_batch, updated_mean, self.mean))
        self.m2.copy_(torch.where(has_batch, updated_m2, self.m2))
        self.count.copy_(torch.where(has_batch, total, self.count))

    def normalize(self, values: Tensor) -> Tensor:
        safe = torch.nan_to_num(values).to(torch.float64)
        result = (safe - self.mean) / torch.sqrt(self.variance + self.epsilon)
        return result.clamp(-self.clip, self.clip).to(values.dtype)

    def state_dict(self) -> dict[str, Tensor | float]:
        return {
            "count": self.count.clone(),
            "mean": self.mean.clone(),
            "m2": self.m2.clone(),
            "epsilon": self.epsilon,
            "clip": self.clip,
        }

    def load_state_dict(self, state: Mapping[str, Tensor | float]) -> None:
        self.count.copy_(torch.as_tensor(state["count"], device=self.device))
        self.mean.copy_(torch.as_tensor(state["mean"], device=self.device))
        self.m2.copy_(torch.as_tensor(state["m2"], device=self.device))


class TensorPotentialReward:
    """Sensor-derived distance/ZEM shaping with canonical pop score separate."""

    def __init__(
        self, num_envs: int, *, device: str | torch.device, dtype: torch.dtype
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        self.reference_distance = torch.full(
            (num_envs,), 100.0, device=self.device, dtype=dtype
        )
        self.reference_zem = torch.full(
            (num_envs,), 30.0, device=self.device, dtype=dtype
        )
        self.previous_phi_distance = torch.full(
            (num_envs,), -1.0, device=self.device, dtype=dtype
        )
        self.previous_phi_zem = torch.full_like(self.previous_phi_distance, -1.0)
        self.previous_target = torch.full(
            (num_envs,), -1, device=self.device, dtype=torch.int64
        )
        self.popped_count = torch.zeros(num_envs, device=self.device, dtype=torch.int64)

    def reset(self, mask: Tensor) -> None:
        self.reference_distance[mask] = 100
        self.reference_zem[mask] = 30
        self.previous_phi_distance[mask] = -1
        self.previous_phi_zem[mask] = -1
        self.previous_target[mask] = -1
        self.popped_count[mask] = 0

    def compute(
        self,
        canonical_pop_reward: Tensor,
        features: TensorEstimatedRocketFeatures,
        target_index: Tensor,
        target_available: Tensor,
        target_state: Tensor,
        terminated: Tensor,
        normalized_action: Tensor,
        sin_alpha: Tensor,
        sin_beta: Tensor,
    ) -> Tensor:
        relative_position = target_state[:, :3] - features.position
        relative_velocity = features.velocity - target_state[:, 3:6]
        distance = torch.linalg.vector_norm(relative_position, dim=-1)
        line_of_sight = relative_position / distance.clamp_min(1e-9).unsqueeze(-1)
        speed_squared = relative_velocity.square().sum(-1)
        closing = (relative_velocity * line_of_sight).sum(-1)
        time_to_go = torch.clamp(distance * closing, min=0) / speed_squared.clamp_min(
            1e-6
        )
        zem = torch.linalg.vector_norm(
            relative_position - relative_velocity * time_to_go.unsqueeze(-1), dim=-1
        )
        zem = torch.where(speed_squared > 1e-6, zem, distance)
        switched = target_index != self.previous_target
        acquired = switched & target_available
        self.reference_distance.copy_(
            torch.where(acquired, distance.clamp_min(5.0), self.reference_distance)
        )
        self.reference_zem.copy_(
            torch.where(acquired, zem.clamp_min(2.0), self.reference_zem)
        )
        phi_distance = -torch.minimum(
            distance / self.reference_distance, torch.full_like(distance, 2.0)
        )
        phi_zem = -torch.minimum(zem / self.reference_zem, torch.full_like(zem, 2.0))
        phi_distance = torch.where(target_available, phi_distance, -1.0)
        phi_zem = torch.where(target_available, phi_zem, -1.0)
        suppress_jump = switched & (canonical_pop_reward > 0)
        approach = torch.where(
            suppress_jump,
            0.0,
            80.0 * (phi_distance - self.previous_phi_distance),
        )
        zem_reward = torch.where(
            suppress_jump, 0.0, 20.0 * (phi_zem - self.previous_phi_zem)
        )
        self.popped_count += canonical_pop_reward.to(torch.int64)
        progress = 1.0 - torch.exp(-0.25 * self.popped_count.to(self.dtype))
        termination_penalty = torch.where(terminated, -200.0 + 100.0 * progress, 0.0)
        pop_reward = canonical_pop_reward * (
            500.0
            + 300.0 * (1.0 - torch.exp(-0.2 * (self.popped_count - 1).clamp_min(0)))
        )
        stability = -0.5 * (sin_alpha.square() + sin_beta.square())
        tvc_cost = -0.1 * normalized_action[:, 1:3].square().sum(-1)
        self.previous_phi_distance.copy_(phi_distance)
        self.previous_phi_zem.copy_(phi_zem)
        self.previous_target.copy_(torch.where(target_available, target_index, -1))
        return (
            termination_penalty
            + pop_reward
            + approach
            + zem_reward
            + stability
            + tvc_cost
        )


class TensorAgentAdapter:
    """Own all agent-side recurrent state without access to oracle state."""

    def __init__(
        self,
        num_envs: int,
        *,
        sampling_rate: float,
        ground_elevation: float,
        handoff_altitude_agl: float,
        device: str | torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.dtype = dtype
        self.estimator = TensorSensorEstimator(
            num_envs,
            sampling_rate=sampling_rate,
            ground_elevation=ground_elevation,
            device=device,
            dtype=dtype,
        )
        self.handoff = TensorAltitudeHandoff(
            num_envs,
            ground_elevation=ground_elevation,
            altitude_agl=handoff_altitude_agl,
            device=device,
        )
        self.bootstrap = TensorBootstrapController(
            num_envs,
            sampling_rate=sampling_rate,
            device=device,
            dtype=dtype,
        )
        self.normalizer = TensorRunningNormalizer(OBSERVATION_SIZE, device=device)
        self.reward_model = TensorPotentialReward(num_envs, device=device, dtype=dtype)
        self.previous_action = torch.zeros(
            (num_envs, ACTION_SIZE), device=device, dtype=dtype
        )
        self.previous_action[:, 3] = 1.0
        self.current: TensorPreparedObservation | None = None

    def prepare(self, observation: TensorAgentObservation) -> TensorPreparedObservation:
        sensors_finite = self.estimator.update(observation)
        features = self.estimator.features
        controller_active = self.handoff.update(features, sensors_finite)
        target_index, target_available, target = select_nearest_target(
            observation, features
        )
        raw, sin_alpha, sin_beta = build_historical_observation(
            features, target, self.previous_action
        )
        self.normalizer.update(raw, controller_active & sensors_finite)
        self.current = TensorPreparedObservation(
            raw=raw,
            normalized=self.normalizer.normalize(raw),
            controller_active=controller_active.clone(),
            sensors_finite=sensors_finite,
            target_index=target_index,
            target_available=target_available,
            target_state=target,
            sin_alpha=sin_alpha,
            sin_beta=sin_beta,
        )
        return self.current

    def physical_action(
        self,
        normalized_action: Tensor,
        *,
        max_roll_torque: float,
        max_gimbal_angle: float,
        throttle_low: float,
        throttle_high: float,
    ) -> Tensor:
        result = torch.empty_like(normalized_action)
        result[:, 0] = normalized_action[:, 0] * max_roll_torque
        result[:, 1:3] = normalized_action[:, 1:3] * max_gimbal_angle
        result[:, 3] = throttle_low + (normalized_action[:, 3] + 1) * 0.5 * (
            throttle_high - throttle_low
        )
        return result

    def select_control(self, policy_action: Tensor) -> Tensor:
        if self.current is None:
            raise RuntimeError("prepare must be called before select_control")
        bootstrap = self.bootstrap.compute(
            self.estimator.features,
            self.current.sensors_finite,
            self.current.controller_active,
        )
        return torch.where(
            self.current.controller_active.unsqueeze(-1), policy_action, bootstrap
        )

    def reset(self, mask: Tensor) -> None:
        mask = torch.as_tensor(mask, device=self.device, dtype=torch.bool).clone()
        self.estimator.reset(mask)
        self.handoff.reset(mask)
        self.bootstrap.reset(mask)
        self.reward_model.reset(mask)
        self.previous_action[mask] = 0
        self.previous_action[mask, 3] = 1

    def state_dict(self) -> dict[str, object]:
        current = None
        if self.current is not None:
            current = {
                name: getattr(self.current, name).clone()
                for name in self.current.__dataclass_fields__
            }
        return {
            "estimator": self.estimator.state_dict(),
            "handoff_active": self.handoff.active.clone(),
            "bootstrap_integral": self.bootstrap.integral.clone(),
            "bootstrap_previous_error": self.bootstrap.previous_error.clone(),
            "bootstrap_has_previous": self.bootstrap.has_previous.clone(),
            "normalizer": self.normalizer.state_dict(),
            "reward": {
                "reference_distance": self.reward_model.reference_distance.clone(),
                "reference_zem": self.reward_model.reference_zem.clone(),
                "previous_phi_distance": self.reward_model.previous_phi_distance.clone(),
                "previous_phi_zem": self.reward_model.previous_phi_zem.clone(),
                "previous_target": self.reward_model.previous_target.clone(),
                "popped_count": self.reward_model.popped_count.clone(),
            },
            "previous_action": self.previous_action.clone(),
            "current": current,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        self.estimator.load_state_dict(state["estimator"])  # type: ignore[arg-type]
        self.handoff.active.copy_(
            torch.as_tensor(state["handoff_active"], device=self.device)
        )
        for target, name in (
            (self.bootstrap.integral, "bootstrap_integral"),
            (self.bootstrap.previous_error, "bootstrap_previous_error"),
            (self.bootstrap.has_previous, "bootstrap_has_previous"),
        ):
            target.copy_(torch.as_tensor(state[name], device=self.device))
        self.normalizer.load_state_dict(state["normalizer"])  # type: ignore[arg-type]
        reward = state["reward"]
        if not isinstance(reward, Mapping):
            raise TypeError("reward checkpoint must be a mapping")
        for name in reward:
            getattr(self.reward_model, name).copy_(
                torch.as_tensor(reward[name], device=self.device)
            )
        self.previous_action.copy_(
            torch.as_tensor(state["previous_action"], device=self.device)
        )
        current = state["current"]
        if current is None:
            self.current = None
        elif isinstance(current, Mapping):
            self.current = TensorPreparedObservation(
                **{
                    name: torch.as_tensor(current[name], device=self.device)
                    for name in TensorPreparedObservation.__dataclass_fields__
                }
            )
        else:
            raise TypeError("current observation checkpoint must be a mapping")
