"""Batched PyTorch 6-DoF flight *surrogate* for CUDA throughput experiments.

This module is deliberately isolated from ``BalloonPoppingGymEnv``.  It is a
non-canonical prototype and has **not** been validated against RocketPy.  In
particular, the aerodynamic, mass, atmosphere, wind, actuator, and contact
models below are intentionally simple.  Results from this surrogate must not
be presented as RocketPy or competition-environment results.

The experiment answers a narrower engineering question: if flight dynamics,
target detection, reward, and episode state are represented as fixed-shape
PyTorch tensors, how much throughput can batched CPU or CUDA execution provide?

State layout (one row per independent environment) is ``[B, 13]``::

    position_xyz_world (3), velocity_xyz_world (3),
    quaternion_wxyz_body_to_world (4), angular_velocity_xyz_body (3)

Actions are ``[B, 4]`` and normalized to ``[-1, 1]``.  Their order mirrors the
current E2E adapter rather than the internal actuator tensor::

    roll, tvc_x, tvc_y, throttle

Throttle maps from ``[-1, 1]`` to ``[0, 1]``.  Gimbal commands map linearly to
``+/- max_gimbal_angle`` and roll maps to a deliberately simplified body-axis
torque.  A separate ``[B, 4]`` actuator tensor is integrated with first-order
lags.  The coupled rigid-body and actuator equations use a
fixed-step RK4 update so they have static control flow suitable for batching.
All step outputs remain on the configured device.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import Tensor


NON_CANONICAL_PROTOTYPE = True


@dataclass(frozen=True)
class TensorFlightConfig:
    """Constants for the intentionally simplified tensor flight model."""

    dt: float = 0.01
    gravity: float = 9.80665
    mass: float = 12.0
    max_thrust: float = 800.0
    drag_force_coefficient: float = 0.008
    angular_damping: float = 0.08
    inertia_x: float = 2.0
    inertia_y: float = 2.0
    inertia_z: float = 0.15
    thrust_lever_arm: float = 0.55
    max_gimbal_angle: float = 0.17453292519943295  # 10 degrees
    max_roll_torque: float = 8.0
    roll_time_constant: float = 0.05
    throttle_time_constant: float = 0.08
    gimbal_time_constant: float = 0.05
    initial_altitude: float = 2.0
    initial_vertical_speed: float = 55.0
    target_min_altitude: float = 180.0
    target_max_altitude: float = 320.0
    target_lateral_span: float = 80.0
    target_max_horizontal_speed: float = 1.0
    target_radius: float = 1.5
    ground_altitude: float = 0.0
    max_steps: int = 2_000
    progress_reward_scale: float = 0.01
    step_penalty: float = 0.001
    hit_reward: float = 10.0
    crash_penalty: float = 2.0

    def __post_init__(self) -> None:
        positive = {
            "dt": self.dt,
            "mass": self.mass,
            "max_thrust": self.max_thrust,
            "inertia_x": self.inertia_x,
            "inertia_y": self.inertia_y,
            "inertia_z": self.inertia_z,
            "roll_time_constant": self.roll_time_constant,
            "throttle_time_constant": self.throttle_time_constant,
            "gimbal_time_constant": self.gimbal_time_constant,
            "target_radius": self.target_radius,
            "max_steps": self.max_steps,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"configuration values must be positive: {invalid}")
        if self.target_max_altitude < self.target_min_altitude:
            raise ValueError("target_max_altitude must be >= target_min_altitude")


class TensorStep(NamedTuple):
    """Device-resident result of one batched transition."""

    state: Tensor
    reward: Tensor
    terminated: Tensor
    truncated: Tensor
    hit: Tensor
    crashed: Tensor
    distance_to_target: Tensor


def _quat_multiply(lhs: Tensor, rhs: Tensor) -> Tensor:
    """Hamilton product for scalar-first quaternions."""

    lw, lx, ly, lz = lhs.unbind(dim=-1)
    rw, rx, ry, rz = rhs.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def _quat_rotate(quaternion: Tensor, vector: Tensor) -> Tensor:
    """Rotate body-frame vectors into the world frame."""

    q_vector = quaternion[..., 1:]
    twice_cross = 2.0 * torch.linalg.cross(q_vector, vector, dim=-1)
    return (
        vector
        + quaternion[..., :1] * twice_cross
        + torch.linalg.cross(q_vector, twice_cross, dim=-1)
    )


def _normalize_quaternion(state: Tensor) -> Tensor:
    quaternion = torch.nn.functional.normalize(state[..., 6:10], dim=-1)
    return torch.cat((state[..., :6], quaternion, state[..., 10:]), dim=-1)


def _segment_distance(
    rocket_start: Tensor,
    rocket_end: Tensor,
    target_start: Tensor,
    target_end: Tensor,
) -> Tensor:
    """Minimum geometric distance between paired swept path segments.

    The two closest-point parameters are independent.  This deliberately
    mirrors the official v0.1.1 pop geometry rather than assuming that the
    rocket and target must reach their closest points at the same fraction of
    the control interval.
    """

    direction_rocket = rocket_end - rocket_start
    direction_target = target_end - target_start
    offset = rocket_start - target_start

    a_coeff = (direction_rocket * direction_rocket).sum(dim=-1)
    b_coeff = (direction_rocket * direction_target).sum(dim=-1)
    c_coeff = (direction_rocket * offset).sum(dim=-1)
    e_coeff = (direction_target * direction_target).sum(dim=-1)
    f_coeff = (direction_target * offset).sum(dim=-1)

    # The official implementation treats segments shorter than one micron as
    # points.  Safe denominators only protect the unused degenerate candidates;
    # endpoint candidates still provide the exact point/segment answer.
    epsilon = 1e-12
    safe_a = a_coeff.clamp_min(epsilon)
    safe_e = e_coeff.clamp_min(epsilon)

    zero = torch.zeros_like(a_coeff)
    one = torch.ones_like(a_coeff)
    s_at_target_start = (-c_coeff / safe_a).clamp(0.0, 1.0)
    s_at_target_end = ((b_coeff - c_coeff) / safe_a).clamp(0.0, 1.0)
    t_at_rocket_start = (f_coeff / safe_e).clamp(0.0, 1.0)
    t_at_rocket_end = ((b_coeff + f_coeff) / safe_e).clamp(0.0, 1.0)

    normal = torch.linalg.cross(direction_rocket, direction_target, dim=-1)
    denominator = (normal * normal).sum(dim=-1)
    denominator_safe = denominator.clamp_min(torch.finfo(offset.dtype).tiny)
    s_interior = (torch.linalg.cross(direction_target, offset, dim=-1) * normal).sum(
        dim=-1
    ) / denominator_safe
    t_interior = (torch.linalg.cross(direction_rocket, offset, dim=-1) * normal).sum(
        dim=-1
    ) / denominator_safe
    solvable = denominator > 0.0
    s_interior = torch.where(solvable, s_interior.clamp(0.0, 1.0), zero)
    t_interior = torch.where(solvable, t_interior.clamp(0.0, 1.0), zero)

    s_candidates = torch.stack(
        (zero, one, s_at_target_start, s_at_target_end, s_interior), dim=-1
    )
    t_candidates = torch.stack(
        (t_at_rocket_start, t_at_rocket_end, zero, one, t_interior), dim=-1
    )
    separation = (
        offset[:, None, :]
        + s_candidates[:, :, None] * direction_rocket[:, None, :]
        - t_candidates[:, :, None] * direction_target[:, None, :]
    )
    distance_squared = (separation * separation).sum(dim=-1)
    return distance_squared.amin(dim=-1).clamp_min(0.0).sqrt()


class TensorFlightBatch:
    """Independent single-balloon episodes advanced as one tensor batch.

    This class intentionally does not implement the Gymnasium API.  Avoiding
    Python dictionaries, NumPy conversion, and per-environment callbacks in the
    hot loop is part of the experiment.  ``reset`` and ``step`` return tensors
    on ``device``; callers should only copy aggregate diagnostics to the host.
    """

    state_size = 13
    action_size = 4
    actuator_size = 4

    def __init__(
        self,
        batch_size: int,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        config: TensorFlightConfig | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not dtype.is_floating_point:
            raise TypeError("dtype must be floating point")

        self.batch_size = batch_size
        requested_device = torch.device(device)
        if requested_device.type == "cuda" and requested_device.index is None:
            requested_device = torch.device("cuda", torch.cuda.current_device())
        self.device = requested_device
        self.dtype = dtype
        self.config = config or TensorFlightConfig()
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but torch.cuda.is_available() is false"
            )

        self._inertia = torch.tensor(
            [
                self.config.inertia_x,
                self.config.inertia_y,
                self.config.inertia_z,
            ],
            device=self.device,
            dtype=self.dtype,
        )
        self._actuator_time_constants = torch.tensor(
            [
                self.config.roll_time_constant,
                self.config.gimbal_time_constant,
                self.config.gimbal_time_constant,
                self.config.throttle_time_constant,
            ],
            device=self.device,
            dtype=self.dtype,
        )
        self._gravity = torch.tensor(
            [0.0, 0.0, -self.config.gravity],
            device=self.device,
            dtype=self.dtype,
        )
        self._integrate_function = self._rk4_integrate
        self._compiled = False

        self.state = torch.empty(
            (batch_size, self.state_size), device=self.device, dtype=self.dtype
        )
        self.actuator_state = torch.empty(
            (batch_size, self.actuator_size), device=self.device, dtype=self.dtype
        )
        self.target_position = torch.empty(
            (batch_size, 3), device=self.device, dtype=self.dtype
        )
        self.target_velocity = torch.empty_like(self.target_position)
        self.step_count = torch.empty(batch_size, device=self.device, dtype=torch.int64)
        self.terminated = torch.empty(batch_size, device=self.device, dtype=torch.bool)
        self.truncated = torch.empty_like(self.terminated)
        self.hit = torch.empty_like(self.terminated)
        self.reset()

    @property
    def compiled(self) -> bool:
        return self._compiled

    def enable_compile(self, *, mode: str = "reduce-overhead") -> None:
        """Compile the RK4 tensor kernel; compilation failures are caller-visible.

        PyTorch compilation is optional and version/platform dependent.  The
        benchmark catches both immediate and first-call failures and restores
        eager mode, so inability to compile never invalidates the experiment.
        """

        compile_function = getattr(torch, "compile", None)
        if compile_function is None:
            raise RuntimeError("this PyTorch version does not provide torch.compile")
        self._integrate_function = compile_function(
            self._rk4_integrate, dynamic=False, mode=mode
        )
        self._compiled = True

    def disable_compile(self) -> None:
        """Restore eager RK4 execution."""

        self._integrate_function = self._rk4_integrate
        self._compiled = False

    def reset(self, *, seed: int | None = None, mask: Tensor | None = None) -> Tensor:
        """Reset all or selected rows and return the device-resident state.

        ``mask`` enables asynchronous vector rollouts without copying done
        indices to the host.  It must be a boolean ``[B]`` tensor on the same
        device.  Omitting it resets the complete batch.
        """

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)

        if mask is None:
            reset_mask = torch.ones(
                self.batch_size, device=self.device, dtype=torch.bool
            )
        else:
            if mask.shape != (self.batch_size,):
                raise ValueError(
                    f"mask shape must be {(self.batch_size,)}, got {tuple(mask.shape)}"
                )
            if mask.device != self.device or mask.dtype != torch.bool:
                raise ValueError("mask must be boolean and on the environment device")
            # Callers may pass one of this environment's own done tensors.
            # Clone before clearing those tensors so the remaining masked
            # assignments cannot lose their selection through aliasing.
            reset_mask = mask.clone()

        initial_state = torch.zeros_like(self.state)
        initial_state[:, 2] = self.config.initial_altitude
        initial_state[:, 5] = self.config.initial_vertical_speed
        initial_state[:, 6] = 1.0
        self.state = torch.where(reset_mask[:, None], initial_state, self.state)
        self.actuator_state = torch.where(
            reset_mask[:, None],
            torch.zeros_like(self.actuator_state),
            self.actuator_state,
        )
        self.step_count.masked_fill_(reset_mask, 0)
        self.terminated.masked_fill_(reset_mask, False)
        self.truncated.masked_fill_(reset_mask, False)
        self.hit.masked_fill_(reset_mask, False)

        uniform = torch.rand(
            (self.batch_size, 4),
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )
        span = self.config.target_lateral_span
        sampled_target_position = torch.empty_like(self.target_position)
        sampled_target_position[:, 0:2] = (uniform[:, 0:2] * 2.0 - 1.0) * span
        altitude_span = (
            self.config.target_max_altitude - self.config.target_min_altitude
        )
        sampled_target_position[:, 2] = (
            self.config.target_min_altitude + uniform[:, 2] * altitude_span
        )
        horizontal_speed = self.config.target_max_horizontal_speed
        angle = uniform[:, 3] * (2.0 * torch.pi)
        sampled_target_velocity = torch.zeros_like(self.target_velocity)
        sampled_target_velocity[:, 0] = torch.cos(angle) * horizontal_speed
        sampled_target_velocity[:, 1] = torch.sin(angle) * horizontal_speed
        self.target_position = torch.where(
            reset_mask[:, None], sampled_target_position, self.target_position
        )
        self.target_velocity = torch.where(
            reset_mask[:, None], sampled_target_velocity, self.target_velocity
        )
        return self.state

    def reset_done(self, *, seed: int | None = None) -> Tensor:
        """Reset only terminated or truncated rows without a host round-trip."""

        return self.reset(seed=seed, mask=self.terminated | self.truncated)

    def set_target(
        self,
        position: Tensor,
        velocity: Tensor | None = None,
    ) -> None:
        """Set one target per environment without moving data off the device."""

        expected_shape = (self.batch_size, 3)
        if position.shape != expected_shape:
            raise ValueError(
                f"position shape must be {expected_shape}, got {tuple(position.shape)}"
            )
        if position.device != self.device or position.dtype != self.dtype:
            raise ValueError("position must already have the environment device/dtype")
        self.target_position.copy_(position)
        if velocity is None:
            self.target_velocity.zero_()
        else:
            if velocity.shape != expected_shape:
                raise ValueError(
                    f"velocity shape must be {expected_shape}, got {tuple(velocity.shape)}"
                )
            if velocity.device != self.device or velocity.dtype != self.dtype:
                raise ValueError(
                    "velocity must already have the environment device/dtype"
                )
            self.target_velocity.copy_(velocity)

    def _normalized_action_to_command(self, action: Tensor) -> Tensor:
        action = action.clamp(-1.0, 1.0)
        return torch.stack(
            (
                action[:, 0] * self.config.max_roll_torque,
                action[:, 1] * self.config.max_gimbal_angle,
                action[:, 2] * self.config.max_gimbal_angle,
                0.5 * (action[:, 3] + 1.0),
            ),
            dim=-1,
        )

    def _derivatives(
        self, state: Tensor, actuator_state: Tensor, command: Tensor
    ) -> tuple[Tensor, Tensor]:
        velocity = state[:, 3:6]
        quaternion = state[:, 6:10]
        angular_velocity = state[:, 10:13]

        roll_torque = actuator_state[:, 0]
        gimbal_x = actuator_state[:, 1]
        gimbal_y = actuator_state[:, 2]
        throttle = actuator_state[:, 3].clamp(0.0, 1.0)
        sin_x, cos_x = torch.sin(gimbal_x), torch.cos(gimbal_x)
        sin_y, cos_y = torch.sin(gimbal_y), torch.cos(gimbal_y)
        body_thrust_direction = torch.stack(
            (sin_y, -sin_x * cos_y, cos_x * cos_y), dim=-1
        )
        body_thrust = (
            throttle.unsqueeze(-1) * self.config.max_thrust * body_thrust_direction
        )
        world_thrust = _quat_rotate(quaternion, body_thrust)
        speed = torch.linalg.vector_norm(velocity, dim=-1, keepdim=True)
        drag = -self.config.drag_force_coefficient * speed * velocity
        acceleration = (world_thrust + drag) / self.config.mass + self._gravity

        lever_arm = self.config.thrust_lever_arm
        torque = torch.stack(
            (
                lever_arm * body_thrust[:, 1],
                -lever_arm * body_thrust[:, 0],
                roll_torque,
            ),
            dim=-1,
        )
        angular_momentum = angular_velocity * self._inertia
        gyroscopic = torch.linalg.cross(angular_velocity, angular_momentum, dim=-1)
        angular_acceleration = (
            torque - gyroscopic - self.config.angular_damping * angular_velocity
        ) / self._inertia

        pure_angular_velocity = torch.cat(
            (torch.zeros_like(angular_velocity[:, :1]), angular_velocity), dim=-1
        )
        quaternion_rate = 0.5 * _quat_multiply(quaternion, pure_angular_velocity)

        state_rate = torch.cat(
            (
                velocity,
                acceleration,
                quaternion_rate,
                angular_acceleration,
            ),
            dim=-1,
        )
        actuator_rate = (command - actuator_state) / self._actuator_time_constants
        return state_rate, actuator_rate

    def _rk4_integrate(
        self, state: Tensor, actuator_state: Tensor, command: Tensor
    ) -> tuple[Tensor, Tensor]:
        dt = self.config.dt
        k1_state, k1_actuator = self._derivatives(state, actuator_state, command)
        k2_state, k2_actuator = self._derivatives(
            state + 0.5 * dt * k1_state,
            actuator_state + 0.5 * dt * k1_actuator,
            command,
        )
        k3_state, k3_actuator = self._derivatives(
            state + 0.5 * dt * k2_state,
            actuator_state + 0.5 * dt * k2_actuator,
            command,
        )
        k4_state, k4_actuator = self._derivatives(
            state + dt * k3_state,
            actuator_state + dt * k3_actuator,
            command,
        )
        next_state = state + (dt / 6.0) * (
            k1_state + 2.0 * k2_state + 2.0 * k3_state + k4_state
        )
        next_actuator = actuator_state + (dt / 6.0) * (
            k1_actuator + 2.0 * k2_actuator + 2.0 * k3_actuator + k4_actuator
        )
        next_state = _normalize_quaternion(next_state)
        next_actuator = torch.stack(
            (
                next_actuator[:, 0].clamp(
                    -self.config.max_roll_torque, self.config.max_roll_torque
                ),
                next_actuator[:, 1].clamp(
                    -self.config.max_gimbal_angle,
                    self.config.max_gimbal_angle,
                ),
                next_actuator[:, 2].clamp(
                    -self.config.max_gimbal_angle,
                    self.config.max_gimbal_angle,
                ),
                next_actuator[:, 3].clamp(0.0, 1.0),
            ),
            dim=-1,
        )
        return next_state, next_actuator

    def step(self, action: Tensor) -> TensorStep:
        """Advance all episodes once while retaining every output on-device."""

        expected_shape = (self.batch_size, self.action_size)
        if action.shape != expected_shape:
            raise ValueError(
                f"action shape must be {expected_shape}, got {tuple(action.shape)}"
            )
        if action.device != self.device or action.dtype != self.dtype:
            raise ValueError("action must already have the environment device/dtype")

        finite_action = torch.isfinite(action).all(dim=-1)
        safe_action = torch.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
        previously_done = self.terminated | self.truncated
        active = ~previously_done
        command = self._normalized_action_to_command(safe_action)
        integrated_state, integrated_actuator = self._integrate_function(
            self.state, self.actuator_state, command
        )
        finite_integration = torch.isfinite(integrated_state).all(
            dim=-1
        ) & torch.isfinite(integrated_actuator).all(dim=-1)
        valid_transition = finite_action & finite_integration
        integrate_mask = active & valid_transition
        next_state = torch.where(integrate_mask[:, None], integrated_state, self.state)
        next_actuator = torch.where(
            integrate_mask[:, None], integrated_actuator, self.actuator_state
        )

        target_start = self.target_position
        integrated_target = target_start + self.config.dt * self.target_velocity
        target_end = torch.where(
            integrate_mask[:, None], integrated_target, target_start
        )
        swept_distance = _segment_distance(
            self.state[:, :3], next_state[:, :3], target_start, target_end
        )
        new_hit = integrate_mask & (swept_distance <= self.config.target_radius)
        new_crash = (
            active
            & ((next_state[:, 2] <= self.config.ground_altitude) | ~valid_transition)
            & ~new_hit
        )

        next_step_count = self.step_count + active.to(torch.int64)
        new_terminated = self.terminated | new_hit | new_crash
        new_truncated = self.truncated | (
            active & ~new_terminated & (next_step_count >= self.config.max_steps)
        )

        old_distance = torch.linalg.vector_norm(
            self.state[:, :3] - target_start, dim=-1
        )
        new_distance = torch.linalg.vector_norm(next_state[:, :3] - target_end, dim=-1)
        progress = old_distance - new_distance
        reward = (
            self.config.progress_reward_scale * progress
            - self.config.step_penalty
            + self.config.hit_reward * new_hit.to(self.dtype)
            - self.config.crash_penalty * new_crash.to(self.dtype)
        )
        reward = torch.where(active, reward, torch.zeros_like(reward))

        self.state = next_state
        self.actuator_state = next_actuator
        self.target_position = target_end
        self.step_count = next_step_count
        self.terminated = new_terminated
        self.truncated = new_truncated
        self.hit = self.hit | new_hit
        return TensorStep(
            state=self.state,
            reward=reward,
            terminated=self.terminated,
            truncated=self.truncated,
            hit=new_hit,
            crashed=new_crash,
            distance_to_target=new_distance,
        )
