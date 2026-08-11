"""Pure-tensor Scenario 1 rocket dynamics used by the Phase 2 fidelity work.

The model in this module never calls ActiveRocketPy.  Canonical objects are
used once, by :mod:`experiments.cuda.phase2_oracle`, to extract immutable
tables and constants.  Every evaluation after construction is a PyTorch-only
operation and supports leading batch dimensions.

This is intentionally a competition-specific port of ``Flight.u_dot_generalized``.
It is not a general RocketPy replacement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import torch


Tensor = torch.Tensor
WindProvider = Callable[[Tensor], Tensor]

ACTUATOR_ORDER = ("roll", "tvc_x", "tvc_y", "throttle")
STATE_SIZE = 13


def _as_tensor(
    value: object,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    return torch.as_tensor(value, device=device, dtype=dtype)


class LinearTable:
    """Device-side one-dimensional linear interpolation with extrapolation."""

    def __init__(self, x: Tensor, y: Tensor) -> None:
        if x.ndim != 1 or x.numel() < 2:
            raise ValueError("table x must be one-dimensional with at least two rows")
        if y.shape[0] != x.shape[0]:
            raise ValueError("table x and y must have the same first dimension")
        if not bool(torch.all(x[1:] > x[:-1])):
            raise ValueError("table x must be strictly increasing")
        self.x = x
        self.y = y

    def __call__(self, query: Tensor) -> Tensor:
        query = torch.as_tensor(query, device=self.x.device, dtype=self.x.dtype)
        # ``broadcast_to`` commonly produces a non-contiguous view. Making the
        # copy explicit avoids an implicit copy (and warning) in searchsorted.
        query = query.contiguous()
        upper = torch.searchsorted(self.x, query, right=False)
        upper = upper.clamp(1, self.x.numel() - 1)
        lower = upper - 1
        x0 = self.x[lower]
        x1 = self.x[upper]
        fraction = (query - x0) / (x1 - x0)
        y0 = self.y[lower]
        y1 = self.y[upper]
        if self.y.ndim > 1:
            fraction = fraction.unsqueeze(-1)
        return y0 + fraction * (y1 - y0)


@dataclass(frozen=True)
class ActuatorSpec:
    lower: float
    upper: float
    rate_limit: float | None
    time_constant: float | None
    initial: float


class Scenario1ActuatorBank:
    """Batched clone of ActiveRocketPy's filter, rate-limit, clamp ordering."""

    def __init__(
        self,
        num_envs: int,
        specs: Sequence[ActuatorSpec],
        *,
        demand_rate: float,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        if len(specs) != len(ACTUATOR_ORDER):
            raise ValueError(f"expected {len(ACTUATOR_ORDER)} actuator specs")
        if demand_rate <= 0:
            raise ValueError("demand_rate must be positive")
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.dtype = dtype
        self.demand_rate = float(demand_rate)
        lower = [spec.lower for spec in specs]
        upper = [spec.upper for spec in specs]
        maximum_change = [
            torch.inf if spec.rate_limit is None else spec.rate_limit / demand_rate
            for spec in specs
        ]
        alpha = [
            1.0
            if spec.time_constant is None or spec.time_constant <= 0
            else 1.0 / (1.0 + spec.time_constant * demand_rate)
            for spec in specs
        ]
        initial = [spec.initial for spec in specs]
        self.lower = _as_tensor(lower, device=self.device, dtype=dtype)
        self.upper = _as_tensor(upper, device=self.device, dtype=dtype)
        self.maximum_change = _as_tensor(
            maximum_change,
            device=self.device,
            dtype=dtype,
        )
        self.alpha = _as_tensor(alpha, device=self.device, dtype=dtype)
        self.initial = _as_tensor(initial, device=self.device, dtype=dtype)
        self.output = self.initial.expand(num_envs, -1).clone()

    def reset(self, mask: Tensor | None = None) -> None:
        if mask is None:
            self.output.copy_(self.initial)
            return
        reset_mask = torch.as_tensor(mask, device=self.device, dtype=torch.bool).clone()
        if reset_mask.shape != (self.num_envs,):
            raise ValueError(f"reset mask must have shape ({self.num_envs},)")
        self.output[reset_mask] = self.initial

    def update(self, command: Tensor, *, validate_finite: bool = True) -> Tensor:
        command = torch.as_tensor(command, device=self.device, dtype=self.dtype)
        if command.shape != self.output.shape:
            raise ValueError(f"command must have shape {tuple(self.output.shape)}")
        if validate_finite and not bool(torch.isfinite(command).all()):
            raise ValueError("actuator command must be finite")
        filtered = self.alpha * command + (1.0 - self.alpha) * self.output
        change = torch.clamp(
            filtered - self.output,
            min=-self.maximum_change,
            max=self.maximum_change,
        )
        self.output.copy_(torch.clamp(self.output + change, self.lower, self.upper))
        return self.output


@dataclass(frozen=True)
class RocketRHSComponents:
    total_mass: Tensor
    total_mass_dot: Tensor
    total_mass_ddot: Tensor
    center_of_mass: Tensor
    center_of_mass_dot: Tensor
    center_of_mass_ddot: Tensor
    inertia: Tensor
    inertia_dot: Tensor
    atmosphere: Tensor
    wind: Tensor
    thrust: Tensor
    aerodynamic_force: Tensor
    aerodynamic_moment: Tensor
    control_moment: Tensor
    translation_acceleration: Tensor
    angular_acceleration: Tensor
    quaternion_derivative: Tensor
    rhs: Tensor


class Scenario1TensorRocket:
    """Float64-first batched port of Scenario 1's generalized 6-DoF RHS."""

    _TIME_COLUMNS = {
        "mass": 0,
        "mass_dot": 1,
        "mass_ddot": 2,
        "com": 3,
        "com_dot": 4,
        "com_ddot": 5,
        "inertia": slice(6, 15),
        "inertia_dot": slice(15, 24),
        "thrust": 24,
    }

    def __init__(
        self,
        data: Mapping[str, object],
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
        wind_provider: WindProvider | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        self.wind_provider = wind_provider
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("rocket model supports float32 or float64")

        atmosphere_data = data["atmosphere"]
        if not isinstance(atmosphere_data, Mapping):
            raise TypeError("atmosphere data must be a mapping")
        self.atmosphere_tables: dict[str, LinearTable] = {}
        for name in (
            "density",
            "pressure",
            "dynamic_viscosity",
            "speed_of_sound",
            "wind_velocity_x",
            "wind_velocity_y",
            "gravity",
        ):
            table = atmosphere_data[name]
            if not isinstance(table, Mapping):
                raise TypeError(f"atmosphere table {name} must be a mapping")
            self.atmosphere_tables[name] = LinearTable(
                _as_tensor(table["x"], device=self.device, dtype=dtype),
                _as_tensor(table["y"], device=self.device, dtype=dtype),
            )

        time_data = data["time"]
        if not isinstance(time_data, Mapping):
            raise TypeError("time data must be a mapping")
        self.time_table = LinearTable(
            _as_tensor(time_data["x"], device=self.device, dtype=dtype),
            _as_tensor(time_data["y"], device=self.device, dtype=dtype),
        )

        constants = data["constants"]
        if not isinstance(constants, Mapping):
            raise TypeError("constants must be a mapping")
        self.burn_duration = float(constants["burn_duration"])
        self.reference_area = float(constants["reference_area"])
        self.reference_length = float(constants["reference_length"])
        self.drag_power_on = float(constants["drag_power_on"])
        self.drag_power_off = float(constants["drag_power_off"])
        self.nozzle_to_cdm = float(constants["nozzle_to_cdm"])
        self.nozzle_area = float(constants["nozzle_area"])
        reference_pressure = constants["reference_pressure"]
        self.reference_pressure = (
            None if reference_pressure is None else float(reference_pressure)
        )
        self.volume = float(constants["volume"])
        self.cp_eccentricity_x = float(constants["cp_eccentricity_x"])
        self.cp_eccentricity_y = float(constants["cp_eccentricity_y"])
        self.thrust_eccentricity_x = float(constants["thrust_eccentricity_x"])
        self.thrust_eccentricity_y = float(constants["thrust_eccentricity_y"])
        self.earth_rotation = _as_tensor(
            constants["earth_rotation"],
            device=self.device,
            dtype=dtype,
        )
        self.nozzle_gyration = _as_tensor(
            constants["nozzle_gyration"],
            device=self.device,
            dtype=dtype,
        )

        surfaces = data["surfaces"]
        if not isinstance(surfaces, Sequence) or len(surfaces) != 2:
            raise ValueError("Scenario 1 must contain one nose and one fin set")
        self.surfaces = tuple(dict(surface) for surface in surfaces)

    def _time_properties(self, elapsed: Tensor) -> tuple[Tensor, ...]:
        values = self.time_table(elapsed)
        columns = self._TIME_COLUMNS
        return (
            values[..., columns["mass"]],
            values[..., columns["mass_dot"]],
            values[..., columns["mass_ddot"]],
            values[..., columns["com"]],
            values[..., columns["com_dot"]],
            values[..., columns["com_ddot"]],
            values[..., columns["inertia"]].reshape(*elapsed.shape, 3, 3),
            values[..., columns["inertia_dot"]].reshape(*elapsed.shape, 3, 3),
            values[..., columns["thrust"]],
        )

    def _atmosphere(self, altitude: Tensor) -> tuple[Tensor, ...]:
        return tuple(
            self.atmosphere_tables[name](altitude)
            for name in (
                "density",
                "pressure",
                "dynamic_viscosity",
                "speed_of_sound",
                "wind_velocity_x",
                "wind_velocity_y",
                "gravity",
            )
        )

    @staticmethod
    def _transformation(quaternion: Tensor) -> Tensor:
        quaternion = quaternion / torch.linalg.vector_norm(
            quaternion,
            dim=-1,
            keepdim=True,
        ).clamp_min(torch.finfo(quaternion.dtype).tiny)
        q0, q1, q2, q3 = quaternion.unbind(-1)
        row0 = torch.stack(
            (
                1 - 2 * (q2.square() + q3.square()),
                2 * (q1 * q2 - q0 * q3),
                2 * (q1 * q3 + q0 * q2),
            ),
            dim=-1,
        )
        row1 = torch.stack(
            (
                2 * (q1 * q2 + q0 * q3),
                1 - 2 * (q1.square() + q3.square()),
                2 * (q2 * q3 - q0 * q1),
            ),
            dim=-1,
        )
        row2 = torch.stack(
            (
                2 * (q1 * q3 - q0 * q2),
                2 * (q2 * q3 + q0 * q1),
                1 - 2 * (q1.square() + q2.square()),
            ),
            dim=-1,
        )
        return torch.stack((row0, row1, row2), dim=-2)

    @staticmethod
    def _fin_clalpha(surface: Mapping[str, object], mach: Tensor) -> Tensor:
        beta = torch.where(
            mach < 0.8,
            torch.sqrt(torch.clamp(1 - mach.square(), min=0.0)),
            torch.where(
                mach < 1.1,
                torch.full_like(mach, (1 - 0.8**2) ** 0.5),
                torch.sqrt(torch.clamp(mach.square() - 1, min=0.0)),
            ),
        )
        clalpha_2d = (2 * torch.pi) / beta
        aspect_ratio = float(surface["aspect_ratio"])
        gamma_c = float(surface["gamma_c"])
        cosine = torch.cos(
            torch.as_tensor(gamma_c, device=mach.device, dtype=mach.dtype)
        )
        correlation = 2 * torch.pi * aspect_ratio / (clalpha_2d * cosine)
        single = (
            clalpha_2d
            * correlation
            * (float(surface["fin_area"]) / float(surface["reference_area"]))
            * cosine
            / (2 + correlation * torch.sqrt(1 + (2 / correlation).square()))
        )
        return (
            float(surface["fin_number_correction"])
            * float(surface["lift_interference_factor"])
            * single
        )

    def _surface_force_moment(
        self,
        surface: Mapping[str, object],
        stream_velocity: Tensor,
        stream_speed: Tensor,
        stream_mach: Tensor,
        density: Tensor,
        angular_velocity: Tensor,
    ) -> tuple[Tensor, Tensor]:
        lateral_square = (
            stream_velocity[..., 0].square() + stream_velocity[..., 1].square()
        )
        safe_speed = stream_speed.clamp_min(torch.finfo(stream_speed.dtype).tiny)
        stream_vzn = torch.clamp(stream_velocity[..., 2] / safe_speed, -1.0, 1.0)
        active = (lateral_square != 0) & (-stream_vzn < 1)
        attack = torch.minimum(
            torch.acos(-stream_vzn),
            torch.full_like(stream_vzn, torch.pi / 10),
        )
        if surface["kind"] == "nose":
            clalpha = torch.full_like(
                stream_mach,
                float(surface["clalpha"]),
            )
        elif surface["kind"] == "fins":
            clalpha = self._fin_clalpha(surface, stream_mach)
        else:  # pragma: no cover - extractor rejects this first
            raise ValueError(f"unsupported surface kind {surface['kind']!r}")
        lift = (
            0.5
            * density
            * stream_speed.square()
            * float(surface["reference_area"])
            * clalpha
            * attack
        )
        lateral_norm = torch.sqrt(lateral_square).clamp_min(
            torch.finfo(stream_speed.dtype).tiny
        )
        lift_x = torch.where(active, lift * stream_velocity[..., 0] / lateral_norm, 0)
        lift_y = torch.where(active, lift * stream_velocity[..., 1] / lateral_norm, 0)
        zero = torch.zeros_like(lift_x)
        force = torch.stack((lift_x, lift_y, zero), dim=-1)
        cp_z = float(surface["cp_to_cdm"])
        moment = torch.stack((-cp_z * lift_y, cp_z * lift_x, zero), dim=-1)

        if surface["kind"] == "fins":
            single = self._fin_clalpha(surface, stream_mach) / (
                float(surface["fin_number_correction"])
                * float(surface["lift_interference_factor"])
            )
            cld = (
                2
                * float(surface["roll_damping_interference_factor"])
                * float(surface["n"])
                * single
                * torch.cos(
                    torch.as_tensor(
                        float(surface["cant_angle"]),
                        device=stream_speed.device,
                        dtype=stream_speed.dtype,
                    )
                )
                * float(surface["roll_geometrical_constant"])
                / (
                    float(surface["reference_area"])
                    * float(surface["reference_length"]) ** 2
                )
            )
            forcing = (
                0.5
                * density
                * stream_speed.square()
                * float(surface["reference_area"])
                * float(surface["reference_length"])
                * float(surface["roll_forcing_coefficient"])
                * float(surface["cant_angle"])
            )
            damping = (
                0.5
                * density
                * stream_speed
                * float(surface["reference_area"])
                * float(surface["reference_length"]) ** 2
                * cld
                * angular_velocity[..., 2]
                / 2
            )
            moment = moment + torch.stack((zero, zero, forcing - damping), dim=-1)
        return force, moment

    def components(
        self,
        elapsed: Tensor | float,
        state: Tensor,
        actuator_output: Tensor,
        *,
        validate_finite: bool = True,
    ) -> RocketRHSComponents:
        state = torch.as_tensor(state, device=self.device, dtype=self.dtype)
        actuator_output = torch.as_tensor(
            actuator_output,
            device=self.device,
            dtype=self.dtype,
        )
        if state.shape[-1] != STATE_SIZE:
            raise ValueError("state must end in 13 elements")
        if actuator_output.shape != (*state.shape[:-1], 4):
            raise ValueError(
                "actuator output must match state batch shape and end in 4"
            )
        if validate_finite and not bool(
            torch.isfinite(state).all() and torch.isfinite(actuator_output).all()
        ):
            raise ValueError("state and actuator output must be finite")
        elapsed_tensor = torch.as_tensor(elapsed, device=self.device, dtype=self.dtype)
        elapsed_tensor = torch.broadcast_to(elapsed_tensor, state.shape[:-1])

        position = state[..., :3]
        velocity = state[..., 3:6]
        quaternion = state[..., 6:10]
        angular_velocity = state[..., 10:13]
        roll, tvc_x, tvc_y, throttle = actuator_output.unbind(-1)

        (
            total_mass,
            total_mass_dot,
            total_mass_ddot,
            com_z,
            com_dot_z,
            com_ddot_z,
            inertia,
            inertia_dot,
            base_thrust,
        ) = self._time_properties(elapsed_tensor)
        zero = torch.zeros_like(total_mass)
        center_of_mass = torch.stack((zero, zero, com_z), dim=-1)
        center_of_mass_dot = torch.stack((zero, zero, com_dot_z), dim=-1)
        center_of_mass_ddot = torch.stack((zero, zero, com_ddot_z), dim=-1)

        (
            density,
            pressure,
            dynamic_viscosity,
            speed_of_sound,
            wind_x,
            wind_y,
            gravity,
        ) = self._atmosphere(position[..., 2])
        atmosphere = torch.stack(
            (density, pressure, dynamic_viscosity, speed_of_sound, gravity),
            dim=-1,
        )
        wind = torch.stack((wind_x, wind_y), dim=-1)
        if self.wind_provider is not None:
            wind = wind + self.wind_provider(position[..., 2])
            wind_x, wind_y = wind.unbind(-1)

        transformation = self._transformation(quaternion)
        transformation_t = transformation.transpose(-1, -2)
        wind_world = torch.stack((wind_x, wind_y, zero), dim=-1)
        free_stream_world = wind_world - velocity
        free_stream_speed = torch.linalg.vector_norm(free_stream_world, dim=-1)

        burning = (elapsed_tensor > 0) & (elapsed_tensor < self.burn_duration)
        pressure_thrust = torch.zeros_like(base_thrust)
        if self.reference_pressure is not None:
            pressure_thrust = (self.reference_pressure - pressure) * self.nozzle_area
        net_thrust = torch.where(
            burning,
            torch.clamp_min(base_thrust + pressure_thrust, 0),
            0,
        )
        effective_thrust = net_thrust * throttle
        tvc_x_rad = torch.deg2rad(tvc_x)
        tvc_y_rad = torch.deg2rad(tvc_y)
        thrust_z = effective_thrust * torch.sqrt(
            torch.clamp(
                1 - torch.sin(tvc_x_rad).square() - torch.sin(tvc_y_rad).square(),
                min=0,
            )
        )
        thrust = torch.stack((zero, zero, thrust_z), dim=-1)

        drag_coefficient = torch.where(
            burning,
            torch.full_like(total_mass, self.drag_power_on),
            torch.full_like(total_mass, self.drag_power_off),
        )
        drag_z = (
            -0.5
            * density
            * free_stream_speed.square()
            * self.reference_area
            * drag_coefficient
        )
        aerodynamic_force = torch.stack((zero, zero, drag_z), dim=-1)
        aerodynamic_moment = torch.zeros_like(aerodynamic_force)

        velocity_body = torch.matmul(
            transformation_t,
            velocity.unsqueeze(-1),
        ).squeeze(-1)
        for surface in self.surfaces:
            cp = torch.stack(
                (zero, zero, torch.full_like(zero, float(surface["cp_to_cdm"]))),
                dim=-1,
            )
            component_velocity = velocity_body + torch.linalg.cross(
                angular_velocity,
                cp,
                dim=-1,
            )
            component_altitude = (
                position[..., 2]
                + torch.matmul(
                    transformation,
                    cp.unsqueeze(-1),
                ).squeeze(-1)[..., 2]
            )
            component_wind_x = self.atmosphere_tables["wind_velocity_x"](
                component_altitude
            )
            component_wind_y = self.atmosphere_tables["wind_velocity_y"](
                component_altitude
            )
            if self.wind_provider is not None:
                component_gust = self.wind_provider(component_altitude)
                component_wind_x = component_wind_x + component_gust[..., 0]
                component_wind_y = component_wind_y + component_gust[..., 1]
            component_wind_world = torch.stack(
                (component_wind_x, component_wind_y, zero),
                dim=-1,
            )
            component_wind_body = torch.matmul(
                transformation_t,
                component_wind_world.unsqueeze(-1),
            ).squeeze(-1)
            component_stream = component_wind_body - component_velocity
            component_speed = torch.linalg.vector_norm(component_stream, dim=-1)
            component_mach = component_speed / speed_of_sound
            force, moment = self._surface_force_moment(
                surface,
                component_stream,
                component_speed,
                component_mach,
                density,
                angular_velocity,
            )
            aerodynamic_force = aerodynamic_force + force
            aerodynamic_moment = aerodynamic_moment + moment

        aerodynamic_moment = aerodynamic_moment + torch.stack(
            (
                self.cp_eccentricity_y * aerodynamic_force[..., 2],
                -self.cp_eccentricity_x * aerodynamic_force[..., 2],
                self.cp_eccentricity_x * aerodynamic_force[..., 1]
                - self.cp_eccentricity_y * aerodynamic_force[..., 0],
            ),
            dim=-1,
        )
        control_moment = torch.stack(
            (
                torch.sin(tvc_x_rad) * effective_thrust * self.nozzle_to_cdm
                + self.thrust_eccentricity_y * thrust_z,
                torch.sin(tvc_y_rad) * effective_thrust * self.nozzle_to_cdm
                - self.thrust_eccentricity_x * thrust_z,
                roll,
            ),
            dim=-1,
        )
        total_moment = aerodynamic_moment + control_moment

        net_gravitational_force = (
            -total_mass * gravity + density * self.volume * gravity
        )
        weight_body = torch.matmul(
            transformation_t,
            torch.stack((zero, zero, net_gravitational_force), dim=-1).unsqueeze(-1),
        ).squeeze(-1)

        cross_matrix = torch.zeros(
            (*state.shape[:-1], 3, 3),
            device=self.device,
            dtype=self.dtype,
        )
        cross_matrix[..., 0, 1] = -com_z
        cross_matrix[..., 1, 0] = com_z
        h_matrix = (
            torch.matmul(cross_matrix, -cross_matrix) * total_mass[..., None, None]
        )
        inertia_cm = inertia - h_matrix

        nozzle = torch.stack(
            (zero, zero, torch.full_like(zero, self.nozzle_to_cdm)),
            dim=-1,
        )
        t00 = total_mass[..., None] * center_of_mass
        t03 = (
            2 * total_mass_dot[..., None] * (nozzle - center_of_mass)
            - 2 * total_mass[..., None] * center_of_mass_dot
        )
        t04 = (
            thrust
            - total_mass[..., None] * center_of_mass_ddot
            - 2 * total_mass_dot[..., None] * center_of_mass_dot
            + total_mass_ddot[..., None] * (nozzle - center_of_mass)
        )
        t05 = total_mass_dot[..., None, None] * self.nozzle_gyration - inertia_dot
        t20 = (
            torch.linalg.cross(
                torch.linalg.cross(angular_velocity, t00, dim=-1),
                angular_velocity,
                dim=-1,
            )
            + torch.linalg.cross(angular_velocity, t03, dim=-1)
            + t04
            + weight_body
            + aerodynamic_force
        )
        inertia_w = torch.matmul(inertia, angular_velocity.unsqueeze(-1)).squeeze(-1)
        t21 = (
            torch.linalg.cross(inertia_w, angular_velocity, dim=-1)
            + torch.matmul(t05, angular_velocity.unsqueeze(-1)).squeeze(-1)
            - torch.linalg.cross(weight_body, center_of_mass, dim=-1)
            + total_moment
        )
        angular_acceleration = torch.linalg.solve(
            inertia_cm,
            (t21 + torch.linalg.cross(t20, center_of_mass, dim=-1)).unsqueeze(-1),
        ).squeeze(-1)

        body_acceleration = t20 / total_mass[..., None] - torch.linalg.cross(
            center_of_mass, angular_acceleration, dim=-1
        )
        translation_acceleration = torch.matmul(
            transformation,
            body_acceleration.unsqueeze(-1),
        ).squeeze(-1) - 2 * torch.linalg.cross(
            self.earth_rotation.expand_as(velocity),
            velocity,
            dim=-1,
        )

        q0, q1, q2, q3 = quaternion.unbind(-1)
        w1, w2, w3 = angular_velocity.unbind(-1)
        quaternion_derivative = 0.5 * torch.stack(
            (
                -w1 * q1 - w2 * q2 - w3 * q3,
                w1 * q0 + w3 * q2 - w2 * q3,
                w2 * q0 - w3 * q1 + w1 * q3,
                w3 * q0 + w2 * q1 - w1 * q2,
            ),
            dim=-1,
        )
        rhs = torch.cat(
            (
                velocity,
                translation_acceleration,
                quaternion_derivative,
                angular_acceleration,
            ),
            dim=-1,
        )
        return RocketRHSComponents(
            total_mass=total_mass,
            total_mass_dot=total_mass_dot,
            total_mass_ddot=total_mass_ddot,
            center_of_mass=center_of_mass,
            center_of_mass_dot=center_of_mass_dot,
            center_of_mass_ddot=center_of_mass_ddot,
            inertia=inertia,
            inertia_dot=inertia_dot,
            atmosphere=atmosphere,
            wind=wind,
            thrust=thrust,
            aerodynamic_force=aerodynamic_force,
            aerodynamic_moment=aerodynamic_moment,
            control_moment=control_moment,
            translation_acceleration=translation_acceleration,
            angular_acceleration=angular_acceleration,
            quaternion_derivative=quaternion_derivative,
            rhs=rhs,
        )

    def rhs(
        self,
        elapsed: Tensor | float,
        state: Tensor,
        actuator_output: Tensor,
    ) -> Tensor:
        # The integrator hot path deliberately avoids a data-dependent Python
        # scalar read, which would synchronize CUDA and break torch.compile.
        # The environment boundary remains responsible for masking bad rows.
        return self.components(
            elapsed,
            state,
            actuator_output,
            validate_finite=False,
        ).rhs


def rk4_step(
    model: Scenario1TensorRocket,
    elapsed: Tensor | float,
    state: Tensor,
    actuator_output: Tensor,
    *,
    dt: float = 0.01,
    substeps: int = 1,
) -> Tensor:
    """Integrate a held-control interval with fixed-step classical RK4."""
    if dt <= 0:
        raise ValueError("dt must be positive")
    if substeps < 1:
        raise ValueError("substeps must be positive")
    result = torch.as_tensor(state, device=model.device, dtype=model.dtype)
    time = torch.as_tensor(elapsed, device=model.device, dtype=model.dtype)
    step = dt / substeps
    for _ in range(substeps):
        k1 = model.rhs(time, result, actuator_output)
        k2 = model.rhs(time + step / 2, result + step * k1 / 2, actuator_output)
        k3 = model.rhs(time + step / 2, result + step * k2 / 2, actuator_output)
        k4 = model.rhs(time + step, result + step * k3, actuator_output)
        result = result + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        time = time + step
    return result


def dopri5_step(
    model: Scenario1TensorRocket,
    elapsed: Tensor | float,
    state: Tensor,
    actuator_output: Tensor,
    *,
    dt: float = 0.01,
    initial_rhs: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """One Dormand--Prince 5(4) step and its FSAL endpoint derivative.

    Supplying ``initial_rhs`` reproduces SciPy RK45's FSAL cache.  In the
    official environment that cache is not invalidated when an actuator changes
    at a control node.  Passing ``None`` evaluates a fresh first stage instead.
    """
    if dt <= 0:
        raise ValueError("dt must be positive")
    y0 = torch.as_tensor(state, device=model.device, dtype=model.dtype)
    t0 = torch.as_tensor(elapsed, device=model.device, dtype=model.dtype)
    if initial_rhs is None:
        k1 = model.rhs(t0, y0, actuator_output)
    else:
        k1 = torch.as_tensor(initial_rhs, device=model.device, dtype=model.dtype)
        if k1.shape != y0.shape:
            raise ValueError("initial_rhs must have the same shape as state")

    # SciPy RK45's Dormand--Prince tableau. The seventh stage is the endpoint
    # derivative and becomes the first stage of the next step (FSAL).
    k2 = model.rhs(t0 + dt * (1 / 5), y0 + dt * ((1 / 5) * k1), actuator_output)
    k3 = model.rhs(
        t0 + dt * (3 / 10),
        y0 + dt * ((3 / 40) * k1 + (9 / 40) * k2),
        actuator_output,
    )
    k4 = model.rhs(
        t0 + dt * (4 / 5),
        y0 + dt * ((44 / 45) * k1 - (56 / 15) * k2 + (32 / 9) * k3),
        actuator_output,
    )
    k5 = model.rhs(
        t0 + dt * (8 / 9),
        y0
        + dt
        * (
            (19372 / 6561) * k1
            - (25360 / 2187) * k2
            + (64448 / 6561) * k3
            - (212 / 729) * k4
        ),
        actuator_output,
    )
    k6 = model.rhs(
        t0 + dt,
        y0
        + dt
        * (
            (9017 / 3168) * k1
            - (355 / 33) * k2
            + (46732 / 5247) * k3
            + (49 / 176) * k4
            - (5103 / 18656) * k5
        ),
        actuator_output,
    )
    result = y0 + dt * (
        (35 / 384) * k1
        + (500 / 1113) * k3
        + (125 / 192) * k4
        - (2187 / 6784) * k5
        + (11 / 84) * k6
    )
    endpoint_rhs = model.rhs(t0 + dt, result, actuator_output)
    return result, endpoint_rhs
