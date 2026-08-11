"""ActiveRocketPy oracle adapter and Phase 2 fidelity report.

All imports of the official simulator live here.  The tensor model itself stays
free of RocketPy, SciPy, Gymnasium, and NumPy callbacks during RHS evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch

from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters
from experiments.cuda.phase2_rocket import (
    ACTUATOR_ORDER,
    ActuatorSpec,
    RocketRHSComponents,
    Scenario1TensorRocket,
    dopri5_step,
    rk4_step,
)


def _official_action(
    *,
    launch: bool,
    roll: float = 0.0,
    tvc_x: float = 0.0,
    tvc_y: float = 0.0,
    throttle: float = 1.0,
) -> dict[str, object]:
    return {
        "launch": launch,
        "launch_inclination_heading": np.array([90.0, 0.0]),
        "tvc": np.array([tvc_x, tvc_y]),
        "throttle": throttle,
        "roll": roll,
    }


@dataclass
class Scenario1Oracle:
    env: BalloonPoppingEnv
    flight: object
    model_data: dict[str, object]
    launch_time: float

    def close(self) -> None:
        self.env.close()


def build_scenario1_oracle(
    *,
    seed: int = 2031,
    time_table_step: float = 0.00125,
) -> Scenario1Oracle:
    """Build a one-balloon Scenario 1 flight at the first free-flight node."""
    parameters, _ = load_scenario_parameters(1)
    parameters["balloon"]["num"] = 1
    warnings.filterwarnings(
        "ignore",
        message=r"Actuator .* output change .* exceeds rate limit.*",
    )
    env = BalloonPoppingEnv(render_mode=None, parameters=parameters)
    env.reset(seed=seed)
    env.step(_official_action(launch=False, throttle=0.0))
    env.step(_official_action(launch=True, throttle=1.0))
    env.step(_official_action(launch=True, throttle=1.0))
    flight = env._rocket_flight
    if flight is None:  # pragma: no cover - official contract regression
        env.close()
        raise RuntimeError("official environment did not create a flight")
    launch_time = float(flight.rocket.motor.burn_start_time)
    model_data = extract_scenario1_model_data(
        flight,
        time_table_step=time_table_step,
    )
    return Scenario1Oracle(env, flight, model_data, launch_time)


def _function_values(function: object, times: np.ndarray) -> np.ndarray:
    return np.asarray(
        [function.get_value_opt(float(t)) for t in times], dtype=np.float64
    )


def _first_derivative(function: object, times: np.ndarray) -> np.ndarray:
    return np.asarray(
        [function.differentiate_complex_step(float(t)) for t in times],
        dtype=np.float64,
    )


def _second_derivative(function: object, times: np.ndarray) -> np.ndarray:
    return np.asarray(
        [function.differentiate(float(t), order=2) for t in times],
        dtype=np.float64,
    )


def _matrix_values(
    rocket: object, times: np.ndarray, *, derivative: bool
) -> np.ndarray:
    if derivative:
        return np.asarray(
            [
                list(
                    map(
                        list,
                        rocket.get_inertia_tensor_derivative_at_time(float(t)),
                    )
                )
                for t in times
            ],
            dtype=np.float64,
        )
    return np.asarray(
        [list(map(list, rocket.get_inertia_tensor_at_time(float(t)))) for t in times],
        dtype=np.float64,
    )


def _atmosphere_table(function: object) -> dict[str, np.ndarray]:
    source = np.asarray(function.source, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 2:
        raise TypeError("Scenario 1 atmosphere functions must have Nx2 sources")
    return {"x": source[:, 0].copy(), "y": source[:, 1].copy()}


def _surface_data(flight: object) -> list[dict[str, object]]:
    rocket = flight.rocket
    result: list[dict[str, object]] = []
    for surface, _position in rocket.aerodynamic_surfaces:
        class_name = type(surface).__name__
        common: dict[str, object] = {
            "reference_area": float(surface.reference_area),
            "reference_length": float(surface.reference_length),
            "cp_to_cdm": float(rocket.surfaces_cp_to_cdm[surface].z),
        }
        if class_name == "NoseCone":
            common.update(
                kind="nose",
                clalpha=float(surface.clalpha.get_value_opt(0.0)),
            )
        elif class_name == "TrapezoidalFins":
            forcing, _damping, cant_angle = surface.roll_parameters
            common.update(
                kind="fins",
                n=int(surface.n),
                cant_angle=float(cant_angle),
                aspect_ratio=float(surface.AR),
                gamma_c=float(surface.gamma_c),
                fin_area=float(surface.Af),
                fin_number_correction=float(surface.fin_num_correction(surface.n)),
                lift_interference_factor=float(surface.lift_interference_factor),
                roll_damping_interference_factor=float(
                    surface.roll_damping_interference_factor
                ),
                roll_geometrical_constant=float(surface.roll_geometrical_constant),
                roll_forcing_coefficient=float(forcing.get_value_opt(0.0)),
            )
        else:
            raise TypeError(f"unsupported Scenario 1 aerodynamic surface {class_name}")
        result.append(common)
    return result


def extract_scenario1_model_data(
    flight: object,
    *,
    time_table_step: float = 0.00125,
) -> dict[str, object]:
    """Extract immutable constants/tables from an initialized canonical flight."""
    if time_table_step <= 0:
        raise ValueError("time_table_step must be positive")
    rocket = flight.rocket
    motor = rocket.motor
    environment = flight.env
    burn_duration = float(motor.burn_out_time - motor.burn_start_time)
    count = math.ceil((burn_duration + 1.0) / time_table_step)
    elapsed = np.arange(count + 1, dtype=np.float64) * time_table_step
    absolute_time = elapsed + float(motor.burn_start_time)

    mass = _function_values(rocket.total_mass, absolute_time)
    mass_dot = _function_values(rocket.total_mass_flow_rate, absolute_time)
    mass_ddot = _first_derivative(rocket.total_mass_flow_rate, absolute_time)
    com = _function_values(rocket.com_to_cdm_function, absolute_time)
    com_dot = _first_derivative(rocket.com_to_cdm_function, absolute_time)
    com_ddot = _second_derivative(rocket.com_to_cdm_function, absolute_time)
    inertia = _matrix_values(rocket, absolute_time, derivative=False)
    inertia_dot = _matrix_values(rocket, absolute_time, derivative=True)
    thrust = _function_values(motor.thrust, absolute_time)
    time_values = np.column_stack(
        (
            mass,
            mass_dot,
            mass_ddot,
            com,
            com_dot,
            com_ddot,
            inertia.reshape(-1, 9),
            inertia_dot.reshape(-1, 9),
            thrust,
        )
    )

    atmosphere = {
        name: _atmosphere_table(getattr(environment, name))
        for name in (
            "density",
            "pressure",
            "dynamic_viscosity",
            "speed_of_sound",
            "wind_velocity_x",
            "wind_velocity_y",
            "gravity",
        )
    }
    zero_aero_inputs = (0.0,) * 7
    return {
        "atmosphere": atmosphere,
        "time": {"x": elapsed, "y": time_values},
        "constants": {
            "burn_duration": burn_duration,
            "reference_area": float(rocket.area),
            "reference_length": float(2 * rocket.radius),
            "drag_power_on": float(rocket.power_on_drag_7d(*zero_aero_inputs)),
            "drag_power_off": float(rocket.power_off_drag_7d(*zero_aero_inputs)),
            "nozzle_to_cdm": float(rocket.nozzle_to_cdm),
            "nozzle_area": float(motor.nozzle_area),
            "reference_pressure": motor.reference_pressure,
            "volume": float(rocket.volume),
            "cp_eccentricity_x": float(rocket.cp_eccentricity_x),
            "cp_eccentricity_y": float(rocket.cp_eccentricity_y),
            "thrust_eccentricity_x": float(rocket.thrust_eccentricity_x),
            "thrust_eccentricity_y": float(rocket.thrust_eccentricity_y),
            "earth_rotation": np.asarray(
                environment.earth_rotation_vector,
                dtype=np.float64,
            ),
            "nozzle_gyration": np.asarray(
                list(map(list, rocket.nozzle_gyration_tensor)),
                dtype=np.float64,
            ),
        },
        "surfaces": _surface_data(flight),
    }


def scenario1_actuator_specs(flight: object) -> tuple[tuple[ActuatorSpec, ...], float]:
    rocket = flight.rocket
    actuators = (
        rocket.roll_control,
        rocket.thrust_vector_control.x,
        rocket.thrust_vector_control.y,
        rocket.throttle_control,
    )
    demand_rates = {float(actuator.demand_rate) for actuator in actuators}
    if len(demand_rates) != 1:
        raise ValueError("Scenario 1 actuators must share one demand rate")
    specs = tuple(
        ActuatorSpec(
            lower=float(actuator.actuator_range[0]),
            upper=float(actuator.actuator_range[1]),
            rate_limit=(
                None
                if actuator.actuator_rate_limit is None
                else float(actuator.actuator_rate_limit)
            ),
            time_constant=(
                None
                if actuator.actuator_time_constant is None
                else float(actuator.actuator_time_constant)
            ),
            initial=float(actuator.actuator_initial_output),
        )
        for actuator in actuators
    )
    return specs, demand_rates.pop()


def canonical_actuator_outputs(flight: object) -> np.ndarray:
    rocket = flight.rocket
    return np.asarray(
        (
            rocket.roll_control.roll_torque,
            rocket.thrust_vector_control.gimbal_angle_x,
            rocket.thrust_vector_control.gimbal_angle_y,
            rocket.throttle_control.throttle,
        ),
        dtype=np.float64,
    )


def apply_canonical_actuator_command(flight: object, command: np.ndarray) -> np.ndarray:
    command = np.asarray(command, dtype=np.float64)
    if command.shape != (4,) or not np.isfinite(command).all():
        raise ValueError(f"command must be finite with order {ACTUATOR_ORDER}")
    rocket = flight.rocket
    rocket.roll_control.roll_torque = float(command[0])
    rocket.thrust_vector_control.gimbal_angle_x = float(command[1])
    rocket.thrust_vector_control.gimbal_angle_y = float(command[2])
    rocket.throttle_control.throttle = float(command[3])
    return canonical_actuator_outputs(flight)


def canonical_components(
    flight: object,
    absolute_time: float,
    state: np.ndarray,
    actuator_output: np.ndarray,
) -> dict[str, np.ndarray]:
    """Evaluate canonical component values at a state without advancing it."""
    state = np.asarray(state, dtype=np.float64)
    actuator_output = np.asarray(actuator_output, dtype=np.float64)
    if state.shape != (13,) or actuator_output.shape != (4,):
        raise ValueError(
            "canonical component shapes must be state=(13,), actuator=(4,)"
        )
    rocket = flight.rocket
    environment = flight.env
    rocket.roll_control._actuator_output = float(actuator_output[0])
    rocket.thrust_vector_control.x._actuator_output = float(actuator_output[1])
    rocket.thrust_vector_control.y._actuator_output = float(actuator_output[2])
    rocket.throttle_control._actuator_output = float(actuator_output[3])

    before = len(flight._Flight__post_processed_variables)
    rhs = np.asarray(
        flight.u_dot_generalized(absolute_time, state, post_processing=True),
        dtype=np.float64,
    )
    row = np.asarray(flight._Flight__post_processed_variables.pop(), dtype=np.float64)
    if len(flight._Flight__post_processed_variables) != before:
        raise RuntimeError("canonical post-processing stack was not restored")

    total_mass = float(rocket.total_mass.get_value_opt(absolute_time))
    total_mass_dot = float(rocket.total_mass_flow_rate.get_value_opt(absolute_time))
    total_mass_ddot = float(
        rocket.total_mass_flow_rate.differentiate_complex_step(absolute_time)
    )
    com_z = float(rocket.com_to_cdm_function.get_value_opt(absolute_time))
    com_dot_z = float(
        rocket.com_to_cdm_function.differentiate_complex_step(absolute_time)
    )
    com_ddot_z = float(rocket.com_to_cdm_function.differentiate(absolute_time, order=2))
    inertia = np.asarray(
        list(map(list, rocket.get_inertia_tensor_at_time(absolute_time))),
        dtype=np.float64,
    )
    inertia_dot = np.asarray(
        list(map(list, rocket.get_inertia_tensor_derivative_at_time(absolute_time))),
        dtype=np.float64,
    )
    altitude = float(state[2])
    density = float(environment.density.get_value_opt(altitude))
    pressure = float(environment.pressure.get_value_opt(altitude))
    viscosity = float(environment.dynamic_viscosity.get_value_opt(altitude))
    sound = float(environment.speed_of_sound.get_value_opt(altitude))
    gravity = float(environment.gravity.get_value_opt(altitude))
    wind = np.asarray(
        (
            environment.wind_velocity_x.get_value_opt(altitude),
            environment.wind_velocity_y.get_value_opt(altitude),
        ),
        dtype=np.float64,
    )
    thrust = np.asarray((0.0, 0.0, row[13]), dtype=np.float64)
    effective_thrust = (
        max(
            rocket.motor.thrust.get_value_opt(absolute_time)
            + rocket.motor.pressure_thrust(pressure),
            0,
        )
        * actuator_output[3]
    )
    tvc_x = np.deg2rad(actuator_output[1])
    tvc_y = np.deg2rad(actuator_output[2])
    control_moment = np.asarray(
        (
            np.sin(tvc_x) * effective_thrust * rocket.nozzle_to_cdm
            + rocket.thrust_eccentricity_y * row[13],
            np.sin(tvc_y) * effective_thrust * rocket.nozzle_to_cdm
            - rocket.thrust_eccentricity_x * row[13],
            actuator_output[0],
        ),
        dtype=np.float64,
    )
    total_moment = row[10:13]
    return {
        "total_mass": np.asarray(total_mass),
        "total_mass_dot": np.asarray(total_mass_dot),
        "total_mass_ddot": np.asarray(total_mass_ddot),
        "center_of_mass": np.asarray((0.0, 0.0, com_z)),
        "center_of_mass_dot": np.asarray((0.0, 0.0, com_dot_z)),
        "center_of_mass_ddot": np.asarray((0.0, 0.0, com_ddot_z)),
        "inertia": inertia,
        "inertia_dot": inertia_dot,
        "atmosphere": np.asarray((density, pressure, viscosity, sound, gravity)),
        "wind": wind,
        "thrust": thrust,
        "aerodynamic_force": row[7:10],
        "aerodynamic_moment": total_moment - control_moment,
        "control_moment": control_moment,
        "translation_acceleration": rhs[3:6],
        "angular_acceleration": rhs[10:13],
        "quaternion_derivative": rhs[6:10],
        "rhs": rhs,
    }


def _tensor_components_as_numpy(
    components: RocketRHSComponents,
) -> dict[str, np.ndarray]:
    return {
        field: np.asarray(getattr(components, field).detach().cpu(), dtype=np.float64)[
            0
        ]
        for field in components.__dataclass_fields__
    }


def _distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "rmse": float(np.sqrt(np.mean(array**2))),
        "p50": float(np.percentile(array, 50)),
        "p99": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
    }


def _norm_error(actual: np.ndarray, expected: np.ndarray) -> float:
    difference = np.asarray(actual) - np.asarray(expected)
    return float(np.linalg.norm(difference.reshape(-1)))


def _attitude_error_degrees(actual: np.ndarray, expected: np.ndarray) -> float:
    actual = actual / np.linalg.norm(actual)
    expected = expected / np.linalg.norm(expected)
    cosine = np.clip(abs(np.dot(actual, expected)), 0.0, 1.0)
    return float(np.rad2deg(2 * np.arccos(cosine)))


def _command(step: int) -> np.ndarray:
    return np.asarray(
        (
            2.0 * np.sin(step * 0.017),
            3.0 * np.sin(step * 0.011),
            3.0 * np.cos(step * 0.013),
            0.85 + 0.15 * np.cos(step * 0.007),
        ),
        dtype=np.float64,
    )


def _state_error_groups() -> dict[str, list[float]]:
    return {
        "position_m": [],
        "velocity_m_s": [],
        "attitude_deg": [],
        "angular_rate_rad_s": [],
    }


def _append_state_errors(
    groups: dict[str, list[float]],
    actual: np.ndarray,
    expected: np.ndarray,
) -> None:
    groups["position_m"].append(_norm_error(actual[:3], expected[:3]))
    groups["velocity_m_s"].append(_norm_error(actual[3:6], expected[3:6]))
    groups["attitude_deg"].append(_attitude_error_degrees(actual[6:10], expected[6:10]))
    groups["angular_rate_rad_s"].append(_norm_error(actual[10:13], expected[10:13]))


def generate_phase2_report(
    *,
    steps: int = 256,
    seed: int = 2031,
    time_table_step: float = 0.00125,
) -> dict[str, object]:
    if steps < 1:
        raise ValueError("steps must be positive")
    oracle = build_scenario1_oracle(seed=seed, time_table_step=time_table_step)
    try:
        flight = oracle.flight
        model = Scenario1TensorRocket(oracle.model_data, dtype=torch.float64)
        component_errors: dict[str, list[float]] = {
            name: [] for name in RocketRHSComponents.__dataclass_fields__
        }
        official_rk4_errors = {
            substeps: _state_error_groups() for substeps in (1, 2, 4)
        }
        fresh_reference_errors = {
            substeps: _state_error_groups() for substeps in (1, 2, 4)
        }
        stale_fsal_errors = _state_error_groups()
        shortened_event_intervals: list[float] = []

        compared = 0
        for step in range(steps):
            if flight._step_state["finished"]:
                break
            phase = flight.flight_phases[flight._step_state["phase_index"]]
            if getattr(phase, "solver", None) is None:
                break
            cached_rhs = np.asarray(phase.solver.f, dtype=np.float64).copy()
            actual_actuator = apply_canonical_actuator_command(flight, _command(step))
            absolute_time = float(flight.t)
            elapsed = absolute_time - oracle.launch_time
            state = np.asarray(flight.y_sol, dtype=np.float64).copy()
            canonical = canonical_components(
                flight,
                absolute_time,
                state,
                actual_actuator,
            )
            tensor = _tensor_components_as_numpy(
                model.components(
                    elapsed,
                    torch.from_numpy(state[None]),
                    torch.from_numpy(actual_actuator[None]),
                )
            )
            for name in component_errors:
                component_errors[name].append(
                    _norm_error(tensor[name], canonical[name])
                )

            predictions = {
                substeps: np.asarray(
                    rk4_step(
                        model,
                        elapsed,
                        torch.from_numpy(state[None]),
                        torch.from_numpy(actual_actuator[None]),
                        substeps=substeps,
                    )[0]
                )
                for substeps in (1, 2, 4)
            }
            fresh_dopri, _ = dopri5_step(
                model,
                elapsed,
                torch.from_numpy(state[None]),
                torch.from_numpy(actual_actuator[None]),
            )
            stale_dopri, _ = dopri5_step(
                model,
                elapsed,
                torch.from_numpy(state[None]),
                torch.from_numpy(actual_actuator[None]),
                initial_rhs=torch.from_numpy(cached_rhs[None]),
            )
            fresh_reference = np.asarray(fresh_dopri[0])
            stale_prediction = np.asarray(stale_dopri[0])
            flight.step_simulation()
            expected = np.asarray(flight.y_sol, dtype=np.float64)
            actual_interval = float(flight.t) - absolute_time
            if actual_interval <= 1e-12:
                # The canonical step API may expose one terminal no-op after an
                # impact root has already ended the flight. It is not an ODE
                # transition, so do not let the repeated state enter the RHS
                # or integrator distributions.
                for values in component_errors.values():
                    values.pop()
                break
            if not np.isclose(actual_interval, 0.01, rtol=0.0, atol=1e-10):
                shortened_event_intervals.append(actual_interval)
                compared += 1
                continue
            for substeps, prediction in predictions.items():
                _append_state_errors(
                    official_rk4_errors[substeps],
                    prediction,
                    expected,
                )
                _append_state_errors(
                    fresh_reference_errors[substeps],
                    prediction,
                    fresh_reference,
                )
            _append_state_errors(stale_fsal_errors, stale_prediction, expected)
            compared += 1

        return {
            "schema_version": 1,
            "scenario": 1,
            "seed": seed,
            "dtype": "float64",
            "time_table_step": time_table_step,
            "compared_steps": compared,
            "full_integrator_steps": compared - len(shortened_event_intervals),
            "shortened_event_intervals": shortened_event_intervals,
            "rhs_component_errors": {
                name: _distribution(values) for name, values in component_errors.items()
            },
            "integrator_errors": {
                "rk4_vs_official_stale_fsal": {
                    str(substeps): {
                        name: _distribution(values) for name, values in groups.items()
                    }
                    for substeps, groups in official_rk4_errors.items()
                },
                "rk4_vs_fresh_dopri5": {
                    str(substeps): {
                        name: _distribution(values) for name, values in groups.items()
                    }
                    for substeps, groups in fresh_reference_errors.items()
                },
                "dopri5_stale_fsal_vs_official": {
                    name: _distribution(values)
                    for name, values in stale_fsal_errors.items()
                },
            },
        }
    finally:
        oracle.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2031)
    parser.add_argument("--time-table-step", type=float, default=0.00125)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = generate_phase2_report(
        steps=args.steps,
        seed=args.seed,
        time_table_step=args.time_table_step,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
