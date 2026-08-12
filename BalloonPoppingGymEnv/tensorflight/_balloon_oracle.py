"""Private canonical balloon capture and fidelity reports."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters
from BalloonPoppingGymEnv.tensorflight._rocket_oracle import _atmosphere_table
from BalloonPoppingGymEnv.tensorflight._rocket_oracle import (
    canonical_actuator_outputs,
    extract_scenario1_model_data,
    scenario1_actuator_specs,
)
from BalloonPoppingGymEnv.tensorflight.balloons import (
    BalloonBatchParameters,
    TensorBalloonWorld,
)
from BalloonPoppingGymEnv.tensorflight.effects import SensorEffectConfig
from BalloonPoppingGymEnv.tensorflight.environment import (
    TensorFlightEnvironment,
    TensorFlightEnvironmentConfig,
)
from rocketpy.simulation.monte_carlo import MonteCarlo


@contextlib.contextmanager
def _capture_monte_carlo_flights() -> Iterator[list[object]]:
    """Capture exact realized Flight objects without modifying official code."""
    captured: list[object] = []
    method_name = "_MonteCarlo__run_single_simulation"
    original = getattr(MonteCarlo, method_name)

    def capture(instance: object) -> object:
        flight = original(instance)
        captured.append(flight)
        return flight

    setattr(MonteCarlo, method_name, capture)
    try:
        yield captured
    finally:
        setattr(MonteCarlo, method_name, original)


@dataclass
class CanonicalBalloonFixture:
    model_data: dict[str, object]
    parameters: BalloonBatchParameters
    trajectories: np.ndarray
    release_steps: np.ndarray
    dt: float
    max_time: float


def _official_action(
    *,
    launch: bool,
    command: np.ndarray | None = None,
    inclination_heading: tuple[float, float] = (90.0, 0.0),
) -> dict[str, object]:
    values = np.asarray(
        (0.0, 0.0, 0.0, 1.0) if command is None else command,
        dtype=np.float64,
    )
    return {
        "launch": launch,
        "launch_inclination_heading": np.asarray(inclination_heading),
        "roll": float(values[0]),
        "tvc": values[1:3].copy(),
        "throttle": float(values[3]),
    }


def _full_command(step: int) -> np.ndarray:
    return np.asarray(
        (
            1.5 * np.sin(step * 0.031),
            2.0 * np.sin(step * 0.023),
            2.0 * np.cos(step * 0.019),
            0.9 + 0.1 * np.cos(step * 0.013),
        ),
        dtype=np.float64,
    )


def build_canonical_balloon_fixture(
    *, seed: int = 0, num_balloons: int = 8
) -> CanonicalBalloonFixture:
    if num_balloons < 1:
        raise ValueError("num_balloons must be positive")
    scenario, _ = load_scenario_parameters(1)
    scenario["balloon"]["num"] = num_balloons
    env = BalloonPoppingEnv(render_mode=None, parameters=scenario)
    try:
        with _capture_monte_carlo_flights() as flights:
            with contextlib.redirect_stdout(io.StringIO()):
                env.reset(seed=seed)
        if len(flights) != num_balloons:
            raise RuntimeError("canonical Monte Carlo flight capture was incomplete")
        canonical = np.transpose(env._balloon_flights.copy(), (0, 2, 1))
        initial_state = canonical[:, 0, :]
        elevation = float(env._rocketpy_env.elevation)
        origin_offset = initial_state[:, :3] - np.asarray((0.0, 0.0, elevation))
        quaternion = np.asarray(
            [flight.initial_solution[7:11] for flight in flights], dtype=np.float64
        )
        dry_mass = np.asarray(
            [flight.rocket.mass for flight in flights], dtype=np.float64
        )
        volume = np.asarray(
            [flight.rocket.volume for flight in flights], dtype=np.float64
        )
        inertia = np.asarray(
            [
                (
                    flight.rocket.I_11_without_motor,
                    flight.rocket.I_22_without_motor,
                    flight.rocket.I_33_without_motor,
                )
                for flight in flights
            ],
            dtype=np.float64,
        )
        rail_exit_time = np.asarray(
            [flight.out_of_rail_time for flight in flights], dtype=np.float64
        )
        reference_flight = flights[-1]
        environment = reference_flight.env
        surface = reference_flight.rocket.aerodynamic_surfaces[0].component
        motor = reference_flight.rocket.motor
        model_data: dict[str, object] = {
            "atmosphere": {
                name: _atmosphere_table(getattr(environment, name))
                for name in (
                    "density",
                    "wind_velocity_x",
                    "wind_velocity_y",
                    "gravity",
                )
            },
            "constants": {
                "earth_rotation": np.asarray(
                    environment.earth_rotation_vector, dtype=np.float64
                ),
                "drag_area": float(surface.reference_area),
                "drag_coefficient": float(surface.cD_0(0, 0, 0, 0, 0, 0, 0)),
                "propellant_initial_mass": float(motor.propellant_initial_mass),
                "burn_duration": float(motor.burn_out_time - motor.burn_start_time),
                "thrust": float(motor.thrust.get_value_opt(0.1)),
                "elevation": elevation,
            },
        }
        parameters = BalloonBatchParameters(
            initial_state=torch.from_numpy(initial_state[None]),
            origin_offset=torch.from_numpy(origin_offset[None]),
            quaternion=torch.from_numpy(quaternion[None]),
            dry_mass=torch.from_numpy(dry_mass[None]),
            volume=torch.from_numpy(volume[None]),
            inertia=torch.from_numpy(inertia[None]),
            release_step=torch.from_numpy(env._balloon_release_at_step[None]),
            rail_exit_time=torch.from_numpy(rail_exit_time[None]),
        )
        return CanonicalBalloonFixture(
            model_data=model_data,
            parameters=parameters,
            trajectories=canonical,
            release_steps=env._balloon_release_at_step.copy(),
            dt=float(scenario["simulation"]["time_step"]),
            max_time=float(scenario["simulation"]["max_time"]),
        )
    finally:
        env.close()


def _distribution(values: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(np.mean(values**2))),
        "p50": float(np.percentile(values, 50)),
        "p99": float(np.percentile(values, 99)),
        "p99_9": float(np.percentile(values, 99.9)),
        "max": float(np.max(values)),
    }


def generate_balloon_fidelity_report(
    *,
    seed: int = 0,
    num_balloons: int = 8,
    steps: int | None = None,
    substeps: int = 2,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device = "cpu",
) -> dict[str, object]:
    fixture = build_canonical_balloon_fixture(seed=seed, num_balloons=num_balloons)
    available_steps = fixture.trajectories.shape[1] - 1
    compared_steps = available_steps if steps is None else min(steps, available_steps)
    world = TensorBalloonWorld(
        fixture.model_data,
        fixture.parameters,
        dt=fixture.dt,
        max_time=fixture.max_time,
        substeps=substeps,
        device=device,
        dtype=dtype,
    )
    position_errors: list[np.ndarray] = []
    velocity_errors: list[np.ndarray] = []
    status_mismatches = 0
    release_time_errors: list[float] = []
    first_released = np.full(num_balloons, -1, dtype=np.int64)
    for step in range(1, compared_steps + 1):
        world.step()
        actual = world.current_state.detach().cpu().numpy()[0]
        expected = fixture.trajectories[:, step]
        position_errors.append(np.linalg.norm(actual[:, :3] - expected[:, :3], axis=1))
        velocity_errors.append(np.linalg.norm(actual[:, 3:] - expected[:, 3:], axis=1))
        expected_status = (step >= fixture.release_steps).astype(np.int8)
        actual_status = world.status.detach().cpu().numpy()[0]
        status_mismatches += int(np.count_nonzero(actual_status != expected_status))
        newly_released = (actual_status == 1) & (first_released < 0)
        first_released[newly_released] = step
    # Even a schedule entry of zero is still ground in reset observation; the
    # official environment changes it to released during the first step.
    transition_steps = np.maximum(fixture.release_steps, 1)
    for actual, expected in zip(first_released, transition_steps):
        if actual >= 0:
            release_time_errors.append(abs(float(actual - expected)) * fixture.dt)

    position = np.concatenate(position_errors)
    velocity = np.concatenate(velocity_errors)
    return {
        "schema_version": 1,
        "scenario": 1,
        "seed": seed,
        "num_balloons": num_balloons,
        "steps": compared_steps,
        "substeps": substeps,
        "dtype": str(dtype).removeprefix("torch."),
        "device": str(device),
        "position_error_m": _distribution(position),
        "velocity_error_m_s": _distribution(velocity),
        "release_time_error_s": _distribution(np.asarray(release_time_errors)),
        "status_mismatches": status_mismatches,
        "full_horizon_state_allocated": False,
        "state_storage_elements": world.state_storage_elements,
    }


def generate_full_environment_report(
    *,
    seed: int = 0,
    num_balloons: int = 100,
    steps: int = 256,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device = "cpu",
    integrator: str = "official_fsal",
    critical_distance_float64: bool = False,
) -> dict[str, object]:
    """Compare the combined float64 tensor environment with the official env."""
    balloon_fixture = build_canonical_balloon_fixture(
        seed=seed, num_balloons=num_balloons
    )
    scenario, _ = load_scenario_parameters(1)
    scenario["balloon"]["num"] = num_balloons
    official = BalloonPoppingEnv(render_mode=None, parameters=scenario)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            official.reset(seed=seed)
        official.step(_official_action(launch=False, command=np.zeros(4)))
        official.step(_official_action(launch=True))
        official.step(_official_action(launch=True))
        flight = official._rocket_flight
        if flight is None:
            raise RuntimeError("canonical rocket did not launch")
        phase = flight.flight_phases[flight._step_state["phase_index"]]
        cached_rhs = np.asarray(phase.solver.f, dtype=np.float64)
        actuator_specs, demand_rate = scenario1_actuator_specs(flight)
        sensor_config = SensorEffectConfig.from_mapping(scenario["rocket"]["sensors"])
        tensor = TensorFlightEnvironment(
            extract_scenario1_model_data(flight),
            balloon_fixture.model_data,
            balloon_fixture.parameters,
            torch.from_numpy(np.asarray(flight.y_sol, dtype=np.float64)[None]),
            actuator_specs,
            actuator_demand_rate=demand_rate,
            rocket_elapsed=float(flight.t - flight.rocket.motor.burn_start_time),
            start_step=official.current_step,
            initial_cached_rhs=torch.from_numpy(cached_rhs[None]),
            sensor_config=sensor_config,
            config=TensorFlightEnvironmentConfig(
                dt=balloon_fixture.dt,
                max_time=balloon_fixture.max_time,
                elevation=float(scenario["environment"]["elevation"]),
                balloon_radius=float(scenario["balloon"]["radius"]),
                integrator=integrator,
                balloon_substeps=1,
                critical_distance_float64=critical_distance_float64,
            ),
            device=device,
            dtype=dtype,
            seed=seed,
        )
        tensor.actuators.output.copy_(
            torch.as_tensor(
                canonical_actuator_outputs(flight)[None], device=device, dtype=dtype
            )
        )

        rocket_position: list[float] = []
        rocket_velocity: list[float] = []
        rocket_attitude: list[float] = []
        rocket_rate: list[float] = []
        balloon_position: list[float] = []
        balloon_velocity: list[float] = []
        sensor_errors: list[float] = []
        closest_errors: list[float] = []
        reward_mismatches = 0
        status_mismatches = 0
        termination_mismatches = 0
        compared = 0

        for step in range(steps):
            if bool(tensor.terminated[0].item()) or bool(tensor.truncated[0].item()):
                break
            command = _full_command(step)
            previous_rocket = np.asarray(official._rocket_states[:3]).copy()
            previous_balloons = official._balloon_states[:, :3].copy()
            official_observation, official_reward, terminated, truncated, _ = (
                official.step(_official_action(launch=True, command=command))
            )
            tensor_step = tensor.step(
                torch.as_tensor(command[None], device=device, dtype=dtype)
            )
            actual_rocket = tensor.oracle_state().rocket_state[0].detach().cpu().numpy()
            expected_rocket = np.asarray(official._rocket_states, dtype=np.float64)
            rocket_position.append(
                float(np.linalg.norm(actual_rocket[:3] - expected_rocket[:3]))
            )
            rocket_velocity.append(
                float(np.linalg.norm(actual_rocket[3:6] - expected_rocket[3:6]))
            )
            q_actual = actual_rocket[6:10] / np.linalg.norm(actual_rocket[6:10])
            q_expected = expected_rocket[6:10] / np.linalg.norm(expected_rocket[6:10])
            rocket_attitude.append(
                float(
                    np.rad2deg(2 * np.arccos(np.clip(abs(q_actual @ q_expected), 0, 1)))
                )
            )
            rocket_rate.append(
                float(np.linalg.norm(actual_rocket[10:] - expected_rocket[10:]))
            )
            actual_balloons = tensor.balloons.current_state[0].detach().cpu().numpy()
            balloon_position.extend(
                np.linalg.norm(
                    actual_balloons[:, :3] - official._balloon_states[:, :3], axis=1
                ).tolist()
            )
            balloon_velocity.extend(
                np.linalg.norm(
                    actual_balloons[:, 3:] - official._balloon_states[:, 3:], axis=1
                ).tolist()
            )
            sensor_errors.extend(
                np.abs(
                    tensor_step.observation.rocket_sensors[0].detach().cpu().numpy()
                    - np.asarray(
                        official_observation["rocket_sensors"], dtype=np.float64
                    )
                ).tolist()
            )
            if np.isfinite(previous_rocket).all():
                canonical_distance = np.sqrt(
                    official._segment_distance_squared_batch(
                        previous_rocket,
                        official._rocket_states[:3],
                        previous_balloons,
                        official._balloon_states[:, :3],
                    )
                )
                closest_errors.extend(
                    np.abs(
                        tensor.closest_distance[0].detach().cpu().numpy()
                        - canonical_distance
                    ).tolist()
                )
            reward_mismatches += int(float(tensor_step.reward[0]) != official_reward)
            status_mismatches += int(
                np.count_nonzero(
                    tensor_step.observation.balloon_status[0].detach().cpu().numpy()
                    != official._balloon_status[:, 0]
                )
            )
            termination_mismatches += int(
                bool(tensor_step.terminated[0]) != bool(terminated)
                or bool(tensor_step.truncated[0]) != bool(truncated)
            )
            compared += 1

        closest = np.asarray(closest_errors or [0.0], dtype=np.float64)
        ambiguity_band = max(float(np.percentile(closest, 99.9)) + 0.01, 0.05)
        return {
            "schema_version": 1,
            "scenario": 1,
            "seed": seed,
            "num_balloons": num_balloons,
            "compared_steps": compared,
            "dtype": str(dtype).removeprefix("torch."),
            "device": str(device),
            "integrator": integrator,
            "critical_distance_float64": critical_distance_float64,
            "rocket_position_error_m": _distribution(np.asarray(rocket_position)),
            "rocket_velocity_error_m_s": _distribution(np.asarray(rocket_velocity)),
            "rocket_attitude_error_deg": _distribution(np.asarray(rocket_attitude)),
            "rocket_rate_error_rad_s": _distribution(np.asarray(rocket_rate)),
            "balloon_position_error_m": _distribution(np.asarray(balloon_position)),
            "balloon_velocity_error_m_s": _distribution(np.asarray(balloon_velocity)),
            "sensor_absolute_error": _distribution(np.asarray(sensor_errors)),
            "closest_distance_error_m": _distribution(closest),
            "ambiguity_band_m": ambiguity_band,
            "reward_mismatches": reward_mismatches,
            "status_mismatches": status_mismatches,
            "termination_mismatches": termination_mismatches,
        }
    finally:
        official.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-balloons", type=int, default=8)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--substeps", type=int, default=2)
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--full-environment", action="store_true")
    parser.add_argument(
        "--integrator", choices=("rk4", "official_fsal"), default="official_fsal"
    )
    parser.add_argument("--critical-distance-float64", action="store_true")
    args = parser.parse_args()
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    if args.full_environment:
        report = generate_full_environment_report(
            seed=args.seed,
            num_balloons=args.num_balloons,
            steps=256 if args.steps is None else args.steps,
            dtype=dtype,
            device=args.device,
            integrator=args.integrator,
            critical_distance_float64=args.critical_distance_float64,
        )
    else:
        report = generate_balloon_fidelity_report(
            seed=args.seed,
            num_balloons=args.num_balloons,
            steps=args.steps,
            substeps=args.substeps,
            dtype=dtype,
            device=args.device,
        )
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
