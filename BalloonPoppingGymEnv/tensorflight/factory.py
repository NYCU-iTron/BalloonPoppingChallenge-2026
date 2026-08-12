"""Canonical Scenario 1 extraction and TensorFlight construction."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import rocketpy.simulation.flight as active_flight_module

from BalloonPoppingGymEnv.envs import balloon_world as balloon_world_module
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters
from BalloonPoppingGymEnv.tensorflight._rocket_oracle import (
    build_scenario1_oracle,
    canonical_actuator_outputs,
    scenario1_actuator_specs,
)
from BalloonPoppingGymEnv.tensorflight.rocket import ActuatorSpec
from BalloonPoppingGymEnv.tensorflight.balloons import Scenario1BalloonSampler
from BalloonPoppingGymEnv.tensorflight.effects import SensorEffectConfig
from BalloonPoppingGymEnv.tensorflight.environment import (
    TensorFlightEnvironment,
    TensorFlightEnvironmentConfig,
)
from BalloonPoppingGymEnv.tensorflight._balloon_oracle import (
    build_canonical_balloon_fixture,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
MODULE_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class TensorFlightSource:
    rocket_model_data: dict[str, object]
    balloon_model_data: dict[str, object]
    initial_rocket_state: np.ndarray
    initial_cached_rhs: np.ndarray
    initial_actuator_output: np.ndarray
    actuator_specs: tuple[ActuatorSpec, ...]
    actuator_demand_rate: float
    rocket_elapsed: float
    start_step: int
    sensor_config: SensorEffectConfig
    dt: float
    max_time: float
    elevation: float
    balloon_radius: float
    num_balloons: int
    release_interval: float
    max_roll_torque: float
    max_gimbal_angle: float
    throttle_low: float
    throttle_high: float
    source_hashes: dict[str, str]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensorflight_source_hashes() -> dict[str, str]:
    paths = {
        "scenario_1": PACKAGE_ROOT
        / "envs"
        / "scenario_parameters"
        / "scenario_1_parameters.yaml",
        "active_flight": Path(active_flight_module.__file__).resolve(),
        "official_balloon_world": Path(balloon_world_module.__file__).resolve(),
        "rocket_oracle": MODULE_ROOT / "_rocket_oracle.py",
        "rocket": MODULE_ROOT / "rocket.py",
        "balloons": MODULE_ROOT / "balloons.py",
        "effects": MODULE_ROOT / "effects.py",
        "environment": MODULE_ROOT / "environment.py",
        "balloon_oracle": MODULE_ROOT / "_balloon_oracle.py",
        "factory": MODULE_ROOT / "factory.py",
        "observations": MODULE_ROOT / "observations.py",
        "ppo": MODULE_ROOT / "ppo.py",
        "trainer": MODULE_ROOT / "trainer.py",
        "numpy_deployment_agent": PACKAGE_ROOT
        / "agents"
        / "numpy_tensorflight_agent.py",
    }
    return {name: _sha256(path) for name, path in paths.items()}


def build_scenario1_source(*, seed: int = 2111) -> TensorFlightSource:
    """Extract immutable Scenario 1 data before entering the tensor hot path."""
    oracle = build_scenario1_oracle(seed=seed)
    try:
        balloon = build_canonical_balloon_fixture(seed=seed, num_balloons=1)
        scenario, _ = load_scenario_parameters(1)
        phase = oracle.flight.flight_phases[oracle.flight._step_state["phase_index"]]
        actuator_specs, demand_rate = scenario1_actuator_specs(oracle.flight)
        control = scenario["rocket"]["control"]
        throttle_range = control["throttle_range"]
        return TensorFlightSource(
            rocket_model_data=oracle.model_data,
            balloon_model_data=balloon.model_data,
            initial_rocket_state=np.asarray(
                oracle.flight.y_sol, dtype=np.float64
            ).copy(),
            initial_cached_rhs=np.asarray(phase.solver.f, dtype=np.float64).copy(),
            initial_actuator_output=canonical_actuator_outputs(oracle.flight).copy(),
            actuator_specs=actuator_specs,
            actuator_demand_rate=demand_rate,
            rocket_elapsed=float(oracle.flight.t - oracle.launch_time),
            start_step=int(oracle.env.current_step),
            sensor_config=SensorEffectConfig.from_mapping(
                scenario["rocket"]["sensors"]
            ),
            dt=float(scenario["simulation"]["time_step"]),
            max_time=float(scenario["simulation"]["max_time"]),
            elevation=float(scenario["environment"]["elevation"]),
            balloon_radius=float(scenario["balloon"]["radius"]),
            num_balloons=int(scenario["balloon"]["num"]),
            release_interval=float(scenario["balloon"]["release_interval"]),
            max_roll_torque=float(control["max_roll_torque"]),
            max_gimbal_angle=float(control["max_gimbal_angle"]),
            throttle_low=float(throttle_range[0]),
            throttle_high=float(throttle_range[1]),
            source_hashes=tensorflight_source_hashes(),
        )
    finally:
        oracle.close()


def make_tensorflight_environment(
    source: TensorFlightSource,
    num_envs: int,
    *,
    seed: int,
    device: str | torch.device,
    dtype: torch.dtype = torch.float32,
    critical_distance_float64: bool = False,
) -> TensorFlightEnvironment:
    target = torch.device(device)
    sampler = Scenario1BalloonSampler(
        source.num_balloons,
        seed=seed,
        release_interval=source.release_interval,
        dt=source.dt,
        elevation=source.elevation,
        device=target,
        dtype=dtype,
    )
    parameters = sampler(num_envs)
    rocket_state = torch.as_tensor(
        source.initial_rocket_state, device=target, dtype=dtype
    ).expand(num_envs, -1)
    cached_rhs = torch.as_tensor(
        source.initial_cached_rhs, device=target, dtype=dtype
    ).expand(num_envs, -1)
    environment = TensorFlightEnvironment(
        source.rocket_model_data,
        source.balloon_model_data,
        parameters,
        rocket_state,
        source.actuator_specs,
        actuator_demand_rate=source.actuator_demand_rate,
        rocket_elapsed=source.rocket_elapsed,
        start_step=source.start_step,
        initial_cached_rhs=cached_rhs,
        sensor_config=source.sensor_config,
        balloon_reset_sampler=sampler,
        config=TensorFlightEnvironmentConfig(
            dt=source.dt,
            max_time=source.max_time,
            elevation=source.elevation,
            balloon_radius=source.balloon_radius,
            rocket_substeps=1,
            balloon_substeps=1,
            integrator="rk4",
            critical_distance_float64=critical_distance_float64,
        ),
        device=target,
        dtype=dtype,
        seed=seed + 1,
    )
    environment.actuators.output.copy_(
        torch.as_tensor(source.initial_actuator_output, device=target, dtype=dtype)
    )
    return environment
