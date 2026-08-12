"""Holdout transfer gates against the unmodified official simulator."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import torch

from BalloonPoppingGymEnv.agents.numpy_tensorflight_agent import (
    NumpyTensorFlightAgent,
)
from BalloonPoppingGymEnv.agents.base_agent import BaseAgent
from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters
from experiments.cuda.phase2_oracle import (
    canonical_actuator_outputs,
    extract_scenario1_model_data,
    scenario1_actuator_specs,
)
from experiments.cuda.phase3_effects import SensorEffectConfig
from experiments.cuda.phase3_environment import (
    TensorAgentObservation,
    TensorFlightEnvironment,
    TensorFlightEnvironmentConfig,
)
from experiments.cuda.phase3_oracle import (
    _distribution,
    build_canonical_balloon_fixture,
    generate_balloon_fidelity_report,
    generate_full_environment_report,
)
from experiments.cuda.phase4_factory import build_phase4_source
from experiments.cuda.phase4_ppo import PPOHyperparameters
from experiments.cuda.phase4_training import Phase4Trainer, Phase4TrainingConfig


Tensor = torch.Tensor
DEFAULT_HOLDOUT_SEEDS = (2201, 2203, 2207)


@dataclass(frozen=True)
class TransferGates:
    balloon_position_p99_m: float = 0.05
    balloon_velocity_p99_m_s: float = 0.02
    closest_distance_p99_m: float = 0.05
    rocket_position_p99_m: float = 0.05
    sensor_component_p99: float = 0.05
    action_p99: float = 0.10
    termination_time_s: float = 0.05


def _action_vector(action: Mapping[str, object]) -> np.ndarray:
    tvc = np.asarray(action["tvc"], dtype=np.float64)
    return np.asarray(
        (float(action["roll"]), tvc[0], tvc[1], float(action["throttle"])),
        dtype=np.float64,
    )


def _tensor_observation(observation: TensorAgentObservation) -> dict[str, object]:
    return {
        "simulation_time": float(observation.simulation_time[0].item()),
        "balloon_status": observation.balloon_status[0].detach().cpu().numpy().copy(),
        "balloon_states": observation.balloon_states[0].detach().cpu().numpy().copy(),
        "rocket_sensors": observation.rocket_sensors[0].detach().cpu().numpy().copy(),
    }


def _finite_distribution(values: list[float]) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return _distribution(finite if finite.size else np.asarray([0.0]))


def _official_step(
    environment: BalloonPoppingEnv, action: Mapping[str, object]
) -> tuple[dict[str, object], float, bool, bool, dict[str, object]]:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=r"Actuator .* output change .* exceeds rate limit.*"
        )
        return environment.step(action)


def _create_policy_fixture(path: Path, *, seed: int) -> dict[str, object]:
    """Export a deterministic, explicitly untrained deployment smoke policy."""
    source = build_phase4_source(seed=seed)
    trainer = Phase4Trainer(
        source,
        Phase4TrainingConfig(
            num_envs=1,
            horizon=1,
            updates=1,
            hidden_size=32,
            handoff_altitude_agl=40.0,
            seed=seed,
            device="cpu",
            ppo=PPOHyperparameters(epochs=1, minibatch_size=1),
        ),
    )
    trainer.export_deployment(path)
    return {
        "kind": "deterministic_untrained_smoke_policy",
        "seed": seed,
        "hidden_size": 32,
        "handoff_altitude_agl": 40.0,
        "source_hashes": source.source_hashes,
    }


def _launch_to_anchor(
    official: BalloonPoppingEnv,
    official_agent: BaseAgent,
    tensor_agent: BaseAgent,
    observation: dict[str, object],
    *,
    max_steps: int,
) -> dict[str, object]:
    """Advance through reset/prelaunch/launch to the first finite flight sensor."""
    for _ in range(max_steps):
        official_action = official_agent.get_action(observation)
        tensor_action = tensor_agent.get_action(observation)
        np.testing.assert_allclose(
            _action_vector(official_action),
            _action_vector(tensor_action),
            atol=0,
            rtol=0,
        )
        observation, _, terminated, truncated, _ = _official_step(
            official, official_action
        )
        if terminated or truncated:
            raise RuntimeError("official flight ended before the transfer anchor")
        if (
            official._rocket_flight is not None
            and np.isfinite(observation["rocket_sensors"]).all()
        ):
            return observation
    raise RuntimeError("could not reach the first post-launch sensor observation")


def _build_exact_tensor_environment(
    official: BalloonPoppingEnv,
    balloon_fixture,
    scenario: Mapping[str, object],
    *,
    seed: int,
    dtype: torch.dtype,
    device: str | torch.device,
) -> TensorFlightEnvironment:
    flight = official._rocket_flight
    if flight is None:
        raise RuntimeError("official flight must be launched before tensor anchoring")
    phase = flight.flight_phases[flight._step_state["phase_index"]]
    actuator_specs, demand_rate = scenario1_actuator_specs(flight)
    sensor_config = SensorEffectConfig.from_mapping(scenario["rocket"]["sensors"])
    environment = TensorFlightEnvironment(
        extract_scenario1_model_data(flight),
        balloon_fixture.model_data,
        balloon_fixture.parameters,
        torch.as_tensor(np.asarray(flight.y_sol)[None], dtype=dtype, device=device),
        actuator_specs,
        actuator_demand_rate=demand_rate,
        rocket_elapsed=float(flight.t - flight.rocket.motor.burn_start_time),
        start_step=official.current_step,
        initial_cached_rhs=torch.as_tensor(
            np.asarray(phase.solver.f)[None], dtype=dtype, device=device
        ),
        sensor_config=sensor_config,
        config=TensorFlightEnvironmentConfig(
            dt=balloon_fixture.dt,
            max_time=balloon_fixture.max_time,
            elevation=float(scenario["environment"]["elevation"]),
            balloon_radius=float(scenario["balloon"]["radius"]),
            rocket_substeps=1,
            balloon_substeps=1,
            integrator="rk4",
        ),
        dtype=dtype,
        device=device,
        seed=seed,
    )
    environment.actuators.output.copy_(
        torch.as_tensor(
            canonical_actuator_outputs(flight)[None], dtype=dtype, device=device
        )
    )
    return environment


def generate_closed_loop_transfer_report(
    deployment_path: str | Path | None = None,
    *,
    seed: int,
    num_balloons: int = 100,
    max_steps: int = 2_000,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
    agent_factory: Callable[[dict[str, object]], BaseAgent] | None = None,
    policy_label: str = "deterministic_untrained_smoke_policy",
) -> dict[str, object]:
    """Run one policy independently on official and tensor observations."""
    scenario, given_parameters = load_scenario_parameters(1)
    scenario["balloon"]["num"] = num_balloons
    given_parameters["balloon"]["num"] = num_balloons
    balloon_fixture = build_canonical_balloon_fixture(
        seed=seed, num_balloons=num_balloons
    )
    official = BalloonPoppingEnv(render_mode=None, parameters=scenario)
    if agent_factory is None:
        if deployment_path is None:
            raise ValueError("deployment_path is required without agent_factory")

        def agent_factory(parameters: dict[str, object]) -> BaseAgent:
            return NumpyTensorFlightAgent(parameters, artifact_path=deployment_path)

    official_agent = agent_factory(given_parameters)
    tensor_agent = agent_factory(given_parameters)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            official_observation, _ = official.reset(seed=seed)
        official_observation = _launch_to_anchor(
            official,
            official_agent,
            tensor_agent,
            official_observation,
            max_steps=int(
                float(scenario["simulation"]["max_time"])
                / float(scenario["simulation"]["time_step"])
            )
            + 2,
        )
        tensor = _build_exact_tensor_environment(
            official,
            balloon_fixture,
            scenario,
            seed=seed,
            dtype=dtype,
            device=device,
        )
        tensor_observation = _tensor_observation(tensor.observation())

        rocket_position: list[float] = []
        rocket_velocity: list[float] = []
        rocket_attitude: list[float] = []
        sensor_errors: list[float] = []
        action_errors: list[float] = []
        closest_errors: list[float] = []
        official_hits: list[dict[str, object]] = []
        tensor_hits: list[dict[str, object]] = []
        official_score = 0
        tensor_score = 0
        official_terminated = official_truncated = False
        tensor_terminated = tensor_truncated = False
        official_end_time: float | None = None
        tensor_end_time: float | None = None
        compared = 0

        for _ in range(max_steps):
            official_active = not (official_terminated or official_truncated)
            tensor_active = not (tensor_terminated or tensor_truncated)
            both_active = official_active and tensor_active
            official_action = (
                official_agent.get_action(official_observation)
                if official_active
                else None
            )
            tensor_action = (
                tensor_agent.get_action(tensor_observation) if tensor_active else None
            )
            if both_active:
                official_command = _action_vector(official_action)
                tensor_command = _action_vector(tensor_action)
                action_errors.append(
                    float(np.max(np.abs(official_command - tensor_command)))
                )
                previous_rocket = np.asarray(official._rocket_states[:3]).copy()
                previous_balloons = official._balloon_states[:, :3].copy()
            else:
                previous_rocket = previous_balloons = None

            if official_active:
                previous_status = official._balloon_status[:, 0].copy()
                (
                    official_observation,
                    reward,
                    official_terminated,
                    official_truncated,
                    _,
                ) = _official_step(official, official_action)
                official_score += int(reward)
                time_value = float(official_observation["simulation_time"])
                official_new_hits = np.flatnonzero(
                    (previous_status != 2) & (official._balloon_status[:, 0] == 2)
                )
                for identity in official_new_hits:
                    official_hits.append(
                        {"identity": int(identity), "time": time_value}
                    )

            if tensor_active:
                tensor_command = _action_vector(tensor_action)
                tensor_step = tensor.step(
                    torch.as_tensor(tensor_command[None], dtype=dtype, device=device)
                )
                tensor_observation = _tensor_observation(tensor_step.observation)
                tensor_score += int(tensor_step.reward[0].item())
                for identity in (
                    torch.nonzero(tensor.hit_mask[0], as_tuple=False).flatten().tolist()
                ):
                    tensor_hits.append(
                        {
                            "identity": int(identity),
                            "time": float(tensor_observation["simulation_time"]),
                        }
                    )
                tensor_terminated = bool(tensor_step.terminated[0].item())
                tensor_truncated = bool(tensor_step.truncated[0].item())

            if both_active:
                actual_rocket = tensor.rocket_state[0].detach().cpu().numpy()
                expected_rocket = np.asarray(official._rocket_states, dtype=np.float64)
                rocket_position.append(
                    float(np.linalg.norm(actual_rocket[:3] - expected_rocket[:3]))
                )
                rocket_velocity.append(
                    float(np.linalg.norm(actual_rocket[3:6] - expected_rocket[3:6]))
                )
                q_actual = actual_rocket[6:10] / np.linalg.norm(actual_rocket[6:10])
                q_expected = expected_rocket[6:10] / np.linalg.norm(
                    expected_rocket[6:10]
                )
                rocket_attitude.append(
                    float(
                        np.rad2deg(
                            2 * np.arccos(np.clip(abs(q_actual @ q_expected), 0, 1))
                        )
                    )
                )
                sensor_errors.extend(
                    np.abs(
                        np.asarray(tensor_observation["rocket_sensors"])
                        - np.asarray(official_observation["rocket_sensors"])
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
                compared += 1
            if (
                official_terminated or official_truncated
            ) and official_end_time is None:
                official_end_time = float(official_observation["simulation_time"])
            if (tensor_terminated or tensor_truncated) and tensor_end_time is None:
                tensor_end_time = float(tensor_observation["simulation_time"])
            if (official_terminated or official_truncated) and (
                tensor_terminated or tensor_truncated
            ):
                break

        last_official_pop = official_hits[-1]["time"] if official_hits else None
        last_tensor_pop = tensor_hits[-1]["time"] if tensor_hits else None
        return {
            "schema_version": 1,
            "mode": "closed_loop_independent_observations",
            "policy": policy_label,
            "scenario": 1,
            "seed": seed,
            "num_balloons": num_balloons,
            "compared_steps": compared,
            "dtype": str(dtype).removeprefix("torch."),
            "device": str(device),
            "official_score": official_score,
            "tensor_score": tensor_score,
            "official_hits": official_hits,
            "tensor_hits": tensor_hits,
            "last_pop_time": {
                "official": last_official_pop,
                "tensor": last_tensor_pop,
            },
            "termination": {
                "official_terminated": bool(official_terminated),
                "official_truncated": bool(official_truncated),
                "tensor_terminated": tensor_terminated,
                "tensor_truncated": tensor_truncated,
                "official_time": official_end_time,
                "tensor_time": tensor_end_time,
            },
            "action_max_component_error": _finite_distribution(action_errors),
            "rocket_position_error_m": _finite_distribution(rocket_position),
            "rocket_velocity_error_m_s": _finite_distribution(rocket_velocity),
            "rocket_attitude_error_deg": _finite_distribution(rocket_attitude),
            "sensor_absolute_error": _finite_distribution(sensor_errors),
            "closest_distance_error_m": _finite_distribution(closest_errors),
            "hit_coverage": bool(official_hits or tensor_hits),
        }
    finally:
        official.close()


def evaluate_transfer_gates(
    balloon_only: list[dict[str, object]],
    open_loop: list[dict[str, object]],
    closed_loop: list[dict[str, object]],
    *,
    gates: TransferGates = TransferGates(),
) -> dict[str, object]:
    failures: list[str] = []
    for report in balloon_only:
        seed = report["seed"]
        if report["status_mismatches"] != 0:
            failures.append(f"balloon-only seed {seed}: status mismatch")
        if report["release_time_error_s"]["max"] != 0:
            failures.append(f"balloon-only seed {seed}: release-time mismatch")
        if report["position_error_m"]["p99"] > gates.balloon_position_p99_m:
            failures.append(f"balloon-only seed {seed}: position p99")
        if report["velocity_error_m_s"]["p99"] > gates.balloon_velocity_p99_m_s:
            failures.append(f"balloon-only seed {seed}: velocity p99")
    for report in open_loop:
        seed = report["seed"]
        if report["reward_mismatches"] != 0:
            failures.append(f"open-loop seed {seed}: reward mismatch")
        if report["status_mismatches"] != 0:
            failures.append(f"open-loop seed {seed}: status mismatch")
        if report["termination_mismatches"] != 0:
            failures.append(f"open-loop seed {seed}: termination mismatch")
        if report["closest_distance_error_m"]["p99"] > gates.closest_distance_p99_m:
            failures.append(f"open-loop seed {seed}: closest-distance p99")
        if report["rocket_position_error_m"]["p99"] > gates.rocket_position_p99_m:
            failures.append(f"open-loop seed {seed}: rocket-position p99")
        if report["sensor_absolute_error"]["p99"] > gates.sensor_component_p99:
            failures.append(f"open-loop seed {seed}: sensor p99")
    for report in closed_loop:
        seed = report["seed"]
        if report["official_score"] != report["tensor_score"]:
            failures.append(f"closed-loop seed {seed}: score mismatch")
        official_events = [event["identity"] for event in report["official_hits"]]
        tensor_events = [event["identity"] for event in report["tensor_hits"]]
        if official_events != tensor_events:
            failures.append(f"closed-loop seed {seed}: hit identity mismatch")
        if report["action_max_component_error"]["p99"] > gates.action_p99:
            failures.append(f"closed-loop seed {seed}: action p99")
        termination = report["termination"]
        for name in ("terminated", "truncated"):
            if termination[f"official_{name}"] != termination[f"tensor_{name}"]:
                failures.append(f"closed-loop seed {seed}: {name} mismatch")
        official_time = termination["official_time"]
        tensor_time = termination["tensor_time"]
        if official_time is None or tensor_time is None:
            failures.append(f"closed-loop seed {seed}: incomplete termination coverage")
        elif abs(official_time - tensor_time) > gates.termination_time_s:
            failures.append(f"closed-loop seed {seed}: termination time")
    hit_coverage = any(bool(report["hit_coverage"]) for report in closed_loop)
    coverage_failures = [] if hit_coverage else ["closed-loop hit coverage missing"]
    return {
        "passed": not failures and hit_coverage,
        "numerical_and_lifecycle_passed": not failures,
        "failures": failures,
        "coverage_failures": coverage_failures,
        "gates": {
            "balloon_position_p99_m": gates.balloon_position_p99_m,
            "balloon_velocity_p99_m_s": gates.balloon_velocity_p99_m_s,
            "closest_distance_p99_m": gates.closest_distance_p99_m,
            "rocket_position_p99_m": gates.rocket_position_p99_m,
            "sensor_component_p99": gates.sensor_component_p99,
            "action_p99": gates.action_p99,
            "termination_time_s": gates.termination_time_s,
        },
        "closed_loop_hit_coverage": hit_coverage,
    }


def generate_phase5_report(
    *,
    seeds: tuple[int, ...] = DEFAULT_HOLDOUT_SEEDS,
    closed_loop_seeds: tuple[int, ...] | None = None,
    num_balloons: int = 100,
    open_loop_steps: int = 1_024,
    closed_loop_steps: int = 2_000,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
) -> dict[str, object]:
    if not seeds:
        raise ValueError("at least one holdout seed is required")
    closed_seeds = seeds[:2] if closed_loop_seeds is None else closed_loop_seeds
    balloon_only = [
        generate_balloon_fidelity_report(
            seed=seed,
            num_balloons=num_balloons,
            substeps=1,
            dtype=dtype,
            device=device,
        )
        for seed in seeds
    ]
    open_loop = [
        generate_full_environment_report(
            seed=seed,
            num_balloons=num_balloons,
            steps=open_loop_steps,
            dtype=dtype,
            device=device,
            integrator="rk4",
        )
        for seed in seeds
    ]
    with tempfile.TemporaryDirectory(prefix="phase5_policy_") as directory:
        deployment = Path(directory) / "deployment.npz"
        policy_fixture = _create_policy_fixture(deployment, seed=seeds[0] + 50_000)
        closed_loop = [
            generate_closed_loop_transfer_report(
                deployment,
                seed=seed,
                num_balloons=num_balloons,
                max_steps=closed_loop_steps,
                dtype=dtype,
                device=device,
            )
            for seed in closed_seeds
        ]
    decision = evaluate_transfer_gates(balloon_only, open_loop, closed_loop)
    return {
        "schema_version": 1,
        "scenario": 1,
        "holdout_seeds": list(seeds),
        "closed_loop_seeds": list(closed_seeds),
        "num_balloons": num_balloons,
        "dtype": str(dtype).removeprefix("torch."),
        "device": str(device),
        "policy_fixture": policy_fixture,
        "balloon_only": balloon_only,
        "open_loop": open_loop,
        "closed_loop": closed_loop,
        "decision": decision,
    }


def _parse_seeds(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(",") if item)
    if not result:
        raise argparse.ArgumentTypeError("expected comma-separated integer seeds")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=_parse_seeds, default=DEFAULT_HOLDOUT_SEEDS)
    parser.add_argument("--closed-loop-seeds", type=_parse_seeds)
    parser.add_argument("--num-balloons", type=int, default=100)
    parser.add_argument("--open-loop-steps", type=int, default=1_024)
    parser.add_argument("--closed-loop-steps", type=int, default=2_000)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    report = generate_phase5_report(
        seeds=args.seeds,
        closed_loop_seeds=args.closed_loop_seeds,
        num_balloons=args.num_balloons,
        open_loop_steps=args.open_loop_steps,
        closed_loop_steps=args.closed_loop_steps,
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
