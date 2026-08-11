"""End-to-end Phase 3 tensor-environment dtype and device benchmark."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters
from experiments.cuda.phase2_oracle import (
    build_scenario1_oracle,
    canonical_actuator_outputs,
    scenario1_actuator_specs,
)
from experiments.cuda.phase3_balloon import sample_scenario1_balloon_parameters
from experiments.cuda.phase3_effects import SensorEffectConfig
from experiments.cuda.phase3_environment import (
    TensorFlightEnvironment,
    TensorFlightEnvironmentConfig,
)
from experiments.cuda.phase3_oracle import build_canonical_balloon_fixture


@dataclass(frozen=True)
class BenchmarkResult:
    device: str
    dtype: str
    critical_distance_float64: bool
    num_envs: int
    num_balloons: int
    steps: int
    repeats: int
    transitions_per_second_median: float
    transitions_per_second_min: float
    transitions_per_second_max: float
    peak_memory_mib: float | None


@dataclass(frozen=True)
class BenchmarkSource:
    rocket_model_data: dict[str, object]
    balloon_model_data: dict[str, object]
    rocket_state: np.ndarray
    cached_rhs: np.ndarray
    actuator_output: np.ndarray
    actuator_specs: tuple[object, ...]
    actuator_demand_rate: float
    rocket_elapsed: float
    sensor_config: SensorEffectConfig
    dt: float
    max_time: float
    elevation: float
    balloon_radius: float


def build_source(seed: int) -> BenchmarkSource:
    oracle = build_scenario1_oracle(seed=seed)
    try:
        balloon = build_canonical_balloon_fixture(seed=seed, num_balloons=1)
        phase = oracle.flight.flight_phases[oracle.flight._step_state["phase_index"]]
        specs, demand_rate = scenario1_actuator_specs(oracle.flight)
        scenario, _ = load_scenario_parameters(1)
        return BenchmarkSource(
            rocket_model_data=oracle.model_data,
            balloon_model_data=balloon.model_data,
            rocket_state=np.asarray(oracle.flight.y_sol, dtype=np.float64).copy(),
            cached_rhs=np.asarray(phase.solver.f, dtype=np.float64).copy(),
            actuator_output=canonical_actuator_outputs(oracle.flight).copy(),
            actuator_specs=tuple(specs),
            actuator_demand_rate=demand_rate,
            rocket_elapsed=float(oracle.flight.t - oracle.launch_time),
            sensor_config=SensorEffectConfig.from_mapping(
                scenario["rocket"]["sensors"]
            ),
            dt=balloon.dt,
            max_time=balloon.max_time,
            elevation=float(scenario["environment"]["elevation"]),
            balloon_radius=float(scenario["balloon"]["radius"]),
        )
    finally:
        oracle.close()


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_case(
    source: BenchmarkSource,
    *,
    device: str,
    dtype: torch.dtype,
    critical_distance_float64: bool,
    num_envs: int,
    num_balloons: int,
    steps: int,
    warmup_steps: int,
    repeats: int,
    seed: int,
) -> BenchmarkResult:
    target = torch.device(device)
    parameters = sample_scenario1_balloon_parameters(
        num_envs,
        num_balloons,
        seed=seed,
        dt=source.dt,
        elevation=source.elevation,
        device=target,
        dtype=dtype,
    )
    rocket_state = torch.as_tensor(
        source.rocket_state, device=target, dtype=dtype
    ).expand(num_envs, -1)
    cached_rhs = torch.as_tensor(source.cached_rhs, device=target, dtype=dtype).expand(
        num_envs, -1
    )
    environment = TensorFlightEnvironment(
        source.rocket_model_data,
        source.balloon_model_data,
        parameters,
        rocket_state,
        source.actuator_specs,
        actuator_demand_rate=source.actuator_demand_rate,
        rocket_elapsed=source.rocket_elapsed,
        start_step=3,
        initial_cached_rhs=cached_rhs,
        sensor_config=source.sensor_config,
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
        seed=seed,
    )
    environment.actuators.output.copy_(
        torch.as_tensor(source.actuator_output, device=target, dtype=dtype)
    )
    generator = torch.Generator(device=target).manual_seed(seed + 1)
    action_bank = torch.rand(
        (max(steps, warmup_steps), num_envs, 4),
        generator=generator,
        device=target,
        dtype=dtype,
    )
    action_bank[..., 0] = (action_bank[..., 0] * 2 - 1) * 10
    action_bank[..., 1:3] = (action_bank[..., 1:3] * 2 - 1) * 15

    for index in range(warmup_steps):
        environment.step(action_bank[index])
    _synchronize(target)
    if target.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target)

    samples: list[float] = []
    for repeat in range(repeats):
        environment.reset()
        _synchronize(target)
        started = time.perf_counter()
        for index in range(steps):
            environment.step(action_bank[index])
        _synchronize(target)
        elapsed = time.perf_counter() - started
        samples.append(num_envs * steps / elapsed)
    peak_memory = (
        torch.cuda.max_memory_allocated(target) / 2**20
        if target.type == "cuda"
        else None
    )
    return BenchmarkResult(
        device=str(target),
        dtype=str(dtype).removeprefix("torch."),
        critical_distance_float64=critical_distance_float64,
        num_envs=num_envs,
        num_balloons=num_balloons,
        steps=steps,
        repeats=repeats,
        transitions_per_second_median=statistics.median(samples),
        transitions_per_second_min=min(samples),
        transitions_per_second_max=max(samples),
        peak_memory_mib=peak_memory,
    )


def build_report(
    *,
    batch_sizes: list[int],
    num_balloons: int,
    steps: int,
    warmup_steps: int,
    repeats: int,
    seed: int,
    include_float64: bool,
) -> dict[str, object]:
    source = build_source(seed)
    cases: list[tuple[str, torch.dtype, bool]] = [("cpu", torch.float32, False)]
    if torch.cuda.is_available():
        cases.extend(
            (
                ("cuda", torch.float32, False),
                ("cuda", torch.float32, True),
            )
        )
        if include_float64:
            cases.append(("cuda", torch.float64, False))
    results = [
        benchmark_case(
            source,
            device=device,
            dtype=dtype,
            critical_distance_float64=critical,
            num_envs=batch_size,
            num_balloons=num_balloons,
            steps=steps,
            warmup_steps=warmup_steps,
            repeats=repeats,
            seed=seed,
        )
        for batch_size in batch_sizes
        for device, dtype, critical in cases
    ]
    gate_batch = max(batch_sizes)
    cpu = next(
        result
        for result in results
        if result.num_envs == gate_batch and result.device == "cpu"
    )
    cuda_candidates = [
        result
        for result in results
        if result.num_envs == gate_batch
        and result.device.startswith("cuda")
        and result.dtype == "float32"
    ]
    if cuda_candidates:
        best_cuda = max(
            cuda_candidates, key=lambda item: item.transitions_per_second_median
        )
        speedup = (
            best_cuda.transitions_per_second_median / cpu.transitions_per_second_median
        )
        gate_passed: bool | None = speedup >= 2.0
        fallback = "gpu_tensorflight" if gate_passed else "cpu_tensorflight"
        throughput_winner = {
            "dtype": best_cuda.dtype,
            "critical_distance_float64": best_cuda.critical_distance_float64,
        }
    else:
        speedup = None
        gate_passed = None
        fallback = "cpu_tensorflight"
        throughput_winner = None
    return {
        "schema_version": 1,
        "scenario": 1,
        "num_balloons": num_balloons,
        "gate_batch": gate_batch,
        "results": [asdict(result) for result in results],
        "cuda_speedup_vs_tensor_cpu": speedup,
        "cuda_2x_gate_passed": gate_passed,
        "throughput_winner": throughput_winner,
        "fallback": fallback,
        "notes": [
            "Environment transitions include rocket, online balloons, sensors, "
            "actuators, swept geometry, reward, and termination.",
            "The throughput winner is valid only among candidates that pass the "
            "separate canonical fidelity report.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", default="64,512,2048")
    parser.add_argument("--num-balloons", type=int, default=100)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2061)
    parser.add_argument("--skip-float64", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    batch_sizes = [int(value) for value in args.batch_sizes.split(",")]
    if any(value < 1 for value in batch_sizes):
        raise ValueError("batch sizes must be positive")
    report = build_report(
        batch_sizes=batch_sizes,
        num_balloons=args.num_balloons,
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        repeats=args.repeats,
        seed=args.seed,
        include_float64=not args.skip_float64,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
