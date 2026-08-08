"""Profile the canonical ActiveRocketPy stepping path used by the challenge.

This script deliberately uses the official environment unchanged.  Scenario 0
with one static balloon isolates rocket initialization and ``step_simulation``
from Scenario 1's balloon Monte Carlo reset cost.  It is a measurement tool,
not a CUDA implementation.

Examples
--------
    .venv/Scripts/python experiments/cuda/profile_official_env.py
    .venv/Scripts/python experiments/cuda/profile_official_env.py --profile
    .venv/Scripts/python experiments/cuda/profile_official_env.py --scenario 1
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import statistics
import time
from dataclasses import dataclass

import numpy as np

from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters


@dataclass(frozen=True)
class Sample:
    reset_seconds: float
    launch_seconds: float
    step_seconds: float
    steps: int
    mean_rhs_evaluations: float

    @property
    def steps_per_second(self) -> float:
        return self.steps / self.step_seconds


def constant_action() -> dict[str, object]:
    """A stable-enough vertical command for measuring simulation throughput."""

    return {
        "launch": True,
        "launch_inclination_heading": np.array([90.0, 0.0]),
        "tvc": np.zeros(2),
        "roll": 0.0,
        "throttle": 1.0,
    }


def run_sample(scenario: int, balloons: int, requested_steps: int) -> Sample:
    parameters, _ = load_scenario_parameters(scenario)
    parameters["balloon"]["num"] = balloons

    dt = float(parameters["simulation"]["time_step"])
    # Give the requested measurement room to finish without changing dt.
    parameters["simulation"]["max_time"] = max(
        float(parameters["simulation"]["max_time"]),
        (requested_steps + 10) * dt,
    )

    env = BalloonPoppingEnv(render_mode=None, parameters=parameters)

    start = time.perf_counter()
    env.reset(seed=0)
    reset_seconds = time.perf_counter() - start

    action = constant_action()
    start = time.perf_counter()
    _, _, terminated, truncated, _ = env.step(action)
    launch_seconds = time.perf_counter() - start

    steps = 0
    start = time.perf_counter()
    for _ in range(requested_steps):
        if terminated or truncated:
            break
        _, _, terminated, truncated, _ = env.step(action)
        steps += 1
    step_seconds = time.perf_counter() - start

    evaluations = env._rocket_flight.function_evaluations_per_time_step  # noqa: SLF001
    mean_evaluations = statistics.fmean(evaluations) if evaluations else 0.0
    env.close()

    return Sample(
        reset_seconds=reset_seconds,
        launch_seconds=launch_seconds,
        step_seconds=step_seconds,
        steps=steps,
        mean_rhs_evaluations=mean_evaluations,
    )


def print_summary(samples: list[Sample]) -> None:
    def median(field: str) -> float:
        return statistics.median(getattr(sample, field) for sample in samples)

    rates = [sample.steps_per_second for sample in samples]
    print(f"repeats:                 {len(samples)}")
    print(f"median reset:            {median('reset_seconds'):.6f} s")
    print(f"median launch/init step: {median('launch_seconds'):.6f} s")
    print(f"median physics rate:     {statistics.median(rates):.1f} steps/s")
    print(
        "median RK RHS calls:    "
        f"{median('mean_rhs_evaluations'):.2f} evaluations/control-step"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=int, choices=(0, 1), default=0)
    parser.add_argument("--balloons", type=int, default=1)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--profile",
        action="store_true",
        help="print the top cumulative Python call sites for one extra sample",
    )
    args = parser.parse_args()

    if args.balloons < 1 or args.steps < 1 or args.repeats < 1:
        parser.error("balloons, steps and repeats must all be positive")

    samples = [
        run_sample(args.scenario, args.balloons, args.steps)
        for _ in range(args.repeats)
    ]
    print_summary(samples)

    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()
        run_sample(args.scenario, args.balloons, args.steps)
        profiler.disable()

        stream = io.StringIO()
        pstats.Stats(profiler, stream=stream).sort_stats("cumtime").print_stats(30)
        print("\nTop cumulative call sites")
        print(stream.getvalue())


if __name__ == "__main__":
    main()
