"""Measure a physics-only one-process-per-flight CPU scaling upper bound.

The historical ``e2e`` and ``improve-rl-navigator`` training scripts use 20
``SubprocVecEnv`` workers.  Every spawned process here owns one canonical
ActiveRocketPy environment and advances one flight, but it deliberately runs
all requested steps inside the child.  It therefore excludes the real
Stable-Baselines3 per-step action/observation IPC and worker barrier.  Use the
result to isolate physics scaling, not to select the final PPO worker count.

Process startup and environment initialization are warmed before timing, so
the reported rate is steady-state physics throughput rather than launch cost.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import statistics
import time

from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters

from experiments.cuda.profile_official_env import constant_action


_WORKER_ENV: BalloonPoppingEnv | None = None


def _initialize_worker(scenario: int, balloons: int, requested_steps: int) -> None:
    global _WORKER_ENV

    parameters, _ = load_scenario_parameters(scenario)
    parameters["balloon"]["num"] = balloons
    dt = float(parameters["simulation"]["time_step"])
    parameters["simulation"]["max_time"] = max(
        float(parameters["simulation"]["max_time"]),
        (requested_steps + 100) * dt,
    )

    _WORKER_ENV = BalloonPoppingEnv(render_mode=None, parameters=parameters)
    _WORKER_ENV.reset(seed=0)
    _WORKER_ENV.step(constant_action())


def _advance(requested_steps: int) -> tuple[int, float]:
    if _WORKER_ENV is None:
        raise RuntimeError("worker environment was not initialized")

    action = constant_action()
    completed = 0
    start = time.perf_counter()
    for _ in range(requested_steps):
        _, _, terminated, truncated, _ = _WORKER_ENV.step(action)
        completed += 1
        if terminated or truncated:
            break
    return completed, time.perf_counter() - start


def benchmark_worker_count(
    workers: int,
    scenario: int,
    balloons: int,
    requested_steps: int,
) -> tuple[float, float]:
    context = mp.get_context("spawn")
    with context.Pool(
        workers,
        initializer=_initialize_worker,
        initargs=(scenario, balloons, requested_steps),
    ) as pool:
        # Ensure every process has finished its expensive imports and launch.
        pool.map(_advance, [5] * workers, chunksize=1)

        start = time.perf_counter()
        results = pool.map(_advance, [requested_steps] * workers, chunksize=1)
        wall_seconds = time.perf_counter() - start

    total_steps = sum(result[0] for result in results)
    worker_rates = [steps / seconds for steps, seconds in results]
    return total_steps / wall_seconds, statistics.median(worker_rates)


def parse_worker_counts(raw: str) -> list[int]:
    try:
        counts = [int(item) for item in raw.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "workers must be comma-separated integers"
        ) from error
    if not counts or any(count < 1 for count in counts):
        raise argparse.ArgumentTypeError("all worker counts must be positive")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=int, choices=(0, 1), default=0)
    parser.add_argument("--balloons", type=int, default=1)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument(
        "--workers",
        type=parse_worker_counts,
        default=parse_worker_counts("1,2,4,8,12,16,20"),
    )
    args = parser.parse_args()

    if args.balloons < 1 or args.steps < 1:
        parser.error("balloons and steps must be positive")

    print("workers  aggregate steps/s  median worker steps/s")
    for workers in args.workers:
        aggregate_rate, median_worker_rate = benchmark_worker_count(
            workers,
            args.scenario,
            args.balloons,
            args.steps,
        )
        print(f"{workers:7d}  {aggregate_rate:17.1f}  {median_worker_rate:21.1f}")


if __name__ == "__main__":
    main()
