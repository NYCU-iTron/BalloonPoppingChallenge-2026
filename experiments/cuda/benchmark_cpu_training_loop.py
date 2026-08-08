"""Benchmark synchronous CPU E2E rollout collection with per-step IPC.

Unlike ``benchmark_cpu_parallel.py``, every parent step performs a batched MLP
forward pass, sends one post-launch action to every child, receives one 29-D
observation from every child, and waits for the complete worker barrier.  It is
still not a full PPO-update benchmark, but it measures the hot rollout boundary
that a SubprocVecEnv-style architecture must cross every 0.01 seconds.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import statistics
import time
import traceback
import warnings
from dataclasses import dataclass
from multiprocessing.connection import Connection

import numpy as np

from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters
from experiments.cuda.agent_contracts import (
    AgentObservation,
    EstimatedAltitudeHandoff,
    HistoricalBootstrapController,
    HistoricalE2EObservationBuilder,
    HistoricalSensorEstimator,
    PostLaunchAction,
)


@dataclass(frozen=True)
class RolloutBenchmark:
    workers: int
    steps: int
    wall_seconds: float

    @property
    def transitions_per_second(self) -> float:
        return self.workers * self.steps / self.wall_seconds

    @property
    def barriers_per_second(self) -> float:
        return self.steps / self.wall_seconds


class FixedBatchPolicy:
    """Deterministic 29→256→256→4 NumPy MLP used only for timing."""

    def __init__(self, *, seed: int = 2026) -> None:
        generator = np.random.default_rng(seed)
        self.weight_1 = generator.normal(0.0, 0.02, size=(29, 256))
        self.bias_1 = np.zeros(256)
        self.weight_2 = generator.normal(0.0, 0.02, size=(256, 256))
        self.bias_2 = np.zeros(256)
        self.weight_out = generator.normal(0.0, 0.02, size=(256, 4))
        # Bias throttle towards full so timing episodes remain airborne.
        self.bias_out = np.array([0.0, 0.0, 0.0, 4.0])

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        hidden = np.tanh(observation @ self.weight_1 + self.bias_1)
        hidden = np.tanh(hidden @ self.weight_2 + self.bias_2)
        return np.tanh(hidden @ self.weight_out + self.bias_out)


def scale_normalized_action(
    normalized: np.ndarray,
    control: dict[str, object],
) -> PostLaunchAction:
    action = np.clip(np.asarray(normalized, dtype=np.float64), -1.0, 1.0)
    if action.shape != (4,):
        raise ValueError("normalized post-launch action must have shape (4,)")
    throttle_low, throttle_high = control["throttle_range"]
    throttle = float(throttle_low) + 0.5 * (action[3] + 1.0) * (
        float(throttle_high) - float(throttle_low)
    )
    return PostLaunchAction(
        roll=action[0] * float(control["max_roll_torque"]),
        tvc=action[1:3] * float(control["max_gimbal_angle"]),
        throttle=throttle,
    )


def _launch_action(launch: bool, throttle: float) -> dict[str, object]:
    return {
        "launch": launch,
        "launch_inclination_heading": np.array([90.0, 0.0]),
        "tvc": np.zeros(2),
        "roll": 0.0,
        "throttle": throttle,
    }


def _start_episode(
    env: BalloonPoppingEnv,
    estimator: HistoricalSensorEstimator,
    builder: HistoricalE2EObservationBuilder,
    *,
    seed: int,
) -> np.ndarray:
    observation, _ = env.reset(seed=seed)
    estimator.reset(np.array([90.0, 0.0]))
    bootstrap = HistoricalBootstrapController(sampling_rate=1.0 / estimator.dt)
    handoff = EstimatedAltitudeHandoff(
        ground_elevation=estimator.ground_elevation,
        altitude_agl=40.0,
    )
    observation, *_ = env.step(_launch_action(False, 0.0))
    observation, *_ = env.step(_launch_action(True, 1.0))
    for _ in range(5_000):
        observation, _, terminated, truncated, _ = env.step(
            _launch_action(True, 1.0)
            if not np.isfinite(observation["rocket_sensors"]).all()
            else bootstrap.compute(
                AgentObservation.from_official(observation),
                estimator.estimate,
            ).as_official_action(np.array([90.0, 0.0]))
        )
        if terminated or truncated:
            raise RuntimeError("bootstrap flight ended before the 40 m handoff")
        agent_observation = AgentObservation.from_official(observation)
        estimate = estimator.update(agent_observation)
        if handoff.controller_active(agent_observation, estimate):
            break
    else:
        raise RuntimeError("bootstrap controller did not reach the 40 m handoff")
    target = agent_observation.balloon_states[0]
    return builder.build(estimate, target)


def _worker(
    connection: Connection,
    scenario: int,
    balloons: int,
    seed: int,
) -> None:
    env: BalloonPoppingEnv | None = None
    try:
        # Rapid policy commands intentionally exercise the official actuator
        # limiter.  Per-step warnings would dominate the IPC measurement.
        warnings.filterwarnings(
            "ignore",
            message=r"Actuator .* output change .* exceeds rate limit.*",
        )
        warnings.filterwarnings(
            "ignore",
            message=r"The Sensor class .* experimental development.*",
        )
        warnings.filterwarnings(
            "ignore",
            message=r"The following arguments have no effect for a chosen solver.*",
        )
        parameters, _ = load_scenario_parameters(scenario)
        parameters["balloon"]["num"] = balloons
        sensors = parameters["rocket"]["sensors"]
        control = parameters["rocket"]["control"]
        env = BalloonPoppingEnv(render_mode=None, parameters=parameters)
        estimator = HistoricalSensorEstimator(
            sampling_rate=float(sensors["sampling_rate"]),
            ground_elevation=float(parameters["environment"]["elevation"]),
        )
        builder = HistoricalE2EObservationBuilder()
        episode = 0
        initial = _start_episode(env, estimator, builder, seed=seed)
        connection.send((True, initial))

        while _run_episode(
            connection,
            env,
            estimator,
            builder,
            control,
            next_seed=seed + episode + 1,
        ):
            episode += 1
    except BaseException as error:
        try:
            connection.send((False, f"{error}\n{traceback.format_exc()}"))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if env is not None:
            env.close()
        connection.close()


def _run_episode(
    connection: Connection,
    env: BalloonPoppingEnv,
    estimator: HistoricalSensorEstimator,
    builder: HistoricalE2EObservationBuilder,
    control: dict[str, object],
    *,
    next_seed: int,
) -> bool:
    """Serve one episode, then reset before returning the terminal response."""
    terminated = False
    truncated = False
    while not (terminated or truncated):
        normalized = connection.recv()
        if normalized is None:
            return False
        physical = scale_normalized_action(normalized, control)
        observation, _, terminated, truncated, _ = env.step(
            physical.as_official_action(np.array([90.0, 0.0]))
        )
        if terminated or truncated:
            next_observation = _start_episode(
                env,
                estimator,
                builder,
                seed=next_seed,
            )
        else:
            agent_observation = AgentObservation.from_official(observation)
            estimate = estimator.update(agent_observation)
            next_observation = builder.build(
                estimate,
                agent_observation.balloon_states[0],
                previous_action=physical,
            )
        connection.send((True, next_observation))
    return True


def _receive(connection: Connection) -> np.ndarray:
    success, payload = connection.recv()
    if not success:
        raise RuntimeError(f"rollout worker failed:\n{payload}")
    return np.asarray(payload, dtype=np.float32)


def benchmark_worker_count(
    workers: int,
    *,
    scenario: int = 0,
    balloons: int = 1,
    steps: int = 300,
    warmup_steps: int = 10,
) -> RolloutBenchmark:
    if workers < 1 or balloons < 1 or steps < 1 or warmup_steps < 0:
        raise ValueError("workers, balloons and steps must be positive")
    context = mp.get_context("spawn")
    processes: list[mp.Process] = []
    parents: list[Connection] = []
    for worker_index in range(workers):
        parent, child = context.Pipe(duplex=True)
        process = context.Process(
            target=_worker,
            args=(child, scenario, balloons, 10_000 + worker_index),
        )
        process.start()
        child.close()
        parents.append(parent)
        processes.append(process)

    policy = FixedBatchPolicy()
    try:
        observations = np.stack([_receive(parent) for parent in parents])

        def advance() -> None:
            nonlocal observations
            actions = policy(observations)
            for parent, action in zip(parents, actions):
                parent.send(action)
            observations = np.stack([_receive(parent) for parent in parents])

        for _ in range(warmup_steps):
            advance()
        start = time.perf_counter()
        for _ in range(steps):
            advance()
        wall_seconds = time.perf_counter() - start
    finally:
        for parent in parents:
            try:
                parent.send(None)
            except (BrokenPipeError, EOFError, OSError):
                pass
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        for parent in parents:
            parent.close()

    return RolloutBenchmark(
        workers=workers,
        steps=steps,
        wall_seconds=wall_seconds,
    )


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
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument(
        "--workers",
        type=parse_worker_counts,
        default=parse_worker_counts("1,2,4,8,12,16,20"),
    )
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")

    print("workers  transitions/s median [min, max]  barriers/s median")
    for workers in args.workers:
        samples = [
            benchmark_worker_count(
                workers,
                scenario=args.scenario,
                balloons=args.balloons,
                steps=args.steps,
                warmup_steps=args.warmup_steps,
            )
            for _ in range(args.repeats)
        ]
        rates = [sample.transitions_per_second for sample in samples]
        barriers = [sample.barriers_per_second for sample in samples]
        print(
            f"{workers:7d}  {statistics.median(rates):20.1f} "
            f"[{min(rates):.1f}, {max(rates):.1f}]  "
            f"{statistics.median(barriers):17.1f}"
        )


if __name__ == "__main__":
    main()
