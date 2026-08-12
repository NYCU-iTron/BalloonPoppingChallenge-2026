"""Evaluate a deployment artifact in the unmodified official environment."""

from __future__ import annotations

import contextlib
import io
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from BalloonPoppingGymEnv.agents.numpy_tensorflight_agent import TensorFlightAgent
from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters


@dataclass(frozen=True)
class OfficialEvaluationResult:
    """One deterministic policy rollout in ActiveRocketPy."""

    scenario: int
    seed: int
    score: int
    hit_indices: tuple[int, ...]
    steps: int
    final_time: float
    terminated: bool
    truncated: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _evaluate_deployment(
    deployment_path: str | Path | None = None,
    *,
    scenario: int = 1,
    seeds: Iterable[int] = (0,),
    max_steps: int | None = None,
) -> list[OfficialEvaluationResult]:
    """Run a NumPy deployment in the official simulator for each seed.

    No TensorFlight dynamics participate in this function.  A non-zero score
    therefore proves the exported policy can hit a balloon in ActiveRocketPy,
    though it is not by itself a competitive-score claim.
    """

    results: list[OfficialEvaluationResult] = []
    for seed_value in seeds:
        seed = int(seed_value)
        parameters, given_parameters = load_scenario_parameters(scenario)
        parameters["scenario"]["random_seed"] = seed
        environment = BalloonPoppingEnv(render_mode=None, parameters=parameters)
        agent = TensorFlightAgent(given_parameters, artifact_path=deployment_path)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                observation, info = environment.reset(seed=seed)
            limit = (
                int(
                    float(parameters["simulation"]["max_time"])
                    / float(parameters["simulation"]["time_step"])
                )
                + 2
                if max_steps is None
                else int(max_steps)
            )
            if limit < 1:
                raise ValueError("max_steps must be positive")
            terminated = truncated = False
            steps = 0
            while not (terminated or truncated):
                if steps >= limit:
                    raise RuntimeError(
                        f"official evaluation exceeded {limit} steps without ending"
                    )
                action = agent.get_action(observation)
                observation, _, terminated, truncated, info = environment.step(action)
                steps += 1
            status = np.asarray(observation["balloon_status"]).reshape(-1)
            hit_indices = tuple(int(index) for index in np.flatnonzero(status == 2))
            results.append(
                OfficialEvaluationResult(
                    scenario=scenario,
                    seed=seed,
                    score=int(info["popped_count"]),
                    hit_indices=hit_indices,
                    steps=steps,
                    final_time=float(observation["simulation_time"]),
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                )
            )
        finally:
            environment.close()
    return results


def evaluate_deployment(
    deployment_path: str | Path | None = None,
    *,
    scenario: int = 1,
    seeds: Iterable[int] = (0,),
    max_steps: int | None = None,
) -> list[OfficialEvaluationResult]:
    """Run a deployment in ActiveRocketPy while silencing known library chatter."""

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=r"Actuator .* output change .* exceeds rate limit.*"
        )
        warnings.filterwarnings("ignore", message=r"The Sensor class .*experimental.*")
        warnings.filterwarnings(
            "ignore", message=r"The following arguments have no effect.*"
        )
        warnings.filterwarnings(
            "ignore", message=r"Exact chosen launch time is not available.*"
        )
        warnings.filterwarnings(
            "ignore", message=r"This class is still under testing.*"
        )
        return _evaluate_deployment(
            deployment_path,
            scenario=scenario,
            seeds=seeds,
            max_steps=max_steps,
        )
