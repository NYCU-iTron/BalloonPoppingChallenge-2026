"""Benchmark the Phase 4 environment/learning grid without hiding null scores."""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import statistics
from dataclasses import asdict
from pathlib import Path

import torch

from experiments.cuda.phase4_factory import build_phase4_source
from experiments.cuda.phase4_ppo import PPOHyperparameters
from experiments.cuda.phase4_training import Phase4Trainer, Phase4TrainingConfig


def _integers(value: str) -> list[int]:
    result = [int(item) for item in value.split(",")]
    if not result or any(item < 1 for item in result):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return result


def _median_optional(values: list[float | None]) -> float | None:
    finite = [value for value in values if value is not None]
    return statistics.median(finite) if finite else None


def _run_case(
    source,
    *,
    num_envs: int,
    horizon: int,
    steps_per_env: int,
    repeat: int,
    seed: int,
    device: str,
    epochs: int,
    minibatch_size: int,
) -> dict[str, object]:
    updates = math.ceil(steps_per_env / horizon)
    config = Phase4TrainingConfig(
        num_envs=num_envs,
        horizon=horizon,
        updates=updates,
        seed=seed,
        device=device,
        ppo=PPOHyperparameters(epochs=epochs, minibatch_size=minibatch_size),
    )
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    trainer = Phase4Trainer(source, config)
    results = [trainer.train_update() for _ in range(updates)]
    total_transitions = updates * num_envs * horizon
    rollout_seconds = sum(result.rollout.seconds for result in results)
    total_seconds = sum(result.update_seconds for result in results)
    completed = sum(result.rollout.completed_episodes for result in results)
    score_weighted = sum(
        (result.rollout.mean_official_score or 0) * result.rollout.completed_episodes
        for result in results
    )
    shaped_weighted = sum(
        (result.rollout.mean_shaped_return or 0) * result.rollout.completed_episodes
        for result in results
    )
    ppo_samples = sum(result.ppo.samples for result in results)
    policy_reward_weighted = sum(
        (result.rollout.mean_policy_shaped_reward or 0) * result.ppo.samples
        for result in results
    )
    peak_memory = (
        torch.cuda.max_memory_allocated(device) if device.startswith("cuda") else None
    )
    result = {
        "num_envs": num_envs,
        "horizon": horizon,
        "steps_per_env": updates * horizon,
        "updates": updates,
        "repeat": repeat,
        "seed": seed,
        "total_transitions": total_transitions,
        "rollout_seconds": rollout_seconds,
        "total_seconds": total_seconds,
        "environment_transitions_per_second": total_transitions / rollout_seconds,
        "end_to_end_transitions_per_second": total_transitions / total_seconds,
        "ppo_samples_per_second": (
            ppo_samples / max(total_seconds - rollout_seconds, 1e-12)
        ),
        "ppo_samples": ppo_samples,
        "mean_policy_shaped_reward": (
            policy_reward_weighted / ppo_samples if ppo_samples else None
        ),
        "completed_episodes": completed,
        "mean_official_score": score_weighted / completed if completed else None,
        "mean_shaped_return": shaped_weighted / completed if completed else None,
        "reward_wallclock_auc": results[-1].reward_wallclock_auc,
        "shaped_return_wallclock_auc": results[-1].shaped_return_wallclock_auc,
        "policy_reward_wallclock_auc": results[-1].policy_reward_wallclock_auc,
        "time_to_score_seconds": results[-1].time_to_score_seconds,
        "peak_memory_bytes": peak_memory,
        "rollout_buffer_bytes": trainer.rollout.allocated_bytes,
        "final_ppo": asdict(results[-1].ppo),
    }
    del trainer
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def _aggregate(cases: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[int, int], list[dict[str, object]]] = {}
    for case in cases:
        key = (int(case["num_envs"]), int(case["horizon"]))
        grouped.setdefault(key, []).append(case)
    result: list[dict[str, object]] = []
    for (num_envs, horizon), repeats in sorted(grouped.items()):
        result.append(
            {
                "num_envs": num_envs,
                "horizon": horizon,
                "repeats": len(repeats),
                "environment_transitions_per_second_median": statistics.median(
                    float(item["environment_transitions_per_second"])
                    for item in repeats
                ),
                "end_to_end_transitions_per_second_median": statistics.median(
                    float(item["end_to_end_transitions_per_second"]) for item in repeats
                ),
                "mean_official_score_median": _median_optional(
                    [item["mean_official_score"] for item in repeats]  # type: ignore[list-item]
                ),
                "mean_shaped_return_median": _median_optional(
                    [item["mean_shaped_return"] for item in repeats]  # type: ignore[list-item]
                ),
                "mean_policy_shaped_reward_median": _median_optional(
                    [item["mean_policy_shaped_reward"] for item in repeats]  # type: ignore[list-item]
                ),
                "reward_wallclock_auc_median": statistics.median(
                    float(item["reward_wallclock_auc"]) for item in repeats
                ),
                "shaped_return_wallclock_auc_median": statistics.median(
                    float(item["shaped_return_wallclock_auc"]) for item in repeats
                ),
                "policy_reward_wallclock_auc_median": statistics.median(
                    float(item["policy_reward_wallclock_auc"]) for item in repeats
                ),
                "time_to_score_seconds_median": _median_optional(
                    [item["time_to_score_seconds"] for item in repeats]  # type: ignore[list-item]
                ),
                "completed_episodes": sum(
                    int(item["completed_episodes"]) for item in repeats
                ),
                "ppo_samples": sum(int(item["ppo_samples"]) for item in repeats),
                "peak_memory_bytes_max": max(
                    (
                        int(item["peak_memory_bytes"])
                        for item in repeats
                        if item["peak_memory_bytes"] is not None
                    ),
                    default=None,
                ),
            }
        )
    return result


def _decision(aggregate: list[dict[str, object]]) -> dict[str, object]:
    reached = [
        case for case in aggregate if case["time_to_score_seconds_median"] is not None
    ]
    if reached:
        selected = min(
            reached, key=lambda case: float(case["time_to_score_seconds_median"])
        )
        return {
            "status": "selected_by_time_to_score",
            "num_envs": selected["num_envs"],
            "horizon": selected["horizon"],
        }
    completed = [case for case in aggregate if int(case["completed_episodes"]) > 0]
    if completed:
        selected = max(
            completed,
            key=lambda case: (
                float(case["mean_official_score_median"] or -math.inf),
                float(case["shaped_return_wallclock_auc_median"]),
            ),
        )
        return {
            "status": "provisional_by_score_then_shaped_auc",
            "num_envs": selected["num_envs"],
            "horizon": selected["horizon"],
            "warning": "No case reached the requested score threshold.",
        }
    active = [case for case in aggregate if int(case["ppo_samples"]) > 0]
    if active:
        reward_reference = max(
            active,
            key=lambda case: float(case["policy_reward_wallclock_auc_median"]),
        )
        fastest = max(
            active,
            key=lambda case: float(case["end_to_end_transitions_per_second_median"]),
        )
        return {
            "status": "learning_signal_inconclusive",
            "throughput_reference_num_envs": fastest["num_envs"],
            "throughput_reference_horizon": fastest["horizon"],
            "policy_reward_reference_num_envs": reward_reference["num_envs"],
            "policy_reward_reference_horizon": reward_reference["horizon"],
            "warning": (
                "Policy updates ran, but no episode completed. Neither throughput "
                "nor a single short-run shaped-reward AUC selects a PPO setting."
            ),
        }
    fastest = max(
        aggregate,
        key=lambda case: float(case["end_to_end_transitions_per_second_median"]),
    )
    return {
        "status": "learning_signal_inconclusive",
        "throughput_reference_num_envs": fastest["num_envs"],
        "throughput_reference_horizon": fastest["horizon"],
        "warning": (
            "No completed episode was observed; throughput alone must not select "
            "the PPO configuration. Increase --steps-per-env."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-envs", type=_integers, default=_integers("512,1024,2048,4096")
    )
    parser.add_argument("--horizons", type=_integers, default=_integers("16,32,64,128"))
    # The historical sensor-derived 40 m handoff occurs around step 583 in
    # Scenario 1, so the default must extend past it to exercise PPO.
    parser.add_argument("--steps-per-env", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--minibatch-size", type=int, default=65_536)
    parser.add_argument("--seed", type=int, default=2151)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.steps_per_env < 1 or args.repeats < 1:
        parser.error("steps-per-env and repeats must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")

    source = build_phase4_source(seed=args.seed)
    cases: list[dict[str, object]] = []
    for num_envs in args.num_envs:
        for horizon in args.horizons:
            for repeat in range(args.repeats):
                case = _run_case(
                    source,
                    num_envs=num_envs,
                    horizon=horizon,
                    steps_per_env=args.steps_per_env,
                    repeat=repeat,
                    # Keep policy/environment RNG seeds aligned across the
                    # grid. Different B shapes consume different amounts, but
                    # horizon must not silently change the experiment seed.
                    seed=args.seed + 10_000 * repeat,
                    device=args.device,
                    epochs=args.epochs,
                    minibatch_size=args.minibatch_size,
                )
                cases.append(case)
                print(json.dumps(case, sort_keys=True))
    aggregate = _aggregate(cases)
    report = {
        "schema_version": 1,
        "device": args.device,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "platform": platform.platform(),
        "requested_steps_per_env": args.steps_per_env,
        "epochs": args.epochs,
        "minibatch_size": args.minibatch_size,
        "cases": cases,
        "aggregate": aggregate,
        "decision": _decision(aggregate),
        "source_hashes": source.source_hashes,
    }
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
