"""Command-line interface for TensorFlight training and official validation."""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from BalloonPoppingGymEnv.tensorflight.evaluation import evaluate_deployment

if TYPE_CHECKING:
    from BalloonPoppingGymEnv.tensorflight.trainer import TensorFlightTrainer


def _seeds(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "seeds must be comma-separated integers"
        ) from error
    if not result:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return result


def _training_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("train", help="train PPO in batched TensorFlight")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/tensorflight"))
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=65_536)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2121)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint-every", type=int, default=10)


def _evaluation_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "evaluate", help="evaluate deployment.npz in unmodified ActiveRocketPy"
    )
    parser.add_argument("artifact", nargs="?", type=Path)
    parser.add_argument("--scenario", type=int, choices=(0, 1), default=1)
    parser.add_argument("--seeds", type=_seeds, default=(0,))
    parser.add_argument("--max-steps", type=int)
    parser.add_argument(
        "--require-hit",
        action="store_true",
        help="return a failing exit status when every rollout scores zero",
    )


def _export_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "export", help="export deployment and self-contained agent from checkpoint"
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/tensorflight/export")
    )


def _selection_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "select",
        help="select a checkpoint by scores from the unmodified official simulator",
    )
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--scenario", type=int, choices=(0, 1), default=1)
    parser.add_argument("--seeds", type=_seeds, default=(0,))
    parser.add_argument("--max-steps", type=int)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/tensorflight/best")
    )
    parser.add_argument(
        "--require-hit",
        action="store_true",
        help="return a failing exit status if the selected policy never scores",
    )


def _trainer_from_checkpoint(path: Path) -> TensorFlightTrainer:
    import torch

    from BalloonPoppingGymEnv.tensorflight.factory import build_scenario1_source
    from BalloonPoppingGymEnv.tensorflight.ppo import PPOHyperparameters
    from BalloonPoppingGymEnv.tensorflight.trainer import (
        TensorFlightTrainer,
        TensorFlightTrainingConfig,
    )

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    values = dict(checkpoint["config"])
    values["ppo"] = PPOHyperparameters(**values["ppo"])
    config = TensorFlightTrainingConfig(**values)
    source = build_scenario1_source(seed=config.seed)
    trainer = TensorFlightTrainer(source, config)
    trainer.load_checkpoint(path)
    return trainer


def _train(args: argparse.Namespace) -> int:
    from BalloonPoppingGymEnv.tensorflight.factory import build_scenario1_source
    from BalloonPoppingGymEnv.tensorflight.ppo import PPOHyperparameters
    from BalloonPoppingGymEnv.tensorflight.trainer import (
        TensorFlightTrainer,
        TensorFlightTrainingConfig,
    )

    if args.checkpoint_every < 1:
        raise ValueError("checkpoint-every must be positive")
    output = args.output_dir.resolve()
    checkpoint_path = output / "training.ckpt"
    deployment_path = output / "deployment.npz"
    agent_path = output / "submission_agent.py"
    checkpoint_dir = output / "checkpoints"
    ppo = PPOHyperparameters(
        epochs=args.epochs,
        minibatch_size=args.minibatch_size,
    )
    config = TensorFlightTrainingConfig(
        num_envs=args.num_envs,
        horizon=args.horizon,
        updates=args.updates,
        seed=args.seed,
        device=args.device,
        ppo=ppo,
    )
    source = build_scenario1_source(seed=args.seed)
    trainer = TensorFlightTrainer(source, config)
    if args.resume is not None:
        trainer.load_checkpoint(args.resume)
    for _ in range(max(config.updates - trainer.update_index, 0)):
        metrics = trainer.train_update()
        print(json.dumps(asdict(metrics), sort_keys=True), flush=True)
        if trainer.update_index % args.checkpoint_every == 0:
            trainer.save_checkpoint(checkpoint_path)
            trainer.save_checkpoint(
                checkpoint_dir / f"update_{trainer.update_index:06d}.ckpt"
            )
    trainer.save_checkpoint(checkpoint_path)
    trainer.export_deployment(deployment_path)
    trainer.export_self_contained_agent(agent_path, deployment_path=deployment_path)
    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "deployment": str(deployment_path),
                "submission_agent": str(agent_path),
            },
            sort_keys=True,
        )
    )
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    results = evaluate_deployment(
        args.artifact,
        scenario=args.scenario,
        seeds=args.seeds,
        max_steps=args.max_steps,
    )
    payload = [result.as_dict() for result in results]
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.require_hit and not any(result.score > 0 for result in results):
        return 2
    return 0


def _export(args: argparse.Namespace) -> int:
    output = args.output_dir.resolve()
    deployment = output / "deployment.npz"
    agent = output / "submission_agent.py"
    trainer = _trainer_from_checkpoint(args.checkpoint)
    trainer.export_deployment(deployment)
    trainer.export_self_contained_agent(agent, deployment_path=deployment)
    print(json.dumps({"deployment": str(deployment), "submission_agent": str(agent)}))
    return 0


def _select(args: argparse.Namespace) -> int:
    candidates: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="tensorflight_selection_") as directory:
        temporary = Path(directory)
        for index, checkpoint in enumerate(args.checkpoints):
            trainer = _trainer_from_checkpoint(checkpoint)
            deployment = temporary / f"candidate_{index}.npz"
            trainer.export_deployment(deployment)
            results = evaluate_deployment(
                deployment,
                scenario=args.scenario,
                seeds=args.seeds,
                max_steps=args.max_steps,
            )
            scores = [result.score for result in results]
            candidate = {
                "checkpoint": str(checkpoint.resolve()),
                "update": trainer.update_index,
                "scores": scores,
                "mean_score": sum(scores) / len(scores),
                "max_score": max(scores),
                "results": [result.as_dict() for result in results],
            }
            candidates.append(candidate)
            print(json.dumps(candidate, sort_keys=True), flush=True)

    best = max(
        candidates,
        key=lambda candidate: (
            float(candidate["mean_score"]),
            int(candidate["max_score"]),
            -int(candidate["update"]),
        ),
    )
    output = args.output_dir.resolve()
    deployment = output / "deployment.npz"
    agent = output / "submission_agent.py"
    trainer = _trainer_from_checkpoint(Path(str(best["checkpoint"])))
    trainer.export_deployment(deployment)
    trainer.export_self_contained_agent(agent, deployment_path=deployment)
    report = {
        "scenario": args.scenario,
        "seeds": list(args.seeds),
        "selected": best,
        "candidates": candidates,
        "deployment": str(deployment),
        "submission_agent": str(agent),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "selection.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.require_hit and int(best["max_score"]) == 0:
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="balloon-tensorflight")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _training_parser(subparsers)
    _evaluation_parser(subparsers)
    _export_parser(subparsers)
    _selection_parser(subparsers)
    args = parser.parse_args(argv)
    if args.command == "train":
        return _train(args)
    if args.command == "evaluate":
        return _evaluate(args)
    if args.command == "export":
        return _export(args)
    if args.command == "select":
        return _select(args)
    raise AssertionError(f"unknown command: {args.command}")
