"""Release-surface tests for TensorFlight users."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

from BalloonPoppingGymEnv.agents import TensorFlightAgent
from BalloonPoppingGymEnv.tensorflight import cli as tensorflight_cli
from BalloonPoppingGymEnv.tensorflight.cli import main
from BalloonPoppingGymEnv.tensorflight.evaluation import (
    OfficialEvaluationResult,
    evaluate_deployment,
)


ROOT = Path(__file__).resolve().parents[1]
PACKAGED_DEPLOYMENT = ROOT / "BalloonPoppingGymEnv" / "agents" / "deployment.npz"


def test_packaged_deployment_is_available_without_an_experiment_path() -> None:
    assert PACKAGED_DEPLOYMENT.is_file()
    assert PACKAGED_DEPLOYMENT.stat().st_size > 100_000
    assert TensorFlightAgent.__module__.startswith("BalloonPoppingGymEnv.agents")
    with np.load(PACKAGED_DEPLOYMENT, allow_pickle=False) as artifact:
        metadata = json.loads(str(artifact["metadata_json"]))
    assert metadata["training_update"] == 80
    assert metadata["global_transitions"] == 20_971_520
    assert metadata["training_num_envs"] == 4096


def test_cli_help_exposes_train_export_and_official_evaluation(capsys) -> None:
    for command in ("train", "export", "evaluate", "select"):
        try:
            main([command, "--help"])
        except SystemExit as error:
            assert error.code == 0
        assert command in capsys.readouterr().out


def test_checkpoint_selection_uses_official_scores_and_prefers_winner(
    monkeypatch, tmp_path: Path
) -> None:
    class FakeTrainer:
        def __init__(self, update: int) -> None:
            self.update_index = update

        @staticmethod
        def export_deployment(path: Path) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"deployment")

        @staticmethod
        def export_self_contained_agent(path: Path, *, deployment_path: Path) -> None:
            assert deployment_path.is_file()
            path.write_text("# agent\n", encoding="utf-8")

    monkeypatch.setattr(
        tensorflight_cli,
        "_trainer_from_checkpoint",
        lambda path: FakeTrainer(int(path.stem.removeprefix("update_"))),
    )

    def fake_evaluate(path, **_):
        score = 2 if Path(path).stem == "candidate_0" else 1
        return [
            OfficialEvaluationResult(
                scenario=1,
                seed=7,
                score=score,
                hit_indices=tuple(range(score)),
                steps=10,
                final_time=0.1,
                terminated=True,
                truncated=False,
            )
        ]

    monkeypatch.setattr(tensorflight_cli, "evaluate_deployment", fake_evaluate)
    output = tmp_path / "selected"
    result = main(
        [
            "select",
            str(tmp_path / "update_80.ckpt"),
            str(tmp_path / "update_90.ckpt"),
            "--seeds",
            "7",
            "--output-dir",
            str(output),
            "--require-hit",
        ]
    )

    report = json.loads((output / "selection.json").read_text(encoding="utf-8"))
    assert result == 0
    assert report["selected"]["update"] == 80
    assert report["selected"]["scores"] == [2]
    assert (output / "deployment.npz").is_file()
    assert (output / "submission_agent.py").is_file()


def test_packaged_gpu_policy_pops_balloon_in_unmodified_activerocketpy() -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=r"Actuator .* output change .*rate limit.*"
        )
        warnings.filterwarnings("ignore", message=r"The Sensor class .*experimental.*")
        warnings.filterwarnings(
            "ignore", message=r"The following arguments have no effect.*"
        )
        result = evaluate_deployment(scenario=0, seeds=(0,))[0]

    assert result.score == 2
    assert result.hit_indices == (0, 1)
    assert result.terminated
    assert not result.truncated
