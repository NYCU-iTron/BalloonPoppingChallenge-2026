"""Phase 4 gates for the device-native PPO and deployment boundary."""

from __future__ import annotations

import ast
import importlib.util
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch

from BalloonPoppingGymEnv.agents.numpy_tensorflight_agent import (
    NumpyTensorFlightAgent,
)
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters
from experiments.cuda.agent_contracts import (
    AgentObservation,
    EstimatedRocketFeatures,
    HistoricalE2EObservationBuilder,
    HistoricalSensorEstimator,
    PostLaunchAction,
)
from experiments.cuda.phase3_environment import TensorAgentObservation
from experiments.cuda.phase4_factory import build_phase4_source
from experiments.cuda.phase4_observation import (
    TensorAltitudeHandoff,
    TensorEstimatedRocketFeatures,
    TensorRunningNormalizer,
    TensorSensorEstimator,
    build_historical_observation,
    select_nearest_target,
)
from experiments.cuda.phase4_ppo import (
    DeviceRolloutBuffer,
    PPOHyperparameters,
    compute_gae,
)
from experiments.cuda.phase4_training import Phase4Trainer, Phase4TrainingConfig


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def phase4_source():
    return build_phase4_source(seed=2141)


def _features() -> tuple[EstimatedRocketFeatures, TensorEstimatedRocketFeatures]:
    values = {
        "position": np.asarray((3.0, -2.0, 61.0)),
        "velocity": np.asarray((4.0, 5.0, 6.0)),
        "specific_force": np.asarray((0.2, -0.3, 9.7)),
        "attitude_quaternion": np.asarray((0.97, 0.1, -0.05, 0.2)),
        "angular_rate": np.asarray((0.01, -0.02, 0.03)),
    }
    values["attitude_quaternion"] /= np.linalg.norm(values["attitude_quaternion"])
    scalar = EstimatedRocketFeatures(**values)
    tensor = TensorEstimatedRocketFeatures(
        **{
            name: torch.as_tensor(value, dtype=torch.float64).unsqueeze(0)
            for name, value in values.items()
        }
    )
    return scalar, tensor


def _assert_nested_equal(left: object, right: object) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left.cpu(), right.cpu())
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_nested_equal(left_item, right_item)
    else:
        assert left == right


def test_policy_modules_cannot_import_or_name_oracle_state() -> None:
    paths = (
        ROOT / "experiments" / "cuda" / "phase4_observation.py",
        ROOT / "BalloonPoppingGymEnv" / "agents" / "numpy_tensorflight_agent.py",
    )
    for path in paths:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        assert "TensorOracleState" not in names
        assert "OracleRocketState" not in names
        assert "rocket_states" not in source
    deployment_source = paths[1].read_text(encoding="utf-8")
    assert "import torch" not in deployment_source


def test_tensor_observation_matches_historical_29_feature_fixture() -> None:
    scalar, tensor = _features()
    target = np.asarray((15.0, 7.0, 90.0, -1.0, 2.0, 0.5))
    previous = PostLaunchAction(roll=1.25, tvc=np.asarray((2.5, -3.5)), throttle=0.8)
    expected = HistoricalE2EObservationBuilder().build(scalar, target, previous)

    actual, _, _ = build_historical_observation(
        tensor,
        torch.as_tensor(target, dtype=torch.float64).unsqueeze(0),
        torch.as_tensor(previous.as_vector(), dtype=torch.float64).unsqueeze(0),
    )

    np.testing.assert_allclose(actual[0].numpy(), expected, rtol=2e-6, atol=2e-6)


def test_tensor_estimator_matches_historical_sensor_fixture() -> None:
    scalar = HistoricalSensorEstimator(sampling_rate=100.0, ground_elevation=20.0)
    tensor = TensorSensorEstimator(
        1,
        sampling_rate=100.0,
        ground_elevation=20.0,
        device="cpu",
        dtype=torch.float64,
    )
    sensors = np.asarray((0.1, -0.2, 0.3, 1, 2, 3, 4, 5, 61, 6, 7, 8), dtype=float)
    for time in (0.02, 0.03, 0.04):
        scalar_result = scalar.update(
            AgentObservation(
                simulation_time=time,
                balloon_status=np.asarray([1]),
                balloon_states=np.zeros((1, 6)),
                rocket_sensors=sensors,
            )
        )
        tensor.update(
            TensorAgentObservation(
                simulation_time=torch.tensor([time], dtype=torch.float64),
                balloon_status=torch.tensor([[1]]),
                balloon_states=torch.zeros((1, 1, 6), dtype=torch.float64),
                rocket_sensors=torch.as_tensor(sensors).unsqueeze(0),
            )
        )
    np.testing.assert_allclose(
        tensor.features.attitude_quaternion[0].numpy(),
        scalar_result.attitude_quaternion,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        tensor.features.position[0].numpy(), scalar_result.position
    )


def test_selector_and_handoff_use_only_sensor_derived_features() -> None:
    _, features = _features()
    observation = TensorAgentObservation(
        simulation_time=torch.tensor([0.5], dtype=torch.float64),
        balloon_status=torch.tensor([[1, 1]]),
        balloon_states=torch.tensor(
            [[[100.0, 0.0, 61.0, 0, 0, 0], [4.0, -1.0, 61.0, 0, 0, 0]]],
            dtype=torch.float64,
        ),
        rocket_sensors=torch.zeros((1, 12), dtype=torch.float64),
    )
    target_index, available, _ = select_nearest_target(observation, features)
    handoff = TensorAltitudeHandoff(
        1, ground_elevation=20.0, altitude_agl=40.0, device="cpu"
    )

    assert target_index.tolist() == [1]
    assert available.tolist() == [True]
    assert handoff.update(features, torch.tensor([True])).tolist() == [True]


def test_normalizer_only_updates_active_finite_rows() -> None:
    normalizer = TensorRunningNormalizer(3, device="cpu")
    values = torch.tensor([[1.0, 2.0, 3.0], [float("nan")] * 3, [9.0, 9.0, 9.0]])
    normalizer.update(values, torch.tensor([True, True, False]))

    assert normalizer.count.item() == 1
    torch.testing.assert_close(
        normalizer.mean, torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    )
    assert torch.isfinite(normalizer.normalize(values)).all()


def test_gae_bootstraps_truncation_without_crossing_reset() -> None:
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    values = torch.zeros_like(rewards)
    bootstrap = torch.tensor([[10.0], [20.0], [30.0]])
    terminated = torch.tensor([[False], [False], [True]])
    truncated = torch.tensor([[False], [True], [False]])

    advantages, returns = compute_gae(
        rewards,
        values,
        bootstrap,
        terminated,
        truncated,
        gamma=0.9,
        gae_lambda=0.8,
    )

    # t=2 is terminal: no V(s'). t=1 is truncated: include V(s') but do not
    # carry t=2's advantage across the reset. t=0 may carry t=1 normally.
    torch.testing.assert_close(advantages[:, 0], torch.tensor([24.4, 20.0, 3.0]))
    torch.testing.assert_close(returns, advantages)


def test_rollout_storage_is_device_resident() -> None:
    buffer = DeviceRolloutBuffer(4, 8, device="cpu")
    assert buffer.transition_count == 32
    assert all(
        value.device.type == "cpu"
        for value in vars(buffer).values()
        if isinstance(value, torch.Tensor)
    )
    assert buffer.allocated_bytes > 0


def test_checkpoint_resume_reproduces_next_update_exactly(
    phase4_source, tmp_path: Path
) -> None:
    config = Phase4TrainingConfig(
        num_envs=3,
        horizon=3,
        updates=2,
        hidden_size=32,
        handoff_altitude_agl=0.0,
        device="cpu",
        ppo=PPOHyperparameters(epochs=2, minibatch_size=9, target_kl=None),
    )
    uninterrupted = Phase4Trainer(phase4_source, config)
    uninterrupted.train_update()
    checkpoint = tmp_path / "training.ckpt"
    uninterrupted.save_checkpoint(checkpoint)

    resumed = Phase4Trainer(phase4_source, config)
    resumed.load_checkpoint(checkpoint)
    uninterrupted.train_update()
    resumed.train_update()

    _assert_nested_equal(uninterrupted.model.state_dict(), resumed.model.state_dict())
    _assert_nested_equal(
        uninterrupted.optimizer.state_dict(), resumed.optimizer.state_dict()
    )
    _assert_nested_equal(
        uninterrupted.adapter.state_dict(), resumed.adapter.state_dict()
    )
    _assert_nested_equal(
        uninterrupted.environment.rocket_state, resumed.environment.rocket_state
    )
    assert uninterrupted.global_transitions == resumed.global_transitions
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert saved.keys() >= {
        "model",
        "optimizer",
        "environment",
        "adapter",
        "progress",
        "rng",
        "source_hashes",
    }
    assert saved["rng"].keys() >= {"torch_cpu", "action", "shuffle"}
    incompatible = Phase4Trainer(phase4_source, replace(config, score_threshold=2.0))
    with pytest.raises(ValueError, match="training config"):
        incompatible.load_checkpoint(checkpoint)


def test_numpy_deployment_forward_matches_torch_actor(
    phase4_source, tmp_path: Path
) -> None:
    config = Phase4TrainingConfig(
        num_envs=2,
        horizon=2,
        updates=1,
        hidden_size=32,
        handoff_altitude_agl=0.0,
        device="cpu",
        ppo=PPOHyperparameters(epochs=1, minibatch_size=4),
    )
    trainer = Phase4Trainer(phase4_source, config)
    trainer.train_update()
    artifact = tmp_path / "deployment.npz"
    trainer.export_deployment(artifact)
    embedded_source = tmp_path / "submission_agent.py"
    trainer.export_self_contained_agent(embedded_source, deployment_path=artifact)
    scenario, _ = load_scenario_parameters(1)
    agent = NumpyTensorFlightAgent(scenario, artifact_path=artifact)
    raw = np.linspace(-2.0, 2.0, 29, dtype=np.float64)

    numpy_action = agent._policy(raw)
    normalized = trainer.adapter.normalizer.normalize(
        torch.as_tensor(raw, dtype=torch.float32).unsqueeze(0)
    )
    with torch.no_grad():
        torch_action = torch.tanh(trainer.model.actor_mean(normalized))[0].numpy()

    np.testing.assert_allclose(numpy_action, torch_action, rtol=2e-6, atol=2e-6)
    generated = embedded_source.read_text(encoding="utf-8")
    assert 'EMBEDDED_DEPLOYMENT_BASE64 = ""' not in generated
    assert "import torch" not in generated
    spec = importlib.util.spec_from_file_location("submission_agent", embedded_source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    embedded_agent = module.NumpyTensorFlightAgent(scenario)
    np.testing.assert_allclose(
        embedded_agent._policy(raw), torch_action, rtol=2e-6, atol=2e-6
    )
    launch_observation = {
        "simulation_time": 0.01,
        "balloon_status": np.asarray([1]),
        "balloon_states": np.asarray([[0, 0, 50, 0, 0, 0]], dtype=float),
        "rocket_sensors": np.full(12, np.nan),
    }
    action = agent.get_action(launch_observation)
    assert action["launch"] is True
    assert set(action) == {
        "launch",
        "launch_inclination_heading",
        "roll",
        "tvc",
        "throttle",
    }
    assert np.isfinite(action["launch_inclination_heading"]).all()
    assert np.isfinite(action["tvc"]).all()
    assert np.isfinite([action["roll"], action["throttle"]]).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_phase4_rollout_and_resume_stay_on_cuda(phase4_source, tmp_path: Path) -> None:
    config = Phase4TrainingConfig(
        num_envs=8,
        horizon=2,
        updates=1,
        hidden_size=32,
        handoff_altitude_agl=0.0,
        device="cuda",
        ppo=PPOHyperparameters(epochs=1, minibatch_size=16),
    )
    trainer = Phase4Trainer(phase4_source, config)
    metrics = trainer.train_update()
    checkpoint = tmp_path / "cuda_training.ckpt"
    trainer.save_checkpoint(checkpoint)
    resumed = Phase4Trainer(phase4_source, config)
    resumed.load_checkpoint(checkpoint)
    trainer.train_update()
    resumed.train_update()

    assert metrics.ppo.samples == 16
    assert trainer.environment.rocket_state.device.type == "cuda"
    assert trainer.rollout.observations.device.type == "cuda"
    assert next(trainer.model.parameters()).device.type == "cuda"
    assert asdict(metrics)["rollout"]["transitions_per_second"] > 0
    _assert_nested_equal(trainer.model.state_dict(), resumed.model.state_dict())
