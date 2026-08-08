"""Tests for the isolated, non-canonical CUDA flight experiment."""

from __future__ import annotations

import pytest
import numpy as np


torch = pytest.importorskip("torch")

from experiments.cuda.tensor_flight import (  # noqa: E402
    NON_CANONICAL_PROTOTYPE,
    TensorFlightBatch,
    TensorFlightConfig,
    _segment_distance,
)


def test_step_shapes_device_and_quaternion_normalization() -> None:
    environment = TensorFlightBatch(8, device="cpu")
    action = torch.zeros((8, 4), dtype=environment.dtype)

    result = environment.step(action)

    assert NON_CANONICAL_PROTOTYPE is True
    assert result.state.shape == (8, 13)
    assert result.reward.shape == (8,)
    assert result.terminated.dtype == torch.bool
    assert result.truncated.dtype == torch.bool
    for tensor in result:
        assert tensor.device.type == "cpu"
    quaternion_norm = torch.linalg.vector_norm(result.state[:, 6:10], dim=-1)
    torch.testing.assert_close(quaternion_norm, torch.ones(8))


def test_normalized_actions_are_clamped_before_actuator_mapping() -> None:
    low = TensorFlightBatch(2, device="cpu")
    high = TensorFlightBatch(2, device="cpu")
    high.state.copy_(low.state)
    high.actuator_state.copy_(low.actuator_state)
    high.target_position.copy_(low.target_position)
    high.target_velocity.copy_(low.target_velocity)

    clamped_action = torch.tensor([[-1.0, 1.0, -1.0, 1.0]]).repeat(2, 1)
    excessive_action = torch.tensor([[-5.0, 5.0, -5.0, 5.0]]).repeat(2, 1)

    clamped_result = low.step(clamped_action)
    excessive_result = high.step(excessive_action)

    torch.testing.assert_close(clamped_result.state, excessive_result.state)
    torch.testing.assert_close(low.actuator_state, high.actuator_state)


def test_swept_hit_terminates_and_subsequent_step_is_frozen() -> None:
    config = TensorFlightConfig(
        dt=0.1,
        gravity=1e-9,
        max_thrust=1e-9,
        drag_force_coefficient=0.0,
        initial_vertical_speed=20.0,
        target_radius=0.2,
    )
    environment = TensorFlightBatch(1, device="cpu", config=config)
    target = torch.tensor([[0.0, 0.0, 3.0]], dtype=environment.dtype)
    environment.set_target(target)
    action = torch.tensor([[0.0, 0.0, 0.0, -1.0]], dtype=environment.dtype)

    result = environment.step(action)

    assert result.hit.tolist() == [True]
    assert result.terminated.tolist() == [True]
    assert result.reward.item() >= config.hit_reward - 0.1
    frozen_state = result.state.clone()

    result_after_done = environment.step(torch.ones_like(action))

    torch.testing.assert_close(result_after_done.state, frozen_state)
    torch.testing.assert_close(result_after_done.reward, torch.zeros(1))


def test_moving_crossing_segments_match_official_pop_geometry() -> None:
    rocket_start = torch.tensor([[0.0, 0.0, 0.0]])
    rocket_end = torch.tensor([[10.0, 0.0, 0.0]])
    target_start = torch.tensor([[8.0, -5.0, 0.0]])
    target_end = torch.tensor([[8.0, 5.0, 0.0]])

    distance = _segment_distance(rocket_start, rocket_end, target_start, target_end)

    # The paths intersect even though the two objects reach the intersection
    # at different fractions of the control interval.
    torch.testing.assert_close(distance, torch.zeros(1), atol=1e-6, rtol=0.0)


def test_segment_distance_matches_official_geometry_on_seeded_batch() -> None:
    try:
        from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
    except ModuleNotFoundError:
        pytest.skip("canonical simulation stack is unavailable")

    generator = np.random.default_rng(2026)
    rocket_start = np.array([-4.0, 1.5, 2.0])
    rocket_end = np.array([6.0, -2.0, 5.0])
    target_start = generator.normal(size=(128, 3)) * 8.0
    target_end = target_start + generator.normal(size=(128, 3)) * 3.0
    target_end[0] = target_start[0]  # point/segment branch
    target_start[1] = np.array([1.0, -5.0, 3.5])
    target_end[1] = np.array([1.0, 5.0, 3.5])

    expected = np.sqrt(
        BalloonPoppingEnv._segment_distance_squared_batch(
            rocket_start,
            rocket_end,
            target_start,
            target_end,
        )
    )
    actual = _segment_distance(
        torch.tensor(np.repeat(rocket_start[None, :], 128, axis=0)),
        torch.tensor(np.repeat(rocket_end[None, :], 128, axis=0)),
        torch.tensor(target_start),
        torch.tensor(target_end),
    )

    np.testing.assert_allclose(actual.numpy(), expected, atol=1e-10, rtol=1e-10)


def test_masked_reset_only_reinitializes_done_rows() -> None:
    environment = TensorFlightBatch(3, device="cpu")
    original_state = environment.state.clone()
    environment.state[1, 0] = 123.0
    environment.terminated[1] = True

    environment.reset_done(seed=7)

    torch.testing.assert_close(environment.state[0], original_state[0])
    torch.testing.assert_close(environment.state[2], original_state[2])
    torch.testing.assert_close(environment.state[1], original_state[1])
    assert environment.terminated.tolist() == [False, False, False]


def test_masked_reset_accepts_an_aliased_done_tensor() -> None:
    environment = TensorFlightBatch(2, device="cpu")
    original_target = environment.target_position.clone()
    environment.terminated[1] = True
    environment.hit[1] = True

    environment.reset(seed=11, mask=environment.terminated)

    assert environment.terminated.tolist() == [False, False]
    assert environment.hit.tolist() == [False, False]
    torch.testing.assert_close(environment.target_position[0], original_target[0])
    assert not torch.equal(environment.target_position[1], original_target[1])


def test_non_finite_action_terminates_with_finite_state() -> None:
    environment = TensorFlightBatch(1, device="cpu")
    action = torch.tensor([[float("nan"), 0.0, 0.0, 0.0]])

    result = environment.step(action)

    assert result.crashed.tolist() == [True]
    assert result.terminated.tolist() == [True]
    assert torch.isfinite(result.state).all()


def test_step_rejects_host_or_wrong_shape_actions() -> None:
    environment = TensorFlightBatch(4, device="cpu")
    with pytest.raises(ValueError, match="action shape"):
        environment.step(torch.zeros((4, 3)))
    with pytest.raises(ValueError, match="device/dtype"):
        environment.step(torch.zeros((4, 4), dtype=torch.float64))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_transition_stays_on_cuda() -> None:
    environment = TensorFlightBatch(64, device="cuda")
    action = torch.zeros((64, 4), device="cuda", dtype=environment.dtype)

    result = environment.step(action)
    torch.cuda.synchronize()

    assert all(tensor.device.type == "cuda" for tensor in result)
