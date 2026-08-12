"""Fidelity gates for the online balloon world and tensor environment."""

# ruff: noqa: E402 -- optional PyTorch must be checked before tensor imports.

from __future__ import annotations

import ast
from dataclasses import fields
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from BalloonPoppingGymEnv.tensorflight.rocket import ActuatorSpec, Scenario1ActuatorBank
from BalloonPoppingGymEnv.tensorflight.balloons import (
    Scenario1BalloonSampler,
    TensorBalloonWorld,
    sample_scenario1_balloon_parameters,
)
from BalloonPoppingGymEnv.tensorflight.effects import (
    LinearGustProfile,
    SensorEffectConfig,
    TensorSensorSuite,
)
from BalloonPoppingGymEnv.tensorflight.environment import (
    TensorAgentObservation,
    TensorEnvironmentStep,
    TensorOracleState,
    segment_distance,
)
from BalloonPoppingGymEnv.tensorflight._balloon_oracle import (
    build_canonical_balloon_fixture,
    generate_balloon_fidelity_report,
    generate_full_environment_report,
)


ROOT = Path(__file__).resolve().parents[1]
TENSOR_MODULES = (
    ROOT / "BalloonPoppingGymEnv" / "tensorflight" / "balloons.py",
    ROOT / "BalloonPoppingGymEnv" / "tensorflight" / "effects.py",
    ROOT / "BalloonPoppingGymEnv" / "tensorflight" / "environment.py",
)


@pytest.fixture(scope="module")
def balloon_report() -> dict[str, object]:
    return generate_balloon_fidelity_report(
        seed=2053,
        num_balloons=2,
        steps=None,
        substeps=1,
        dtype=torch.float64,
    )


@pytest.fixture(scope="module")
def full_report() -> dict[str, object]:
    return generate_full_environment_report(seed=2054, num_balloons=2, steps=300)


def test_training_modules_have_no_oracle_or_host_array_dependencies() -> None:
    forbidden_roots = {"numpy", "scipy", "rocketpy", "gymnasium"}
    for path in TENSOR_MODULES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert imported.isdisjoint(forbidden_roots), (path.name, imported)


def test_training_step_exposes_only_official_observation_fields() -> None:
    assert {field.name for field in fields(TensorAgentObservation)} == {
        "simulation_time",
        "balloon_status",
        "balloon_states",
        "rocket_sensors",
    }
    assert {field.name for field in fields(TensorEnvironmentStep)} == {
        "observation",
        "reward",
        "terminated",
        "truncated",
    }
    assert "rocket_state" in {field.name for field in fields(TensorOracleState)}


def test_sampler_is_reproducible_positive_and_uses_release_permutations() -> None:
    first = sample_scenario1_balloon_parameters(3, 100, seed=91, dtype=torch.float64)
    second = sample_scenario1_balloon_parameters(3, 100, seed=91, dtype=torch.float64)

    for field in fields(first):
        assert torch.equal(getattr(first, field.name), getattr(second, field.name))
    assert bool((first.dry_mass > 0).all())
    assert bool((first.volume > 0).all())
    assert bool((first.inertia > 0).all())
    expected = torch.arange(100, dtype=torch.int64) * 50
    assert torch.equal(
        torch.sort(first.release_step, dim=1).values, expected.expand(3, -1)
    )


def test_balloon_world_has_bounded_online_state_and_alias_safe_masked_reset() -> None:
    fixture = build_canonical_balloon_fixture(seed=82, num_balloons=2)
    parameters = fixture.parameters
    world = TensorBalloonWorld(
        fixture.model_data,
        parameters,
        dt=fixture.dt,
        max_time=fixture.max_time,
        substeps=1,
        dtype=torch.float64,
    )
    world.step()
    reset_mask = world.current_step > 0
    world.reset(reset_mask)

    assert world.state_storage_elements == 3 * 1 * 2 * 6
    assert world.previous_state.shape == (1, 2, 6)
    assert world.current_state.shape == (1, 2, 6)
    assert world.next_state.shape == (1, 2, 6)
    assert world.current_step.tolist() == [0]
    assert not any(
        isinstance(value, torch.Tensor) and value.ndim == 4
        for value in vars(world).values()
    )


def test_balloon_world_can_resample_stochastic_parameters_on_reset() -> None:
    fixture = build_canonical_balloon_fixture(seed=83, num_balloons=2)
    sampler = Scenario1BalloonSampler(2, seed=84, dtype=torch.float64)
    initial = sampler(2)
    world = TensorBalloonWorld(
        fixture.model_data,
        initial,
        dtype=torch.float64,
        reset_sampler=sampler,
    )
    before = world.parameters.initial_state.clone()
    world.current_step[1] = 7

    world.reset(torch.tensor([True, False]))

    assert not torch.equal(world.parameters.initial_state[0], before[0])
    torch.testing.assert_close(world.parameters.initial_state[1], before[1])
    torch.testing.assert_close(
        world.current_state[0], world.parameters.initial_state[0]
    )
    assert world.current_step.tolist() == [0, 7]


def test_full_horizon_balloon_fidelity_gate(balloon_report) -> None:
    assert balloon_report["steps"] == 14_999
    assert balloon_report["full_horizon_state_allocated"] is False
    assert balloon_report["status_mismatches"] == 0
    assert balloon_report["release_time_error_s"]["max"] == 0.0
    assert balloon_report["position_error_m"]["p99"] <= 0.02
    # The sole rail/free phase-discontinuity sample can have a larger point
    # error; the full-horizon p99 must remain tightly bounded.
    assert balloon_report["velocity_error_m_s"]["p99"] <= 0.02


def test_independent_segment_parameters_detect_crossing_paths() -> None:
    rocket_start = torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float64)
    rocket_end = torch.tensor([[[10.0, 0.0, 0.0]]], dtype=torch.float64)
    balloon_start = torch.tensor([[[8.0, -5.0, 0.0]]], dtype=torch.float64)
    balloon_end = torch.tensor([[[8.0, 5.0, 0.0]]], dtype=torch.float64)

    distance = segment_distance(rocket_start, rocket_end, balloon_start, balloon_end)

    torch.testing.assert_close(distance, torch.zeros_like(distance), atol=1e-12, rtol=0)


def test_gust_profile_interpolates_per_environment_and_decays() -> None:
    generator = torch.Generator().manual_seed(11)
    profile = LinearGustProfile.sample(
        2,
        max_height=100.0,
        altitude_spacing=10.0,
        max_gust_speed=3.0,
        decay_height=50.0,
        generator=generator,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    at_nodes = profile(torch.tensor([[0.0, 10.0], [0.0, 10.0]]))
    halfway = profile(torch.tensor([5.0, 5.0]))

    torch.testing.assert_close(at_nodes[:, 0, 0], profile.x_nodes[:, 0])
    torch.testing.assert_close(at_nodes[:, 1, 1], profile.y_nodes[:, 1])
    torch.testing.assert_close(
        halfway[:, 0], (profile.x_nodes[:, 0] + profile.x_nodes[:, 1]) / 2
    )


def test_sensor_suite_matches_zero_noise_official_shape_and_gravity_contract() -> None:
    suite = TensorSensorSuite(
        1,
        SensorEffectConfig(),
        device="cpu",
        dtype=torch.float64,
        seed=5,
    )
    state = torch.zeros((1, 13), dtype=torch.float64)
    state[:, 6] = 1.0
    state[:, :3] = torch.tensor([[1.0, 2.0, 20.0]])
    state[:, 3:6] = torch.tensor([[4.0, 5.0, 6.0]])
    rhs = torch.zeros_like(state)
    rhs[:, 3:6] = torch.tensor([[0.0, 0.0, 12.0]])

    measured = suite.measure(state, rhs, gravity=torch.tensor([9.8]))

    assert measured.shape == (1, 12)
    torch.testing.assert_close(
        measured[0, 3:6], torch.tensor([0.0, 0.0, 2.2], dtype=torch.float64)
    )
    torch.testing.assert_close(measured[0, 6:9], state[0, :3])
    torch.testing.assert_close(measured[0, 9:12], state[0, 3:6])


def test_optional_actuator_lpf_keeps_filter_rate_limit_clamp_order() -> None:
    bank = Scenario1ActuatorBank(
        1,
        [ActuatorSpec(-10.0, 10.0, 2.0, 0.5, 0.0)] * 4,
        demand_rate=100.0,
        dtype=torch.float64,
    )
    result = bank.update(torch.full((1, 4), 10.0, dtype=torch.float64))

    # alpha=1/51 gives 0.196 filtered demand, then the 0.02/step rate limit.
    torch.testing.assert_close(result, torch.full_like(result, 0.02))


def test_combined_float64_environment_matches_official(full_report) -> None:
    assert full_report["compared_steps"] < 300
    assert full_report["rocket_position_error_m"]["p99"] <= 1e-4
    assert full_report["rocket_position_error_m"]["max"] <= 1e-3
    assert full_report["balloon_position_error_m"]["p99"] <= 0.02
    assert full_report["closest_distance_error_m"]["p99"] <= 0.05
    assert full_report["ambiguity_band_m"] == max(
        0.05, full_report["closest_distance_error_m"]["p99_9"] + 0.01
    )
    assert full_report["sensor_absolute_error"]["p99"] <= 1e-4
    assert full_report["sensor_absolute_error"]["max"] <= 1e-3
    assert full_report["reward_mismatches"] == 0
    assert full_report["status_mismatches"] == 0
    assert full_report["termination_mismatches"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_online_world_stays_on_cuda() -> None:
    fixture = build_canonical_balloon_fixture(seed=44, num_balloons=2)
    world = TensorBalloonWorld(
        fixture.model_data,
        fixture.parameters,
        dt=fixture.dt,
        max_time=fixture.max_time,
        substeps=1,
        device="cuda",
        dtype=torch.float32,
    )

    world.step()

    assert world.current_state.device.type == "cuda"
    assert world.status.device.type == "cuda"
    assert torch.isfinite(world.current_state).all()
