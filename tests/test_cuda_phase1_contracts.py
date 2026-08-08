"""Phase 1 contracts for the canonical-oracle CUDA experiment."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from experiments.cuda.agent_contracts import (
    AgentObservation,
    EstimatedAltitudeHandoff,
    EstimatedRocketFeatures,
    HistoricalE2EObservationBuilder,
    HistoricalSensorEstimator,
    LaunchPlannerProtocol,
    MaskedRunningNormalizer,
    NearestEstimatedTargetSelector,
    OracleRocketState,
    PostLaunchAction,
    TargetSelectorProtocol,
    TimedLaunchPlanner,
)
from experiments.cuda.benchmark_cpu_training_loop import (
    FixedBatchPolicy,
    benchmark_worker_count,
    scale_normalized_action,
)
from experiments.cuda.canonical_oracle import (
    ACTION_FIELDS,
    EVENT_FIELDS,
    FIXED_CASES,
    CanonicalTrace,
    OracleCase,
    record_case,
)


def _observation(*, sensor_altitude: float | None = None) -> AgentObservation:
    sensors = np.full(12, np.nan)
    if sensor_altitude is not None:
        sensors[:] = 0.0
        sensors[8] = sensor_altitude
    return AgentObservation(
        simulation_time=0.02,
        balloon_status=np.array([1, 0]),
        balloon_states=np.array(
            [
                [10.0, 0.0, 80.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 30.0, 0.0, 0.0, 0.0],
            ]
        ),
        rocket_sensors=sensors,
    )


def test_agent_observation_copies_only_official_fields() -> None:
    raw = {
        "simulation_time": 1.0,
        "balloon_status": np.array([[1]]),
        "balloon_states": np.zeros((1, 6)),
        "rocket_sensors": np.zeros(12),
        "rocket_states": np.arange(13),
    }

    observation = AgentObservation.from_official(raw)
    raw["rocket_sensors"][0] = 99.0

    assert observation.rocket_sensors[0] == 0.0
    assert not hasattr(observation, "rocket_states")


def test_launch_planner_has_no_estimate_or_handoff_responsibility() -> None:
    planner = TimedLaunchPlanner(launch_time=0.02)
    decision = planner.plan(_observation())

    assert isinstance(planner, LaunchPlannerProtocol)
    assert decision.launch is True
    assert not hasattr(decision, "controller_active")
    parameters = tuple(inspect.signature(planner.plan).parameters)
    assert parameters == ("observation",)


def test_handoff_uses_sensor_estimate_and_rejects_oracle_state() -> None:
    estimator = HistoricalSensorEstimator(sampling_rate=100.0, ground_elevation=20.0)
    handoff = EstimatedAltitudeHandoff(ground_elevation=20.0, altitude_agl=40.0)

    before = _observation()
    estimate = estimator.update(before)
    oracle = OracleRocketState(np.array([0.0, 0.0, 1_000.0, *np.zeros(10)]))

    assert handoff.controller_active(before, estimate) is False
    with pytest.raises(TypeError, match="EstimatedRocketFeatures"):
        handoff.controller_active(before, oracle)  # type: ignore[arg-type]

    after = _observation(sensor_altitude=60.01)
    estimated_after = estimator.update(after)
    assert handoff.controller_active(after, estimated_after) is True


def test_selector_accepts_estimated_features_not_oracle_state() -> None:
    observation = _observation(sensor_altitude=20.0)
    estimate = EstimatedRocketFeatures(
        position=np.array([0.0, 0.0, 20.0]),
        velocity=np.zeros(3),
        specific_force=np.zeros(3),
        attitude_quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_rate=np.zeros(3),
    )
    selector = NearestEstimatedTargetSelector()

    assert isinstance(selector, TargetSelectorProtocol)
    assert selector.select_target(observation, estimate) == 0
    with pytest.raises(TypeError, match="EstimatedRocketFeatures"):
        selector.select_target(  # type: ignore[arg-type]
            observation,
            OracleRocketState(np.zeros(13)),
        )


def test_historical_e2e_builder_is_finite_and_29_dimensional() -> None:
    estimator = HistoricalSensorEstimator(sampling_rate=100.0, ground_elevation=20.0)
    observation = _observation(sensor_altitude=61.0)
    estimate = estimator.update(observation)
    previous = PostLaunchAction(roll=1.0, tvc=np.array([2.0, -3.0]), throttle=0.8)

    result = HistoricalE2EObservationBuilder().build(
        estimate,
        observation.balloon_states[0],
        previous,
    )

    assert result.shape == (29,)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()
    np.testing.assert_allclose(result[-4:], np.array([2.0, -3.0, 1.0, 0.8]))


def test_normalizer_updates_only_active_finite_rows() -> None:
    normalizer = MaskedRunningNormalizer(size=3)
    batch = np.array([[1.0, 2.0, 3.0], [np.nan, np.nan, np.nan], [9.0, 9.0, 9.0]])

    accepted = normalizer.update(
        batch,
        controller_active=np.array([True, True, False]),
        sensors_finite=np.array([True, False, True]),
    )

    assert accepted.tolist() == [True, False, False]
    assert normalizer.count == 1
    np.testing.assert_allclose(normalizer.mean, batch[0])
    assert np.isfinite(normalizer.normalize(batch)).all()


def test_rollout_policy_and_action_scaling_use_post_launch_order() -> None:
    policy = FixedBatchPolicy(seed=1)
    action = policy(np.zeros((2, 29)))
    control = {
        "max_roll_torque": 10.0,
        "max_gimbal_angle": 15.0,
        "throttle_range": [0.0, 1.0],
    }

    physical = scale_normalized_action(np.array([1.0, -1.0, 0.5, 0.0]), control)

    assert action.shape == (2, 4)
    np.testing.assert_allclose(physical.as_vector(), [10.0, -15.0, 7.5, 0.5])


def test_short_canonical_trace_locks_launch_boundary_and_round_trips(
    tmp_path: Path,
) -> None:
    case = OracleCase(
        name="test_launch",
        scenario=0,
        seed=2026,
        balloons=1,
        steps=4,
        schedule="vertical_full",
    )
    trace = record_case(case)
    output = tmp_path / "trace.npz"
    trace.save(output)
    loaded = CanonicalTrace.load(output)

    assert loaded.arrays["action"].shape == (5, len(ACTION_FIELDS))
    assert loaded.arrays["sensor_finite"].tolist() == [False, False, False, True, True]
    assert np.isnan(loaded.arrays["rocket_state"][:3]).all()
    assert np.isfinite(loaded.arrays["rocket_state"][3:]).all()
    assert np.isfinite(loaded.arrays["rhs"][3:]).all()
    assert loaded.arrays["events"][0, EVENT_FIELDS.index("reset")]
    assert loaded.arrays["events"][1, EVENT_FIELDS.index("pre_launch")]
    assert loaded.arrays["events"][2, EVENT_FIELDS.index("launch_step")]
    assert loaded.arrays["events"][3, EVENT_FIELDS.index("first_post_launch")]
    assert loaded.arrays["balloon_release"][0, 0]
    assert not loaded.arrays["balloon_release"][1:, 0].any()
    assert loaded.arrays["geometry_hit"].tolist() == [True, False]


def test_timeout_case_records_truncation_separately_from_termination() -> None:
    trace = record_case(FIXED_CASES["timeout_without_launch"])

    assert trace.arrays["truncated"][-1]
    assert not trace.arrays["terminated"].any()
    assert trace.arrays["events"][-1, EVENT_FIELDS.index("truncated")]
    assert not trace.arrays["events"][:, EVENT_FIELDS.index("launch_step")].any()
    assert not trace.arrays["events"][:, EVENT_FIELDS.index("first_post_launch")].any()


def test_fixed_oracle_corpus_names_all_required_regimes() -> None:
    assert {
        "launch_control",
        "burnout",
        "impact",
        "timeout_without_launch",
        "moving_balloon",
    } <= FIXED_CASES.keys()


def test_tensor_baseline_report_uses_first_post_launch_anchor() -> None:
    pytest.importorskip("torch")
    from experiments.cuda.compare_canonical_trace import compare_trace

    trace = record_case(
        OracleCase(
            name="comparison",
            scenario=0,
            seed=42,
            balloons=1,
            steps=5,
            schedule="vertical_full",
        )
    )

    report = compare_trace(trace)

    assert report["anchor_frame"] == 3
    assert report["compared_transitions"] == 2
    assert report["non_canonical_prototype"] is True
    assert "position_l2_m" in report["state_error"]


def test_synchronous_rollout_baseline_crosses_a_step_barrier() -> None:
    sample = benchmark_worker_count(1, steps=2, warmup_steps=0)

    assert sample.steps == 2
    assert sample.transitions_per_second > 0
    assert sample.barriers_per_second > 0
