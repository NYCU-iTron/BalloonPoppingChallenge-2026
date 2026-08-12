"""Holdout transfer report and decision-gate tests."""

# ruff: noqa: E402 -- optional PyTorch must be checked before tensor imports.

from __future__ import annotations

import copy

import pytest

pytest.importorskip("torch")

from BalloonPoppingGymEnv.tensorflight.validation import evaluate_transfer_gates


def _distribution(value: float) -> dict[str, float]:
    return {
        "rmse": value,
        "p50": value,
        "p99": value,
        "p99_9": value,
        "max": value,
    }


def _open_loop_report() -> dict[str, object]:
    return {
        "seed": 1,
        "reward_mismatches": 0,
        "status_mismatches": 0,
        "termination_mismatches": 0,
        "closest_distance_error_m": _distribution(0.01),
        "rocket_position_error_m": _distribution(0.02),
        "sensor_absolute_error": _distribution(0.01),
    }


def _balloon_report() -> dict[str, object]:
    return {
        "seed": 0,
        "status_mismatches": 0,
        "release_time_error_s": _distribution(0.0),
        "position_error_m": _distribution(0.01),
        "velocity_error_m_s": _distribution(0.01),
    }


def _closed_loop_report() -> dict[str, object]:
    return {
        "seed": 2,
        "official_score": 1,
        "tensor_score": 1,
        "official_hits": [{"identity": 7, "time": 3.0}],
        "tensor_hits": [{"identity": 7, "time": 3.01}],
        "action_max_component_error": _distribution(0.01),
        "termination": {
            "official_terminated": True,
            "official_truncated": False,
            "tensor_terminated": True,
            "tensor_truncated": False,
            "official_time": 8.0,
            "tensor_time": 8.01,
        },
        "hit_coverage": True,
    }


def test_transfer_gates_accept_matching_holdouts() -> None:
    decision = evaluate_transfer_gates(
        [_balloon_report()], [_open_loop_report()], [_closed_loop_report()]
    )

    assert decision["passed"]
    assert decision["failures"] == []
    assert decision["closed_loop_hit_coverage"]


def test_transfer_gates_reject_event_and_lifecycle_drift() -> None:
    report = copy.deepcopy(_closed_loop_report())
    report["tensor_hits"][0]["identity"] = 8
    report["termination"]["tensor_terminated"] = False
    report["termination"]["tensor_truncated"] = True
    report["termination"]["tensor_time"] = None

    decision = evaluate_transfer_gates(
        [_balloon_report()], [_open_loop_report()], [report]
    )

    assert not decision["passed"]
    assert any("hit identity mismatch" in item for item in decision["failures"])
    assert any("terminated mismatch" in item for item in decision["failures"])
    assert any("truncated mismatch" in item for item in decision["failures"])
    assert any(
        "incomplete termination coverage" in item for item in decision["failures"]
    )


def test_transfer_gates_reject_open_loop_metric_regression() -> None:
    report = _open_loop_report()
    report["closest_distance_error_m"] = _distribution(0.051)
    report["rocket_position_error_m"] = _distribution(0.051)
    report["sensor_absolute_error"] = _distribution(0.051)
    report["reward_mismatches"] = 1

    decision = evaluate_transfer_gates(
        [_balloon_report()], [report], [_closed_loop_report()]
    )

    assert not decision["passed"]
    assert len(decision["failures"]) == 4


def test_transfer_gate_does_not_claim_event_parity_without_a_hit() -> None:
    report = _closed_loop_report()
    report["official_score"] = 0
    report["tensor_score"] = 0
    report["official_hits"] = []
    report["tensor_hits"] = []
    report["hit_coverage"] = False

    decision = evaluate_transfer_gates(
        [_balloon_report()], [_open_loop_report()], [report]
    )

    assert decision["numerical_and_lifecycle_passed"]
    assert not decision["passed"]
    assert decision["coverage_failures"] == ["closed-loop hit coverage missing"]


def test_transfer_gates_reject_balloon_only_regression() -> None:
    report = _balloon_report()
    report["status_mismatches"] = 1
    report["position_error_m"] = _distribution(0.051)

    decision = evaluate_transfer_gates(
        [report], [_open_loop_report()], [_closed_loop_report()]
    )

    assert not decision["passed"]
    assert any("status mismatch" in item for item in decision["failures"])
    assert any("position p99" in item for item in decision["failures"])
