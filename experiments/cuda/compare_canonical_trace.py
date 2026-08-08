"""Quantify the current non-canonical TensorFlight baseline against an oracle.

This deliberately reports error rather than asserting fidelity.  The existing
tensor model is a throughput surrogate; Phase 2 will replace its simplified
equations only after the oracle and comparison pipeline are stable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters
from experiments.cuda.canonical_oracle import CanonicalTrace, EVENT_FIELDS
from experiments.cuda.tensor_flight import TensorFlightBatch, TensorFlightConfig


def _distribution(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {name: float("nan") for name in ("rmse", "p50", "p95", "p99", "max")}
    return {
        "rmse": float(np.sqrt(np.mean(finite * finite))),
        "p50": float(np.percentile(finite, 50)),
        "p95": float(np.percentile(finite, 95)),
        "p99": float(np.percentile(finite, 99)),
        "max": float(np.max(finite)),
    }


def _attitude_error_degrees(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    actual_norm = actual / np.linalg.norm(actual, axis=1, keepdims=True)
    expected_norm = expected / np.linalg.norm(expected, axis=1, keepdims=True)
    dot = np.abs(np.sum(actual_norm * expected_norm, axis=1))
    return np.degrees(2.0 * np.arccos(np.clip(dot, 0.0, 1.0)))


def _normalized_post_launch_action(
    action: np.ndarray,
    control: dict[str, object],
) -> np.ndarray:
    throttle_low, throttle_high = control["throttle_range"]
    throttle_width = float(throttle_high) - float(throttle_low)
    if throttle_width <= 0:
        raise ValueError("throttle_range must have positive width")
    return np.array(
        [
            action[6] / float(control["max_roll_torque"]),
            action[3] / float(control["max_gimbal_angle"]),
            action[4] / float(control["max_gimbal_angle"]),
            2.0 * (action[5] - float(throttle_low)) / throttle_width - 1.0,
        ],
        dtype=np.float64,
    )


def compare_trace(trace: CanonicalTrace) -> dict[str, object]:
    trace.validate()
    first_post_column = EVENT_FIELDS.index("first_post_launch")
    anchors = np.flatnonzero(trace.arrays["events"][:, first_post_column])
    if anchors.size != 1:
        raise ValueError("trace must contain exactly one first_post_launch event")
    anchor = int(anchors[0])
    canonical = trace.arrays["rocket_state"]
    if not np.isfinite(canonical[anchor]).all():
        raise ValueError("first post-launch canonical state must be finite")

    scenario = int(trace.metadata["scenario"])
    parameters, _ = load_scenario_parameters(scenario)
    control = parameters["rocket"]["control"]
    elevation = float(parameters["environment"]["elevation"])
    config = TensorFlightConfig(
        dt=float(trace.metadata["time_step"]),
        ground_altitude=elevation,
        initial_altitude=elevation,
        target_radius=float(parameters["balloon"]["radius"]),
    )
    environment = TensorFlightBatch(
        1,
        device="cpu",
        dtype=torch.float64,
        config=config,
    )
    environment.state.copy_(torch.from_numpy(canonical[anchor : anchor + 1]))

    canonical_actuator = trace.arrays["actuator_output"][anchor]
    initial_actuator = canonical_actuator.copy()
    initial_actuator[1:3] = np.radians(initial_actuator[1:3])
    environment.actuator_state.copy_(torch.from_numpy(initial_actuator.reshape(1, 4)))
    first_target = trace.arrays["balloon_states"][anchor, 0]
    environment.set_target(
        torch.from_numpy(first_target[:3].reshape(1, 3)),
        torch.from_numpy(first_target[3:6].reshape(1, 3)),
    )

    predicted_state: list[np.ndarray] = []
    predicted_target: list[np.ndarray] = []
    predicted_actuator: list[np.ndarray] = []
    predicted_hit: list[bool] = []
    predicted_terminated: list[bool] = []
    predicted_truncated: list[bool] = []
    compared_rows: list[int] = []

    for row in range(anchor + 1, canonical.shape[0]):
        action = trace.arrays["action"][row]
        if not np.isfinite(action).all():
            continue
        normalized = _normalized_post_launch_action(action, control)
        result = environment.step(torch.from_numpy(normalized.reshape(1, 4)))
        predicted_state.append(result.state[0].numpy().copy())
        predicted_target.append(environment.target_position[0].numpy().copy())
        predicted_actuator.append(environment.actuator_state[0].numpy().copy())
        predicted_hit.append(bool(result.hit[0]))
        predicted_terminated.append(bool(result.terminated[0]))
        predicted_truncated.append(bool(result.truncated[0]))
        compared_rows.append(row)

    if not compared_rows:
        raise ValueError("trace has no post-launch transitions to compare")

    rows = np.asarray(compared_rows, dtype=np.int64)
    actual = np.asarray(predicted_state)
    expected = canonical[rows]
    target_actual = np.asarray(predicted_target)
    target_expected = trace.arrays["balloon_states"][rows, 0, :3]
    actuator_actual = np.asarray(predicted_actuator)
    actuator_expected = trace.arrays["actuator_output"][rows].copy()
    actuator_actual[:, 1:3] = np.degrees(actuator_actual[:, 1:3])

    canonical_hit = trace.arrays["reward"][rows] > 0
    canonical_terminated = trace.arrays["terminated"][rows]
    canonical_truncated = trace.arrays["truncated"][rows]

    report: dict[str, object] = {
        "trace_schema_version": trace.metadata["trace_schema_version"],
        "case": trace.metadata["case"],
        "scenario": scenario,
        "seed": trace.metadata["seed"],
        "compared_transitions": int(rows.size),
        "anchor_frame": anchor,
        "non_canonical_prototype": True,
        "state_error": {
            "position_l2_m": _distribution(
                np.linalg.norm(actual[:, 0:3] - expected[:, 0:3], axis=1)
            ),
            "velocity_l2_mps": _distribution(
                np.linalg.norm(actual[:, 3:6] - expected[:, 3:6], axis=1)
            ),
            "attitude_angle_deg": _distribution(
                _attitude_error_degrees(actual[:, 6:10], expected[:, 6:10])
            ),
            "angular_rate_l2_radps": _distribution(
                np.linalg.norm(actual[:, 10:13] - expected[:, 10:13], axis=1)
            ),
        },
        "target_position_l2_m": _distribution(
            np.linalg.norm(target_actual - target_expected, axis=1)
        ),
        "actuator_error": {
            "roll_abs_nm": _distribution(
                np.abs(actuator_actual[:, 0] - actuator_expected[:, 0])
            ),
            "tvc_l2_deg": _distribution(
                np.linalg.norm(
                    actuator_actual[:, 1:3] - actuator_expected[:, 1:3], axis=1
                )
            ),
            "throttle_abs": _distribution(
                np.abs(actuator_actual[:, 3] - actuator_expected[:, 3])
            ),
        },
        "discrete_agreement": {
            "hit": float(np.mean(np.asarray(predicted_hit) == canonical_hit)),
            "terminated": float(
                np.mean(np.asarray(predicted_terminated) == canonical_terminated)
            ),
            "truncated": float(
                np.mean(np.asarray(predicted_truncated) == canonical_truncated)
            ),
        },
        "warning": (
            "This is a baseline measurement of a simplified throughput surrogate, "
            "not a RocketPy fidelity claim."
        ),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = compare_trace(CanonicalTrace.load(args.trace))
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
