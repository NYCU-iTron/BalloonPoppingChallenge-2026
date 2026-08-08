"""Record versioned ActiveRocketPy traces without modifying the official env.

The trace is a development oracle, not agent input.  It intentionally records
both the four observation fields visible to an agent and privileged diagnostic
values such as the true 13-state, actuator outputs, and the active RHS.

Examples
--------
Record one short launch/control trace::

    python -m experiments.cuda.canonical_oracle --case launch_control \
        --output .artifacts/cuda/launch_control.npz

Record the complete fixed corpus::

    python -m experiments.cuda.canonical_oracle --corpus-dir .artifacts/cuda/oracle
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters


TRACE_SCHEMA_VERSION = 1
ACTION_FIELDS = (
    "launch",
    "inclination",
    "heading",
    "tvc_x",
    "tvc_y",
    "throttle",
    "roll",
)
ACTUATOR_FIELDS = ("roll", "tvc_x", "tvc_y", "throttle")
EVENT_FIELDS = (
    "reset",
    "pre_launch",
    "launch_step",
    "first_post_launch",
    "burnout_crossing",
    "hit",
    "impact",
    "truncated",
)

OfficialAction = dict[str, object]
ActionSchedule = Callable[[int, float], OfficialAction]


@dataclass(frozen=True)
class OracleCase:
    name: str
    scenario: int
    seed: int
    balloons: int
    steps: int
    schedule: str
    max_time: float | None = None


FIXED_CASES = {
    "launch_control": OracleCase(
        name="launch_control",
        scenario=0,
        seed=2026,
        balloons=1,
        steps=400,
        schedule="control_sweep",
    ),
    "burnout": OracleCase(
        name="burnout",
        scenario=0,
        seed=2027,
        balloons=1,
        steps=3_105,
        schedule="vertical_full",
    ),
    "impact": OracleCase(
        name="impact",
        scenario=0,
        seed=2028,
        balloons=1,
        steps=3_000,
        schedule="cutoff",
    ),
    "timeout_without_launch": OracleCase(
        name="timeout_without_launch",
        scenario=0,
        seed=2029,
        balloons=1,
        steps=10,
        schedule="never_launch",
        max_time=0.05,
    ),
    "moving_balloon": OracleCase(
        name="moving_balloon",
        scenario=1,
        seed=2030,
        balloons=1,
        steps=400,
        schedule="control_sweep",
    ),
}


@dataclass(frozen=True)
class CanonicalTrace:
    metadata: dict[str, object]
    arrays: dict[str, np.ndarray]

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(self.arrays)
        payload["metadata_json"] = np.asarray(
            json.dumps(self.metadata, sort_keys=True, separators=(",", ":"))
        )
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: Path) -> CanonicalTrace:
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"]))
            arrays = {
                name: archive[name].copy()
                for name in archive.files
                if name != "metadata_json"
            }
        trace = cls(metadata=metadata, arrays=arrays)
        trace.validate()
        return trace

    def validate(self) -> None:
        if self.metadata.get("trace_schema_version") != TRACE_SCHEMA_VERSION:
            raise ValueError("unsupported canonical trace schema")
        required = {
            "step_index",
            "time",
            "action",
            "rocket_sensors",
            "rocket_state",
            "rhs",
            "actuator_output",
            "balloon_states",
            "balloon_status",
            "balloon_release",
            "swept_distance",
            "reward",
            "popped_count",
            "terminated",
            "truncated",
            "sensor_finite",
            "phase_index",
            "events",
        }
        missing = required - self.arrays.keys()
        if missing:
            raise ValueError(f"canonical trace is missing arrays: {sorted(missing)}")
        frames = self.arrays["time"].shape[0]
        if any(self.arrays[name].shape[0] != frames for name in required):
            raise ValueError("all canonical trace arrays must have one row per frame")
        if self.arrays["action"].shape != (frames, len(ACTION_FIELDS)):
            raise ValueError("action array has the wrong shape")
        if self.arrays["events"].shape != (frames, len(EVENT_FIELDS)):
            raise ValueError("events array has the wrong shape")


def _base_action(*, launch: bool, throttle: float) -> OfficialAction:
    return {
        "launch": launch,
        "launch_inclination_heading": np.array([90.0, 0.0]),
        "tvc": np.zeros(2),
        "roll": 0.0,
        "throttle": throttle,
    }


def _launch_state(call_index: int) -> bool:
    # call 0 is the explicit pre-launch frame; call 1 issues launch.
    return call_index >= 1


def control_sweep_action(call_index: int, dt: float) -> OfficialAction:
    launch = _launch_state(call_index)
    action = _base_action(launch=launch, throttle=1.0 if launch else 0.0)
    if call_index < 2:
        return action

    elapsed = (call_index - 2) * dt
    window = int(elapsed / 0.5) % 4
    signs = (0.0, 1.0, -1.0, 0.0)
    sign = signs[window]
    action["tvc"] = np.array([15.0 * sign, -15.0 * sign])
    action["roll"] = 10.0 * sign
    action["throttle"] = (1.0, 0.0, 1.0, 0.5)[window]
    return action


def vertical_full_action(call_index: int, dt: float) -> OfficialAction:
    del dt
    return _base_action(launch=_launch_state(call_index), throttle=1.0)


def cutoff_action(call_index: int, dt: float) -> OfficialAction:
    del dt
    return _base_action(
        launch=_launch_state(call_index),
        throttle=0.0 if call_index >= 2 else float(call_index == 1),
    )


def never_launch_action(call_index: int, dt: float) -> OfficialAction:
    del call_index, dt
    return _base_action(launch=False, throttle=0.0)


SCHEDULES: dict[str, ActionSchedule] = {
    "control_sweep": control_sweep_action,
    "vertical_full": vertical_full_action,
    "cutoff": cutoff_action,
    "never_launch": never_launch_action,
}


def _action_vector(action: Mapping[str, object]) -> np.ndarray:
    attitude = np.asarray(action["launch_inclination_heading"], dtype=np.float64)
    tvc = np.asarray(action["tvc"], dtype=np.float64)
    return np.array(
        [
            float(bool(action["launch"])),
            attitude[0],
            attitude[1],
            tvc[0],
            tvc[1],
            float(action["throttle"]),
            float(action["roll"]),
        ]
    )


def _git_revision(path: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def _scenario_hash(parameters: Mapping[str, object]) -> str:
    encoded = json.dumps(parameters, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _actuator_outputs(env: BalloonPoppingEnv) -> np.ndarray:
    if env._rocket_flight is None:  # noqa: SLF001
        return np.full(4, np.nan)
    rocket = env._rocket_flight.rocket  # noqa: SLF001
    return np.array(
        [
            rocket.roll_control.roll_torque,
            rocket.thrust_vector_control.gimbal_angle_x,
            rocket.thrust_vector_control.gimbal_angle_y,
            rocket.throttle_control.throttle,
        ],
        dtype=np.float64,
    )


def _active_rhs(env: BalloonPoppingEnv) -> tuple[np.ndarray, int]:
    flight = env._rocket_flight  # noqa: SLF001
    if flight is None or not np.isfinite(env._rocket_states).all():  # noqa: SLF001
        return np.full(13, np.nan), -1
    phase_index = min(
        int(flight._step_state["phase_index"]),  # noqa: SLF001
        len(flight.flight_phases) - 2,
    )
    derivative = flight.flight_phases[phase_index].derivative
    if derivative is None:
        return np.full(13, np.nan), phase_index
    try:
        rhs = np.asarray(derivative(flight.t, flight.y_sol), dtype=np.float64)
    except (AttributeError, IndexError, RuntimeError, ValueError):
        return np.full(13, np.nan), phase_index
    return rhs.reshape(13), phase_index


def _geometry_oracle(radius: float) -> dict[str, np.ndarray]:
    rocket_start = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    rocket_end = np.array([[10.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    balloon_start = np.array([[8.0, -5.0, 0.0], [8.0, 3.0, 0.0]])
    balloon_end = np.array([[8.0, 5.0, 0.0], [8.0, 3.0, 0.0]])
    distance = np.empty(2)
    for index in range(2):
        squared = BalloonPoppingEnv._segment_distance_squared_batch(
            rocket_start[index],
            rocket_end[index],
            balloon_start[index : index + 1],
            balloon_end[index : index + 1],
        )
        distance[index] = np.sqrt(squared[0])
    return {
        "geometry_rocket_start": rocket_start,
        "geometry_rocket_end": rocket_end,
        "geometry_balloon_start": balloon_start,
        "geometry_balloon_end": balloon_end,
        "geometry_distance": distance,
        "geometry_hit": distance <= radius,
    }


def record_case(case: OracleCase) -> CanonicalTrace:
    parameters, _ = load_scenario_parameters(case.scenario)
    parameters["balloon"]["num"] = case.balloons
    if case.max_time is not None:
        parameters["simulation"]["max_time"] = case.max_time
    dt = float(parameters["simulation"]["time_step"])
    schedule = SCHEDULES[case.schedule]
    env = BalloonPoppingEnv(render_mode=None, parameters=parameters)

    rows: dict[str, list[np.ndarray | float | int | bool]] = {
        "step_index": [],
        "time": [],
        "action": [],
        "rocket_sensors": [],
        "rocket_state": [],
        "rhs": [],
        "actuator_output": [],
        "balloon_states": [],
        "balloon_status": [],
        "balloon_release": [],
        "swept_distance": [],
        "reward": [],
        "popped_count": [],
        "terminated": [],
        "truncated": [],
        "sensor_finite": [],
        "phase_index": [],
        "events": [],
    }

    previous_status: np.ndarray | None = None

    def append_frame(
        observation: Mapping[str, object],
        info: Mapping[str, object],
        *,
        step_index: int,
        action: Mapping[str, object] | None,
        reward: float,
        terminated: bool,
        truncated: bool,
        events: np.ndarray,
        swept_distance: np.ndarray | None = None,
    ) -> None:
        rhs, phase_index = _active_rhs(env)
        rows["step_index"].append(step_index)
        rows["time"].append(float(observation["simulation_time"]))
        rows["action"].append(
            np.full(len(ACTION_FIELDS), np.nan)
            if action is None
            else _action_vector(action)
        )
        sensors = np.asarray(observation["rocket_sensors"], dtype=np.float64)
        rows["rocket_sensors"].append(sensors.copy())
        rows["rocket_state"].append(
            np.asarray(info["rocket_states"], dtype=np.float64).copy()
        )
        rows["rhs"].append(rhs)
        rows["actuator_output"].append(_actuator_outputs(env))
        rows["balloon_states"].append(
            np.asarray(observation["balloon_states"], dtype=np.float64).copy()
        )
        nonlocal previous_status
        current_status = (
            np.asarray(observation["balloon_status"], dtype=np.int64).reshape(-1).copy()
        )
        rows["balloon_status"].append(current_status)
        released = (
            current_status == 1
            if previous_status is None
            else (previous_status == 0) & (current_status == 1)
        )
        rows["balloon_release"].append(released)
        previous_status = current_status
        rows["swept_distance"].append(
            np.full(case.balloons, np.nan) if swept_distance is None else swept_distance
        )
        rows["reward"].append(float(reward))
        rows["popped_count"].append(int(info["popped_count"]))
        rows["terminated"].append(bool(terminated))
        rows["truncated"].append(bool(truncated))
        rows["sensor_finite"].append(bool(np.isfinite(sensors).all()))
        rows["phase_index"].append(phase_index)
        rows["events"].append(events.copy())

    try:
        observation, info = env.reset(seed=case.seed)
        reset_events = np.zeros(len(EVENT_FIELDS), dtype=bool)
        reset_events[EVENT_FIELDS.index("reset")] = True
        append_frame(
            observation,
            info,
            step_index=0,
            action=None,
            reward=0.0,
            terminated=False,
            truncated=False,
            events=reset_events,
        )

        launch_time: float | None = None
        burnout_seen = False
        for call_index in range(case.steps):
            action = schedule(call_index, dt)
            was_launched = env.rocket_launched
            sensors_were_finite = bool(np.isfinite(env._rocket_sensors).all())  # noqa: SLF001
            balloon_start = env._balloon_states[:, :3].copy()  # noqa: SLF001
            rocket_start = (
                None
                if env._sweep_origin is None  # noqa: SLF001
                else env._sweep_origin.copy()  # noqa: SLF001
            )
            with warnings.catch_warnings():
                # Rate-limit saturation is intentional in the control corpus;
                # the actual limited output is recorded explicitly below.
                warnings.filterwarnings(
                    "ignore",
                    message=r"Actuator .* output change .* exceeds rate limit.*",
                )
                observation, reward, terminated, truncated, info = env.step(action)

            launched_now = not was_launched and env.rocket_launched
            sensors_are_finite = bool(
                np.isfinite(np.asarray(observation["rocket_sensors"])).all()
            )
            first_post_launch = (
                was_launched and not sensors_were_finite and sensors_are_finite
            )
            if launch_time is None and launched_now:
                launch_time = float(observation["simulation_time"])

            swept_distance = None
            rocket_state = np.asarray(info["rocket_states"], dtype=np.float64)
            if rocket_start is not None and np.isfinite(rocket_state[:3]).all():
                squared = BalloonPoppingEnv._segment_distance_squared_batch(
                    rocket_start,
                    rocket_state[:3],
                    balloon_start,
                    np.asarray(observation["balloon_states"], dtype=np.float64)[:, :3],
                )
                swept_distance = np.sqrt(squared)

            events = np.zeros(len(EVENT_FIELDS), dtype=bool)
            if not env.rocket_launched:
                events[EVENT_FIELDS.index("pre_launch")] = True
            if launched_now:
                events[EVENT_FIELDS.index("launch_step")] = True
            if first_post_launch:
                events[EVENT_FIELDS.index("first_post_launch")] = True
            if reward > 0:
                events[EVENT_FIELDS.index("hit")] = True
            if terminated:
                events[EVENT_FIELDS.index("impact")] = True
            if truncated:
                events[EVENT_FIELDS.index("truncated")] = True

            if launch_time is not None and env._rocket_flight is not None:  # noqa: SLF001
                burn_time = float(parameters["rocket"]["motor"]["burn_time"])
                burnout = env._rocket_flight.t >= launch_time + burn_time  # noqa: SLF001
                if burnout and not burnout_seen:
                    events[EVENT_FIELDS.index("burnout_crossing")] = True
                    burnout_seen = True

            append_frame(
                observation,
                info,
                step_index=call_index + 1,
                action=action,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                events=events,
                swept_distance=swept_distance,
            )
            if terminated or truncated:
                break
    finally:
        env.close()

    root = Path(__file__).resolve().parents[2]
    arrays = {name: np.asarray(values) for name, values in rows.items()}
    arrays.update(_geometry_oracle(float(parameters["balloon"]["radius"])))
    metadata: dict[str, object] = {
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "case": case.name,
        "scenario": case.scenario,
        "seed": case.seed,
        "balloons": case.balloons,
        "requested_steps": case.steps,
        "recorded_frames": int(arrays["time"].shape[0]),
        "schedule": case.schedule,
        "time_step": dt,
        "action_fields": ACTION_FIELDS,
        "actuator_fields": ACTUATOR_FIELDS,
        "event_fields": EVENT_FIELDS,
        "scenario_sha256": _scenario_hash(parameters),
        "repository_revision": _git_revision(root),
        "active_rocketpy_revision": _git_revision(root / "ActiveRocketPy"),
        "oracle_only_fields": (
            "rocket_state",
            "rhs",
            "actuator_output",
            "phase_index",
            "swept_distance",
        ),
    }
    trace = CanonicalTrace(metadata=metadata, arrays=arrays)
    trace.validate()
    return trace


def _record_one(case: OracleCase, output: Path) -> None:
    trace = record_case(case)
    trace.save(output)
    event_counts = trace.arrays["events"].sum(axis=0)
    print(f"saved {trace.arrays['time'].shape[0]} frames to {output}")
    print(
        "events: "
        + ", ".join(
            f"{name}={int(count)}" for name, count in zip(EVENT_FIELDS, event_counts)
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=tuple(FIXED_CASES), default="launch_control")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        help="record every fixed case into this directory",
    )
    parser.add_argument("--steps", type=int, help="override the selected case length")
    args = parser.parse_args()

    if args.corpus_dir is not None:
        for case in FIXED_CASES.values():
            _record_one(case, args.corpus_dir / f"{case.name}.npz")
        return

    case = FIXED_CASES[args.case]
    if args.steps is not None:
        if args.steps < 1:
            parser.error("steps must be positive")
        case = OracleCase(
            name=case.name,
            scenario=case.scenario,
            seed=case.seed,
            balloons=case.balloons,
            steps=args.steps,
            schedule=case.schedule,
            max_time=case.max_time,
        )
    output = args.output or Path(".artifacts/cuda/oracle") / f"{case.name}.npz"
    _record_one(case, output)


if __name__ == "__main__":
    main()
