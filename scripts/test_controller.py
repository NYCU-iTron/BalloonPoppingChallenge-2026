"""Bench test for the attitude controller against the real simulator.

Drives the actual BalloonPoppingEnv rocket with a scripted desired_acc profile:
no navigator, no selector, no reward. The estimator sits in the loop because
that is what the controller sees in flight, while the pass/fail metrics come
from info["rocket_states"], the simulator's ground truth.

Each case climbs vertically to clear the low-altitude tilt limiter, then steps
the command and measures how the vehicle tracks it.

Run directly for a report table, or under pytest for pass/fail.
"""

import copy

import numpy as np

from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
from BalloonPoppingGymEnv.agents.gnc.controller import Controller
from BalloonPoppingGymEnv.agents.gnc.estimator import Estimator
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters

SCENARIO = 1
CLIMB_SECONDS = 16.0      # vertical climb before the step command
DURATION = 26.0          # simulated seconds per case
GRAVITY_Z = 9.81


def simulate(command, climb_seconds=CLIMB_SECONDS, duration=DURATION) -> dict:
    scenario_parameters, given_parameters = load_scenario_parameters(SCENARIO)

    # Balloons play no part here; one keeps flight generation cheap.
    parameters = copy.deepcopy(scenario_parameters)
    parameters["balloon"]["num"] = 1

    env = BalloonPoppingEnv(render_mode=None, parameters=parameters)
    controller = Controller(given_parameters)
    estimator = Estimator(given_parameters)
    controller.reset()
    estimator.reset()

    dt = parameters["simulation"]["time_step"]
    observation, info = env.reset(seed=parameters["scenario"]["random_seed"])

    command = np.asarray(command, dtype=float)
    climb = np.array([0.0, 0.0, 2.0])
    heading = np.array([90.0, 0.0])

    log = {"time": [], "tilt": [], "estimated_tilt": [], "gimbal": [],
           "throttle": [], "altitude": [], "speed": []}

    for _ in range(int(duration / dt)):
        sim_time = float(observation["simulation_time"])
        rocket_state = estimator.estimate_rocket(observation)
        desired_acc = climb if sim_time < climb_seconds else command

        controller.update(rocket_state, sim_time)
        tvc, roll, throttle = controller.compute(desired_acc)

        observation, _reward, terminated, _truncated, info = env.step({
            "launch": True,
            "launch_inclination_heading": heading,
            "tvc": tvc,
            "roll": roll,
            "throttle": throttle,
        })

        truth = np.asarray(info["rocket_states"], dtype=float)
        if np.all(np.isfinite(truth)):
            log["time"].append(sim_time)
            log["tilt"].append(_tilt_degrees(truth[6:10]))
            log["estimated_tilt"].append(_tilt_degrees(rocket_state[9:13]))
            log["gimbal"].append(float(np.max(np.abs(tvc))))
            log["throttle"].append(throttle)
            log["altitude"].append(truth[2])
            log["speed"].append(float(np.linalg.norm(truth[3:6])))

        if terminated:
            break

    env.close()

    out = {key: np.asarray(value) for key, value in log.items()}
    out["target"] = np.degrees(np.arctan2(np.hypot(command[0], command[1]),
                                          command[2] + GRAVITY_Z))
    out["step_time"] = climb_seconds
    out["dt"] = dt
    return out


def _tilt_degrees(quaternion) -> float:
    """Angle between the body thrust axis and world up, from a wxyz quaternion."""
    qw, qx, qy, qz = np.asarray(quaternion, dtype=float)
    body_z = np.array([2 * (qx * qz + qw * qy),
                       2 * (qy * qz - qw * qx),
                       1 - 2 * (qx * qx + qy * qy)])
    return float(np.degrees(np.arctan2(np.hypot(body_z[0], body_z[1]), body_z[2])))


def metrics(log: dict) -> dict:
    """Response quality over the window after the step command."""
    empty = {"target": log["target"], "settled": np.nan, "error": np.nan,
             "rise": np.nan, "overshoot": np.nan, "saturation": np.nan,
             "drift": np.nan, "altitude": np.nan, "speed": np.nan}
    if log["time"].size == 0:
        return empty

    after = log["time"] >= log["step_time"]
    tilt, time = log["tilt"][after], log["time"][after]
    if tilt.size == 0:
        return empty

    target = log["target"]
    settled = float(tilt[-max(int(0.5 / log["dt"]), 1):].mean())

    if target > 1.0:
        reached = np.flatnonzero(tilt >= 0.9 * target)
        rise = float(time[reached[0]] - log["step_time"]) if reached.size else np.nan
        overshoot = float(tilt.max() / target - 1.0)
    else:
        rise, overshoot = 0.0, 0.0

    return {
        "target": target,
        "settled": settled,
        "error": settled - target,
        "rise": rise,
        "overshoot": overshoot,
        "saturation": float(np.mean(log["gimbal"][after] >= 14.9)),
        "drift": float(np.max(np.abs(log["estimated_tilt"] - log["tilt"]))),
        "altitude": float(log["altitude"][-1]),
        "speed": float(log["speed"][-1]),
    }


# name, desired_acc after the climb, max |steady error| deg, max overshoot
CASES = [
    ("hold vertical",         (0.0, 0.0, 0.0),   3.0, 0.35),
    ("step tilt east",        (4.0, 0.0, 0.0),   5.0, 0.35),
    ("step tilt diagonal",    (3.0, 3.0, 0.0),   5.0, 0.35),
    ("tilt while climbing",   (4.0, 0.0, 2.0),   5.0, 0.35),
    ("tilt while descending", (2.0, 0.0, -3.0),  8.0, 0.50),
]


def run_all(verbose: bool = True) -> list:
    results = []
    if verbose:
        print(f"{'case':<24}{'target':>8}{'settled':>9}{'err':>7}{'rise':>8}"
              f"{'over':>7}{'sat':>6}{'estdrift':>10}{'alt':>8}{'speed':>7}")
        print("-" * 96)

    for name, command, error_tolerance, overshoot_tolerance in CASES:
        m = metrics(simulate(command))
        ok = (abs(m["error"]) <= error_tolerance
              and m["overshoot"] <= overshoot_tolerance)
        results.append((name, m, ok))
        if verbose:
            print(f"{name:<24}{m['target']:>7.1f}°{m['settled']:>8.1f}°{m['error']:>6.1f}°"
                  f"{m['rise']:>7.2f}s{m['overshoot']:>6.0%}{m['saturation']:>6.0%}"
                  f"{m['drift']:>9.1f}°{m['altitude']:>7.0f}m{m['speed']:>6.0f}  "
                  f"{'ok' if ok else 'FAIL'}")

    if verbose:
        failed = [name for name, _, ok in results if not ok]
        print("-" * 96)
        print("all cases passed" if not failed else f"failed: {', '.join(failed)}")
    return results


def test_controller_tracks_commanded_tilt():
    for name, m, ok in run_all(verbose=False):
        assert ok, (f"{name}: settled {m['settled']:.1f} deg "
                    f"against target {m['target']:.1f} deg")


def test_estimator_attitude_stays_close_to_truth():
    """Guards the controller's input: gyro integration must not drift away."""
    m = metrics(simulate((4.0, 0.0, 0.0)))
    assert m["drift"] < 5.0, f"estimated tilt drifted {m['drift']:.1f} deg from truth"


if __name__ == "__main__":
    run_all()
