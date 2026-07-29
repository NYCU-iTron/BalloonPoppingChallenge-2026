"""Bench test for the attitude controller against the real simulator.

Drives the actual BalloonPoppingEnv rocket with a scripted desired_acc profile:
no navigator, no selector, no reward. The estimator sits in the loop because
that is what the controller sees in flight, while the metrics come from
info["rocket_states"], the simulator's ground truth.

The pass criterion is acceleration tracking -- does the vehicle actually deliver
the commanded desired_acc -- not the tilt it uses to get there. Tilt is only a
means: the fins oppose a sustained lateral command (once the vehicle is turning,
the velocity vector swings round faster than the body, so the angle of attack
reverses and the aerodynamic normal force pushes back), and the controller has
to lean further than a thrust-only calculation would suggest to overcome it.
Grading tilt against a thrust-only formula would flag correct behaviour.

Cases span the engagement band (the balloon column never exceeds ~185 m, so
that is where the mission happens), plus command reversal, large-angle slew,
gusts and the unpowered coast after burnout.
"""

import copy

import numpy as np

from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
from BalloonPoppingGymEnv.agents.gnc.controller import Controller
from BalloonPoppingGymEnv.agents.gnc.estimator import Estimator
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters

SCENARIO = 1
CLIMB_ACC = np.array([0.0, 0.0, 2.0])
SETTLE_WINDOW = 1.0      # (s) averaging window at the end of each case


def simulate(command, climb_seconds=8.0, duration=20.0, gusts=False) -> dict:
    scenario_parameters, given_parameters = load_scenario_parameters(SCENARIO)

    parameters = copy.deepcopy(scenario_parameters)
    parameters["balloon"]["num"] = 1          # balloons play no part here
    if gusts:
        parameters["environment"]["gust"]["enable"] = True

    env = BalloonPoppingEnv(render_mode=None, parameters=parameters)
    controller = Controller(given_parameters)
    estimator = Estimator(given_parameters)
    controller.reset()
    estimator.reset()

    dt = parameters["simulation"]["time_step"]
    observation, info = env.reset(seed=parameters["scenario"]["random_seed"])
    heading = np.array([90.0, 0.0])

    log = {"time": [], "commanded": [], "achieved": [], "tilt": [],
           "estimated_tilt": [], "gimbal": [], "throttle": [], "trim": [],
           "altitude": [], "speed": []}
    previous_velocity = None

    for _ in range(int(duration / dt)):
        sim_time = float(observation["simulation_time"])
        rocket_state = estimator.estimate_rocket(observation)
        desired_acc = CLIMB_ACC if sim_time < climb_seconds else np.asarray(
            command(sim_time - climb_seconds) if callable(command) else command, dtype=float)

        controller.update(rocket_state, sim_time)
        tvc, roll, throttle = controller.compute(desired_acc)
        trim = float(np.linalg.norm(controller.acc_trim))

        observation, _reward, terminated, _truncated, info = env.step({
            "launch": True,
            "launch_inclination_heading": heading,
            "tvc": tvc,
            "roll": roll,
            "throttle": throttle,
        })

        truth = np.asarray(info["rocket_states"], dtype=float)
        if np.all(np.isfinite(truth)):
            velocity = truth[3:6]
            if previous_velocity is not None:
                log["time"].append(sim_time)
                log["commanded"].append(desired_acc.copy())
                log["achieved"].append((velocity - previous_velocity) / dt)
                log["tilt"].append(_tilt_degrees(truth[6:10]))
                log["estimated_tilt"].append(_tilt_degrees(rocket_state[9:13]))
                log["gimbal"].append(float(np.max(np.abs(tvc))))
                log["throttle"].append(throttle)
                log["trim"].append(trim)
                log["altitude"].append(truth[2])
                log["speed"].append(float(np.linalg.norm(velocity)))
            previous_velocity = velocity.copy()

        if terminated:
            break

    env.close()

    out = {key: np.asarray(value) for key, value in log.items()}
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
    blank = dict.fromkeys(["acc_error", "peak_error", "rise", "tilt", "saturation",
                           "drift", "trim", "altitude", "speed"], np.nan)
    if log["time"].size == 0:
        return blank

    after = log["time"] >= log["step_time"]
    if not after.any():
        return blank

    time = log["time"][after]
    commanded = log["commanded"][after]
    achieved = log["achieved"][after]
    error = np.linalg.norm(achieved - commanded, axis=1)
    window = max(int(SETTLE_WINDOW / log["dt"]), 1)

    # Time for the delivered acceleration to reach 90% of the command along the
    # commanded direction. Undefined when nothing was asked for.
    magnitude = float(np.linalg.norm(commanded[-1]))
    if magnitude > 0.5:
        direction = commanded[-1] / magnitude
        projection = achieved @ direction
        reached = np.flatnonzero(projection >= 0.9 * magnitude)
        rise = float(time[reached[0]] - log["step_time"]) if reached.size else np.nan
    else:
        rise = 0.0

    return {
        "acc_error": float(error[-window:].mean()),
        "peak_error": float(error.max()),
        "rise": rise,
        "tilt": float(log["tilt"][after][-window:].mean()),
        "saturation": float(np.mean(log["gimbal"][after] >= 14.9)),
        "drift": float(np.max(np.abs(log["estimated_tilt"] - log["tilt"]))),
        "trim": float(log["trim"][after].max()),
        "altitude": float(log["altitude"][-1]),
        "speed": float(log["speed"][-1]),
    }


def _reversal(elapsed):
    """Flip the lateral command every two seconds."""
    return (4.0 if int(elapsed // 2.0) % 2 == 0 else -4.0, 0.0, 0.0)


# name, command, simulate kwargs, max steady |acc error| (m/s^2)
CASES = [
    ("hold vertical",           (0.0, 0.0, 0.0),  {},                              0.5),
    ("tilt in engagement band", (4.0, 0.0, 0.0),  {"climb_seconds": 8.0},          2.0),
    ("tilt above limiter",      (4.0, 0.0, 0.0),  {"climb_seconds": 16.0,
                                                   "duration": 26.0},              2.0),
    ("large angle slew",        (9.0, 0.0, 0.0),  {"climb_seconds": 16.0,
                                                   "duration": 26.0},              2.5),
    ("command reversal",        _reversal,        {"climb_seconds": 16.0,
                                                   "duration": 28.0},              5.0),
    ("gust disturbance",        (4.0, 0.0, 0.0),  {"climb_seconds": 5.0,
                                                   "gusts": True},                 2.5),
    # Unpowered: nothing can be tracked once the motor is out, so this only
    # guards against divergence -- finite states and a sane estimator.
    ("past burnout",            (4.0, 0.0, 0.0),  {"climb_seconds": 16.0,
                                                   "duration": 40.0},             99.0),
]


def run_all(verbose: bool = True) -> list:
    results = []
    if verbose:
        print(f"{'case':<26}{'accerr':>9}{'peak':>8}{'rise':>8}{'tilt':>8}"
              f"{'sat':>6}{'drift':>8}{'trim':>7}{'alt':>8}{'speed':>7}")
        print("-" * 97)

    for name, command, kwargs, tolerance in CASES:
        m = metrics(simulate(command, **kwargs))
        ok = np.isfinite(m["acc_error"]) and m["acc_error"] <= tolerance
        results.append((name, m, ok))
        if verbose:
            print(f"{name:<26}{m['acc_error']:>9.2f}{m['peak_error']:>8.2f}"
                  f"{m['rise']:>7.2f}s{m['tilt']:>7.1f}°{m['saturation']:>6.0%}"
                  f"{m['drift']:>7.1f}°{m['trim']:>7.2f}{m['altitude']:>7.0f}m"
                  f"{m['speed']:>6.0f}  {'ok' if ok else 'FAIL'}")

    if verbose:
        failed = [name for name, _, ok in results if not ok]
        print("-" * 97)
        print("all cases passed" if not failed else f"failed: {', '.join(failed)}")
    return results


def test_controller_delivers_commanded_acceleration():
    for name, m, ok in run_all(verbose=False):
        assert ok, f"{name}: steady acceleration error {m['acc_error']:.2f} m/s^2"


def test_estimator_attitude_stays_close_to_truth():
    """Guards the controller's input: gyro integration must not drift away."""
    m = metrics(simulate((4.0, 0.0, 0.0)))
    assert m["drift"] < 5.0, f"estimated tilt drifted {m['drift']:.1f} deg from truth"


def test_trim_does_not_run_to_its_limit():
    """A trim pinned at the limit means the aero load exceeds what it can absorb."""
    m = metrics(simulate((4.0, 0.0, 0.0), climb_seconds=16.0, duration=26.0))
    assert m["trim"] < 0.95 * Controller.ACC_TRIM_LIMIT, (
        f"trim reached {m['trim']:.1f} of {Controller.ACC_TRIM_LIMIT} m/s^2")


if __name__ == "__main__":
    run_all()
