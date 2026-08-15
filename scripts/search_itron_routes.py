"""Search fixed Scenario routes against cached Monte Carlo trajectories."""

import argparse
import hashlib
import itertools
import json
import math
import signal
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from BalloonPoppingGymEnv.agents.gnc.selector import Selector
from BalloonPoppingGymEnv.agents.itron_agent import ITronAgent
from BalloonPoppingGymEnv.envs.cached_env import (
    CachedBalloonEnv,
    scenario_parameter_fingerprint,
)
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters
from BalloonPoppingGymEnv.utils.schema import Schema


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAJECTORY_CACHE = REPOSITORY_ROOT / ".cache" / "balloon_trajectories"
DEFAULT_RESULT_CACHE = REPOSITORY_ROOT / ".cache" / "itron_route_search"
ROUTE_EXECUTION_VERSION = 1


@dataclass(frozen=True)
class RouteTask:
    scenario: int
    seed: int
    launch_step: int
    launch_time: float
    targets: tuple[int, ...]
    trajectory_cache_dir: str
    parameter_fingerprint: str
    implementation_fingerprint: str
    lookahead_guidance_weight: float = 0.0
    corridor_targets: tuple[tuple[int, int], ...] = ()
    corridor_guidance_weight: float = 0.0
    corridor_guidance_horizon: float = 0.0
    controller_loop_separation: float = 2.0
    controller_rate_margin: float = 0.8
    controller_feedforward: float = 0.0
    controller_slew_aware: bool = False


class RouteEvaluationTimeout(TimeoutError):
    """One pathological rocket integration exceeded its search allowance."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--launch-start", type=float, default=10.0)
    parser.add_argument("--launch-stop", type=float, default=30.0)
    parser.add_argument("--launch-step", type=float, default=2.0)
    parser.add_argument(
        "--targets",
        type=int,
        nargs="+",
        help="test one explicit route instead of selector candidates",
    )
    parser.add_argument(
        "--append-target",
        type=int,
        nargs="+",
        help="test each listed target after the --targets prefix",
    )
    parser.add_argument(
        "--append-depth",
        type=int,
        default=1,
        help="append permutations of this many --append-target values",
    )
    parser.add_argument("--candidates-per-model", type=int, default=3)
    parser.add_argument("--routes-per-time", type=int, default=6)
    parser.add_argument("--minimum-route-length", type=int, default=7)
    parser.add_argument(
        "--planning-budget",
        type=float,
        default=80.0,
        help="optimistic seconds allowed while generating candidate routes",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--route-timeout",
        type=float,
        default=60.0,
        help="wall-clock seconds allowed per fixed-route simulation",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="evaluate selected routes again even when a result is cached",
    )
    parser.add_argument(
        "--lookahead-weight",
        type=float,
        default=0.0,
        help="following-target terminal-velocity guidance blend (0..1)",
    )
    parser.add_argument(
        "--corridor",
        action="append",
        default=[],
        metavar="COMMITTED:FLYBY",
        help="add a flyby balloon while retaining the committed target",
    )
    parser.add_argument(
        "--corridor-weight",
        type=float,
        default=0.0,
        help="blend towards unbraked flyby ZEM guidance (0..1)",
    )
    parser.add_argument(
        "--corridor-horizon",
        type=float,
        default=0.0,
        help="seconds before predicted flyby to begin shaping; 0 is unlimited",
    )
    parser.add_argument("--controller-loop-separation", type=float, default=2.0)
    parser.add_argument("--controller-rate-margin", type=float, default=0.8)
    parser.add_argument(
        "--controller-feedforward",
        type=float,
        default=0.0,
        help="guidance-direction angular-rate feedforward (0..1)",
    )
    parser.add_argument(
        "--controller-slew-aware",
        action="store_true",
        help="limit issued TVC commands to the actuator's per-step slew",
    )
    parser.add_argument(
        "--trajectory-cache-dir", type=Path, default=DEFAULT_TRAJECTORY_CACHE
    )
    parser.add_argument("--result-cache-dir", type=Path, default=DEFAULT_RESULT_CACHE)
    return parser.parse_args()


def implementation_fingerprint() -> str:
    """Hash code that can change route execution or scoring."""
    paths = list((REPOSITORY_ROOT / "BalloonPoppingGymEnv" / "agents").rglob("*.py"))
    paths.extend(
        [
            REPOSITORY_ROOT / "BalloonPoppingGymEnv" / "envs" / "balloon_world.py",
            REPOSITORY_ROOT / "BalloonPoppingGymEnv" / "envs" / "cached_env.py",
        ]
    )
    digest = hashlib.sha256()
    digest.update(f"route-execution:{ROUTE_EXECUTION_VERSION}".encode("utf-8"))
    for path in sorted(paths):
        digest.update(str(path.relative_to(REPOSITORY_ROOT)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def parse_corridors(values: list[str]) -> tuple[tuple[int, int], ...]:
    """Parse repeatable COMMITTED:FLYBY route guidance pairs."""
    pairs = []
    for value in values:
        try:
            committed_text, flyby_text = value.split(":", maxsplit=1)
            pair = (int(committed_text), int(flyby_text))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid corridor {value!r}; expected COMMITTED:FLYBY"
            ) from exc
        if pair[0] == pair[1]:
            raise ValueError("a corridor flyby must differ from the committed target")
        pairs.append(pair)
    if len({committed for committed, _flyby in pairs}) != len(pairs):
        raise ValueError("each committed target may have only one corridor flyby")
    return tuple(sorted(pairs))


def task_key(task: RouteTask) -> str:
    encoded = json.dumps(asdict(task), sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def launch_steps(
    time_step: float, start: float, stop: float, spacing: float
) -> list[int]:
    if spacing <= 0.0:
        raise ValueError("launch-step must be positive")
    if stop < start:
        raise ValueError("launch-stop must not be earlier than launch-start")

    first = math.ceil((start - 1e-12) / time_step)
    last = math.floor((stop + 1e-12) / time_step)
    stride = max(round(spacing / time_step), 1)
    return list(range(first, last + 1, stride))


def _observation_at_step(env: CachedBalloonEnv, step: int) -> dict:
    if step < 0 or step >= env.num_timesteps:
        raise ValueError(f"launch step {step} is outside the episode")
    status = (step >= np.asarray(env._balloon_release_at_step, dtype=int)).astype(int)
    return {
        Schema.Observation.SIMULATION_TIME: (
            step * float(env.simulation_parameters["time_step"])
        ),
        Schema.Observation.BALLOON_STATUS: status[:, None],
        Schema.Observation.BALLOON_STATES: env._balloon_flights[:, :, step].copy(),
        Schema.Observation.ROCKET_SENSORS: np.full(12, np.nan),
    }


def build_tasks(args: argparse.Namespace) -> list[RouteTask]:
    parameters, given_parameters = load_scenario_parameters(args.scenario)
    parameter_hash = scenario_parameter_fingerprint(parameters)
    implementation_hash = implementation_fingerprint()
    corridors = parse_corridors(args.corridor)
    env = CachedBalloonEnv(
        render_mode=None,
        parameters=parameters,
        cache_dir=args.trajectory_cache_dir,
    )
    try:
        env.reset(seed=args.seed)
        selector = Selector(given_parameters)
        dt = float(parameters["simulation"]["time_step"])
        steps = launch_steps(dt, args.launch_start, args.launch_stop, args.launch_step)
        tasks = []
        seen = set()
        for step in steps:
            observation = _observation_at_step(env, step)
            active = selector.active_states(observation)
            if args.targets:
                if args.append_target:
                    routes = [
                        [*args.targets, *suffix]
                        for suffix in itertools.permutations(
                            (
                                target
                                for target in args.append_target
                                if target not in args.targets
                            ),
                            args.append_depth,
                        )
                    ]
                else:
                    routes = [list(args.targets)]
            else:
                routes = selector.candidate_chains(
                    active,
                    limit_per_model=args.candidates_per_model,
                    time_budget=args.planning_budget,
                )

            status = np.asarray(
                observation[Schema.Observation.BALLOON_STATUS], dtype=int
            ).reshape(-1)
            eligible = []
            for route in routes:
                if len(route) < args.minimum_route_length:
                    continue
                if any(target < 0 or target >= len(status) for target in route):
                    raise ValueError(f"route contains an invalid target: {route}")
                if not all(status[target] == 1 for target in route):
                    if args.targets:
                        raise ValueError(
                            f"explicit route is not fully released at "
                            f"t={step * dt:.2f}: {route}"
                        )
                    continue
                eligible.append(route)

            for route in eligible[: args.routes_per_time]:
                identity = (step, tuple(route))
                if identity in seen:
                    continue
                seen.add(identity)
                tasks.append(
                    RouteTask(
                        scenario=args.scenario,
                        seed=args.seed,
                        launch_step=step,
                        launch_time=step * dt,
                        targets=tuple(route),
                        trajectory_cache_dir=str(args.trajectory_cache_dir.resolve()),
                        parameter_fingerprint=parameter_hash,
                        implementation_fingerprint=implementation_hash,
                        lookahead_guidance_weight=args.lookahead_weight,
                        corridor_targets=corridors,
                        corridor_guidance_weight=args.corridor_weight,
                        corridor_guidance_horizon=args.corridor_horizon,
                        controller_loop_separation=args.controller_loop_separation,
                        controller_rate_margin=args.controller_rate_margin,
                        controller_feedforward=args.controller_feedforward,
                        controller_slew_aware=args.controller_slew_aware,
                    )
                )
    finally:
        env.close()
    return tasks


def evaluate_route(task: RouteTask) -> dict:
    parameters, given_parameters = load_scenario_parameters(task.scenario)
    if scenario_parameter_fingerprint(parameters) != task.parameter_fingerprint:
        raise ValueError("Scenario parameters changed after route tasks were built")

    env = CachedBalloonEnv(
        render_mode=None,
        parameters=parameters,
        cache_dir=task.trajectory_cache_dir,
    )
    agent = ITronAgent(
        given_parameters,
        fixed_launch_time=task.launch_time,
        fixed_target_list=task.targets,
        lookahead_guidance_weight=task.lookahead_guidance_weight,
        corridor_targets=dict(task.corridor_targets),
        corridor_guidance_weight=task.corridor_guidance_weight,
        corridor_guidance_horizon=task.corridor_guidance_horizon,
        controller_kwargs={
            "loop_separation": task.controller_loop_separation,
            "rate_command_margin": task.controller_rate_margin,
            "direction_rate_feedforward": task.controller_feedforward,
            "slew_aware": task.controller_slew_aware,
        },
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            observation, info = env.reset(seed=task.seed)
            previous_status = np.asarray(
                observation[Schema.Observation.BALLOON_STATUS], dtype=int
            ).reshape(-1)
            popped_targets = []
            pop_events = []
            approaches = {}
            all_min_distances = np.full(len(previous_status), np.inf)
            all_min_times = np.full(len(previous_status), np.nan)
            last_pop_time = None
            terminated = truncated = False
            while not (terminated or truncated):
                action = agent.get_action(observation)
                observation, reward, terminated, truncated, info = env.step(action)
                rocket_position = np.asarray(info["rocket_states"], dtype=float)[:3]
                if np.isfinite(rocket_position).all():
                    balloon_positions = np.asarray(
                        observation[Schema.Observation.BALLOON_STATES], dtype=float
                    )[:, :3]
                    distances = np.linalg.norm(
                        balloon_positions - rocket_position,
                        axis=1,
                    )
                    active_mask = (
                        np.asarray(
                            observation[Schema.Observation.BALLOON_STATUS], dtype=int
                        ).reshape(-1)
                        == 1
                    )
                    improved = active_mask & (distances < all_min_distances)
                    all_min_distances[improved] = distances[improved]
                    all_min_times[improved] = float(
                        observation[Schema.Observation.SIMULATION_TIME]
                    )
                if np.isfinite(
                    rocket_position
                ).all() and agent.current_target_idx < len(agent.target_idx_list):
                    engaged = int(agent.target_idx_list[agent.current_target_idx])
                    target_position = np.asarray(
                        observation[Schema.Observation.BALLOON_STATES], dtype=float
                    )[engaged, :3]
                    distance = float(np.linalg.norm(target_position - rocket_position))
                    previous_approach = approaches.get(engaged)
                    if (
                        previous_approach is None
                        or distance < previous_approach["distance"]
                    ):
                        approaches[engaged] = {
                            "distance": distance,
                            "time": float(
                                observation[Schema.Observation.SIMULATION_TIME]
                            ),
                        }
                status = np.asarray(
                    observation[Schema.Observation.BALLOON_STATUS], dtype=int
                ).reshape(-1)
                newly_popped = np.flatnonzero((previous_status != 2) & (status == 2))
                popped_targets.extend(int(target) for target in newly_popped)
                if reward > 0:
                    last_pop_time = float(
                        observation[Schema.Observation.SIMULATION_TIME]
                    )
                    rocket_state = np.asarray(info["rocket_states"], dtype=float)
                    for target in newly_popped:
                        pop_events.append(
                            {
                                "target": int(target),
                                "time": last_pop_time,
                                "rocket_position": rocket_state[:3].tolist(),
                                "rocket_velocity": rocket_state[3:6].tolist(),
                            }
                        )
                previous_status = status.copy()

            planned_targets_popped = [
                target for target in task.targets if target in popped_targets
            ]
            failed_target = next(
                (target for target in task.targets if target not in popped_targets),
                None,
            )
            unpopped = [
                target
                for target in np.argsort(all_min_distances)
                if target not in popped_targets
                and np.isfinite(all_min_distances[target])
            ][:20]
            near_misses = [
                {
                    "target": int(target),
                    "distance": float(all_min_distances[target]),
                    "time": float(all_min_times[target]),
                }
                for target in unpopped
            ]

        return {
            "key": task_key(task),
            "scenario": task.scenario,
            "seed": task.seed,
            "launch_step": task.launch_step,
            "launch_time": task.launch_time,
            "targets": list(task.targets),
            "lookahead_guidance_weight": task.lookahead_guidance_weight,
            "corridor_targets": [list(pair) for pair in task.corridor_targets],
            "corridor_guidance_weight": task.corridor_guidance_weight,
            "corridor_guidance_horizon": task.corridor_guidance_horizon,
            "controller_loop_separation": task.controller_loop_separation,
            "controller_rate_margin": task.controller_rate_margin,
            "controller_feedforward": task.controller_feedforward,
            "controller_slew_aware": task.controller_slew_aware,
            "popped_count": int(info["popped_count"]),
            "popped_targets": popped_targets,
            "planned_targets_popped": planned_targets_popped,
            "failed_target": failed_target,
            "pop_events": pop_events,
            "approaches": {
                str(target): approach for target, approach in approaches.items()
            },
            "near_misses": near_misses,
            "last_pop_time": last_pop_time,
        }
    finally:
        env.close()


def evaluate_route_with_timeout(task: RouteTask, timeout: float) -> dict:
    """Evaluate one route without letting a stiff integration stall the batch."""

    def timeout_handler(_signum, _frame):
        raise RouteEvaluationTimeout(
            f"route exceeded its {timeout:.1f} second wall-time allowance"
        )

    previous_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        return evaluate_route(task)
    except RouteEvaluationTimeout as exc:
        return {
            "key": task_key(task),
            "scenario": task.scenario,
            "seed": task.seed,
            "launch_step": task.launch_step,
            "launch_time": task.launch_time,
            "targets": list(task.targets),
            "lookahead_guidance_weight": task.lookahead_guidance_weight,
            "corridor_targets": [list(pair) for pair in task.corridor_targets],
            "corridor_guidance_weight": task.corridor_guidance_weight,
            "corridor_guidance_horizon": task.corridor_guidance_horizon,
            "controller_loop_separation": task.controller_loop_separation,
            "controller_rate_margin": task.controller_rate_margin,
            "controller_feedforward": task.controller_feedforward,
            "controller_slew_aware": task.controller_slew_aware,
            "popped_count": -1,
            "popped_targets": [],
            "planned_targets_popped": [],
            "last_pop_time": None,
            "status": "timeout",
            "error": str(exc),
        }
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def result_sort_key(result: dict) -> tuple[float, float]:
    last_pop_time = result["last_pop_time"]
    return (
        -int(result["popped_count"]),
        float("inf") if last_pop_time is None else last_pop_time,
    )


def _load_result_cache(path: Path) -> dict[str, dict]:
    results = {}
    if not path.is_file():
        return results
    with path.open("r", encoding="utf-8") as result_file:
        for line_number, line in enumerate(result_file, start=1):
            if not line.strip():
                continue
            try:
                result = json.loads(line)
                results[result["key"]] = result
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(
                    f"Invalid result cache line {line_number}: {path}"
                ) from exc
    return results


def _print_result(result: dict, *, prefix: str = "done") -> None:
    if result.get("status") == "timeout":
        print(
            f"{prefix}: timeout launch={result['launch_time']:.2f} "
            f"route={result['targets']}"
        )
        return
    last = result["last_pop_time"]
    last_text = "--" if last is None else f"{last:.2f}"
    print(
        f"{prefix}: score={result['popped_count']} last={last_text} "
        f"launch={result['launch_time']:.2f} route={result['targets']} "
        f"lookahead={result.get('lookahead_guidance_weight', 0.0):.3f} "
        f"corridor={result.get('corridor_guidance_weight', 0.0):.3f} "
        f"horizon={result.get('corridor_guidance_horizon', 0.0):.2f} "
        f"ctrl=({result.get('controller_loop_separation', 2.0):.2f},"
        f"{result.get('controller_rate_margin', 0.8):.2f},"
        f"{result.get('controller_feedforward', 0.0):.2f},"
        f"{int(result.get('controller_slew_aware', False))}) "
        f"popped={result['popped_targets']}"
    )


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be at least one")
    if args.append_target and not args.targets:
        raise ValueError("append-target requires a targets prefix")
    if args.append_depth < 1:
        raise ValueError("append-depth must be at least one")
    if args.route_timeout <= 0.0:
        raise ValueError("route-timeout must be positive")
    if args.candidates_per_model < 1 or args.routes_per_time < 1:
        raise ValueError("candidate limits must be at least one")
    if args.minimum_route_length < 1 or args.planning_budget <= 0.0:
        raise ValueError("route length and planning budget must be positive")
    if not 0.0 <= args.lookahead_weight <= 1.0:
        raise ValueError("lookahead-weight must be between 0 and 1")
    if not 0.0 <= args.corridor_weight <= 1.0:
        raise ValueError("corridor-weight must be between 0 and 1")
    if args.corridor_horizon < 0.0:
        raise ValueError("corridor-horizon must not be negative")
    if args.controller_loop_separation <= 0.0:
        raise ValueError("controller-loop-separation must be positive")
    if args.controller_rate_margin <= 0.0:
        raise ValueError("controller-rate-margin must be positive")
    if not 0.0 <= args.controller_feedforward <= 1.0:
        raise ValueError("controller-feedforward must be between 0 and 1")

    tasks = build_tasks(args)
    result_path = (
        args.result_cache_dir
        / f"scenario_{args.scenario}_seed_{args.seed}_fixed_routes.jsonl"
    )
    cached = _load_result_cache(result_path)
    pending = [task for task in tasks if args.rerun or task_key(task) not in cached]
    selected_results = (
        []
        if args.rerun
        else [cached[task_key(task)] for task in tasks if task_key(task) in cached]
    )
    print(
        f"routes={len(tasks)} cached={len(tasks) - len(pending)} "
        f"pending={len(pending)} workers={args.workers}"
    )

    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("a", encoding="utf-8") as result_file:
        if args.workers == 1:
            completed = (
                (task, evaluate_route_with_timeout(task, args.route_timeout))
                for task in pending
            )
            for _task, result in completed:
                result_file.write(json.dumps(result, sort_keys=True) + "\n")
                result_file.flush()
                selected_results.append(result)
                _print_result(result)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(
                        evaluate_route_with_timeout, task, args.route_timeout
                    ): task
                    for task in pending
                }
                for future in as_completed(futures):
                    result = future.result()
                    result_file.write(json.dumps(result, sort_keys=True) + "\n")
                    result_file.flush()
                    selected_results.append(result)
                    _print_result(result)

    print("\nBest results:")
    for result in sorted(selected_results, key=result_sort_key)[:20]:
        _print_result(result, prefix="best")


if __name__ == "__main__":
    main()
