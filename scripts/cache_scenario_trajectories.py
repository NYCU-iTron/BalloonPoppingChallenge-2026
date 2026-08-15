"""Generate or verify reusable Monte Carlo trajectory caches for development."""

import argparse
from pathlib import Path

from BalloonPoppingGymEnv.envs.cached_env import CachedBalloonEnv
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_DIR = REPOSITORY_ROOT / ".cache" / "balloon_trajectories"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        type=int,
        default=1,
        help="scenario number to generate (default: 1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        nargs="+",
        default=[0],
        help="one or more reset seeds (default: 0)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"cache directory (default: {DEFAULT_CACHE_DIR})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parameters, _given_parameters = load_scenario_parameters(args.scenario)

    for seed in args.seed:
        env = CachedBalloonEnv(
            render_mode=None,
            parameters=parameters,
            cache_dir=args.cache_dir,
        )
        try:
            env.reset(seed=seed)
            cache_path = env.last_trajectory_cache_path
            source = "loaded" if env.last_trajectory_cache_hit else "generated"
            size_mib = cache_path.stat().st_size / (1024**2)
            print(f"seed {seed}: {source} {cache_path} ({size_mib:.1f} MiB)")
        finally:
            env.close()


if __name__ == "__main__":
    main()
