import argparse
import copy
from pathlib import Path
from BalloonPoppingGymEnv.envs.balloon_generator import generate_balloon_pool

base_config = {
    "environment": {
        "date": [2025, 7, 25, 8], "latitude": 22.1749259, "longitude": 120.8922531, "elevation": 20.0
    },
    "simulation": {
        "time_step": 0.01, "max_time": 150
    },
    "balloon": {
        "radius": 1.5, "mass": 0.8, "inertia": [1, 1, 1],
        "aero_coefficients": {"cL": 0.0, "cQ": 0.0, "cD": 1.0, "moment_damping": -0.01},
        "stochastic": {"latitude_std": 0.0, "longitude_std": 0.0, "mass_std": 0.0, "volume_std": 0.0, "inertia_std": 0.0}
    }
}

LEVELS = {
    1: {
        "name": "easy",
        "total_tracks": 2000,
        "stochastic": {
            "latitude_std": 0.0001,
            "longitude_std": 0.0001,
            "mass_std": 0.01,
            "volume_std": 0.01,
            "inertia_std": 0.005,
        },
    },
    2: {
        "name": "medium",
        "total_tracks": 5000,
        "stochastic": {
            "latitude_std": 0.0002,
            "longitude_std": 0.0002,
            "mass_std": 0.05,
            "volume_std": 0.10,
            "inertia_std": 0.02,
        },
    },
    3: {
        "name": "hard",
        "total_tracks": 30000,
        "stochastic": {
            "latitude_std": 0.0010,
            "longitude_std": 0.0010,
            "mass_std": 0.20,
            "volume_std": 0.50,
            "inertia_std": 0.10,
        },
    },
}


def main():
    parser = argparse.ArgumentParser(description="Generate curriculum trajectory pools")
    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        choices=LEVELS,
        default=list(LEVELS),
        help="Curriculum levels to generate (default: 1 2 3)",
    )
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for level in dict.fromkeys(args.levels):
        level_spec = LEVELS[level]
        config = copy.deepcopy(base_config)
        config["balloon"]["stochastic"].update(level_spec["stochastic"])
        output_path = args.output_dir / f"pool_level_{level}_{level_spec['name']}.npy"
        generate_balloon_pool(
            config=config,
            total_tracks=level_spec["total_tracks"],
            chunk_size=args.chunk_size,
            output_path=str(output_path),
        )


if __name__ == "__main__":
    main()
