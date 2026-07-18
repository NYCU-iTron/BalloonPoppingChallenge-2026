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

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent

    level_1_cfg = copy.deepcopy(base_config)
    level_1_cfg["balloon"]["stochastic"].update({
        "latitude_std": 0.0001,
        "longitude_std": 0.0001,
        "mass_std": 0.01,
        "volume_std": 0.01,
        "inertia_std": 0.005
    })
    output_path = str(script_dir / "pool_level_1_easy.npy")
    generate_balloon_pool(
        config=level_1_cfg,
        total_tracks=2000,
        chunk_size=1000,
        output_path=output_path
    )

    # level_2_cfg = copy.deepcopy(base_config)
    # level_2_cfg["balloon"]["stochastic"].update({
    #     "latitude_std": 0.0002,
    #     "longitude_std": 0.0002,
    #     "mass_std": 0.05,
    #     "volume_std": 0.10,
    #     "inertia_std": 0.02
    # })
    # output_path = str(script_dir / "pool_level_2_medium.npy")
    # generate_balloon_pool(
    #     config=level_2_cfg,
    #     total_tracks=5000,
    #     chunk_size=1000,
    #     output_path=output_path
    # )

    # level_3_cfg = copy.deepcopy(base_config)
    # level_3_cfg["balloon"]["stochastic"].update({
    #     "latitude_std": 0.0010,
    #     "longitude_std": 0.0010,
    #     "mass_std": 0.20,
    #     "volume_std": 0.50,
    #     "inertia_std": 0.10
    # })
    # output_path = str(script_dir / "pool_level_3_hard.npy")
    # generate_balloon_pool(
    #     config=level_3_cfg,
    #     total_tracks=30000,
    #     chunk_size=1000,
    #     output_path=output_path
    # )
