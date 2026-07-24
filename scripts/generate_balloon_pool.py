from pathlib import Path
from BalloonPoppingGymEnv.utils.balloon_generator import generate_balloon_pool


scenario_1_config = {
    "environment": {
        "date": [2025, 7, 25, 8], "latitude": 22.1749259, "longitude": 120.8922531, "elevation": 20.0,
        "atmosphere_data_filename": "TW_Cup_250726_NetCDF4_Ensemble.nc"
    },
    "simulation": {
        "time_step": 0.01, "max_time": 150
    },
    "balloon": {
        "radius": 1.5, "mass": 0.8, "inertia": [1, 1, 1],
        "aero_coefficients": {"cL": 0.0, "cQ": 0.0, "cD": 1.0, "moment_damping": -0.01},
        "stochastic": {
            "latitude_std": 0.001,
            "longitude_std": 0.001,
            "mass_std": 0.2,
            "volume_std": 0.5,
            "inertia_std": 0.1
        }
    }
}

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    output_path = str(script_dir / "pool_scenario_1.npy")

    generate_balloon_pool(
        config=scenario_1_config,
        total_tracks=5000,
        output_path=output_path
    )
