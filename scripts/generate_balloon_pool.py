"""Generate a balloon trajectory pool matching a scenario's own Monte Carlo."""

from pathlib import Path

from BalloonPoppingGymEnv.envs.balloon_generator import generate_balloon_pool
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters

SCENARIO = 1
TOTAL_TRACKS = 5000
CHUNK_SIZE = 1000
SEED = 0
OUTPUT_PATH = Path(__file__).resolve().parent / f"pool_scenario_{SCENARIO}.npy"


def main():
    scenario_parameters, _ = load_scenario_parameters(SCENARIO)
    generate_balloon_pool(
        config=scenario_parameters,
        total_tracks=TOTAL_TRACKS,
        chunk_size=CHUNK_SIZE,
        output_path=str(OUTPUT_PATH),
        seed=SEED,
    )


if __name__ == "__main__":
    main()
