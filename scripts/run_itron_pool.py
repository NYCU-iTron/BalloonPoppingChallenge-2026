from pathlib import Path

import numpy as np

from BalloonPoppingGymEnv.agents.itron_agent import ITronAgent
from BalloonPoppingGymEnv.envs.pool_env import PoolEnv
from BalloonPoppingGymEnv.evaluation.evaluate import load_pool_parameters


def run_for_development():
    scenario_parameters, given_parameters = load_pool_parameters()
    # Match the Scenario 1 evaluation configuration: the agent's interactive
    # selector visualizer owns the 3-D view, while the Gym renderer stays off.
    env = PoolEnv(render_mode=None, parameters=scenario_parameters)

    agent = ITronAgent(
        given_parameters,
        visualize_selector=True,
        visualizer_kwargs={
            "update_interval": 0.25,
            "pick_radius": 14,
        },
    )

    # Load the pool
    script_dir = Path(__file__).resolve().parent
    pool_file = script_dir / "pool_scenario_1.npy"
    trajectory_database = np.load(pool_file, mmap_mode="r")
    pool_capacity = trajectory_database.shape[0]

    # Sample tracks from the pool
    num_balloons = scenario_parameters["balloon"]["num"]
    rng = np.random.default_rng(scenario_parameters["scenario"]["random_seed"])
    sampled_indices = rng.choice(pool_capacity, size=num_balloons, replace=False)
    extracted_tracks = np.array(trajectory_database[sampled_indices])

    env.update_source_trajectories(extracted_tracks)

    observation, info = env.reset(seed=scenario_parameters["scenario"]["random_seed"])
    terminated = truncated = False

    while not (terminated or truncated):
        action = agent.get_action(observation)
        observation, reward, terminated, truncated, info = env.step(action)

        if info["popped_count"] == num_balloons:
            print(
                "\nAll balloons popped at simulation_time: "
                f"{observation['simulation_time']:.2f} sec"
            )
            break

    print(
        f"Popped {info['popped_count']} balloons; "
        f"simulation_time={observation['simulation_time']:.2f} sec"
    )
    if agent.leg_diagnostics:
        print("Leg prediction diagnostics:")
        for leg in agent.leg_diagnostics:
            print(
                f"  #{leg['target']}: time "
                f"{leg['predicted_duration']:.2f}/{leg['actual_duration']:.2f} s, "
                f"speed {leg['predicted_speed']:.2f}/{leg['actual_speed']:.2f} m/s, "
                f"distance {leg['distance']:.1f} m, turn {leg['turn_deg']:.1f} deg, "
                f"climb {leg['climb_sin']:.2f}, entry {leg['incoming_speed']:.1f} m/s, "
                f"direction error {leg['direction_error_deg']:.1f} deg, "
                f"position error {leg['position_error']:.2f} m"
            )


if __name__ == "__main__":
    run_for_development()
