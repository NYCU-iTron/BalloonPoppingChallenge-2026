import numpy as np
from pathlib import Path
from BalloonPoppingGymEnv.evaluation.evaluate import load_pool_parameters
from BalloonPoppingGymEnv.envs.pool_env import PoolEnv
from BalloonPoppingGymEnv.utils.scene import Scene
from BalloonPoppingGymEnv.agents.rl_agent import RLAgent


def run_for_development():
    # Setup env
    scenario_parameters, given_parameters = load_pool_parameters()
    env = PoolEnv(render_mode='matplotlib', parameters=scenario_parameters)

    # Load pool
    script_dir = Path(__file__).resolve().parent.parent
    pool_file = script_dir / "pool_level_1_easy.npy"
    trajectory_database = np.load(pool_file, mmap_mode='r')
    pool_capacity = trajectory_database.shape[0]

    # Sample tracks from pool
    num_balloons = scenario_parameters["balloon"]["num"]
    sampled_indices = np.random.choice(pool_capacity, size=num_balloons, replace=False)
    extracted_tracks = np.array(trajectory_database[sampled_indices])
    env.update_source_trajectories(extracted_tracks)

    scene = Scene()

    # Setup agent
    runs_dir = Path(__file__).resolve().parent.parent / "runs"
    checkpoint_dir = runs_dir / "2026-07-30-0057" / "checkpoints"
    model_path = checkpoint_dir / "rl_model_5600000_steps.zip"
    vecnormalize_path = checkpoint_dir / "rl_model_vecnormalize_5600000_steps.pkl"
    agent = RLAgent(given_parameters, model_path, vecnormalize_path)

    observation, info = env.reset(seed=scenario_parameters["scenario"]["random_seed"])
    terminated = truncated = False

    while not (terminated or truncated):
        action = agent.get_action(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        scene.update(observation, info)

        if info['popped_count'] == num_balloons:
            print(f"\nAll balloons popped at simulation_time: {observation['simulation_time']:.2f} sec")
            break

    scene.draw()


if __name__ == "__main__":
    run_for_development()
