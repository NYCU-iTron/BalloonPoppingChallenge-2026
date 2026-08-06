import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from BalloonPoppingGymEnv.evaluation.evaluate import load_static_parameters
from BalloonPoppingGymEnv.envs.static_balloon_world import BalloonPoppingEnv
from BalloonPoppingGymEnv.envs.e2e_static_env import TARGET_AGL_RANGE, TARGET_RADIUS_MIN, TARGET_RADIUS_FRAC
from BalloonPoppingGymEnv.agents.e2e_agent import E2EAgent
from BalloonPoppingGymEnv.utils.reference_trajectory import generate_reference_trajectory
from BalloonPoppingGymEnv.utils.scene import Scene


def run_for_development():
    # Load scenario parameters
    scenario_parameters, given_parameters = load_static_parameters()
    ground_elevation = float(scenario_parameters["environment"]["elevation"])
    seed = scenario_parameters["scenario"]["random_seed"]

    # Create environment with scenario parameters, turn off rendering to make own plots
    env = BalloonPoppingEnv(render_mode='matplotlib', parameters=scenario_parameters)

    # Sample a single off-axis target
    rng = np.random.default_rng(seed)
    target_agl = rng.uniform(*TARGET_AGL_RANGE)
    target_z = ground_elevation + target_agl
    radius = rng.uniform(TARGET_RADIUS_MIN, TARGET_RADIUS_FRAC * target_agl)
    bearing = rng.uniform(0.0, 2.0 * np.pi)
    target_position = np.array([radius * np.cos(bearing), radius * np.sin(bearing), target_z])
    env.update_balloons(target_position)

    # Setup agent
    runs_dir = Path(__file__).resolve().parent / "runs"
    checkpoint_dir = runs_dir / "2026-08-06-0236" / "checkpoints"
    model_path = checkpoint_dir / "final_model.zip"
    vecnormalize_path = checkpoint_dir / "vecnormalize.pkl"
    agent = E2EAgent(given_parameters, model_path, vecnormalize_path)

    scene = Scene()

    observation, info = env.reset(seed=seed)
    terminated = False
    handoff_pos = None
    handoff_vel = None

    while not terminated:
        was_launch_complete = agent.launch_complete
        action = agent.get_action(observation)
        if agent.launch_complete and not was_launch_complete:
            handoff_pos = agent.rl_observator.rocket_pos.copy()
            handoff_vel = agent.rl_observator.rocket_vel.copy()

        observation, reward, terminated, _, info = env.step(action)

        scene.update(observation, info)

        if info['popped_count'] == 1:
            print(f"\nTarget popped at simulation_time: {observation['simulation_time']:.2f} sec")
            break

    ax = None
    if handoff_pos is not None:
        reference_trajectory = generate_reference_trajectory(
            start_pos=handoff_pos,
            start_vel=handoff_vel,
            target_pos=target_position,
            ground_elevation=ground_elevation,
        )
        ref_pos = reference_trajectory[:, 1:4]

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(projection="3d")
        ax.plot(ref_pos[:, 0], ref_pos[:, 1], ref_pos[:, 2], color="darkorange", linestyle="--", linewidth=1.5)

    scene.draw(ax=ax)

if __name__ == "__main__":
    run_for_development()
