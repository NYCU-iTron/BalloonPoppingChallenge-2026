import numpy as np
import matplotlib.pyplot as plt
from BalloonPoppingGymEnv.envs.static_balloon_world import BalloonPoppingEnv
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters
from BalloonPoppingGymEnv.agents.itron_agent import ITronAgent
from BalloonPoppingGymEnv.utils.setup_logging import setup_logging
from BalloonPoppingGymEnv.utils.render_scene import RenderScene

scenario_number = 2

def run_for_development():

    # Load scenario parameters
    scenario_parameters, given_parameters = load_scenario_parameters(scenario_number)

    # Create environment with scenario parameters turn off rendering to make own plots
    env = BalloonPoppingEnv(render_mode='matplotlib', parameters=scenario_parameters)

    # Instantiate agent with given parameters and any additional user kwargs
    agent = ITronAgent(given_parameters)

    render_scene = RenderScene()

    # use seed=None to randomize environment
    observation, info = env.reset(seed=scenario_parameters["scenario"]["random_seed"])
    terminated = False

    while not terminated:
        action = agent.get_action(observation)
        observation, reward, terminated, _, info = env.step(action)

        render_scene.get_ob(observation, info)

        if info['popped_count'] == scenario_parameters["balloon"]["num"]:
            print(f"\nAll balloons popped at simulation_time: {observation['simulation_time']:.2f} sec")
            break

    render_scene.draw()

    error_buffer = agent.estimator.error_buffer

    if len(error_buffer) > 0:
        error_list = list(error_buffer)
        mean_error = float(np.mean(error_list))

        plt.figure(figsize=(10, 5))
        plt.plot(
            error_list,
            color="darkmagenta",
            linestyle="-",
            marker="o",
            markersize=3,
            alpha=0.7,
            label="Settled Error Rate"
        )
        plt.axhline(
            y=mean_error,
            color="crimson",
            linestyle="--",
            linewidth=1.5,
            label=f"Session Mean: {mean_error:.3f} m/s"
        )

        plt.title("Post-Flight Global Prediction Error Analysis", fontsize=12, fontweight="bold")
        plt.xlabel("Sequential Settled Sample Index")
        plt.ylabel("Normalized Position Error Rate [m/s]")
        plt.legend(loc="upper right")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()
    else:
        print("\n[Diagnostics Warning] Global error buffer is empty. No predictions reached expiration.")

if __name__ == "__main__":
    setup_logging()
    run_for_development()
