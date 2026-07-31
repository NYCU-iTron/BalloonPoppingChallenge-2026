from pathlib import Path
from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters
from BalloonPoppingGymEnv.agents.rl_agent import RLAgent
from BalloonPoppingGymEnv.utils.scene import Scene

scenario_number = 1

def run_for_development():
    scenario_parameters, given_parameters = load_scenario_parameters(scenario_number)
    env = BalloonPoppingEnv(render_mode="matplotlib", parameters=scenario_parameters)

    checkpoint_dir = Path(__file__).resolve().parent / "downloaded_runs" / "runs" / "2026-07-19-1443" / "checkpoints"
    model_path = checkpoint_dir / "final_model.zip"
    agent = RLAgent(given_parameters, model_path)

    scene = Scene()

    observation, info = env.reset(seed=scenario_parameters["scenario"]["random_seed"])
    terminated = False

    while not terminated:
        action = agent.get_action(observation)
        observation, reward, terminated, _, info = env.step(action)

        scene.update(observation, info)

        if info['popped_count'] == scenario_parameters["balloon"]["num"]:
            print(f"\nAll balloons popped at simulation_time: {observation['simulation_time']:.2f} sec")
            break

    scene.draw()

if __name__ == "__main__":
    run_for_development()
