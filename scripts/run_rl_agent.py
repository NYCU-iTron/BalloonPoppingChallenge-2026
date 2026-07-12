import numpy as np
from pathlib import Path

from BalloonPoppingGymEnv.envs.static_balloon_world import BalloonPoppingEnv
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters
from BalloonPoppingGymEnv.agents.rl_agent import RLAgent
from BalloonPoppingGymEnv.utils.scene import Scene
from BalloonPoppingGymEnv.utils.rl_utils import compute_rl_reward

scenario_number = 0

def run_for_development():
    scenario_parameters, given_parameters = load_scenario_parameters(scenario_number)
    env = BalloonPoppingEnv(render_mode=None, parameters=scenario_parameters)

    checkpoint_dir = Path(__file__).resolve().parent / "runs" / "2026-07-12-1444" / "checkpoints"
    model_path = checkpoint_dir / "rl_model_900000_steps.zip"
    agent = RLAgent(given_parameters, str(model_path))

    scene = Scene()

    observation, info = env.reset(seed=scenario_parameters["scenario"]["random_seed"])
    terminated = False

    prev_rl_action = np.zeros(4, dtype=np.float32)
    rl_reward = 0.0

    while not terminated:
        action = agent.get_action(observation)
        observation, reward, terminated, _, info = env.step(action)

        action_delta = agent.rl_action - prev_rl_action
        prev_rl_action = agent.rl_action.copy()

        scene.update(observation, info)

        if info['popped_count'] == scenario_parameters["balloon"]["num"]:
            print(f"\nAll balloons popped at simulation_time: {observation['simulation_time']:.2f} sec")
            break

        rl_reward += compute_rl_reward(observation, info, agent.rocket_state, agent.target_state,
                                       reward, terminated, action_delta)

    print(f"Total RL Reward: {rl_reward}")
    scene.draw()

if __name__ == "__main__":
    run_for_development()
