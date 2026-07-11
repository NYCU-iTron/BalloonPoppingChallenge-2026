import numpy as np

from BalloonPoppingGymEnv.envs.static_balloon_world import BalloonPoppingEnv
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters
from BalloonPoppingGymEnv.agents.rl_agent import RLAgent
from BalloonPoppingGymEnv.utils.setup_logging import setup_logging
from BalloonPoppingGymEnv.utils.scene import Scene
from BalloonPoppingGymEnv.utils.rl_utils import compute_rl_reward

scenario_number = 0

def run_for_development():

    # Load scenario parameters
    scenario_parameters, given_parameters = load_scenario_parameters(scenario_number)

    # Create environment with scenario parameters turn off rendering to make own plots
    env = BalloonPoppingEnv(render_mode='matplotlib', parameters=scenario_parameters)

    # Instantiate agent with given parameters and any additional user kwargs
    agent = RLAgent(given_parameters)

    scene = Scene()

    # use seed=None to randomize environment
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

        rl_reward += compute_rl_reward(agent.rocket_state, agent.target_state, info, action_delta)

    print(f"Total RL Reward: {rl_reward}")
    scene.draw()

if __name__ == "__main__":
    setup_logging()
    run_for_development()
