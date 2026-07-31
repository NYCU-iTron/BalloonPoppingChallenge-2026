from BalloonPoppingGymEnv.evaluation.evaluate import load_static_parameters
from BalloonPoppingGymEnv.envs.static_balloon_world import BalloonPoppingEnv
from BalloonPoppingGymEnv.agents.itron_agent import ITronAgent
from BalloonPoppingGymEnv.utils.scene import Scene


def run_for_development():

    # Load scenario parameters
    scenario_parameters, given_parameters = load_static_parameters()

    # Create environment with scenario parameters turn off rendering to make own plots
    env = BalloonPoppingEnv(render_mode='matplotlib', parameters=scenario_parameters)

    # Instantiate agent with given parameters and any additional user kwargs
    agent = ITronAgent(given_parameters)

    scene = Scene()

    # use seed=None to randomize environment
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
