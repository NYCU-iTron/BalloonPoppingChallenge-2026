from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from BalloonPoppingGymEnv.envs.static_balloon_world import BalloonPoppingEnv
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters
from BalloonPoppingGymEnv.envs.rl_training_env import RLTrainingEnv


def make_custom_env(scenario_params, given_params):
    """
    Helper function required by SubprocVecEnv to spawn isolated environment instances.
    """
    def _init():
        raw_env = BalloonPoppingEnv(render_mode=None, parameters=scenario_params)
        return RLTrainingEnv(raw_env, given_params)
    return _init


def train_parallel():
    scenario_parameters, given_parameters = load_scenario_parameters(2)

    # Define how many parallel universes (CPU cores) you want to occupy
    num_envs = 8

    # Automatically spawn 8 independent background processes running our rocket physics
    train_env = make_vec_env(
        env_id=make_custom_env(scenario_parameters, given_parameters),
        n_envs=num_envs,
        vec_env_cls=SubprocVecEnv
    )

    # Standard PPO Agent now receives a vectorized observation tensor of shape (8, 9)
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,      # Each env collects 2048 steps -> Total batch size = 2048 * 8 = 16384
        batch_size=128     # Mini-batch size optimized for GPU updates
    )

    model.learn(total_timesteps=500000)
    model.save("ppo_rocket_navigator_parallel")


if __name__ == "__main__":
    train_parallel()
