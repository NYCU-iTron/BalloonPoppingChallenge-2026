from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from BalloonPoppingGymEnv.envs.static_balloon_world import BalloonPoppingEnv
from BalloonPoppingGymEnv.envs.rl_navigator_env import RLNavigatorEnv
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters
from BalloonPoppingGymEnv.utils.plotting_callback import PlottingCallback

def make_custom_env(scenario_params, given_params):
    def _init():
        raw_env = BalloonPoppingEnv(render_mode=None, parameters=scenario_params)
        return RLNavigatorEnv(raw_env, given_params)
    return _init

def train():
    scenario_parameters, given_parameters = load_scenario_parameters(2)

    num_envs = 8

    train_env = make_vec_env(
        env_id=make_custom_env(scenario_parameters, given_parameters),
        n_envs=num_envs,
        vec_env_cls=SubprocVecEnv
    )

    plot_callback = PlottingCallback(window_size=20, update_freq=1)

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=128
    )

    model.learn(total_timesteps=50000, callback=plot_callback)
    model.save("rl_navigator")


if __name__ == "__main__":
    train()
