import torch
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CallbackList

from BalloonPoppingGymEnv.envs.static_balloon_world import BalloonPoppingEnv
from BalloonPoppingGymEnv.envs.rl_navigator_env import RLNavigatorEnv
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters
from BalloonPoppingGymEnv.utils.plotting_callback import PlottingCallback
from BalloonPoppingGymEnv.utils.save_callback import SaveCallback

def make_custom_env(scenario_params, given_params):
    def _init():
        raw_env = BalloonPoppingEnv(render_mode=None, parameters=scenario_params)
        return RLNavigatorEnv(raw_env, given_params)
    return _init

def train():
    scenario_parameters, given_parameters = load_scenario_parameters(0)

    num_envs = 10
    train_env = make_vec_env(
        env_id=make_custom_env(scenario_parameters, given_parameters),
        n_envs=num_envs,
        vec_env_cls=SubprocVecEnv
    )

    policy_kwargs = dict(
        activation_fn=torch.nn.Tanh,
        net_arch=dict(
            pi=[128, 128],
            vf=[128, 128]
        )
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        device="cpu",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        policy_kwargs=policy_kwargs
        tensorboard_log=str(scripts_dir / "tensorboard")
    )

    plot_callback = PlottingCallback()
    save_callback = SaveCallback(save_dir=str(current_dir / "models"), save_freq=20000)
    callback_list = CallbackList([plot_callback, save_callback])

    model.learn(total_timesteps=100000, callback=callback_list)
    model.save("rl_navigator")


if __name__ == "__main__":
    train()
