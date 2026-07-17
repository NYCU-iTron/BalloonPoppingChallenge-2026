import torch
from pathlib import Path
from datetime import datetime
from tensorboard import program
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, CheckpointCallback

from BalloonPoppingGymEnv.envs.static_balloon_world import BalloonPoppingEnv
from BalloonPoppingGymEnv.envs.rl_navigator_env import RLNavigatorEnv
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters
from BalloonPoppingGymEnv.utils.metrics_callback import MetricsCallback
from BalloonPoppingGymEnv.utils.rl_utils import RL_FRAME_SKIP

def make_custom_env(scenario_params, given_params):
    def _init():
        raw_env = BalloonPoppingEnv(render_mode=None, parameters=scenario_params)
        return RLNavigatorEnv(raw_env, given_params)
    return _init

def main():
    # -------------------------------- Parameters -------------------------------- #
    # Dependent on CPU cores available
    n_train_envs = 24

    total_timesteps = 15_000_000
    n_evals = 30
    n_saves = 10

    eval_freq = max(total_timesteps // (n_train_envs * n_evals), 1)
    save_freq = max(total_timesteps // (n_train_envs * n_saves), 1)
    print(f"[Config] eval_freq={eval_freq}, save_freq={save_freq}")

    policy_size = 128

    time_step = 0.01
    horizon_seconds = 40.0
    gamma = 1.0 - (RL_FRAME_SKIP * time_step) / horizon_seconds

    entropy_coeff = 0.005

    n_steps = 512
    batch_size = 512
    assert (n_steps * n_train_envs) % batch_size == 0, \
        "batch_size must divide n_steps * n_train_envs"
    print(f"[Config] n_steps={n_steps}, buffer={n_steps * n_train_envs}, "
          f"~{total_timesteps // (n_steps * n_train_envs)} PPO updates")

    # ------------------------------- Environments ------------------------------- #
    scenario_parameters, given_parameters = load_scenario_parameters(0)

    train_env = make_vec_env(
        env_id=make_custom_env(scenario_parameters, given_parameters),
        n_envs=n_train_envs,
        vec_env_cls=SubprocVecEnv
    )

    eval_env = make_vec_env(
        env_id=make_custom_env(scenario_parameters, given_parameters),
        n_envs=1,
        vec_env_cls=SubprocVecEnv
    )

    # -------------------------------- Directories ------------------------------- #
    scripts_dir = Path(__file__).resolve().parent
    runs_dir = scripts_dir / "runs"
    run_dir = runs_dir / datetime.now().strftime("%Y-%m-%d-%H%M")
    run_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------- Tensorboard ------------------------------- #
    try:
        tb = program.TensorBoard()
        tb.configure(argv=[None, "--logdir", str(runs_dir)])
        url = tb.launch()
        print(f"[TensorBoard] serving {runs_dir} at {url}")
    except Exception as exc:
        print(f"[TensorBoard] auto-launch skipped ({exc}). "
              f"Run manually: tensorboard --logdir {runs_dir}")

    # ----------------------------------- Model ---------------------------------- #
    policy_kwargs = dict(
        activation_fn=torch.nn.Tanh,
        net_arch=dict(
            pi=[policy_size, policy_size],
            vf=[policy_size, policy_size]
        )
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        device="cpu",
        learning_rate=3e-4,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=10,
        gamma=gamma,
        gae_lambda=0.95,
        ent_coef=entropy_coeff,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(run_dir / "tensorboard")
    )

    # --------------------------------- CallBacks -------------------------------- #
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=str(run_dir / "checkpoints"),
        name_prefix="rl_model",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / "eval_best"),
        log_path=str(run_dir / "eval_logs"),
        eval_freq=eval_freq,
        n_eval_episodes=3,
        deterministic=True,
    )

    metrics_callback = MetricsCallback()

    callback_list = CallbackList([checkpoint_callback, eval_callback, metrics_callback])

    # --------------------------------- Training --------------------------------- #
    try:
        model.learn(total_timesteps=total_timesteps, callback=callback_list, progress_bar=True)
    except KeyboardInterrupt:
        print("[Training] interrupted by user (Ctrl+C) - saving current model...")
    finally:
        final_model_path = run_dir / "final_model.zip"
        model.save(str(final_model_path))
        print(f"[Training] final model saved -> {final_model_path}")


if __name__ == "__main__":
    main()
