import torch
from pathlib import Path
from datetime import datetime
from tensorboard import program
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, CheckpointCallback

from BalloonPoppingGymEnv.evaluation.evaluate import load_pool_parameters
from BalloonPoppingGymEnv.utils.metrics_callback import MetricsCallback
from BalloonPoppingGymEnv.utils.save_vecnormalize_callback import SaveVecNormalizeCallback
from BalloonPoppingGymEnv.utils.rl_utils import (
    RL_FRAME_SKIP,
    make_custom_env,
    linear_schedule
)

def main():
    # -------------------------------- Parameters -------------------------------- #
    torch.set_num_threads(1)

    n_train_envs = 20
    total_timesteps = 8_000_000

    n_steps = 1024
    batch_size = 1024
    assert (n_steps * n_train_envs) % batch_size == 0
    # PPO updates = total_timesteps // (n_steps * n_train_envs)

    policy_size = 256

    time_step = 0.01
    horizon_seconds = 35.0
    gamma = 1.0 - (RL_FRAME_SKIP * time_step) / horizon_seconds

    entropy_coeff = 1e-4

    n_evals = 25
    n_saves = 10
    eval_freq = max(total_timesteps // (n_train_envs * n_evals), 1)
    save_freq = max(total_timesteps // (n_train_envs * n_saves), 1)

    n_eval_episodes = 10

    # -------------------------------- Directories ------------------------------- #
    scripts_dir = Path(__file__).resolve().parent
    pool_path = scripts_dir / "pool_scenario_1.npy"
    pool_path_list = [
        pool_path
    ]

    runs_dir = scripts_dir / "runs"
    run_dir = runs_dir / datetime.now().strftime("%Y-%m-%d-%H%M")
    run_dir.mkdir(parents=True, exist_ok=True)

    final_path = run_dir / "checkpoints"
    final_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------- Environments ------------------------------- #
    scenario_parameters, given_parameters = load_pool_parameters()

    seed = 0

    train_env = make_vec_env(
        env_id=make_custom_env(scenario_parameters, given_parameters, pool_path_list),
        n_envs=n_train_envs,
        seed=seed,
        vec_env_cls=SubprocVecEnv
    )

    train_env = VecNormalize(
        train_env,
        training=True,
        norm_obs=True,
        clip_obs=10.0,
        norm_reward=False,
        gamma=gamma,
    )

    eval_env = make_vec_env(
        env_id=make_custom_env(scenario_parameters, given_parameters, pool_path_list),
        n_envs=1,
        seed=seed + n_train_envs,
        vec_env_cls=SubprocVecEnv
    )

    eval_env = VecNormalize(
        eval_env,
        training=False,
        norm_obs=True,
        clip_obs=10.0,
        norm_reward=False,
        gamma=gamma,
    )

    # -------------------------------- Tensorboard ------------------------------- #
    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", str(runs_dir)])
    url = tb.launch()
    print(f"[TensorBoard] serving {runs_dir} at {url}")

    # ----------------------------------- Model ---------------------------------- #
    policy_kwargs = dict(
        activation_fn=torch.nn.Tanh,
        log_std_init=-1.0,
        net_arch=dict(
            pi=[policy_size, policy_size],
            vf=[policy_size, policy_size, policy_size]
        )
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        device="cpu",
        learning_rate=linear_schedule(3e-4, 3e-5),
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        gae_lambda=0.97,
        target_kl=0.02,
        ent_coef=entropy_coeff,
        use_sde=True,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(run_dir / "tensorboard")
    )

    # --------------------------------- CallBacks -------------------------------- #
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=str(run_dir / "checkpoints"),
        name_prefix="rl_model",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / "eval_best"),
        callback_on_new_best=SaveVecNormalizeCallback(str(run_dir / "eval_best" / "vecnormalize.pkl")),
        log_path=str(run_dir / "eval_logs"),
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
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
        model.save(str(final_path / "final_model.zip"))
        train_env.save(str(final_path / "vecnormalize.pkl"))
        print(f"[Training] final model saved -> {final_path / 'final_model.zip'}")


if __name__ == "__main__":
    main()
