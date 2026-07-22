import torch
from pathlib import Path
from datetime import datetime
from tensorboard import program
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, CheckpointCallback

from BalloonPoppingGymEnv.envs.pool_env import PoolEnv
from BalloonPoppingGymEnv.envs.rl_navigator_env import RLNavigatorEnv
from BalloonPoppingGymEnv.evaluation.evaluate import load_pool_parameters
from BalloonPoppingGymEnv.utils.metrics_callback import MetricsCallback
from BalloonPoppingGymEnv.utils.rl_utils import RL_FRAME_SKIP

def make_custom_env(scenario_params, given_params, pool_path):
    def _init():
        raw_env = PoolEnv(render_mode=None, parameters=scenario_params)
        return RLNavigatorEnv(raw_env, given_params, pool_path)
    return _init

def linear_schedule(initial_value, final_value):
    def schedule(progress_remaining):
        return final_value + progress_remaining * (initial_value - final_value)
    return schedule

def main():
    # -------------------------------- Parameters -------------------------------- #
    n_train_envs = 26
    total_timesteps = 4_000_000

    n_steps = 1024
    batch_size = 1024
    assert (n_steps * n_train_envs) % batch_size == 0, \
        "batch_size must divide n_steps * n_train_envs"
    print(f"[Config] n_steps={n_steps}, buffer={n_steps * n_train_envs}, "
          f"~{total_timesteps // (n_steps * n_train_envs)} PPO updates")

    policy_size = 256

    time_step = 0.01
    horizon_seconds = 30.0
    gamma = 1.0 - (RL_FRAME_SKIP * time_step) / horizon_seconds

    entropy_coeff = 0.005

    n_evals = 30
    n_saves = 10
    eval_freq = max(total_timesteps // (n_train_envs * n_evals), 1)
    save_freq = max(total_timesteps // (n_train_envs * n_saves), 1)
    print(f"[Config] eval_freq={eval_freq}, save_freq={save_freq}")

    # ------------------------------- Environments ------------------------------- #
    scenario_parameters, given_parameters = load_pool_parameters()

    scripts_dir = Path(__file__).resolve().parent
    pool_path = scripts_dir / "pool_level_1_easy.npy"

    seed = 0

    train_env = make_vec_env(
        env_id=make_custom_env(scenario_parameters, given_parameters, pool_path),
        n_envs=n_train_envs,
        seed=seed,
        vec_env_cls=SubprocVecEnv
    )
    # norm_obs must stay False: the deployment path (RLNavigator) feeds the
    # policy raw observations, so normalizing them here would break
    # train/deploy symmetry. Reward normalization is training-only and tames
    # the +1000 pop spikes for stable PPO value updates; Monitor still logs
    # raw episode rewards, so ep_rew_mean stays interpretable.
    train_env = VecNormalize(
        train_env,
        training=True,
        norm_obs=False,
        norm_reward=True,
        clip_reward=10.0,
        gamma=gamma,
    )

    eval_env = make_vec_env(
        env_id=make_custom_env(scenario_parameters, given_parameters, pool_path),
        n_envs=1,
        seed=seed + n_train_envs,
        vec_env_cls=SubprocVecEnv
    )
    # EvalCallback requires the eval env to mirror the train env's wrapper
    # stack (it syncs normalization stats each eval). Frozen, with raw rewards
    # so eval metrics stay in real units.
    eval_env = VecNormalize(
        eval_env,
        training=False,
        norm_obs=False,
        norm_reward=False,
        gamma=gamma,
    )

    # -------------------------------- Directories ------------------------------- #
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
        learning_rate=linear_schedule(3e-4, 3e-5),
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
        save_vecnormalize=True,
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
        # Reward-normalization running stats; needed to resume training with
        # a consistent reward scale (not needed for deployment: norm_obs=False).
        train_env.save(str(run_dir / "vecnormalize.pkl"))
        print(f"[Training] final model saved -> {final_model_path}")


if __name__ == "__main__":
    main()
