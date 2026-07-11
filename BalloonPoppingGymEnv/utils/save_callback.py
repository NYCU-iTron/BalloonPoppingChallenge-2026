import os
from pathlib import Path
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class SaveCallback(BaseCallback):
    """
    Callback for periodically saving the 'latest' model and saving the 'best'
    model based on the running rolling average of episode rewards.
    """
    def __init__(self, save_dir: str, save_freq=20000, window_size=20, verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.window_size = window_size

        # Combine script root directory with the customized save_dir string
        self.model_dir = Path(save_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Define exact file destinations using modern Path objects strings conversion
        self.best_save_path = str(self.model_dir / "best_model.zip")
        self.latest_save_path = str(self.model_dir / "latest_model.zip")

        # Parallel environment tracking variables
        self.episode_rewards = []
        self.current_rewards = None
        self.best_mean_reward = -float("inf")

    def _on_step(self) -> bool:
        # 1. Periodically save the latest model based on timestep intervals
        if self.n_calls % self.save_freq == 0:
            self.model.save(self.latest_save_path)
            if self.verbose > 0:
                print(f"[SaveCallback] Timestep {self.num_timesteps}: Saved latest checkpoint.")

        # 2. Monitor tracking rewards across parallel channels to catch the best model
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]

        if self.current_rewards is None:
            self.current_rewards = np.zeros(len(dones), dtype=np.float32)

        self.current_rewards += rewards

        for env_idx, done in enumerate(dones):
            if done:
                self.episode_rewards.append(self.current_rewards[env_idx])
                self.current_rewards[env_idx] = 0.0  # Reset cache channel

                # Check performance trends once enough historical metadata is available
                if len(self.episode_rewards) >= self.window_size:
                    current_window = self.episode_rewards[-self.window_size:]
                    mean_reward = float(np.mean(current_window))

                    # Trigger best model save sequence if performance breaks historical records
                    if mean_reward > self.best_mean_reward:
                        self.best_mean_reward = mean_reward
                        self.model.save(self.best_save_path)
                        if self.verbose > 0:
                            print(f"[SaveCallback] New Best Performance Detected! "
                                  f"Mean Reward: {mean_reward:.2f}. Saved best_model.zip")

        return True
