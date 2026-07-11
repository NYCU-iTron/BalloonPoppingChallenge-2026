import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class PlottingCallback(BaseCallback):
    def __init__(self, save_dir: str, window_size=25, verbose=0, update_freq=1):
        super().__init__(verbose)
        self.window_size = window_size
        self.update_freq = update_freq

        self.save_dir = save_dir

        # Recording arrays
        self.episode_rewards = []
        self.moving_avg = []
        self.std_devs = []

        # Parallel tracking variables initialized dynamically on start
        self.current_rewards = None
        self.episode_count = 0

        self.fig = None
        self.ax = None

    def _on_training_start(self) -> None:
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 5))

        # Set the initial scannable layout configuration
        self.ax.set_title("Real-Time RL Training Progress", fontsize=14, fontweight="bold")
        self.ax.set_xlabel("Episode", fontsize=12)
        self.ax.set_ylabel("Total Reward", fontsize=12)

        plt.show(block=False)
        plt.pause(0.1)

    def _on_step(self) -> bool:
        # Fetch operational data arrays across all active parallel workers
        rewards = self.locals["rewards"]  # Shape: (num_envs,)
        dones = self.locals["dones"]      # Shape: (num_envs,)

        # Dynamic lazily allocation matching the precise environment process topology
        if self.current_rewards is None:
            self.current_rewards = np.zeros(len(dones), dtype=np.float32)

        # Accumulate stepwise reward matrices independently for each parallel channel
        self.current_rewards += rewards

        # Scan and capture completed trajectories across all parallel workers
        for env_idx, done in enumerate(dones):
            if done:
                self.episode_count += 1
                completed_reward = self.current_rewards[env_idx]
                self.episode_rewards.append(completed_reward)

                # Instantly clear cache memory for this specific environment channel
                self.current_rewards[env_idx] = 0.0

                # Compute statistical sliding updates safely
                current_window = self.episode_rewards[-self.window_size:]
                avg = float(np.mean(current_window))
                std = float(np.std(current_window)) if len(current_window) > 1 else 0.0

                self.moving_avg.append(avg)
                self.std_devs.append(std)

                # Execute UI render pass upon meeting interval pacing thresholds
                if self.episode_count % self.update_freq == 0:
                    self._update_plot()

                # Console telemetry diagnostics redirection tracking
                if self.verbose and self.episode_count % max(1, self.update_freq) == 0:
                    print(f"[PlotCallback] Episode: {self.episode_count} | "
                          f"Env ID: {env_idx} | "
                          f"Reward: {completed_reward:.2f} | "
                          f"Moving Avg: {avg:.2f}")

        return True

    def _update_plot(self):
        if len(self.episode_rewards) == 0:
            return

        self.ax.clear()
        episodes = list(range(1, len(self.episode_rewards) + 1))

        # Render raw episodic trajectory feedback bounds
        self.ax.plot(episodes, self.episode_rewards, 'b-', alpha=0.25, label="Raw Episode Reward")

        # Overlay moving localized average trending path
        self.ax.plot(episodes, self.moving_avg, 'r-', linewidth=2, label=f"Moving Average (w={self.window_size})")

        # Enclose standard dev variations envelope bounds
        upper_bound = np.array(self.moving_avg) + np.array(self.std_devs)
        lower_bound = np.array(self.moving_avg) - np.array(self.std_devs)
        self.ax.fill_between(episodes, lower_bound, upper_bound, color='red', alpha=0.15, label="±1 Std Dev")

        # Polish scannable HUD chart cosmetics layouts
        self.ax.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='none')
        self.ax.grid(True, linestyle='--', alpha=0.6)
        self.ax.set_title("Real-Time Navigator Performance Trajectory", fontsize=14, fontweight="bold")
        self.ax.set_xlabel("Completed Episodes", fontsize=12)
        self.ax.set_ylabel("Accumulated Reward Score", fontsize=12)

        # Dynamic downsampling spacing configuration for horizontal grid x-ticks
        if len(episodes) > 10:
            self.ax.set_xticks(np.linspace(1, len(episodes), min(10, len(episodes))).astype(int))

        # Flush drawing pipeline requests safely
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def _on_training_end(self) -> None:
        # Wrap up display structures neatly
        plt.ioff()
        self._update_plot()

        # Save output consistently as PNG file asset
        png = f"{self.save_dir}/rl_training_progress.png"
        plt.savefig(png, dpi=150, bbox_inches='tight')

        pdf = f"{self.save_dir}/rl_training_progress.pdf"
        plt.savefig(pdf, dpi=150, bbox_inches='tight')

        if self.verbose:
            print(f"\n[PlotCallback] Training complete. Progress visualization saved as '{output_filename}'")
            if len(self.moving_avg) > 0:
                print(f"[PlotCallback] Final Window Average Reward: {self.moving_avg[-1]:.2f}")

        plt.close(self.fig)
