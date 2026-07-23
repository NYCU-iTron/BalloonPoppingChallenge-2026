import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class MetricsCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.popped, self.lengths, self.closest = [], [], []

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                self.popped.append(info.get("popped_count", 0))
                if "closest_approach" in info:
                    self.closest.append(info["closest_approach"])
                if "episode" in info:
                    self.lengths.append(info["episode"]["l"])
        if self.popped:
            self.logger.record("rollout/mean_popped", float(np.mean(self.popped[-50:])))
        if self.closest:
            self.logger.record("rollout/mean_closest_approach", float(np.mean(self.closest[-50:])))
        if self.lengths:
            self.logger.record("rollout/mean_ep_len", float(np.mean(self.lengths[-50:])))
        return True
