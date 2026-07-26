import pickle
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO

from BalloonPoppingGymEnv.utils.rl_utils import (
    compute_rl_observation,
    scale_rl_action,
)


class RLNavigator:
    def __init__(self, given_parameters, model_path: Path, vecnormalize_path: Path):
        self.given_parameters = given_parameters

        self.model = PPO.load(str(model_path), device="cpu")

        with open(str(vecnormalize_path), "rb") as f:
            self.vec_normalize = pickle.load(f)

    def reset(self):
        pass

    def compute(self, target_state: np.ndarray, rocket_state: np.ndarray) -> np.ndarray:
        if np.isnan(target_state).any():
            return np.full(3, np.nan)

        rl_obs = compute_rl_observation(
            rocket_state=rocket_state,
            target_state=target_state,
        )
        rl_obs = self.vec_normalize.normalize_obs(rl_obs)

        rl_action, _ = self.model.predict(rl_obs, deterministic=True)
        desired_acc = scale_rl_action(rl_action)

        return desired_acc
