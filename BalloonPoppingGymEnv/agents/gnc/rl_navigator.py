import logging
import pickle
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from BalloonPoppingGymEnv.agents.gnc.navigator import Navigator
from BalloonPoppingGymEnv.utils.rl_utils import (
    compute_rl_observation,
    RL_RESIDUAL_ACCEL_LIMIT,
    RL_RESIDUAL_THROTTLE_LIMIT,
)

class RLNavigator:
    """PN guidance with a learned residual correction.

    Mirrors the training-time composition in RLNavigatorEnv: the policy output
    is a normalized ([-1, 1]) CORRECTION scaled onto the proportional-navigation
    command, so zero residual (or a missing model) degrades gracefully to pure
    PN. If the model was trained with VecNormalize observation normalization,
    the saved running stats (vecnormalize.pkl) are loaded and applied before
    every predict -- without them a norm_obs-trained policy would silently see
    inputs on a completely different scale than during training.
    """

    def __init__(self, given_parameters, model_path: str, vecnormalize_path: str | None = None):
        self.logger = logging.getLogger(__name__)
        self.given_parameters = given_parameters

        self.pn_navigator = Navigator(given_parameters)
        self.last_residual = np.zeros(4, dtype=np.float32)

        # Observation normalization stats (None -> raw observations)
        self._obs_rms = None
        self._clip_obs = 10.0
        self._obs_epsilon = 1e-8

        self.model = None
        try:
            self.model = PPO.load(model_path, device="cpu")
        except Exception as e:
            self.logger.error(
                f"Error loading RL model from {model_path}: {e}. "
                "Falling back to pure PN guidance."
            )
            return

        stats_path = self._resolve_vecnormalize_path(model_path, vecnormalize_path)
        if stats_path is None:
            self.logger.warning(
                "No vecnormalize.pkl found next to the model; assuming the "
                "policy was trained on RAW observations. If it was trained "
                "with norm_obs=True this WILL misbehave."
            )
            return
        try:
            with open(stats_path, "rb") as f:
                vec_normalize = pickle.load(f)
            if getattr(vec_normalize, "norm_obs", False):
                self._obs_rms = vec_normalize.obs_rms
                self._clip_obs = float(vec_normalize.clip_obs)
                self._obs_epsilon = float(vec_normalize.epsilon)
                self.logger.info(f"Loaded observation normalization stats from {stats_path}")
        except Exception as e:
            self.logger.error(f"Failed to load VecNormalize stats from {stats_path}: {e}")

    @staticmethod
    def _resolve_vecnormalize_path(model_path, explicit_path):
        """Locate the VecNormalize stats saved alongside a model.

        Search order: the explicit path, `vecnormalize.pkl` in the model's
        directory (final_model / eval_best layout), then the CheckpointCallback
        naming scheme `rl_model_vecnormalize_<steps>_steps.pkl`.
        """
        if explicit_path is not None:
            path = Path(explicit_path)
            return path if path.exists() else None

        model_path = Path(model_path)
        candidates = [model_path.parent / "vecnormalize.pkl"]
        stem = model_path.stem  # e.g. "rl_model_800000_steps"
        if stem.startswith("rl_model_") and stem.endswith("_steps"):
            steps = stem[len("rl_model_"):-len("_steps")]
            candidates.append(model_path.parent / f"rl_model_vecnormalize_{steps}_steps.pkl")
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def reset(self):
        """Resets sequential tracking variables if necessary."""
        self.pn_navigator.reset()
        self.last_residual = np.zeros(4, dtype=np.float32)

    def compute(self, target_state: np.ndarray, rocket_state: np.ndarray) -> tuple[None, None] | tuple[np.ndarray, float]:
        a_pn, throttle_pn = self.pn_navigator.compute(target_state, rocket_state)
        if a_pn is None:
            return None, None

        if self.model is None:
            self.last_residual = np.zeros(4, dtype=np.float32)
            return a_pn, throttle_pn

        rl_obs = compute_rl_observation(rocket_state, target_state)
        if self._obs_rms is not None:
            rl_obs = np.clip(
                (rl_obs - self._obs_rms.mean) / np.sqrt(self._obs_rms.var + self._obs_epsilon),
                -self._clip_obs, self._clip_obs,
            ).astype(np.float32)

        rl_action, _ = self.model.predict(rl_obs, deterministic=True)

        # Normalized residual, exactly like the training action space.
        rl_action = np.clip(np.asarray(rl_action, dtype=np.float32), -1.0, 1.0)
        residual_accel = rl_action[0:3] * RL_RESIDUAL_ACCEL_LIMIT
        residual_throttle = float(rl_action[3]) * RL_RESIDUAL_THROTTLE_LIMIT
        self.last_residual = rl_action.copy()

        a_cmd = a_pn + residual_accel
        throttle = float(np.clip(throttle_pn + residual_throttle, 0.0, 1.0))

        return a_cmd, throttle
