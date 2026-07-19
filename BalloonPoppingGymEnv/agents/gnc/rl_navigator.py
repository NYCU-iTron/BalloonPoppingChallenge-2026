import logging
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
    is a bounded CORRECTION added to the proportional-navigation command, so
    zero residual (or a missing model) degrades gracefully to pure PN.
    """

    def __init__(self, given_parameters, model_path: str):
        self.logger = logging.getLogger(__name__)
        self.given_parameters = given_parameters

        self.pn_navigator = Navigator(given_parameters)
        self.last_residual = np.zeros(4, dtype=np.float32)

        self.model = None
        try:
            self.model = PPO.load(model_path, device="cpu")
        except Exception as e:
            self.logger.error(
                f"Error loading RL model from {model_path}: {e}. "
                "Falling back to pure PN guidance."
            )

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
        rl_action, _ = self.model.predict(rl_obs, deterministic=True)

        # Clip to the shared residual bounds so deployment semantics match the
        # training action space exactly.
        residual_accel = np.clip(rl_action[0:3], -RL_RESIDUAL_ACCEL_LIMIT, RL_RESIDUAL_ACCEL_LIMIT)
        residual_throttle = float(np.clip(rl_action[3], -RL_RESIDUAL_THROTTLE_LIMIT, RL_RESIDUAL_THROTTLE_LIMIT))
        self.last_residual = np.concatenate([residual_accel, [residual_throttle]]).astype(np.float32)

        a_cmd = a_pn + residual_accel
        throttle = float(np.clip(throttle_pn + residual_throttle, 0.0, 1.0))

        return a_cmd, throttle
