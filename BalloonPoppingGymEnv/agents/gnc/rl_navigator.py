import os
import logging
import numpy as np
from stable_baselines3 import PPO

from BalloonPoppingGymEnv.utils.rl_utils import compute_rl_observation

class RLNavigator:
    def __init__(self, given_parameters, model_path="rl_navigator.zip"):
        self.logger = logging.getLogger(__name__)
        self.given_parameters = given_parameters

        # Load the pre-trained stable-baselines3 PPO model onto CPU for inference fast execution
        if os.path.exists(model_path):
            self.model = PPO.load(model_path, device="cpu")
            self.logger.info(f"Successfully loaded RL network deployment from: {model_path}")
        else:
            self.model = None
            self.logger.error(f"Critical Error: Model file assets not discovered at paths: {model_path}")

    def reset(self):
        """Resets sequential tracking variables if necessary."""
        pass

    def compute(self, target_state: np.ndarray | None, rocket_state: np.ndarray) -> tuple[None, None] | tuple[np.ndarray, float]:
        """
        Compute the world-frame lateral acceleration command and throttle using the RL policy.

        Parameters
        ----------
        target_state : np.ndarray | None
            Predicted target state [pos(3), vel(3)] from the estimator, or None.
        rocket_state : np.ndarray
            Estimated state [pos(3), vel(3), acc(3), quat(4), gyro(3)].

        Returns
        -------
        a_cmd_world : np.ndarray | None
            Desired lateral acceleration in the world frame, shape (3,). None when invalid.
        throttle : float | None
            Energy-management throttle in [0, 1]. None when invalid.
        """
        # Defensive fallback validation gate checks
        if target_state is None or np.isnan(target_state).any() or self.model is None:
            return None, None

        rl_obs = compute_rl_observation(rocket_state, target_state)

        # --- Neural Network Model Inference Execution ---
        # deterministic=True disables exploration noise to output optimal actions
        rl_action, _states = self.model.predict(rl_obs, deterministic=True)

        # Extract operational output parameters mappings
        a_cmd_world = rl_action[0:3]
        throttle = float(np.clip(rl_action[3], 0.0, 1.0))

        return a_cmd_world, throttle
