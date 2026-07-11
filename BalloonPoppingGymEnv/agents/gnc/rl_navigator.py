import logging
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO

from BalloonPoppingGymEnv.utils.rl_utils import compute_rl_observation

class RLNavigator:
    def __init__(self, given_parameters):
        self.logger = logging.getLogger(__name__)
        self.given_parameters = given_parameters

        agent_root = Path(__file__).resolve().parent.parent
        model_path = agent_root / "models" / "rl_navigator_0711.zip"
        self.model = PPO.load(str(model_path), device="cpu")

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
