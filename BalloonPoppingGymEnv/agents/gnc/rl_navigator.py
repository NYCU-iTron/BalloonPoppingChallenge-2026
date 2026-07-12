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
        model_path = agent_root / "models" / "final_model.zip"
        self.model = PPO.load(str(model_path), device="cpu")

    def reset(self):
        """Resets sequential tracking variables if necessary."""
        pass

    def compute(self, target_state: np.ndarray, rocket_state: np.ndarray) -> tuple[None, None] | tuple[np.ndarray, float]:
        # Defensive fallback validation gate checks
        if np.isnan(target_state).any() or np.isnan(rocket_state).any():
            return None, None

        if self.model is None:
            self.logger.warning("RL model is not loaded. Returning None for action.")
            return None, None

        rl_obs = compute_rl_observation(rocket_state, target_state)

        # --- Neural Network Model Inference Execution ---
        # deterministic=True disables exploration noise to output optimal actions
        rl_action, _states = self.model.predict(rl_obs, deterministic=True)
        print(f"RL Action: {rl_action}")

        # Extract operational output parameters mappings
        a_cmd_world = rl_action[0:3]
        throttle = float(np.clip(rl_action[3], 0.0, 1.0))

        return a_cmd_world, throttle
