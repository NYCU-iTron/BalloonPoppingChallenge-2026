import pickle
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO

from BalloonPoppingGymEnv.agents.base_agent import BaseAgent
from BalloonPoppingGymEnv.utils.schema import Schema
from BalloonPoppingGymEnv.utils.e2e_utils import RLObservator, scale_rl_action


class E2EAgent(BaseAgent):
    def __init__(self,
                 given_parameters,
                 model_path: Path | None = None,
                 vecnormalize_path: Path | None = None,
                ):
        super().__init__(given_parameters)

        if model_path is None:
            model_path = Path(__file__).resolve().parent / "models" / "e2e_model.zip"

        if vecnormalize_path is None:
            vecnormalize_path = Path(__file__).resolve().parent / "models" / "e2e_vecnormalize.pkl"

        self.rl_observator = RLObservator(given_parameters)
        self.model = PPO.load(str(model_path), device="cpu")

        with open(str(vecnormalize_path), "rb") as f:
            self.vec_normalize = pickle.load(f)

        # Launch-phase attitude-rate-hold PID, same law as
        # utils/e2e_utils.py::launch_schedule -- replayed one action per
        # get_action() call instead of owning its own env.step() loop, since
        # BaseAgent.get_action() only returns a single action per call.
        self.target_altitude = 40
        self.launch_inclination_heading = np.array([90.0, 0.0])
        self.rate_targets = np.zeros(3)
        self.KP = np.array([100.0, 100.0, 100.0])
        self.KI = np.array([0.0, 0.0, 5.0])
        self.KD = np.array([0.0, 0.0, 0.0])

        self.reset()

    def reset(self) -> None:
        self.rl_observator.reset()
        self.rate_errors = np.zeros((3, 1))
        self.launch_complete = False
        self.prev_rl_action = None

    def get_action(self, observation: dict) -> dict:
        self.rl_observator.update_state(observation)
        rocket_sensors = observation[Schema.Observation.ROCKET_SENSORS]

        if not self.launch_complete:
            rocket_pos = rocket_sensors[6:9]
            reached_altitude = np.all(np.isfinite(rocket_pos)) and rocket_pos[2] >= self.target_altitude

            if not reached_altitude:
                if not np.isnan(rocket_sensors[:3]).any():
                    self.rate_errors = np.append(
                        self.rate_errors, (self.rate_targets - rocket_sensors[:3]).reshape(-1, 1), axis=1
                    )
                    error_integral = np.sum(self.rate_errors, axis=1) / self.rl_observator.sampling_rate
                    error_derivative = (
                        (self.rate_errors[:, -1] - self.rate_errors[:, -2]) * self.rl_observator.sampling_rate
                        if self.rate_errors.shape[1] > 1 else np.zeros(3)
                    )
                    torque_cmd = self.KP * self.rate_errors[:, -1] + self.KI * error_integral + self.KD * error_derivative
                else:
                    torque_cmd = np.zeros(3)

                return {
                    "launch": True,
                    "launch_inclination_heading": self.launch_inclination_heading,
                    "tvc": torque_cmd[0:2],
                    "roll": torque_cmd[2],
                    "throttle": 1.0,
                }

            # This observation already cleared target_altitude -- switch to
            # the RL policy starting this same call, no extra step consumed.
            self.launch_complete = True
            self.rl_observator.mark_launch_complete()

        target_state = observation[Schema.Observation.BALLOON_STATES][0]
        rl_obs = self.rl_observator.get_rl_obs(target_state=target_state, action=self.prev_rl_action)
        rl_obs = self.vec_normalize.normalize_obs(rl_obs)

        rl_action, _ = self.model.predict(rl_obs, deterministic=True)
        tvc, roll, throttle = scale_rl_action(rl_action)

        action = {
            "launch": True,
            "launch_inclination_heading": [0, 0],
            "tvc": tvc,
            "roll": roll,
            "throttle": throttle,
        }
        self.prev_rl_action = action

        return action
