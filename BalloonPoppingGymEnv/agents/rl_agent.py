import logging
import numpy as np
from pathlib import Path

from BalloonPoppingGymEnv.agents.base_agent import BaseAgent
from BalloonPoppingGymEnv.agents.gnc.estimator import Estimator
from BalloonPoppingGymEnv.agents.gnc.selector import Selector
from BalloonPoppingGymEnv.agents.gnc.rl_navigator import RLNavigator
from BalloonPoppingGymEnv.agents.gnc.controller import Controller
from BalloonPoppingGymEnv.utils.rl_utils import RL_FRAME_SKIP


class RLAgent(BaseAgent):
    def __init__(self, given_parameters, model_path: str = None):
        super().__init__(given_parameters)
        self.logger = logging.getLogger(__name__)

        if model_path is None:
            model_path = Path(__file__).resolve().parent / "final_model.zip"

        # Initialize GNC components
        self.estimator = Estimator(given_parameters)
        self.selector = Selector(given_parameters)
        self.navigator = RLNavigator(given_parameters, model_path=model_path)
        self.controller = Controller(given_parameters)

        self.rocket_state = None
        self.target_state = None
        self.rl_action = None

        # Action-repeat bookkeeping (mirrors RLNavigatorEnv): re-query the policy
        # once every RL_FRAME_SKIP control steps and hold the command in between.
        self._skip_counter = 0
        self._cached_a_cmd = None
        self._cached_throttle = None

    def reset(self) -> None:
        self.estimator.reset()
        self.selector.reset()
        self.navigator.reset()
        self.controller.reset()

        self._skip_counter = 0
        self._cached_a_cmd = None
        self._cached_throttle = None

    def get_action(self, observation: dict) -> dict:
        is_launched = self.selector.should_launch(observation)
        launch_inclination_heading = self.selector.get_launch_heading(observation)

        if not is_launched:
            return {
                "launch": False,
                "launch_inclination_heading": np.array([90.0, 0.0]),
                "tvc": np.zeros(2),
                "roll": 0.0,
                "throttle": self.controller.throttle_min,
            }

        rocket_state = self.estimator.estimate_rocket(observation)

        # Re-query the RL guidance policy only once per RL_FRAME_SKIP control
        # steps; hold the previous command otherwise. The inner controller below
        # still runs every step.
        if self._skip_counter % RL_FRAME_SKIP == 0 or self._cached_a_cmd is None:
            balloon_states = self.estimator.predict_balloons(observation)
            target_idx = self.selector.select_target(balloon_states, rocket_state)
            target_state = self.estimator.predict_target(observation, target_idx)
            self._cached_a_cmd, self._cached_throttle = self.navigator.compute(target_state, rocket_state)
            self.target_state = target_state
        self._skip_counter += 1

        a_cmd = self._cached_a_cmd
        desired_throttle = self._cached_throttle
        tvc, roll, throttle = self.controller.compute(rocket_state, a_cmd, desired_throttle)

        self.rocket_state = rocket_state
        # The RL action is the residual on top of PN (matches the training
        # action space), not the combined command.
        self.rl_action = self.navigator.last_residual.copy()

        return {
            "launch": is_launched,
            "launch_inclination_heading": launch_inclination_heading,
            "tvc": tvc,
            "roll": roll,
            "throttle": throttle,
        }
