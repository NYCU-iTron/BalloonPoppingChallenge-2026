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
    def __init__(self, given_parameters, model_path: str = None, vecnormalize_path: str = None):
        super().__init__(given_parameters)
        self.logger = logging.getLogger(__name__)

        if model_path is None:
            model_path = Path(__file__).resolve().parent / "model.zip"

        if vecnormalize_path is None:
            vecnormalize_path = Path(__file__).resolve().parent / "vecnormalize.pkl"

        # Init GNC components
        self.estimator = Estimator(given_parameters)
        self.selector = Selector(given_parameters)
        self.navigator = RLNavigator(given_parameters, model_path=model_path, vecnormalize_path=vecnormalize_path)
        self.controller = Controller(given_parameters)

        self.skip_counter = 0

        # Commands
        self.should_launch = None
        self.rl_action = None
        self.a_cmd = None
        self.throttle = None

        # States
        self.rocket_state = None
        self.target_state = None

    def reset(self) -> None:
        self.estimator.reset()
        self.selector.reset()
        self.navigator.reset()
        self.controller.reset()

        self.skip_counter = 0

        # Commands
        self.should_launch = None
        self.rl_action = None
        self.a_cmd = None
        self.throttle = None

        # States
        self.rocket_state = None
        self.target_state = None

    def get_action(self, observation: dict) -> dict:
        if not self.should_launch or self.should_launch is None:
            self.should_launch = self.selector.should_launch(observation)
            self.launch_inclination_heading = self.selector.get_launch_heading(observation)

        if not self.should_launch:
            return {
                "launch": False,
                "launch_inclination_heading": np.zeros(2, dtype=np.float64),
                "tvc": np.zeros(2, dtype=np.float64),
                "roll": 0.0,
                "throttle": 0.0,
            }

        self.rocket_state = self.estimator.estimate_rocket(observation)

        if self.skip_counter % RL_FRAME_SKIP == 0:
            balloon_states = self.estimator.predict_balloons(observation)
            target_idx = self.selector.select_target(balloon_states, self.rocket_state)
            self.target_state = self.estimator.predict_target(observation, target_idx)
            self.a_cmd, self.throttle = self.navigator.compute(self.target_state, self.rocket_state)

        self.skip_counter += 1
        self.rl_action = self.navigator.last_residual.copy()

        tvc, roll, throttle = self.controller.compute(self.rocket_state, self.a_cmd, self.throttle)

        return {
            "launch": self.should_launch,
            "launch_inclination_heading": self.launch_inclination_heading,
            "tvc": tvc,
            "roll": roll,
            "throttle": throttle,
        }
