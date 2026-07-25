import numpy as np
from pathlib import Path

from BalloonPoppingGymEnv.agents.base_agent import BaseAgent
from BalloonPoppingGymEnv.agents.gnc.estimator import Estimator
from BalloonPoppingGymEnv.agents.gnc.selector import Selector
from BalloonPoppingGymEnv.agents.gnc.rl_navigator import RLNavigator
from BalloonPoppingGymEnv.agents.gnc.controller import Controller
from BalloonPoppingGymEnv.utils.rl_utils import RL_FRAME_SKIP


class RLAgent(BaseAgent):
    def __init__(self,
                 given_parameters,
                 model_path: Path | None = None,
                 vecnormalize_path: Path | None = None,
                ):
        super().__init__(given_parameters)

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
        self.launch_inclination_heading = None
        self.desired_acc = None
        self.desired_throttle = None

    def reset(self) -> None:
        self.estimator.reset()
        self.selector.reset()
        self.navigator.reset()
        self.controller.reset()

        self.skip_counter = 0

        # Commands
        self.should_launch = None
        self.launch_inclination_heading = None
        self.desired_acc = None
        self.desired_throttle = None

    def get_action(self, observation: dict) -> dict:
        # Idle state
        if not self.should_launch:
            self.should_launch = self.selector.should_launch(observation)

            # Still in idle state
            if not self.should_launch:
                return {
                    "launch": False,
                    "launch_inclination_heading": np.array([90.0, 0.0]),
                    "tvc": np.zeros(2, dtype=np.float64),
                    "roll": 0.0,
                    "throttle": 0.0,
                }

            # Run get_launch_heading only once after should_launch turn true
            self.launch_inclination_heading = self.selector.get_launch_heading(observation)

        rocket_state = self.estimator.estimate_rocket(observation)

        if self.skip_counter % RL_FRAME_SKIP == 0:
            balloon_states = self.estimator.predict_balloons(observation)
            target_idx = self.selector.select_target(balloon_states, rocket_state)
            target_state = self.estimator.predict_target(observation, target_idx)
            self.desired_acc, self.desired_throttle = self.navigator.compute(target_state, rocket_state)

        self.skip_counter += 1

        tvc, roll, throttle = self.controller.compute(rocket_state, self.desired_acc, self.desired_throttle)

        return {
            "launch": self.should_launch,
            "launch_inclination_heading": self.launch_inclination_heading,
            "tvc": tvc,
            "roll": roll,
            "throttle": throttle,
        }
