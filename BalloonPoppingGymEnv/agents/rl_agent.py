import logging
from BalloonPoppingGymEnv.agents.base_agent import BaseAgent
from BalloonPoppingGymEnv.agents.gnc.estimator import Estimator
from BalloonPoppingGymEnv.agents.gnc.selector import Selector
from BalloonPoppingGymEnv.agents.gnc.rl_navigator import RLNavigator
from BalloonPoppingGymEnv.agents.gnc.controller import Controller


class ITronAgent(BaseAgent):
    def __init__(self, given_parameters):
        super().__init__(given_parameters)
        self.logger = logging.getLogger(__name__)

        # Initialize GNC components
        self.estimator = Estimator(given_parameters)
        self.selector = Selector(given_parameters)
        self.navigator = RLNavigator(given_parameters)
        self.controller = Controller(given_parameters)

    def reset(self) -> None:
        self.estimator.reset()
        self.selector.reset()
        self.navigator.reset()
        self.controller.reset()

    def get_action(self, observation: dict) -> dict:
        rocket_state = self.estimator.estimate_rocket(observation)
        balloon_states = self.estimator.predict_balloons(observation)

        target_idx = self.selector.select_target(balloon_states, rocket_state)
        target_state = self.estimator.predict_target(observation, target_idx)

        a_cmd, desired_throttle = self.navigator.compute(target_state, rocket_state)
        tvc, roll, throttle = self.controller.compute(rocket_state, a_cmd, desired_throttle)

        # Set launch parameters
        is_launched = self.selector.should_launch(observation)
        launch_inclination_heading = self.selector.get_launch_heading(observation)

        return {
            "launch": is_launched,
            "launch_inclination_heading": launch_inclination_heading,
            "tvc": tvc,
            "roll": roll,
            "throttle": throttle,
        }
