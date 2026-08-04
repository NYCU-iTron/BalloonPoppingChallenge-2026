import numpy as np
from BalloonPoppingGymEnv.agents.base_agent import BaseAgent
from BalloonPoppingGymEnv.agents.gnc.selector import Selector
from BalloonPoppingGymEnv.agents.gnc.estimator import Estimator
from BalloonPoppingGymEnv.utils.schema import Schema


class E2EAgent(BaseAgent):
    def __init__(self, given_parameters):
        super().__init__(given_parameters)

        self.selector = Selector(given_parameters)

        self.should_launch = None
        self.launch_inclination_heading = None

    def reset(self) -> None:
        self.selector.reset()

        self.should_launch = None
        self.launch_inclination_heading = None

    def get_action(self, observation: dict) -> dict:
        # Idle
        if not self.should_launch:
            self.should_launch = self.selector.should_launch(observation)

            # Still idle
            if not self.should_launch:
                return {
                    "launch": False,
                    "launch_inclination_heading": np.array([90.0, 0.0]),
                    "tvc": np.zeros(2),
                    "roll": 0.0,
                    "throttle": 0.0,
                }

            self.launch_inclination_heading = self.selector.get_launch_heading(observation)

        # Update states

        return {
            "launch": True,
            "launch_inclination_heading": self.launch_inclination_heading,
            "tvc": tvc,
            "roll": roll,
            "throttle": throttle,
        }
