import numpy as np
from BalloonPoppingGymEnv.agents.base_agent import BaseAgent
from BalloonPoppingGymEnv.agents.gnc.estimator import Estimator
from BalloonPoppingGymEnv.agents.gnc.selector import Selector
from BalloonPoppingGymEnv.agents.gnc.navigator import Navigator
from BalloonPoppingGymEnv.agents.gnc.controller import Controller
from BalloonPoppingGymEnv.utils.schema import Schema
from BalloonPoppingGymEnv.utils.rl_utils import RL_FRAME_SKIP


class ITronAgent(BaseAgent):
    def __init__(self, given_parameters):
        super().__init__(given_parameters)

        self.estimator = Estimator(given_parameters)
        self.selector = Selector(given_parameters)
        self.navigator = Navigator(given_parameters)
        self.controller = Controller(given_parameters)

        self.skip_counter = 0

        self.should_launch = None
        self.launch_inclination_heading = None
        self.desired_acc = None

    def reset(self) -> None:
        self.estimator.reset()
        self.selector.reset()
        self.navigator.reset()
        self.controller.reset()

        self.skip_counter = 0

        self.should_launch = None
        self.launch_inclination_heading = None
        self.desired_acc = None

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
        rocket_state = self.estimator.estimate_rocket(observation)
        self.controller.update(
            rocket_state=rocket_state,
            simulation_time=observation[Schema.Observation.SIMULATION_TIME],
        )

        if self.skip_counter % RL_FRAME_SKIP == 0:
            # Select target
            balloon_states = self.estimator.predict_balloons(observation)
            target_idx = self.selector.select_target(
                rocket_state=rocket_state,
                balloon_states=balloon_states,
            )

            target_state = self.estimator.predict_target(
                observation=observation,
                target_idx=target_idx,
            )

            self.desired_acc = self.navigator.compute(
                rocket_state=rocket_state,
                target_state=target_state,
            )

        self.skip_counter += 1

        tvc, roll, throttle = self.controller.compute(
            desired_acc=self.desired_acc,
        )

        return {
            "launch": True,
            "launch_inclination_heading": self.launch_inclination_heading,
            "tvc": tvc,
            "roll": roll,
            "throttle": throttle,
        }
