from BalloonPoppingGymEnv.agents.base_agent import BaseAgent
from BalloonPoppingGymEnv.agents.gnc.estimator import Estimator
from BalloonPoppingGymEnv.agents.gnc.selector import Selector
from BalloonPoppingGymEnv.agents.gnc.navigator import Navigator
from BalloonPoppingGymEnv.agents.gnc.controller import Controller


class ITronAgent(BaseAgent):
    def __init__(self, given_parameters):
        super().__init__(given_parameters)

        # Initialize GNC components
        self.estimator = Estimator(given_parameters)
        self.selector = Selector(given_parameters)
        self.navigator = Navigator(given_parameters)
        self.controller = Controller(given_parameters)

        self.should_launch = False
        self.launch_inclination_heading = None
        self.target_idx_list = []
        self.current_target_idx = 0

    def reset(self) -> None:
        self.estimator.reset()
        self.selector.reset()
        self.navigator.reset()
        self.controller.reset()

        self.should_launch = False
        self.launch_inclination_heading = None
        self.target_idx_list = []

    def get_action(self, observation: dict) -> dict:
        # Not launch
        if not self.should_launch:
            self.should_launch = self.selector.should_launch(observation)

            # Still not launch
            if not self.should_launch:
                return {
                    "launch": False,
                    "launch_inclination_heading": [90.0, 0.0],
                    "tvc": [0.0, 0.0],
                    "roll": 0.0,
                    "throttle": 0.0,
                }

            # Run only once after should_launch become true
            self.launch_inclination_heading = self.selector.get_launch_heading(observation)
            self.target_idx_list = self.selector.select_targets(observation)

        rocket_state = self.estimator.estimate_rocket(observation)

        balloon_idx = self.target_idx_list[self.current_target_idx]
        is_target_popped = self.selector.check_target_popped(balloon_idx, observation)
        if is_target_popped:
            self.current_target_idx += 1
            balloon_idx = self.target_idx_list[self.current_target_idx]

        # target_state = self.selector.get_target_state(balloon_idx, observation)
        target_state = self.estimator.predict_target(observation, balloon_idx)

        a_cmd, desired_throttle = self.navigator.compute(target_state, rocket_state)
        tvc, roll, throttle = self.controller.compute(rocket_state, a_cmd, desired_throttle)

        return {
            "launch": True,
            "launch_inclination_heading": self.launch_inclination_heading,
            "tvc": tvc,
            "roll": roll,
            "throttle": throttle,
        }
