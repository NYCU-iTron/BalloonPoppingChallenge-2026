from BalloonPoppingGymEnv.agents.base_agent import BaseAgent
from BalloonPoppingGymEnv.agents.gnc.estimator import Estimator
from BalloonPoppingGymEnv.agents.gnc.selector import Selector
from BalloonPoppingGymEnv.agents.gnc.navigator import Navigator
from BalloonPoppingGymEnv.agents.gnc.controller import Controller
from BalloonPoppingGymEnv.utils.schema import Schema

IDLE_ACTION = {
    "launch": False,
    "launch_inclination_heading": [90.0, 0.0],
    "tvc": [0.0, 0.0],
    "roll": 0.0,
    "throttle": 0.0,
}


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
        self.launch_time = None
        self.target_idx_list = []
        self.current_target_idx = 0

    def reset(self) -> None:
        self.estimator.reset()
        self.selector.reset()
        self.navigator.reset()
        self.controller.reset()

        self.should_launch = False
        self.launch_inclination_heading = None
        self.launch_time = None
        self.target_idx_list = []
        self.current_target_idx = 0

    def get_action(self, observation: dict) -> dict:
        simulation_time = float(observation[Schema.Observation.SIMULATION_TIME])

        # Not launch
        if not self.should_launch:
            self.should_launch = self.selector.should_launch(observation)

            # Still not launch
            if not self.should_launch:
                return dict(IDLE_ACTION)

            # Run only once after should_launch become true
            balloon_states = observation[Schema.Observation.BALLOON_STATES]
            self.target_idx_list = self.selector.select_targets(balloon_states)

            # No feasible target set: stay on the pad rather than launching blind.
            if not self.target_idx_list:
                self.should_launch = False
                return dict(IDLE_ACTION)

            self.launch_time = simulation_time

            # Aim the rail at the first target. With a thrust-to-weight ratio
            # near 1.2 the rocket cannot turn hard in flight, so this is the
            # most powerful input available.
            first_target_state = balloon_states[self.target_idx_list[0]]
            self.launch_inclination_heading = self.navigator.get_launch_attitude(
                first_target_state
            )
            self.estimator.set_launch_attitude(self.launch_inclination_heading)

        t_since_launch = simulation_time - self.launch_time
        rocket_state = self.estimator.estimate_rocket(observation)

        balloon_idx = self._current_target(observation)
        if balloon_idx is None:
            # Every selected target is gone; nothing left to steer towards.
            return {
                "launch": True,
                "launch_inclination_heading": self.launch_inclination_heading,
                "tvc": [0.0, 0.0],
                "roll": 0.0,
                "throttle": 0.0,
            }

        # Current target state, not a lead one: the guidance law does its own
        # extrapolation over its time-to-go, so leading here would double it.
        target_state = self.estimator.estimate_target(observation, balloon_idx)

        thrust_dir, desired_throttle = self.navigator.compute(
            target_state, rocket_state, t_since_launch
        )
        tvc, roll, throttle = self.controller.compute(
            rocket_state, thrust_dir, desired_throttle, t_since_launch
        )

        return {
            "launch": True,
            "launch_inclination_heading": self.launch_inclination_heading,
            "tvc": tvc,
            "roll": roll,
            "throttle": throttle,
        }

    def _current_target(self, observation: dict) -> int | None:
        """Index of the target being engaged, skipping any already popped."""
        while self.current_target_idx < len(self.target_idx_list):
            balloon_idx = self.target_idx_list[self.current_target_idx]
            if not self.selector.check_target_popped(balloon_idx, observation):
                return balloon_idx
            self.current_target_idx += 1
        return None
