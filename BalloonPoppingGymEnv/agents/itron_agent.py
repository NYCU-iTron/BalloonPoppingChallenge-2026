import numpy as np

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

        # Rebuilding the chain after every hit measures worse than flying the
        # launch plan through: the search from a mid-plume origin keeps coming
        # back with one or two targets where the original chain had four.
        # Kept behind a switch because the diagnosis is not finished.
        self.replan_after_hit = False

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

            # Run only once after should_launch become true. The selector
            # already searched a chain while deciding to go, so reuse it.
            balloon_states = self.selector.active_states(observation)
            self.target_idx_list = (
                self.selector.pending_targets
                or self.selector.select_targets(balloon_states)
            )

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

        # A target just retired. The plan's remaining legs were priced from a
        # snapshot that is now seconds stale and the error grows down the chain,
        # so rebuild the tail -- but leave the balloon the rocket is already
        # committed to alone. Replanning from scratch at the moment of a hit
        # throws away an approach the guidance has already shaped for, and
        # measured 0.29 balloons worse; the accumulated error lives at the end
        # of the chain, not the front.
        if self.replan_after_hit and self.current_target_idx > 0 and balloon_idx is not None:
            self._replan_tail(observation, rocket_state, t_since_launch, balloon_idx)

        # Chain flown to the end. Any burn still left is worth another chain, and
        # planning here costs nothing that was already committed -- unlike
        # replanning mid-approach, there is no target being flown at to abandon.
        if balloon_idx is None and self.selector.remaining_budget(t_since_launch) > 0.0:
            self._replan_from_here(observation, rocket_state, t_since_launch)
            balloon_idx = self._current_target(observation)

        if balloon_idx is None:
            # Nothing reachable left; stop steering.
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

    def _replan_from_here(
        self, observation: dict, rocket_state, t_since_launch: float
    ) -> None:
        """Start a fresh chain from the rocket's current state."""
        rocket_vel = rocket_state[3:6]
        speed = float(np.linalg.norm(rocket_vel))

        chain = self.selector.select_targets(
            self.selector.active_states(observation),
            origin=rocket_state[0:3],
            incoming_dir=rocket_vel / speed if speed > 1.0 else None,
            time_budget=self.selector.remaining_budget(t_since_launch),
            t_since_launch=t_since_launch,
        )
        if chain:
            self.target_idx_list = chain
            self.current_target_idx = 0

    def _replan_tail(
        self, observation: dict, rocket_state, t_since_launch: float, committed: int
    ) -> None:
        """Rebuild the chain beyond the balloon the rocket is already flying at."""
        states = self.selector.active_states(observation)

        arrival_time, arrival_pos, arrival_dir = self.selector.predict_arrival(
            states[committed], rocket_state[0:3], rocket_state[3:6], t_since_launch
        )

        chain_start = t_since_launch + arrival_time
        budget = self.selector.remaining_budget(chain_start)
        if budget <= 0.0:
            return

        # Roll the field forward to the moment the tail actually begins, and take
        # the committed balloon out of the running.
        tail_states = states.copy()
        tail_states[:, :3] += tail_states[:, 3:6] * arrival_time
        tail_states[committed] = np.nan

        tail = self.selector.select_targets(
            tail_states,
            origin=arrival_pos,
            incoming_dir=arrival_dir,
            time_budget=budget,
            t_since_launch=chain_start,
        )

        self.target_idx_list = [committed] + list(tail or [])
        self.current_target_idx = 0

    def _current_target(self, observation: dict) -> int | None:
        """Index of the target being engaged, skipping any already popped."""
        while self.current_target_idx < len(self.target_idx_list):
            balloon_idx = self.target_idx_list[self.current_target_idx]
            if not self.selector.check_target_popped(balloon_idx, observation):
                return balloon_idx
            self.current_target_idx += 1
        return None
