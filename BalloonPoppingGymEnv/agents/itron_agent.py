import logging

import numpy as np

from BalloonPoppingGymEnv.agents.base_agent import BaseAgent
from BalloonPoppingGymEnv.agents.gnc.controller import Controller
from BalloonPoppingGymEnv.agents.gnc.estimator import Estimator
from BalloonPoppingGymEnv.agents.gnc.navigator import Navigator
from BalloonPoppingGymEnv.agents.gnc.selector import Selector
from BalloonPoppingGymEnv.agents.gnc.target_visualizer import (
    TargetSelectionVisualizer,
)
from BalloonPoppingGymEnv.utils.schema import Schema

IDLE_ACTION = {
    "launch": False,
    "launch_inclination_heading": [90.0, 0.0],
    "tvc": [0.0, 0.0],
    "roll": 0.0,
    "throttle": 0.0,
}


class ITronAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        self.logger = logging.getLogger(__name__)

        # Initialize GNC components
        self.estimator = Estimator(self.given_parameters)
        self.selector = Selector(self.given_parameters)
        self.navigator = Navigator(self.given_parameters)
        self.controller = Controller(
            self.given_parameters, **dict(kwargs.get("controller_kwargs", {}))
        )

        self.lookahead_guidance_weight = float(
            kwargs.get("lookahead_guidance_weight", 0.0)
        )
        if not 0.0 <= self.lookahead_guidance_weight <= 1.0:
            raise ValueError("lookahead_guidance_weight must be between 0 and 1")

        corridor_targets = kwargs.get("corridor_targets", {})
        self.corridor_targets = {
            int(committed): int(flyby)
            for committed, flyby in dict(corridor_targets).items()
        }
        self.corridor_guidance_weight = float(
            kwargs.get("corridor_guidance_weight", 0.0)
        )
        if not 0.0 <= self.corridor_guidance_weight <= 1.0:
            raise ValueError("corridor_guidance_weight must be between 0 and 1")
        self.corridor_guidance_horizon = float(
            kwargs.get("corridor_guidance_horizon", 0.0)
        )
        if self.corridor_guidance_horizon < 0.0:
            raise ValueError("corridor_guidance_horizon must not be negative")
        if any(
            committed == flyby for committed, flyby in self.corridor_targets.items()
        ):
            raise ValueError("a corridor target must differ from its committed target")

        fixed_targets = kwargs.get("fixed_target_list")
        fixed_launch_time = kwargs.get("fixed_launch_time")
        if (fixed_targets is None) != (fixed_launch_time is None):
            raise ValueError(
                "fixed_target_list and fixed_launch_time must be provided together"
            )
        self.fixed_target_list = None
        self.fixed_launch_time = None
        if fixed_targets is not None:
            route = tuple(int(target) for target in fixed_targets)
            if not route:
                raise ValueError("fixed_target_list must not be empty")
            if any(target < 0 for target in route):
                raise ValueError("fixed target indices must be non-negative")
            if len(set(route)) != len(route):
                raise ValueError("fixed_target_list must not contain duplicates")
            if float(fixed_launch_time) < 0.0:
                raise ValueError("fixed_launch_time must be non-negative")
            self.fixed_target_list = route
            self.fixed_launch_time = float(fixed_launch_time)

        self.target_visualizer = None
        if kwargs.get("visualize_selector", False):
            visualizer_kwargs = kwargs.get("visualizer_kwargs", {})
            self.target_visualizer = TargetSelectionVisualizer(**visualizer_kwargs)

        # Rebuilding the chain after every hit measures worse, and retesting it
        # on the beam search -- which has none of the axis search's filtering
        # that first explained the loss -- gave the same answer: 3.92 against
        # 4.29, worse on 8 of 24 seeds and better on none.
        #
        # It does what it was meant to: misses beyond 5 m fall from 14 to 2,
        # because replanning drops a last target it can now see is out of reach.
        # That turns out to be the whole problem. A miss at the end of the chain
        # is free -- the burn left over has no other use -- so giving up on one
        # only forfeits the times it would have connected. Over-planning costs
        # nothing and under-planning costs balloons, which is also why every
        # attempt to make the planner more ambitious came out exactly neutral:
        # the current setting already sits on that boundary.
        self.replan_after_hit = False

        # Once a target has actually popped, the measured exit velocity is more
        # reliable than the launch-time chain's predicted heading. Select the
        # next target from that state. This never abandons an active target: it
        # runs only after `_current_target` has confirmed the previous one popped.
        # With the selector's dual turn model, five fixed scenario-1 pool samples
        # raised the mean score from 4.2 to 4.6 and reduced approach-away events
        # from 4.0 to 2.0 per run.
        self.reselect_after_hit = True
        if self.fixed_target_list is not None:
            self.replan_after_hit = False
            self.reselect_after_hit = False

        self.should_launch = False
        self.launch_inclination_heading = None
        self.launch_time = None
        self.target_idx_list = []
        self.current_target_idx = 0
        self.leg_diagnostics = []
        self._engagement_prediction = None

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
        self.leg_diagnostics.clear()
        self._engagement_prediction = None
        if self.target_visualizer is not None:
            self.target_visualizer.reset()

    def get_action(self, observation: dict) -> dict:
        simulation_time = float(observation[Schema.Observation.SIMULATION_TIME])

        # Not launch
        if not self.should_launch:
            if self.fixed_target_list is not None:
                self.should_launch = self._fixed_route_is_ready(observation)
            else:
                self.should_launch = self.selector.should_launch(observation)

            # Still not launch
            if not self.should_launch:
                self._update_target_visualizer(observation)
                return dict(IDLE_ACTION)

            # Run only once after should_launch become true. The selector
            # already searched a chain while deciding to go, so reuse it.
            balloon_states = self.selector.active_states(observation)
            if self.fixed_target_list is not None:
                self.target_idx_list = list(self.fixed_target_list)
            else:
                self.target_idx_list = (
                    self.selector.pending_targets
                    or self.selector.plan_chain(balloon_states)
                )

            # No feasible target set: stay on the pad rather than launching blind.
            if not self.target_idx_list:
                self.should_launch = False
                self._update_target_visualizer(observation)
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
        self._record_completed_engagement(observation, rocket_state, simulation_time)

        previous_target_position = self.current_target_idx
        balloon_idx = self._current_target(observation)
        target_completed = self.current_target_idx > previous_target_position

        if self.reselect_after_hit and target_completed:
            self._replan_from_here(observation, rocket_state, t_since_launch)
            balloon_idx = self._current_target(observation)

        # A target just retired. The plan's remaining legs were priced from a
        # snapshot that is now seconds stale and the error grows down the chain,
        # so rebuild the tail -- but leave the balloon the rocket is already
        # committed to alone. Replanning from scratch at the moment of a hit
        # throws away an approach the guidance has already shaped for, and
        # measured 0.29 balloons worse; the accumulated error lives at the end
        # of the chain, not the front.
        if (
            self.replan_after_hit
            and self.current_target_idx > 0
            and balloon_idx is not None
        ):
            self._replan_tail(observation, rocket_state, t_since_launch, balloon_idx)

        # Chain flown to the end. Any burn still left is worth another chain, and
        # planning here costs nothing that was already committed -- unlike
        # replanning mid-approach, there is no target being flown at to abandon.
        if (
            self.fixed_target_list is None
            and balloon_idx is None
            and self.selector.remaining_budget(t_since_launch) > 0.0
        ):
            self._replan_from_here(observation, rocket_state, t_since_launch)
            balloon_idx = self._current_target(observation)

        if balloon_idx is None:
            # Nothing reachable left; stop steering.
            self._update_target_visualizer(observation)
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
        next_target_state = None
        next_position = self.current_target_idx + 1
        if next_position < len(self.target_idx_list):
            next_idx = self.target_idx_list[next_position]
            if not self.selector.check_target_popped(next_idx, observation):
                next_target_state = self.estimator.estimate_target(
                    observation, next_idx
                )
        corridor_target_state = None
        corridor_idx = self.corridor_targets.get(balloon_idx)
        if corridor_idx is not None and not self.selector.check_target_popped(
            corridor_idx, observation
        ):
            corridor_target_state = self.estimator.estimate_target(
                observation, corridor_idx
            )
        self._start_engagement_prediction(
            balloon_idx,
            target_state,
            rocket_state,
            simulation_time,
            t_since_launch,
        )
        self._update_target_visualizer(observation, balloon_idx)

        thrust_dir, desired_throttle = self.navigator.compute(
            target_state,
            rocket_state,
            t_since_launch,
            next_target_state=next_target_state,
            lookahead_weight=self.lookahead_guidance_weight,
            corridor_target_state=corridor_target_state,
            corridor_weight=self.corridor_guidance_weight,
            corridor_horizon=self.corridor_guidance_horizon,
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

    def _fixed_route_is_ready(self, observation: dict) -> bool:
        """Whether a configured route can launch without targeting the ground."""
        if self.fixed_target_list is None:
            return False

        simulation_time = float(observation[Schema.Observation.SIMULATION_TIME])
        if simulation_time < self.fixed_launch_time:
            return False

        status = np.asarray(
            observation[Schema.Observation.BALLOON_STATUS], dtype=int
        ).reshape(-1)
        if any(target >= len(status) for target in self.fixed_target_list):
            raise ValueError(
                f"fixed_target_list contains an index outside 0..{len(status) - 1}"
            )

        # A fixed sequence must be fully observable when committed. Otherwise
        # guidance would steer towards a balloon's rail position before release.
        return all(status[target] == 1 for target in self.fixed_target_list)

    def _update_target_visualizer(
        self, observation: dict, current_target: int | None = None
    ) -> None:
        """Refresh the optional selector view without changing agent decisions."""
        if self.target_visualizer is None:
            return
        self.target_visualizer.update(
            observation,
            planned_targets=self.target_idx_list[self.current_target_idx :],
            current_target=current_target,
        )

    def _replan_from_here(
        self, observation: dict, rocket_state, t_since_launch: float
    ) -> None:
        """Start a fresh chain from the rocket's current state."""
        rocket_vel = rocket_state[3:6]
        speed = float(np.linalg.norm(rocket_vel))

        chain = self.selector.plan_chain(
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

        tail = self.selector.plan_chain(
            tail_states,
            origin=arrival_pos,
            incoming_dir=arrival_dir,
            time_budget=budget,
            t_since_launch=chain_start,
        )

        self.target_idx_list = [committed] + list(tail or [])
        self.current_target_idx = 0

    def _start_engagement_prediction(
        self,
        balloon_idx: int,
        target_state: np.ndarray,
        rocket_state: np.ndarray,
        simulation_time: float,
        t_since_launch: float,
    ) -> None:
        """Snapshot the selector's prediction when a new target is engaged."""
        if (
            self._engagement_prediction is not None
            and self._engagement_prediction["target"] == balloon_idx
        ):
            return

        duration, position, velocity = self.selector.predict_arrival_state(
            target_state,
            rocket_state[0:3],
            rocket_state[3:6],
            t_since_launch,
        )
        leg = np.asarray(position, dtype=float) - np.asarray(
            rocket_state[0:3], dtype=float
        )
        distance = float(np.linalg.norm(leg))
        incoming_velocity = np.asarray(rocket_state[3:6], dtype=float)
        incoming_speed = float(np.linalg.norm(incoming_velocity))
        if distance > 1e-9 and incoming_speed > 1e-9:
            turn_deg = float(
                np.degrees(
                    np.arccos(
                        np.clip(
                            np.dot(incoming_velocity, leg)
                            / (incoming_speed * distance),
                            -1.0,
                            1.0,
                        )
                    )
                )
            )
        else:
            turn_deg = 0.0
        self._engagement_prediction = {
            "target": int(balloon_idx),
            "start_time": float(simulation_time),
            "duration": float(duration),
            "position": np.asarray(position, dtype=float),
            "velocity": np.asarray(velocity, dtype=float),
            "distance": distance,
            "turn_deg": turn_deg,
            "climb_sin": float(leg[2] / distance) if distance > 1e-9 else 0.0,
            "incoming_speed": incoming_speed,
            "t_since_launch": float(t_since_launch),
        }

    def _record_completed_engagement(
        self,
        observation: dict,
        rocket_state: np.ndarray,
        simulation_time: float,
    ) -> None:
        """Compare a completed leg with the prediction that selected it."""
        prediction = self._engagement_prediction
        if prediction is None:
            return

        status = np.asarray(
            observation[Schema.Observation.BALLOON_STATUS], dtype=int
        ).reshape(-1)
        target = prediction["target"]
        if target >= len(status) or status[target] != 2:
            return

        predicted_velocity = prediction["velocity"]
        actual_velocity = np.asarray(rocket_state[3:6], dtype=float)
        predicted_speed = float(np.linalg.norm(predicted_velocity))
        actual_speed = float(np.linalg.norm(actual_velocity))
        if predicted_speed > 1e-9 and actual_speed > 1e-9:
            direction_error = float(
                np.degrees(
                    np.arccos(
                        np.clip(
                            np.dot(predicted_velocity, actual_velocity)
                            / (predicted_speed * actual_speed),
                            -1.0,
                            1.0,
                        )
                    )
                )
            )
        else:
            direction_error = np.nan

        actual_duration = float(simulation_time - prediction["start_time"])
        diagnostic = {
            "target": target,
            "predicted_duration": prediction["duration"],
            "actual_duration": actual_duration,
            "predicted_speed": predicted_speed,
            "actual_speed": actual_speed,
            "direction_error_deg": direction_error,
            "position_error": float(
                np.linalg.norm(
                    np.asarray(rocket_state[0:3], dtype=float) - prediction["position"]
                )
            ),
            "distance": prediction.get("distance", np.nan),
            "turn_deg": prediction.get("turn_deg", np.nan),
            "climb_sin": prediction.get("climb_sin", np.nan),
            "incoming_speed": prediction.get("incoming_speed", np.nan),
            "t_since_launch": prediction.get("t_since_launch", np.nan),
        }
        self.leg_diagnostics.append(diagnostic)
        self.logger.info(
            "Target %d leg: dt %.2f predicted / %.2f actual s; "
            "exit speed %.2f / %.2f m/s; direction error %.1f deg; "
            "position error %.2f m",
            target,
            diagnostic["predicted_duration"],
            diagnostic["actual_duration"],
            diagnostic["predicted_speed"],
            diagnostic["actual_speed"],
            diagnostic["direction_error_deg"],
            diagnostic["position_error"],
        )
        self._engagement_prediction = None

    def _current_target(self, observation: dict) -> int | None:
        """Index of the target being engaged, skipping any already popped."""
        while self.current_target_idx < len(self.target_idx_list):
            balloon_idx = self.target_idx_list[self.current_target_idx]
            if not self.selector.check_target_popped(balloon_idx, observation):
                return balloon_idx
            self.current_target_idx += 1
        return None
