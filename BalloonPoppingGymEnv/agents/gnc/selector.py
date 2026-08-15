import logging

import numpy as np
from BalloonPoppingGymEnv.agents.gnc.vehicle import Vehicle
from BalloonPoppingGymEnv.utils.schema import Schema


class Selector:
    def __init__(self, given_parameters):
        self.logger = logging.getLogger(__name__)

        # Shared vehicle model, so the chain is budgeted against the same flight
        # profile guidance will actually fly. Without it the DP picks chains on
        # pure geometry and hands the rocket more balloons than 30 seconds of
        # burn can reach -- the previous run engaged 3 of 8 before burnout.
        self.vehicle = Vehicle(given_parameters)

        # Fraction of the burn to spend; the rest is margin for the estimate
        # being coarse and for a leg going worse than modelled.
        self.time_budget_fraction = 0.95
        self.time_weight = 5.0  # cost per second, to prefer quicker chains

        # --- Chain cost weights -------------------------------------------
        # Lateral manoeuvring is the scarce resource: thrust already sits about
        # 59 degrees off the velocity vector just holding the vehicle up, and
        # 43% of guidance steps run against the acceleration limit. Distance off
        # the plume axis is what costs lateral authority, so it is priced here
        # rather than buried in the method body where it could not be swept.
        self.dist_weight = 10.0  # off-axis distance from the plume axis
        self.angle_weight = 20.0  # turn between consecutive legs, per degree
        self.max_segment_dist = 100.0  # (m) beyond this a leg is a long haul
        self.too_far_weight = 5.0

        # A nominal extra target is worthwhile, but not at any manoeuvre cost.
        # The old lexicographic objective always preferred a deeper chain, even
        # when the extra predicted hit required a foldback the rocket could not
        # actually complete. This makes target count and flight cost comparable.
        self.target_reward = 3000.0

        # A short leg cannot absorb an arbitrary heading change. Charge the
        # estimated arc-length shortfall as an out-and-back recovery, without
        # rejecting the target: a close target straight ahead remains free.
        self.short_turn_recovery_factor = 2.0
        self.conservative_short_turn_recovery_factor = 12.0

        # Transit time is deliberately budgeted at the conservative planning
        # speed, but turn geometry must use the 20--25 m/s velocity measured at
        # balloon crossings.  Assuming the exit velocity instantly aligns with
        # every short leg made a sequence of nearby targets look much easier
        # than it is.  Propagate the heading that the available lateral
        # acceleration can actually reach instead.
        self.turn_geometry_speed = 22.0

        # Beam search over the rocket's own state, as an alternative to the
        # axis-ordered DP. Width is generous because the branching factor is the
        # whole balloon field and a good chain can start with a mediocre leg.
        # Raising it from 96 through 1024 produced the same scenario-1 route, so
        # the remaining error is in the dynamics model rather than beam pruning.
        #
        # `angle_weight` above turned out to be inert here: 5, 10 and 20 give
        # bit-identical results across 24 seeds, because the turn is already
        # priced through the time it costs -- entry speed falls with cos(turn)
        # and time_weight multiplies the resulting leg time. Only 40 moved the
        # answer, and it moved it down.
        self.beam_width = 96
        self.max_chain_length = 8
        self.use_beam_search = True

        # --- Launch timing ------------------------------------------------
        # Every second spent waiting puts the nearest balloon further downwind,
        # and reaching the first target is the single largest item in the burn
        # budget. Measured over the trajectory pool: launching around t=20
        # plans 3.38 targets against 2.12 for the old fixed t=70, with a broad
        # plateau from t=10 to t=25. So launch as soon as a worthwhile chain is
        # reachable rather than waiting for the whole field to be airborne.
        self.earliest_launch_time = 10.0  # (s) below this too few are up
        self.latest_launch_time = 30.0  # (s) stop holding out for a better chain
        self.launch_evaluation_interval = 1.0  # (s) between replans while waiting
        self.min_launch_chain = 5  # targets that make a launch worth taking

        self._last_launch_evaluation = -np.inf
        self.pending_targets = None

        # Balloon and rocket positions are reported as altitude above sea level,
        # so the launch pad sits at z = elevation, not at the coordinate origin.
        elevation = given_parameters[Schema.Given.Section.ENVIRONMENT][
            Schema.Given.Environment.ELEVATION
        ]
        self.pad_origin = np.array([0.0, 0.0, float(elevation)])

    def reset(self):
        self._last_launch_evaluation = -np.inf
        self.pending_targets = None

    @staticmethod
    def active_states(observation: dict) -> np.ndarray:
        """Balloon states with grounded and popped balloons marked NaN.

        The environment reports all balloons every step: unreleased ones sit at
        their launch point and popped ones carry on flying, so position alone
        cannot say which are in play. Only the status flag can. This was
        harmless while the agent waited for the whole field to be airborne;
        launching at t=10 with most balloons still on the ground would
        otherwise hand the planner 80 targets that do not exist yet.
        """
        states = np.array(
            observation[Schema.Observation.BALLOON_STATES], dtype=float
        ).copy()
        status = np.array(
            observation[Schema.Observation.BALLOON_STATUS], dtype=int
        ).flatten()
        states[status != 1] = np.nan
        return states

    def should_launch(self, observation: dict) -> bool:
        """Launch as soon as a worthwhile chain is reachable.

        Waiting is not free: the nearest balloon drifts further away every
        second, and the leg out to the first target is the largest single item
        in the burn budget. So plan on a coarse cadence and go as soon as the
        plan is good enough, falling back to a deadline if it never is.

        The chain found here is kept on ``pending_targets`` so the caller can
        fly it without paying for the search twice.
        """
        simulation_time = float(observation[Schema.Observation.SIMULATION_TIME])
        self.pending_targets = None

        if simulation_time < self.earliest_launch_time:
            return False

        if (
            simulation_time - self._last_launch_evaluation
            < self.launch_evaluation_interval
        ):
            return False
        self._last_launch_evaluation = simulation_time

        targets = self.plan_chain(self.active_states(observation))
        if not targets:
            return False

        deadline_reached = simulation_time >= self.latest_launch_time
        if len(targets) >= self.min_launch_chain or deadline_reached:
            self.pending_targets = targets
            return True

        return False

    def get_launch_heading(self, observation: dict) -> np.ndarray:
        """
        Returns
        -------
        heading: np.ndarray
            [inclination, heading] in degrees based on balloon positions.
        """
        balloon_states = np.array(
            observation[Schema.Observation.BALLOON_STATES], dtype=float
        )
        valid_mask = ~np.isnan(balloon_states[:, 0])
        valid_indices = np.where(valid_mask)[0]
        velocities = balloon_states[valid_indices, 3:]

        mean_vel_xy = np.mean(velocities[:, :2], axis=0)
        vel_norm = np.linalg.norm(mean_vel_xy)
        dir_xy = mean_vel_xy / vel_norm if vel_norm > 1e-5 else np.array([1.0, 0.0])

        # The simulator's heading is a compass bearing (0 = North, 90 = East,
        # clockwise), while dir_xy is an ENU vector -- so the East component
        # goes first in the arctan2.
        heading_deg = np.degrees(np.arctan2(dir_xy[0], dir_xy[1]))
        heading_deg = heading_deg % 360.0

        heading = np.array([90.0, heading_deg])
        return heading

    @staticmethod
    def _position_at(balloon: dict, t: float) -> np.ndarray:
        """Where a balloon will be ``t`` seconds after the plan is made."""
        if not np.isfinite(t):
            return balloon["pos"]
        return balloon["pos"] + balloon["vel"] * t

    def _predict_leg(
        self,
        balloon: dict,
        from_pos: np.ndarray,
        chain_depart: float,
        t_since_launch: float,
        turn_angle: float,
    ) -> tuple[np.ndarray, float, float]:
        """Leg to a balloon that keeps drifting while the rocket is on its way.

        Arrival time and aim point define each other, so solve for them: guess an
        arrival, move the balloon there, re-time the leg. Two passes are enough --
        the aim point shifts by metres on the second.

        Two clocks are in play and they are not the same once the agent replans
        mid-flight: balloons are extrapolated from the moment the plan is made,
        while the mass and thrust curve is read at time since launch.

        Returns the leg vector, its length, and the time it takes.
        """
        arrival = chain_depart
        leg = balloon["pos"] - from_pos
        length = float(np.linalg.norm(leg))
        leg_time = 0.0

        for _ in range(2):
            leg = self._position_at(balloon, arrival) - from_pos
            length = float(np.linalg.norm(leg))
            if length < 1e-9:
                return leg, 0.0, 0.0
            leg_time = self.vehicle.transit_time(
                length,
                turn_angle,
                chain_depart + t_since_launch,
                float(leg[2] / length),
            )
            arrival = chain_depart + leg_time

        return leg, length, leg_time

    def _predict_leg_state(
        self,
        balloon: dict,
        from_pos: np.ndarray,
        chain_depart: float,
        t_since_launch: float,
        turn_angle: float,
        incoming_speed: float | None,
    ) -> tuple[np.ndarray, float, float, float]:
        """Predict one leg while retaining its exit speed."""
        arrival = chain_depart
        leg = balloon["pos"] - from_pos
        length = float(np.linalg.norm(leg))
        leg_time = 0.0
        exit_speed = self.vehicle.planning_speed

        for _ in range(2):
            leg = self._position_at(balloon, arrival) - from_pos
            length = float(np.linalg.norm(leg))
            if length < 1e-9:
                return leg, 0.0, 0.0, max(float(incoming_speed or 0.0), 0.0)
            leg_time, exit_speed = self.vehicle.transit_state(
                length,
                turn_angle,
                chain_depart + t_since_launch,
                float(leg[2] / length),
                incoming_speed,
            )
            arrival = chain_depart + leg_time

        return leg, length, leg_time, exit_speed

    def _predict_leg_from_heading(
        self,
        balloon: dict,
        from_pos: np.ndarray,
        chain_depart: float,
        t_since_launch: float,
        incoming_dir: np.ndarray | None,
    ) -> tuple[np.ndarray, float, float, float]:
        """Solve moving-target arrival and the turn into that leg together.

        Turn cost lengthens arrival time, which lets the balloon drift farther
        and changes the turn again. Previously the beam predicted drift with a
        zero-angle leg and added turn time afterwards, so difficult corners were
        always aimed at a stale, too-near position.
        """
        turn = 0.0
        leg = np.zeros(3)
        length = 0.0
        leg_time = 0.0
        departure_turn = 0.0
        departure_length = 0.0
        if incoming_dir is not None:
            departure_leg = self._position_at(balloon, chain_depart) - from_pos
            departure_length = float(np.linalg.norm(departure_leg))
            if departure_length > 1e-9:
                departure_turn = float(
                    np.arccos(
                        np.clip(
                            np.dot(incoming_dir, departure_leg / departure_length),
                            -1.0,
                            1.0,
                        )
                    )
                )

        for _ in range(6):
            leg, length, leg_time = self._predict_leg(
                balloon,
                from_pos,
                chain_depart,
                t_since_launch,
                turn,
            )
            if length < 1e-9:
                return leg, length, leg_time, 0.0
            leg_dir = leg / length
            updated_turn = (
                0.0
                if incoming_dir is None
                else max(
                    departure_turn,
                    float(np.arccos(np.clip(np.dot(incoming_dir, leg_dir), -1.0, 1.0))),
                )
            )
            if abs(updated_turn - turn) < 1e-4:
                turn = updated_turn
                break
            turn = updated_turn

        # Keep the returned flight time and aim point consistent with the last
        # turn estimate even when a fast target has not quite converged.
        leg, length, leg_time = self._predict_leg(
            balloon,
            from_pos,
            chain_depart,
            t_since_launch,
            turn,
        )
        if length < 1e-9:
            return leg, length, leg_time, 0.0
        leg_dir = leg / length
        turn = (
            0.0
            if incoming_dir is None
            else max(
                departure_turn,
                float(np.arccos(np.clip(np.dot(incoming_dir, leg_dir), -1.0, 1.0))),
            )
        )
        leg_time += self._short_turn_recovery_time(
            departure_length,
            departure_turn,
            chain_depart + t_since_launch,
        )
        return leg, length, leg_time, turn

    def _predict_leg_from_velocity(
        self,
        balloon: dict,
        from_pos: np.ndarray,
        chain_depart: float,
        t_since_launch: float,
        incoming_velocity: np.ndarray | None,
    ) -> tuple[np.ndarray, float, float, float, np.ndarray]:
        """Solve a moving-target leg and propagate its predicted exit velocity."""
        incoming_dir = None
        incoming_speed = None
        if incoming_velocity is not None:
            incoming_velocity = np.asarray(incoming_velocity, dtype=float)
            speed = float(np.linalg.norm(incoming_velocity))
            if speed > 1e-9:
                incoming_dir = incoming_velocity / speed
                incoming_speed = speed

        turn = 0.0
        leg = np.zeros(3)
        length = 0.0
        leg_time = 0.0
        exit_speed = self.vehicle.planning_speed
        departure_turn = 0.0
        departure_length = 0.0
        if incoming_dir is not None:
            departure_leg = self._position_at(balloon, chain_depart) - from_pos
            departure_length = float(np.linalg.norm(departure_leg))
            if departure_length > 1e-9:
                departure_turn = float(
                    np.arccos(
                        np.clip(
                            np.dot(incoming_dir, departure_leg / departure_length),
                            -1.0,
                            1.0,
                        )
                    )
                )

        for _ in range(6):
            leg, length, leg_time, exit_speed = self._predict_leg_state(
                balloon,
                from_pos,
                chain_depart,
                t_since_launch,
                turn,
                incoming_speed,
            )
            if length < 1e-9:
                return leg, length, leg_time, 0.0, np.zeros(3)
            leg_dir = leg / length
            updated_turn = (
                0.0
                if incoming_dir is None
                else max(
                    departure_turn,
                    float(np.arccos(np.clip(np.dot(incoming_dir, leg_dir), -1.0, 1.0))),
                )
            )
            if abs(updated_turn - turn) < 1e-4:
                turn = updated_turn
                break
            turn = updated_turn

        leg, length, leg_time, exit_speed = self._predict_leg_state(
            balloon,
            from_pos,
            chain_depart,
            t_since_launch,
            turn,
            incoming_speed,
        )
        if length < 1e-9:
            return leg, length, leg_time, 0.0, np.zeros(3)

        leg_dir = leg / length
        turn = (
            0.0
            if incoming_dir is None
            else max(
                departure_turn,
                float(np.arccos(np.clip(np.dot(incoming_dir, leg_dir), -1.0, 1.0))),
            )
        )
        leg_time += self._short_turn_recovery_time(
            departure_length,
            departure_turn,
            chain_depart + t_since_launch,
            incoming_speed,
        )
        exit_direction = self._predict_exit_direction(
            incoming_dir,
            leg_dir,
            turn,
            length,
            chain_depart + t_since_launch,
        )
        return leg, length, leg_time, turn, exit_direction * exit_speed

    def _predict_exit_direction(
        self,
        incoming_dir: np.ndarray | None,
        leg_dir: np.ndarray,
        turn_angle: float,
        leg_length: float,
        t_elapsed: float,
    ) -> np.ndarray:
        """Direction retained when a leg is too short to finish its turn.

        A vehicle moving at speed ``v`` needs approximately
        ``v**2 * angle / lateral_accel`` metres to rotate its velocity through
        ``angle``.  The old model used that relation to price lost speed, but
        then returned a velocity exactly along the new leg anyway.  Rotating
        only through the angle the leg can contain preserves the residual
        heading error for the next beam-search step.
        """
        if incoming_dir is None or turn_angle < 1e-9:
            return leg_dir

        lateral_accel = max(
            self.vehicle.max_lateral_accel(t_elapsed),
            self.vehicle.min_transit_accel,
        )
        geometry_speed = max(
            float(getattr(self, "turn_geometry_speed", self.vehicle.planning_speed)),
            self.vehicle.planning_speed,
        )
        achievable_turn = lateral_accel * leg_length / geometry_speed**2
        if achievable_turn >= turn_angle:
            return leg_dir

        incoming_dir = np.asarray(incoming_dir, dtype=float)
        incoming_dir /= np.linalg.norm(incoming_dir)
        axis = np.cross(incoming_dir, leg_dir)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm < 1e-9:
            # Parallel vectors are already handled above.  For a 180-degree
            # reversal, choose any stable axis perpendicular to the approach.
            basis = np.array([1.0, 0.0, 0.0])
            if abs(float(np.dot(incoming_dir, basis))) > 0.9:
                basis = np.array([0.0, 1.0, 0.0])
            axis = np.cross(incoming_dir, basis)
            axis_norm = float(np.linalg.norm(axis))
        axis /= axis_norm

        # Rodrigues' formula rotates the incoming velocity towards the desired
        # leg without changing its magnitude.
        rotated = (
            incoming_dir * np.cos(achievable_turn)
            + np.cross(axis, incoming_dir) * np.sin(achievable_turn)
            + axis * float(np.dot(axis, incoming_dir)) * (1.0 - np.cos(achievable_turn))
        )
        return rotated / np.linalg.norm(rotated)

    def _short_turn_recovery_time(
        self,
        departure_length: float,
        departure_turn: float,
        t_elapsed: float,
        incoming_speed: float | None = None,
    ) -> float:
        """Cost of overshooting a leg too short to contain its initial turn."""
        if departure_turn < 1e-9 or self.short_turn_recovery_factor <= 0.0:
            return 0.0

        lateral_accel = max(
            self.vehicle.max_lateral_accel(t_elapsed),
            self.vehicle.min_transit_accel,
        )
        speed = (
            self.vehicle.planning_speed
            if incoming_speed is None
            else max(float(incoming_speed), 0.0)
        )
        required_arc_length = speed**2 * departure_turn / lateral_accel
        shortfall = max(required_arc_length - departure_length, 0.0)
        return float(self.short_turn_recovery_factor * shortfall / max(speed, 1e-9))

    def predict_arrival_state(
        self,
        target_state: np.ndarray,
        rocket_pos: np.ndarray,
        rocket_vel: np.ndarray,
        t_since_launch: float,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Predict arrival time, position, and exit velocity for diagnostics."""
        balloon = {
            "pos": np.asarray(target_state[:3], dtype=float),
            "vel": np.asarray(target_state[3:6], dtype=float),
        }
        leg, _, leg_time, _, exit_velocity = self._predict_leg_from_velocity(
            balloon,
            np.asarray(rocket_pos, dtype=float),
            0.0,
            t_since_launch,
            np.asarray(rocket_vel, dtype=float),
        )
        return leg_time, np.asarray(rocket_pos, dtype=float) + leg, exit_velocity

    def predict_arrival(
        self,
        target_state: np.ndarray,
        rocket_pos: np.ndarray,
        rocket_vel: np.ndarray,
        t_since_launch: float,
    ) -> tuple[float, np.ndarray, np.ndarray | None]:
        """When and where the rocket will reach the target it is already on.

        Used to replan everything *after* the committed target without
        disturbing the approach to it.

        Returns seconds to arrival, the arrival point, and the direction the
        rocket will be travelling when it gets there.
        """
        rocket_pos = np.asarray(rocket_pos, dtype=float)
        leg_time, arrival_pos, exit_velocity = self.predict_arrival_state(
            target_state, rocket_pos, rocket_vel, t_since_launch
        )
        exit_speed = float(np.linalg.norm(exit_velocity))
        direction = exit_velocity / exit_speed if exit_speed > 1e-9 else None
        return leg_time, arrival_pos, direction

    def select_targets_beam(
        self,
        balloon_states: np.ndarray,
        origin: np.ndarray | None = None,
        incoming_dir: np.ndarray | None = None,
        incoming_velocity: np.ndarray | None = None,
        time_budget: float | None = None,
        t_since_launch: float = 0.0,
    ) -> list[int] | None:
        """Return the highest-ranked chain from the beam search."""
        ranked = self.rank_target_chains_beam(
            balloon_states,
            origin=origin,
            incoming_dir=incoming_dir,
            incoming_velocity=incoming_velocity,
            time_budget=time_budget,
            t_since_launch=t_since_launch,
            limit=1,
        )
        if not ranked:
            return None

        utility, chain = ranked[0]
        self.logger.info(
            f"Beam chain of {len(chain)} within the flight budget: "
            f"{chain} (utility {utility:.1f})"
        )
        return chain

    def rank_target_chains_beam(
        self,
        balloon_states: np.ndarray,
        origin: np.ndarray | None = None,
        incoming_dir: np.ndarray | None = None,
        incoming_velocity: np.ndarray | None = None,
        time_budget: float | None = None,
        t_since_launch: float = 0.0,
        *,
        limit: int = 10,
    ) -> list[tuple[float, list[int]]]:
        """Chain search carried out in the rocket's own frame, with no plume axis.

        The axis-based search sorts balloons along a fitted plume direction and
        only ever moves forwards along it. That buys a directed graph to run a
        DP over, but it is a computational convenience rather than a physical
        constraint, and two of the three things it was doing turned out to be
        harmful: the off-axis penalty measured monotonically worse as it was
        raised, and the "ahead of the origin" filter is what left mid-flight
        replans with nothing to plan.

        What actually constrains the vehicle is its velocity. Speed is won
        almost entirely on the first, steep leg and merely carried afterwards,
        and turning is what scrubs it off -- so the state that matters when
        choosing the next balloon is where the rocket is, how fast, and which
        way it is pointing. This searches over exactly that, and prices each
        turn against the heading the rocket will really hold rather than
        against a direction inferred from the previous chain entry.

        Acyclicity comes for free: every leg costs time and the budget only
        shrinks, so no chain can revisit a balloon.

        Returns up to ``limit`` distinct ``(utility, chain)`` pairs. Keeping
        alternatives is development-only support for closed-loop route trials;
        :meth:`select_targets_beam` still chooses exactly the first entry.
        """
        if limit < 1:
            raise ValueError("limit must be at least one")
        if time_budget is None:
            time_budget = self.vehicle.burn_time * self.time_budget_fraction

        chain_origin = (
            self.pad_origin if origin is None else np.asarray(origin, dtype=float)
        )

        valid = np.where(~np.isnan(balloon_states[:, 0]))[0]
        if valid.size == 0:
            return []

        candidates = [
            {
                "id": int(i),
                "pos": balloon_states[i, :3] - chain_origin,
                "vel": balloon_states[i, 3:6],
            }
            for i in valid
        ]

        # A beam entry: cumulative cost, elapsed, position, velocity, chain.
        # ``incoming_dir`` remains supported for callers that do not know speed.
        if incoming_velocity is not None:
            start_velocity = np.asarray(incoming_velocity, dtype=float)
        elif incoming_dir is not None:
            start_velocity = (
                np.asarray(incoming_dir, dtype=float) * self.vehicle.planning_speed
            )
        else:
            start_velocity = None
        beam = [(0.0, 0.0, np.zeros(3), start_velocity, ())]
        ranked: list[tuple[float, tuple[int, ...]]] = []

        for _ in range(self.max_chain_length):
            expanded = []
            for cost, elapsed, position, velocity, chain in beam:
                for candidate in candidates:
                    if candidate["id"] in chain:
                        continue

                    leg, length, leg_time, turn, exit_velocity = (
                        self._predict_leg_from_velocity(
                            candidate,
                            position,
                            elapsed,
                            t_since_launch,
                            velocity,
                        )
                    )
                    if length < 1e-6:
                        continue

                    finish = elapsed + leg_time
                    if finish > time_budget:
                        continue

                    penalty = 0.0
                    if length > self.max_segment_dist:
                        penalty += (
                            length - self.max_segment_dist
                        ) ** 2 * self.too_far_weight

                    expanded.append(
                        (
                            cost
                            + self.time_weight * leg_time
                            + self.angle_weight * np.degrees(turn)
                            + penalty,
                            finish,
                            position + leg,
                            exit_velocity,
                            chain + (candidate["id"],),
                        )
                    )

            if not expanded:
                break

            expanded.sort(key=lambda e: e[0])
            beam = expanded[: self.beam_width]
            ranked.extend(
                (entry[0] - self.target_reward * len(entry[4]), entry[4])
                for entry in beam
            )

        ranked.sort(key=lambda item: item[0])
        unique = []
        seen = set()
        for utility, chain in ranked:
            if chain in seen:
                continue
            seen.add(chain)
            unique.append((utility, list(chain)))
            if len(unique) >= limit:
                break
        return unique

    def candidate_chains(
        self,
        balloon_states: np.ndarray,
        *,
        limit_per_model: int = 10,
        **kwargs,
    ) -> list[list[int]]:
        """Return diverse permissive and conservative routes for simulation."""
        if not self.use_beam_search:
            route = self.select_targets(balloon_states, **kwargs)
            return [route] if route else []

        original_factor = self.short_turn_recovery_factor
        try:
            self.short_turn_recovery_factor = (
                self.conservative_short_turn_recovery_factor
            )
            conservative = self.rank_target_chains_beam(
                balloon_states, limit=limit_per_model, **kwargs
            )

            self.short_turn_recovery_factor = original_factor
            permissive = self.rank_target_chains_beam(
                balloon_states, limit=limit_per_model, **kwargs
            )
        finally:
            self.short_turn_recovery_factor = original_factor

        # Alternate the two models instead of sorting their utilities together:
        # the conservative model intentionally uses a different cost scale.
        routes = []
        seen = set()
        for rank in range(max(len(conservative), len(permissive))):
            for ranked in (conservative, permissive):
                if rank >= len(ranked):
                    continue
                chain = tuple(ranked[rank][1])
                if chain not in seen:
                    seen.add(chain)
                    routes.append(list(chain))
        return routes

    def plan_chain(self, balloon_states: np.ndarray, **kwargs) -> list[int] | None:
        """Plan with both permissive and conservative short-turn models.

        Use the conservative route whenever it reaches at least as many
        targets; otherwise retain the longer permissive route. This spends
        extra search time to avoid folding back without sacrificing a target
        solely because the turn estimate is cautious.
        """
        if self.use_beam_search:
            original_factor = self.short_turn_recovery_factor
            try:
                self.short_turn_recovery_factor = (
                    self.conservative_short_turn_recovery_factor
                )
                conservative = self.select_targets_beam(balloon_states, **kwargs)

                self.short_turn_recovery_factor = original_factor
                permissive = self.select_targets_beam(balloon_states, **kwargs)
            finally:
                self.short_turn_recovery_factor = original_factor

            if len(conservative or []) >= len(permissive or []):
                return conservative
            return permissive
        return self.select_targets(balloon_states, **kwargs)

    def remaining_budget(self, t_since_launch: float) -> float:
        """Burn time left to spend on a chain, in seconds."""
        return max(
            self.vehicle.burn_time * self.time_budget_fraction - t_since_launch, 0.0
        )

    def select_targets(
        self,
        balloon_states: np.ndarray,
        origin: np.ndarray | None = None,
        incoming_dir: np.ndarray | None = None,
        time_budget: float | None = None,
        t_since_launch: float = 0.0,
        reverse_order: bool = False,
    ) -> list[int] | None:
        """
        Parameters
        ----------
        balloon_states : np.ndarray
            Shape (N, 6) predicted states [x, y, z, vx, vy, vz]; NaN position
            marks inactive (unreleased or popped) balloons.
        origin : np.ndarray | None
            Where the chain starts. The launch pad by default; the rocket's
            current position when replanning in flight.
        incoming_dir : np.ndarray | None
            Unit velocity the chain is entered with, so the first leg pays for
            the turn out of it. None off the pad, where the rail is aimed at the
            first target and there is no turn to pay for.
        time_budget : float | None
            Seconds of burn the chain may use. The whole burn by default.
        t_since_launch : float
            Where the vehicle already is on its mass and thrust curve.
        reverse_order : bool
            Work the plume from its downwind end back towards the pad instead of
            outwards from it. Legs then run against the drift, so balloons close
            on the rocket rather than running from it.

            Measured worse at every launch time from t=10 to t=40, by 0.67 to
            1.17 targets, and measured worse again after drift was modelled --
            drift being the whole reason to expect it to win. The pad sits at
            the upwind end, so the rocket has to cross the entire plume before
            it can start working back, and that crossing costs more than meeting
            the balloons head-on saves.

        Returns
        -------
        list[int] | None
            Target balloon IDs ordered from T1 onwards, or None if no chain is
            reachable.
        """
        # --- 1. 權重與門檻參數設定 ---
        dist_weight = self.dist_weight
        angle_weight = self.angle_weight

        max_dist = self.max_segment_dist
        too_far_weight = self.too_far_weight

        # --- 2. 過濾 Valid 氣球 ---
        valid_mask = ~np.isnan(balloon_states[:, 0])
        valid_indices = np.where(valid_mask)[0]

        max_chain = 8  # 鏈長上限
        if len(valid_indices) == 0:
            return None

        # 一律換算成「相對發射台」座標：主軸擬合、投影 s、離軸距離 d 與發射夾角
        # 都應該以發射台為原點，而不是海平面。
        chain_origin = (
            self.pad_origin if origin is None else np.asarray(origin, dtype=float)
        )
        positions = balloon_states[valid_indices, :3] - chain_origin
        velocities = balloon_states[valid_indices, 3:]

        # --- 3. (r, z) 擬合斜率 + 速度對齊生成 3D 主軸向量 u ---
        r = np.hypot(positions[:, 0], positions[:, 1])
        z = positions[:, 2]

        sum_r2 = np.sum(r**2)
        slope = np.sum(r * z) / sum_r2 if sum_r2 > 1e-6 else 1.0

        mean_vel_xy = np.mean(velocities[:, :2], axis=0)
        vel_norm = np.linalg.norm(mean_vel_xy)
        dir_xy = mean_vel_xy / vel_norm if vel_norm > 1e-5 else np.array([1.0, 0.0])

        if origin is None:
            main_axis = np.array([dir_xy[0], dir_xy[1], slope])
        else:
            # Replanning from inside the plume: the (r, z) slope fit needs an
            # origin the balloons are all in front of. With the rocket among
            # them r loses its meaning and the fitted slope becomes noise, which
            # is what left mid-flight replans finding one target where the
            # launch plan found four. The plume's own mean velocity gives the
            # same axis without depending on where the origin sits.
            main_axis = np.mean(velocities, axis=0)
            if np.linalg.norm(main_axis) < 1e-5:
                main_axis = np.array([dir_xy[0], dir_xy[1], slope])

        u = main_axis / np.linalg.norm(main_axis)

        # --- 4. 計算投影高度 s 與垂直離軸距離 d，並按 s 排序 ---
        s_vals = np.dot(positions, u)
        valid_balloons = []

        for idx, pos, vel, s_val in zip(valid_indices, positions, velocities, s_vals):
            if s_val > 0:  # 只考慮原點前方的氣球
                d_val = np.linalg.norm(pos - s_val * u)
                valid_balloons.append(
                    {"id": int(idx), "pos": pos, "vel": vel, "s": s_val, "d": d_val}
                )

        # K 是上限而不是下限。要求「原點前方至少 8 顆」對從發射台規劃來說沒問題
        # ——整片氣球場都在前方——但中途重規劃時原點已經在羽流內部，s > 0 會擋掉
        # 一半以上的候選，硬性下限會讓尾巴永遠規劃不出東西。
        if not valid_balloons:
            return None
        K = min(max_chain, len(valid_balloons))

        # 依主軸進度 s 由低到高排序 (天然保證單向推進)
        sorted_balloons = sorted(
            valid_balloons, key=lambda b: b["s"], reverse=reverse_order
        )
        N = len(sorted_balloons)

        # --- 5. DP 演算法實作 ---
        dp = np.full((N, K + 1), float("inf"))
        parent = np.full((N, K + 1), -1, dtype=int)
        chain_start = np.array([0.0, 0.0, 0.0])  # 鏈的起點，因 positions 已是相對座標

        # 沿著最佳成本路徑累計的飛行時間，用來剔除燒完前飛不完的鏈
        elapsed = np.full((N, K + 1), float("inf"))
        if time_budget is None:
            time_budget = self.vehicle.burn_time * self.time_budget_fraction

        # Base Case: k = 1 (第一顆目標只算離軸、發射夾角與時間)
        for i in range(N):
            # 目標在抵達時刻的位置，而不是現在的位置。整條鏈是在發射瞬間規劃的，
            # 但最後一顆要 30 秒後才會被追上，那時它已經飄走約 230 公尺 ——
            # 用當下快照算出來的幾何，火箭實際飛的距離是規劃值的 2.31 倍。
            dir_origin, norm_orig, leg_time = self._predict_leg(
                sorted_balloons[i], chain_start, 0.0, t_since_launch, 0.0
            )
            if norm_orig < 1e-5:
                continue

            # Off the pad the rail is aimed at this balloon, so there is no turn.
            # Replanning in flight has to pay for turning out of the velocity the
            # rocket already carries.
            entry_turn = 0.0
            if incoming_dir is not None:
                entry_turn = float(
                    np.arccos(
                        np.clip(np.dot(incoming_dir, dir_origin / norm_orig), -1.0, 1.0)
                    )
                )
                leg_time = self.vehicle.transit_time(
                    norm_orig,
                    entry_turn,
                    t_since_launch,
                    float(dir_origin[2] / norm_orig),
                )

            if leg_time > time_budget:
                continue

            cos_a = np.clip(np.dot(u, dir_origin) / norm_orig, -1.0, 1.0)
            init_angle = np.degrees(np.arccos(cos_a))

            # Base Cost：只考慮離主軸距離與發射角
            dp[i][1] = (
                dist_weight * sorted_balloons[i]["d"]
                + angle_weight * (init_angle + np.degrees(entry_turn))
                + self.time_weight * leg_time
            )
            parent[i][1] = -1
            elapsed[i][1] = leg_time

        # DP State Transition: k = 2 -> K
        for k in range(2, K + 1):
            for i in range(N):
                for j in range(i):  # j < i 確保單向推進 (s_j < s_i)
                    if dp[j][k - 1] == float("inf"):
                        continue

                    # 兩端都取抵達時刻的預測位置：離開 j 的時間是 elapsed[j][k-1]，
                    # 抵達 i 的時間再由這一段本身決定（迭代一次收斂）。
                    depart_time = elapsed[j][k - 1]
                    pos_j = self._position_at(sorted_balloons[j], depart_time)

                    # 1) 計算兩顆氣球之間 (P_j -> P_i) 的距離。近目標本身
                    # 不是錯誤；只有需要急轉時，才由轉角與轉向時間付出成本。
                    v2, segment_dist, _ = self._predict_leg(
                        sorted_balloons[i], pos_j, depart_time, t_since_launch, 0.0
                    )
                    if segment_dist < 1e-5:
                        continue

                    too_far_penalty = 0.0
                    if segment_dist > max_dist:
                        too_far_penalty = (
                            segment_dist - max_dist
                        ) ** 2 * too_far_weight

                    # 2) 計算兩氣球之間的轉向折角 (P_prev -> P_j 與 P_j -> P_i)
                    #    前一顆也要取它自己被抵達那一刻的位置。
                    prev_idx = parent[j][k - 1]
                    prev_pos = (
                        chain_start
                        if prev_idx == -1
                        else self._position_at(
                            sorted_balloons[prev_idx], elapsed[prev_idx][k - 2]
                        )
                    )

                    v1 = pos_j - prev_pos

                    n1, n2 = np.linalg.norm(v1), segment_dist
                    turn_angle = (
                        np.degrees(
                            np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
                        )
                        if (n1 > 1e-5 and n2 > 1e-5)
                        else 0.0
                    )

                    # 3) 可達性：這一段飛完會不會超出燃燒時間
                    leg_time = self.vehicle.transit_time(
                        float(segment_dist),
                        np.radians(turn_angle),
                        elapsed[j][k - 1] + t_since_launch,
                        float(v2[2] / max(segment_dist, 1e-9)),
                    )
                    leg_elapsed = elapsed[j][k - 1] + leg_time
                    if leg_elapsed > time_budget:
                        continue

                    # 總代價 = 前一步 Cost + 離軸懲罰 + 轉向角懲罰 + 過遠懲罰 + 時間
                    cost = (
                        dp[j][k - 1]
                        + dist_weight * sorted_balloons[i]["d"]
                        + angle_weight * turn_angle
                        + too_far_penalty
                        + self.time_weight * leg_time
                    )

                    if cost < dp[i][k]:
                        dp[i][k] = cost
                        parent[i][k] = j
                        elapsed[i][k] = leg_elapsed

        # --- 6. 回溯找出最佳序列 ---
        # K 現在是「上限」而非硬性數量：時間預算可能撐不到 K 顆，這時回傳飛得完
        # 的最長鏈，比回傳 None 或一條飛不完的鏈都有用。
        best_k = 0
        for k in range(K, 0, -1):
            if np.isfinite(dp[:, k]).any():
                best_k = k
                break

        if best_k == 0:
            return None

        curr = int(np.argmin(dp[:, best_k]))
        target_ids = []
        leg_finish_times = []
        for k in range(best_k, 0, -1):
            target_ids.append(sorted_balloons[curr]["id"])
            leg_finish_times.append(elapsed[curr][k])
            curr = parent[curr][k]

        target_ids.reverse()
        leg_finish_times.reverse()

        legs = np.diff([0.0] + leg_finish_times)
        self.logger.info(
            f"Chain of {best_k}/{K} targets within a {time_budget:.1f} s budget: "
            f"{target_ids} (legs {np.round(legs, 1).tolist()} s, "
            f"total {leg_finish_times[-1]:.1f} s)"
        )
        return target_ids

    def check_target_popped(self, target_idx: int, observation: dict) -> bool:
        balloon_status = np.array(
            observation[Schema.Observation.BALLOON_STATUS], dtype=int
        ).flatten()
        target_status = balloon_status[target_idx]
        return target_status == 2

    def get_target_state(self, target_idx: int, observation: dict) -> np.ndarray:
        balloon_states = observation[Schema.Observation.BALLOON_STATES]
        target_state = balloon_states[target_idx]
        return target_state
