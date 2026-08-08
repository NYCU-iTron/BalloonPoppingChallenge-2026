import logging
import numpy as np

from BalloonPoppingGymEnv.agents.gnc.vehicle import Vehicle, GRAVITY
from BalloonPoppingGymEnv.utils.schema import Schema


class Navigator:
    """Guidance law: owns the entire commanded acceleration vector.

    Zero-effort-miss terminal guidance produces the acceleration the vehicle
    should fly; gravity cancellation and saturation against the vehicle's real
    acceleration envelope happen here too, so the command handed down is always
    something the rocket can actually deliver. The output is split into a thrust
    direction and a throttle that come from the same vector -- the autopilot
    below only has to point the rocket, it never re-decides the magnitude.

    This replaces proportional navigation, whose assumptions (high speed, thrust
    to weight far above one, negligible gravity, short engagement) are all false
    for this vehicle, and whose closing-velocity gate left the command at exactly
    zero for the first nine seconds of flight.
    """

    def __init__(self, given_parameters):
        self.logger = logging.getLogger(__name__)
        self.given_parameters = given_parameters
        self.vehicle = Vehicle(given_parameters)

        elevation = given_parameters[Schema.Given.Section.ENVIRONMENT][
            Schema.Given.Environment.ELEVATION
        ]
        self.pad_pos = np.array([0.0, 0.0, float(elevation)])

        # --- Tunable guidance parameters ----------------------------------
        self.nav_constant = 3.0        # N in a = N * ZEM / t_go^2 (3 = energy optimal)
        self.t_go_min = 0.3            # (s) floor, keeps a = N*ZEM/t_go^2 finite
        self.t_go_margin = 1.2         # allow t_go a little past burnout before giving up
        self.min_closing_accel = 0.5   # (m/s^2) floor used when estimating t_go
        self.launch_climb_accel = 1.0  # (m/s^2) vertical margin required off the pad

        # Thrust may not lean further than this from vertical. Past it the
        # vertical component stops holding the vehicle up, and beyond 90 degrees
        # thrust adds to gravity instead of opposing it -- which is how the
        # rocket previously ended up inverted at 164 degrees and fell out of
        # the sky while trying to turn back towards an overshot target.
        self.max_tilt = 60.0           # (deg) from vertical

        # Approach speed is capped at range / terminal_control_time, so the
        # engagement always leaves at least this long to null the remaining
        # miss. Arriving faster than the attitude loop can steer is what turned
        # a 5 m first-pass miss into a 176 m overshoot.
        self.terminal_control_time = 1.2  # (s)
        self.brake_time_constant = 0.8    # (s) how hard excess closing speed is bled off

        # Braking only earns its keep when the approach is still going to miss.
        # Shedding speed on a pass that is already inside the balloon throws away
        # the momentum the next leg needs, and with a fixed 30 s burn that speed
        # is the whole budget. So slow down for a bad approach, not for a close
        # one.
        self.miss_tolerance = float(
            given_parameters[Schema.Given.Section.BALLOON][Schema.Given.Balloon.RADIUS]
        )

        # A cruise ceiling on top of that. The range-proportional cap alone only
        # bites in the last second, by which point shedding the speed would need
        # far more deceleration than the vehicle owns; holding the run-in near
        # this speed instead keeps the braking gentle and always affordable.
        # Shared with the vehicle's transit model so target selection budgets
        # against the speed guidance will actually fly.
        self.max_closing_speed = self.vehicle.cruise_speed

        # --- Diagnostics (read by tests and logging, never by the control path) ---
        self.t_go = None
        self.zem_miss = None
        self.saturated = False
        self.braking = False
        self.tilt_limited = False

    def reset(self):
        self.t_go = None
        self.zem_miss = None
        self.saturated = False
        self.braking = False
        self.tilt_limited = False

    # ------------------------------------------------------------------ #
    # Launch
    # ------------------------------------------------------------------ #

    def get_launch_attitude(self, target_state: np.ndarray) -> np.ndarray:
        """Aim the rail at where the first target will be, not at where it is.

        With a thrust-to-weight ratio near 1.2 the rocket cannot turn hard once
        it is fast, so the launch attitude is the single most powerful input
        this agent has. Pointing it straight up -- as the previous fixed
        inclination of 90 degrees did -- throws that away.

        Parameters
        ----------
        target_state : np.ndarray
            Shape (6,): first target's [pos(3), vel(3)] in world coordinates.

        Returns
        -------
        np.ndarray
            [inclination, heading] in degrees, in the simulator's convention:
            inclination measured from horizontal (90 = vertical) and heading as
            a compass bearing (0 = North, 90 = East, clockwise).
        """
        target_state = np.asarray(target_state, dtype=float).reshape(-1)
        target_pos = target_state[0:3]
        target_vel = target_state[3:6] if target_state.size >= 6 else np.zeros(3)

        # Lead the target: solve for time-of-flight, re-aim, repeat. Two passes
        # are plenty -- the aim point moves only metres on the second one.
        aim_point = target_pos
        for _ in range(2):
            t_go = self._estimate_t_go(
                r_rel=aim_point - self.pad_pos,
                v_rel=target_vel,
                t_since_launch=0.0,
            )
            aim_point = target_pos + target_vel * t_go

        aim = aim_point - self.pad_pos
        horizontal = float(np.hypot(aim[0], aim[1]))

        inclination = float(np.degrees(np.arctan2(aim[2], max(horizontal, 1e-6))))
        # East component first: the simulator's heading is a compass bearing.
        heading = float(np.degrees(np.arctan2(aim[0], aim[1])) % 360.0)

        # A shallow rail is useless if the vertical thrust component cannot beat
        # gravity -- the rocket would simply sink off the pad.
        min_inclination = self.vehicle.min_climb_inclination(0.0, self.launch_climb_accel)
        clamped = float(np.clip(inclination, min_inclination, 90.0))

        self.logger.info(
            f"Launch attitude: aim inclination={inclination:.1f} deg -> {clamped:.1f} deg "
            f"(min climbable {min_inclination:.1f}), heading={heading:.1f} deg, "
            f"lead t_go={t_go:.1f} s"
        )
        return np.array([clamped, heading])

    # ------------------------------------------------------------------ #
    # In-flight guidance
    # ------------------------------------------------------------------ #

    def compute(
        self,
        target_state: np.ndarray | None,
        rocket_state: np.ndarray,
        t_since_launch: float,
    ) -> tuple[None, None] | tuple[np.ndarray, float]:
        """
        Compute the thrust direction and throttle the vehicle should fly.

        Parameters
        ----------
        target_state : np.ndarray | None
            Current target state [pos(3), vel(3)], not lead-extrapolated --
            this law does its own lead over its time-to-go.
        rocket_state : np.ndarray
            Estimated state [pos(3), vel(3), acc(3), quat(4), gyro(3)].
        t_since_launch : float
            Seconds since liftoff, used for the mass and thrust model.

        Returns
        -------
        thrust_dir_world : np.ndarray | None
            Unit vector the thrust axis should point along, world frame.
            None when there is no valid target or no thrust left to steer with.
        throttle : float | None
            Throttle realising the commanded acceleration magnitude.
        """
        self.saturated = False

        if target_state is None or np.isnan(target_state).any():
            self.t_go = None
            self.zem_miss = None
            return None, None

        # Without thrust the TVC produces no moment, so there is nothing to command.
        if not self.vehicle.is_powered(t_since_launch):
            self.t_go = None
            self.zem_miss = None
            return None, None

        rocket_pos = np.asarray(rocket_state, dtype=float)[0:3]
        rocket_vel = np.asarray(rocket_state, dtype=float)[3:6]

        target_state = np.asarray(target_state, dtype=float).reshape(-1)
        target_pos = target_state[0:3]
        target_vel = target_state[3:6] if target_state.size >= 6 else np.zeros(3)

        r_rel = target_pos - rocket_pos
        v_rel = target_vel - rocket_vel

        # --- Zero-effort miss ---------------------------------------------
        # How far the target would be missed by if the vehicle stopped steering
        # now. Unlike proportional navigation this is well defined from rest,
        # and it drives position rather than line-of-sight rate, so the rocket
        # cannot quietly fly past the target while the command reads zero.
        t_go = self._estimate_t_go(r_rel, v_rel, t_since_launch)
        zem = r_rel + v_rel * t_go

        self.t_go = t_go
        self.zem_miss = float(np.linalg.norm(zem))

        a_des = self.nav_constant * zem / (t_go**2)

        # --- Terminal speed management -------------------------------------
        a_des = a_des + self._approach_brake(r_rel, v_rel)

        # --- Gravity cancellation ------------------------------------------
        # Thrust has to supply the manoeuvre *and* hold the vehicle up. Doing
        # this here rather than in the autopilot keeps one owner for the whole
        # acceleration vector.
        gravity_cancel = np.array([0.0, 0.0, GRAVITY])
        a_thrust = self._saturate(gravity_cancel, a_des, t_since_launch)
        a_thrust = self._limit_tilt(a_thrust)

        thrust_norm = float(np.linalg.norm(a_thrust))
        if thrust_norm < 1e-9:
            return None, None

        thrust_dir = a_thrust / thrust_norm

        # --- Throttle: the magnitude of the same vector ---------------------
        # Never a separate heuristic. Throttling also saves no propellant here,
        # so the only reason to sit below full thrust is not needing the
        # acceleration.
        a_max = self.vehicle.max_accel(t_since_launch)
        throttle = float(
            np.clip(
                thrust_norm / max(a_max, 1e-9),
                self.vehicle.throttle_min,
                self.vehicle.throttle_max,
            )
        )

        return thrust_dir, throttle

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _estimate_t_go(
        self, r_rel: np.ndarray, v_rel: np.ndarray, t_since_launch: float
    ) -> float:
        """Time to close the range, from the vehicle's own acceleration budget.

        Solves ``R = Vc*t + 0.5*a*t^2`` rather than dividing range by closing
        speed, so it stays finite from a standing start -- the case that made
        the old ``dist / max(v_closing, 1.0)`` estimate meaningless off the pad.
        """
        distance = float(np.linalg.norm(r_rel))
        if distance < 1e-6:
            return self.t_go_min

        los_hat = r_rel / distance
        closing_speed = -float(np.dot(v_rel, los_hat))

        # Acceleration left over once gravity is held off, i.e. what can
        # actually be spent on closing the range.
        a_max = self.vehicle.max_accel(t_since_launch)
        closing_accel = max(a_max - GRAVITY, self.min_closing_accel)

        t_go = (
            -closing_speed + np.sqrt(closing_speed**2 + 2.0 * closing_accel * distance)
        ) / closing_accel

        # Nothing can be flown under power past burnout, so cap the horizon there.
        t_max = self.vehicle.burn_time_remaining(t_since_launch) * self.t_go_margin
        return float(np.clip(t_go, self.t_go_min, max(t_max, self.t_go_min)))

    def _approach_brake(self, r_rel: np.ndarray, v_rel: np.ndarray) -> np.ndarray:
        """Bleed off closing speed the endgame would not have time to steer out.

        Terminal accuracy is set by how long the vehicle still has to null the
        remaining miss, so the approach speed is capped at
        ``range / terminal_control_time``. Far out the cap is loose and costs
        nothing; close in it keeps the last seconds usable instead of flashing
        past the balloon at full speed.

        Returns the along-line-of-sight deceleration to add to the command
        (zero when the approach is already slow enough).
        """
        self.braking = False

        # Already on track to pass inside the balloon: keep the speed.
        if self.zem_miss is not None and self.zem_miss <= self.miss_tolerance:
            return np.zeros(3)

        distance = float(np.linalg.norm(r_rel))
        if distance < 1e-6:
            return np.zeros(3)

        los_hat = r_rel / distance
        closing_speed = -float(np.dot(v_rel, los_hat))

        max_closing_speed = min(
            distance / self.terminal_control_time, self.max_closing_speed
        )
        excess = closing_speed - max_closing_speed
        if excess <= 0.0:
            return np.zeros(3)

        self.braking = True
        return -(excess / self.brake_time_constant) * los_hat

    def _limit_tilt(self, a_thrust: np.ndarray) -> np.ndarray:
        """Keep the thrust direction within ``max_tilt`` of vertical.

        Rotates the command back towards vertical while preserving its
        magnitude, so the vehicle keeps the vertical authority it needs to stay
        flying. Guidance asking to lean past this is asking for a manoeuvre the
        vehicle cannot survive, not one it cannot perform.
        """
        self.tilt_limited = False

        magnitude = float(np.linalg.norm(a_thrust))
        if magnitude < 1e-9:
            return a_thrust

        tilt = np.arccos(np.clip(a_thrust[2] / magnitude, -1.0, 1.0))
        max_tilt = np.radians(self.max_tilt)
        if tilt <= max_tilt:
            return a_thrust

        self.tilt_limited = True

        horizontal = a_thrust[0:2]
        horizontal_norm = float(np.linalg.norm(horizontal))
        if horizontal_norm < 1e-9:
            # Pointing straight down with no horizontal preference: go straight up.
            return np.array([0.0, 0.0, magnitude])

        horizontal_dir = horizontal / horizontal_norm
        limited_horizontal = horizontal_dir * magnitude * np.sin(max_tilt)
        return np.array([
            limited_horizontal[0],
            limited_horizontal[1],
            magnitude * np.cos(max_tilt),
        ])

    def _saturate(
        self, gravity_cancel: np.ndarray, a_des: np.ndarray, t_since_launch: float
    ) -> np.ndarray:
        """Fit ``gravity_cancel + a_des`` inside the achievable acceleration ball.

        Scaling the whole vector would eat into holding the vehicle up, which for
        a thrust-to-weight ratio near 1.2 means falling out of the sky. So the
        gravity term is kept whole and the manoeuvre term is trimmed instead:
        find the largest ``k`` in [0, 1] with ``|gravity_cancel + k*a_des| <= a_max``.
        """
        a_max = self.vehicle.max_accel(t_since_launch)
        total = gravity_cancel + a_des

        if float(np.linalg.norm(total)) <= a_max:
            return total

        self.saturated = True

        # Not even enough thrust to hover: point everything straight up.
        g_norm = float(np.linalg.norm(gravity_cancel))
        if a_max <= g_norm:
            return gravity_cancel * (a_max / max(g_norm, 1e-9))

        # |g + k*a|^2 = a_max^2  ->  k^2|a|^2 + 2k(g.a) + |g|^2 - a_max^2 = 0
        aa = float(np.dot(a_des, a_des))
        if aa < 1e-12:
            return gravity_cancel

        ga = float(np.dot(gravity_cancel, a_des))
        discriminant = ga**2 - aa * (g_norm**2 - a_max**2)
        k = (-ga + np.sqrt(max(discriminant, 0.0))) / aa

        return gravity_cancel + np.clip(k, 0.0, 1.0) * a_des
