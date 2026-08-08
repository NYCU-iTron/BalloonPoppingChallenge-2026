import logging
import numpy as np
from BalloonPoppingGymEnv.utils.schema import Schema

GRAVITY = 9.81
GRAVITY_WORLD = np.array([0.0, 0.0, -GRAVITY])


class Vehicle:
    """Mass, thrust and control-authority model of the rocket.

    Guidance reads this to know what acceleration it is allowed to ask for, and
    the autopilot reads it to size its gains. Both share one model so they can
    never disagree about what the vehicle is physically able to do -- the split
    that let the old navigator command 30 m/s^2 of lateral acceleration from a
    vehicle with a thrust-to-weight ratio of 1.2.
    """

    def __init__(self, given_parameters):
        self.logger = logging.getLogger(__name__)

        rocket = given_parameters[Schema.Given.Section.ROCKET]
        motor = rocket[Schema.Given.Rocket.MOTOR]
        tank = rocket[Schema.Given.Rocket.TANK]
        body = rocket[Schema.Given.Rocket.ROCKET_BODY]
        control = rocket[Schema.Given.Rocket.CONTROL]

        # ------------------------------ Propulsion ------------------------------ #
        self.max_thrust = float(motor[Schema.Given.Motor.THRUST_SOURCE])
        self.burn_time = float(motor[Schema.Given.Motor.BURN_TIME])

        grain_volume = (
            np.pi
            * (
                motor[Schema.Given.Motor.GRAIN_OUTER_RADIUS] ** 2
                - motor[Schema.Given.Motor.GRAIN_INITIAL_INNER_RADIUS] ** 2
            )
            * motor[Schema.Given.Motor.GRAIN_INITIAL_HEIGHT]
            * motor[Schema.Given.Motor.GRAIN_NUMBER]
        )
        grain_mass = grain_volume * motor[Schema.Given.Motor.GRAIN_DENSITY]

        self.initial_mass = float(
            body[Schema.Given.RocketBody.MASS]
            + motor[Schema.Given.Motor.DRY_MASS]
            + tank[Schema.Given.Tank.INITIAL_LIQUID_MASS]
            + tank[Schema.Given.Tank.INITIAL_GAS_MASS]
            + grain_mass
        )

        # The tank drains on a fixed schedule: throttle scales thrust only, never
        # propellant flow, so throttling down buys nothing and the mass profile is
        # purely a function of time since launch.
        self.mass_flow_rate = float(
            tank[Schema.Given.Tank.LIQUID_MASS_FLOW_RATE_OUT] + grain_mass / self.burn_time
        )

        # ---------------------------- Control authority ---------------------------- #
        # Lever arm from the nozzle to the dry centre of mass, matching how the
        # simulator builds the TVC moment (M = T * sin(gimbal) * lever).
        nozzle_position = float(
            motor[Schema.Given.Motor.MOTOR_POSITION] + motor[Schema.Given.Motor.NOZZLE_POSITION]
        )
        self.tvc_lever = abs(
            nozzle_position - float(body[Schema.Given.RocketBody.CENTER_OF_MASS_WITHOUT_MOTOR])
        )

        inertia = body[Schema.Given.RocketBody.INERTIA]
        self.transverse_inertia = float(inertia[0])
        self.roll_inertia = float(inertia[2])

        # --------------------------- Flight profile --------------------------- #
        # How the vehicle actually gets from one balloon to the next. Guidance
        # flies to this profile and target selection budgets against it, so both
        # agree on how long a chain of balloons will take.
        #
        # Calibrated against three measured legs of a scenario 1 flight:
        # pad -> T1 (202 m, no turn) took 14.7 s against 15.3 s predicted;
        # T1 -> T2 (33 m, hard turn) 8.7 s against 8.7 s; T2 -> T3 (36 m,
        # shallow turn) 5.4 s against 5.3 s.
        # Swept twice, at two different launch geometries, and both times the
        # score came out flat: over 24 seeds, 15 through 30 m/s all score 3.92 to
        # 4.00. Speed buys a longer planned chain (4.04 -> 4.42 targets) and
        # loses exactly as much to legs the rocket then cannot fly (completion
        # 99% -> 89%, and 0 versus 5 misses beyond 5 m). Taken at the low end,
        # where the same score comes with less than half the spread.
        self.cruise_speed = 15.0          # (m/s) ceiling on closing speed

        # Speed carried out of a balloon and into the next leg. Flying straight
        # through keeps it; a sharp corner scrubs it off and the rocket has to
        # build the whole leg's speed again from nothing, which is what makes a
        # leg cost 5.7 s instead of the 4.1 s that six targets need. Pricing it
        # this way makes chain straightness pay for itself, so the planner seeks
        # the collinear chains the vehicle can actually carry speed through.
        self.corner_speed_retention = 1.0  # scales cos(turn); 0 disables carry-over
        self.turn_time_per_radian = 1.0    # (s) left over for physically rotating
        self.terminal_time = 1.2           # (s) settling on each balloon
        self.min_transit_accel = 1.0       # (m/s^2) floor on usable closing accel

        # ------------------------------- Actuators ------------------------------- #
        self.max_gimbal = float(control[Schema.Given.Control.GIMBAL_RANGE])
        self.gimbal_rate_limit = float(control[Schema.Given.Control.GIMBAL_RATE_LIMIT])
        self.max_roll_torque = float(control[Schema.Given.Control.MAX_ROLL_TORQUE])
        throttle_range = control[Schema.Given.Control.THROTTLE_RANGE]
        self.throttle_min = float(throttle_range[0])
        self.throttle_max = float(throttle_range[1])

        self.logger.info(
            f"Vehicle: m0={self.initial_mass:.1f} kg, T={self.max_thrust:.0f} N, "
            f"T/W={self.max_accel(0.0) / GRAVITY:.2f}, burn={self.burn_time:.0f} s, "
            f"mdot={self.mass_flow_rate:.3f} kg/s, tvc_lever={self.tvc_lever:.2f} m"
        )

    # ------------------------------------------------------------------ #
    # All times below are seconds since launch.
    # ------------------------------------------------------------------ #

    def mass(self, t: float) -> float:
        """Vehicle mass, decreasing at a fixed rate until burnout."""
        return self.initial_mass - self.mass_flow_rate * min(max(t, 0.0), self.burn_time)

    def is_powered(self, t: float) -> bool:
        """True while the motor still burns. TVC produces no moment without thrust."""
        return 0.0 <= t < self.burn_time

    def burn_time_remaining(self, t: float) -> float:
        return max(self.burn_time - t, 0.0)

    def max_accel(self, t: float) -> float:
        """Largest acceleration the thrust can produce, in m/s^2."""
        if not self.is_powered(t):
            return 0.0
        return self.max_thrust / self.mass(t)

    def max_lateral_accel(self, t: float) -> float:
        """Lateral acceleration available while still holding altitude.

        Reference figure only -- guidance saturates against the full
        acceleration ball, which is a less conservative and more correct bound.
        """
        a = self.max_accel(t)
        if a <= GRAVITY:
            return 0.0
        return float(np.sqrt(a**2 - GRAVITY**2))

    def min_climb_inclination(self, t: float, required_climb_accel: float = 1.0) -> float:
        """Shallowest rail inclination (deg from horizontal) that still climbs.

        Below this the vertical component of thrust no longer beats gravity and
        the rocket sinks straight off the pad.
        """
        a = self.max_accel(t)
        if a <= GRAVITY:
            return 90.0
        sin_theta = (GRAVITY + required_climb_accel) / a
        if sin_theta >= 1.0:
            return 90.0
        return float(np.degrees(np.arcsin(sin_theta)))

    def transit_time(
        self,
        length: float,
        turn_angle: float,
        t_elapsed: float,
        climb_sin: float = 1.0,
    ) -> float:
        """Estimated time to fly one leg of a target chain.

        Accelerate at whatever gravity leaves along the leg, cap at the cruise
        speed, then pay for the turn into the leg and for settling onto the
        balloon. Deliberately coarse -- it exists so target selection can tell a
        chain the rocket can finish from one it cannot, not to predict a
        trajectory.

        How much gravity takes depends on how steeply the leg climbs: straight
        up it costs the full 9.81 m/s^2 and leaves barely 2, while a level leg
        costs nothing. Charging every leg the full climb, as this did before,
        priced level and shallow legs as if they were vertical and hid chains
        the vehicle can comfortably fly.

        Parameters
        ----------
        length : float
            Leg length in metres.
        turn_angle : float
            Heading change into this leg, in radians.
        t_elapsed : float
            Seconds since launch at the start of the leg, for the mass model.
        climb_sin : float
            Vertical component of the leg direction: 1 straight up, 0 level,
            -1 straight down.
        """
        max_accel = self.max_accel(t_elapsed)
        # Never claim more along-track acceleration than the motor alone could
        # give: on a descending leg the thrust axis cannot point far enough from
        # vertical to add to gravity, so the assist is capped rather than banked.
        closing_accel = float(
            np.clip(max_accel - GRAVITY * climb_sin, self.min_transit_accel, max(max_accel, self.min_transit_accel))
        )

        # Speed carried through the corner into this leg: full cruise straight
        # ahead, nothing at all round a right angle.
        entry_speed = float(
            np.clip(
                self.cruise_speed * self.corner_speed_retention * np.cos(turn_angle),
                0.0,
                self.cruise_speed,
            )
        )

        # Accelerate from there back up to cruise, then hold.
        time_to_cruise = (self.cruise_speed - entry_speed) / closing_accel
        distance_to_cruise = (
            entry_speed * time_to_cruise + 0.5 * closing_accel * time_to_cruise**2
        )

        if length <= distance_to_cruise:
            travel = (
                -entry_speed + np.sqrt(entry_speed**2 + 2.0 * closing_accel * length)
            ) / closing_accel
        else:
            travel = time_to_cruise + (length - distance_to_cruise) / self.cruise_speed

        return float(
            travel + self.turn_time_per_radian * abs(turn_angle) + self.terminal_time
        )

    def pitch_accel_per_gimbal_deg(self, t: float) -> float:
        """Angular acceleration (rad/s^2) produced by one degree of gimbal.

        This is the plant gain the rate loop closes around; it drops to zero at
        burnout, which is exactly when the autopilot must stop trying.
        """
        if not self.is_powered(t):
            return 0.0
        moment_per_deg = self.max_thrust * np.sin(np.radians(1.0)) * self.tvc_lever
        return float(moment_per_deg / self.transverse_inertia)

    def max_angular_accel(self, t: float) -> float:
        """Angular acceleration at full gimbal deflection, in rad/s^2."""
        return self.max_gimbal * self.pitch_accel_per_gimbal_deg(t)

    def gimbal_slew_time(self) -> float:
        """Time for the actuator to travel its whole range, in seconds.

        A rate loop cannot usefully be tuned faster than the actuator serving
        it; asking for that just parks the gimbal on its stop.
        """
        return self.max_gimbal / self.gimbal_rate_limit
