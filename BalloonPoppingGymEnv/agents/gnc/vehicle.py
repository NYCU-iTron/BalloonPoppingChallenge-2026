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

        # ------------------------------- Actuators ------------------------------- #
        self.max_gimbal = float(control[Schema.Given.Control.GIMBAL_RANGE])
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

    def pitch_accel_per_gimbal_deg(self, t: float) -> float:
        """Angular acceleration (rad/s^2) produced by one degree of gimbal.

        This is the plant gain the rate loop closes around; it drops to zero at
        burnout, which is exactly when the autopilot must stop trying.
        """
        if not self.is_powered(t):
            return 0.0
        moment_per_deg = self.max_thrust * np.sin(np.radians(1.0)) * self.tvc_lever
        return float(moment_per_deg / self.transverse_inertia)
