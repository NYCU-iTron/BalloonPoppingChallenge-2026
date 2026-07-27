import logging
import numpy as np


class Navigator:
    """Analytic guidance baseline: proportional navigation + range control.

    Returns the full desired world-frame acceleration. The controller derives
    both of its outputs from it -- attitude from the direction of (a_cmd + g),
    throttle from its magnitude -- so every component of a_cmd is meaningful
    and the thrust magnitude can never contradict the commanded tilt.

    The command splits along the line of sight:
      perpendicular  true PN, drives the LOS rate to zero (the intercept law)
      along the LOS  closing-speed control, which sets how fast the range is
                     closed and so doubles as altitude management. Without it a
                     command that merely cancels gravity would hover, and full
                     thrust would overshoot the balloon column by ~1 km.
    """

    NAV_CONSTANT = 4.0          # classic PN gain, 3-5
    PN_MIN_CLOSING = 5.0        # (m/s) floor on the PN gain term so steering
                                # survives a failed pass instead of vanishing
    APPROACH_GAIN = 0.3         # (1/s) desired closing speed per metre of range
    MAX_CLOSING_SPEED = 30.0    # (m/s) turn radius is v^2/a; with ~7 m/s2 of
                                # lateral authority 30 m/s already costs ~130 m
    SPEED_GAIN = 0.5            # (1/s) P gain on the closing-speed error
    MAX_LATERAL_ACCEL = 8.0     # (m/s2) bounds PN as omega blows up at short range
    MAX_BRAKE_ACCEL = 9.81      # (m/s2) engine off: gravity is the only brake
    MAX_THRUST_ACCEL = 11.8     # (m/s2) T/m at liftoff (worst case); the
                                # achievable set is a ball of this radius on -g
    GRAVITY = np.array([0.0, 0.0, -9.81])

    def __init__(self, given_parameters):
        self.logger = logging.getLogger(__name__)
        self.given_parameters = given_parameters

    def reset(self):
        pass

    def compute(self, rocket_state: np.ndarray, target_state: np.ndarray) -> np.ndarray:
        if np.isnan(target_state).any():
            return np.full(3, np.nan)

        target_pos = target_state[0:3]
        target_vel = target_state[3:6]

        rocket_pos = rocket_state[0:3]
        rocket_vel = rocket_state[3:6]

        # --- Line of sight ------------------------------------------------ #
        los = target_pos - rocket_pos
        distance = float(np.linalg.norm(los))
        if distance < 1e-3:
            return np.zeros(3)
        los_hat = los / distance

        v_rel = target_vel - rocket_vel
        v_closing = -float(np.dot(v_rel, los_hat))      # > 0 while approaching

        # --- Perpendicular: true proportional navigation ------------------ #
        # omega = (r x v_rel) / (r . r);  a = N * Vc * (omega x los_hat)
        omega_los = np.cross(los, v_rel) / np.dot(los, los)
        a_lateral = (self.NAV_CONSTANT
                     * max(v_closing, self.PN_MIN_CLOSING)
                     * np.cross(omega_los, los_hat))
        a_lateral = self._bound(a_lateral, self.MAX_LATERAL_ACCEL)

        # --- Along the LOS: closing-speed control ------------------------- #
        # Slow down as the range shrinks so the terminal turn stays flyable,
        # and push back toward the target whenever the range is opening.
        # Braking is limited to gravity once the engine is cut, so never ask
        # for a speed the remaining range cannot bleed off.
        desired_closing = min(self.APPROACH_GAIN * distance,
                              self.MAX_CLOSING_SPEED,
                              float(np.sqrt(2.0 * self.MAX_BRAKE_ACCEL * distance)))
        a_along = self.SPEED_GAIN * (desired_closing - v_closing) * los_hat

        return self._realizable(a_lateral + a_along)

    def _realizable(self, a_cmd: np.ndarray) -> np.ndarray:
        """Project the request onto what thrust can actually deliver."""
        a_thrust = a_cmd - self.GRAVITY

        # Thrust may not point below the horizon: a hard deceleration is flown
        # by cutting the engine and letting gravity brake, not by flipping the
        # vehicle over to thrust downwards.
        a_thrust[2] = max(a_thrust[2], 0.0)

        # Scaling the sum preserves the commanded direction -- what the attitude
        # loop tracks -- and gives up only magnitude, exactly as throttle
        # saturation would.
        a_thrust = self._bound(a_thrust, self.MAX_THRUST_ACCEL)

        return a_thrust + self.GRAVITY

    @staticmethod
    def _bound(vec: np.ndarray, limit: float) -> np.ndarray:
        norm = float(np.linalg.norm(vec))
        return vec * (limit / norm) if norm > limit else vec
