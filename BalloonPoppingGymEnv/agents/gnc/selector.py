import logging
import numpy as np
from BalloonPoppingGymEnv.utils.schema import Schema


class Selector:
    # Waiting on the pad is free -- propellant only burns after ignition, and
    # the flight is far shorter than the episode -- so the gate is chosen for
    # geometry. It sets the altitude band the vehicle fights in, and lateral
    # authority there is what limits the score: ~3 m/s^2 with a 60 m column,
    # ~6.4 m/s^2 with a 120 m one. A longer column also tolerates a faster
    # climb without flying out of the top. Balloons rise ~6 m/s, so 120 m fires
    # near 20 s and still leaves every release inside the burn.
    LAUNCH_GATE_AGL = 120.0    # (m)

    # Minimum altitude for an opening target. Lower balloons are unreachable at
    # T/W ~1.2, but with the gate above there are now many candidates rather
    # than only the topmost.
    ENGAGE_FLOOR_AGL = 60.0    # (m)

    # Pad tilt toward the stream must stay small; larger tilts lawn-dart.
    # (Measured closest pass: 0 deg 13 m, 5 deg 5 m, 10 deg+ diverges.)
    MAX_LAUNCH_TILT = 5.0      # (deg)

    # Lateral authority actually available, scheduled on altitude because the
    # controller's tilt limiter is what caps it. Measured: ~2.4 m/s^2 at 50 m
    # AGL, ~6 m/s^2 at 100 m, ~7 m/s^2 once the limiter is fully open. A fixed
    # budget overestimates by 4x down low and rules targets reachable that the
    # vehicle cannot in fact fly to.
    LATERAL_BUDGET_LOW = 2.5   # (m/s^2) at/below BUDGET_LOW_ALT
    LATERAL_BUDGET_HIGH = 8.0  # (m/s^2) at/above BUDGET_HIGH_ALT
    BUDGET_LOW_ALT = 50.0      # (m AGL)
    BUDGET_HIGH_ALT = 150.0    # (m AGL)

    SWITCH_HYSTERESIS = 1.5    # keep the current target until this far over budget
    MIN_CHAIN_SPEED = 15.0     # (m/s) below this the course is still forming
    MAX_TIME_TO_GO = 30.0      # (s) slower intercepts outlast the burn
    SLEW_MARGIN = 0.7          # (s) attitude response before a command bites

    def __init__(self, given_parameters):
        self.logger = logging.getLogger(__name__)
        self.given_parameters = given_parameters

        env_cfg = given_parameters[Schema.Given.Section.ENVIRONMENT]
        self.ground_elevation = float(env_cfg[Schema.Given.Environment.ELEVATION])
        balloon_cfg = given_parameters[Schema.Given.Section.BALLOON]
        self.balloon_radius = float(balloon_cfg[Schema.Given.Balloon.RADIUS])
        # Waiting past the final release only shrinks the field, so that time
        # backstops the altitude gate if the balloons climb slower than assumed.
        self.last_release_time = (float(balloon_cfg[Schema.Given.Balloon.NUM]) - 1.0) \
            * float(balloon_cfg[Schema.Given.Balloon.RELEASE_INTERVAL])

        self.current_target_idx = None

    def reset(self):
        # Must clear: the instance is reused across episodes, and hysteresis
        # would otherwise hold last episode's target through the new launch.
        self.current_target_idx = None

    def should_launch(self, observation: dict) -> bool:
        raw_balloons = np.asarray(observation[Schema.Observation.BALLOON_STATES], dtype=float)
        if raw_balloons.size == 0:
            return True

        status = np.asarray(observation[Schema.Observation.BALLOON_STATUS], dtype=int).reshape(-1)
        airborne = status == 1
        if not airborne.any():
            return False

        # Static or descending targets will not come to us -- engage now.
        if float(np.max(raw_balloons[airborne, 5])) <= 0.1:
            return True

        if float(observation[Schema.Observation.SIMULATION_TIME]) >= self.last_release_time:
            return True

        highest_agl = float(np.max(raw_balloons[airborne, 2])) - self.ground_elevation
        return highest_agl >= self.LAUNCH_GATE_AGL

    def get_launch_heading(self, observation: dict) -> np.ndarray:
        """[inclination, heading] in degrees, tipping the rail toward the
        opening target (vertical fallback)."""
        vertical = np.array([90.0, 0.0])

        states = np.asarray(observation[Schema.Observation.BALLOON_STATES], dtype=float)
        if states.size == 0:
            return vertical
        status = np.asarray(observation[Schema.Observation.BALLOON_STATUS], dtype=int).reshape(-1)
        airborne = status == 1
        if not airborne.any():
            return vertical

        # Aim where select_target will aim: engageable balloon with the least
        # horizontal offset, or the centroid while none has cleared the floor.
        agl = states[:, 2] - self.ground_elevation
        climbing = float(np.max(states[airborne, 5])) > 0.1
        engageable = airborne & (agl >= self.ENGAGE_FLOOR_AGL)
        if climbing and engageable.any():
            horizontal_distance = np.hypot(states[:, 0], states[:, 1])
            horizontal_distance[~engageable] = np.inf
            aim_point = states[int(np.argmin(horizontal_distance)), 0:3]
        else:
            aim_point = states[airborne, 0:3].mean(axis=0)

        east, north = float(aim_point[0]), float(aim_point[1])
        horizontal = float(np.hypot(east, north))
        if horizontal < 10.0:
            return vertical

        heading = float(np.degrees(np.arctan2(east, north))) % 360.0
        altitude_agl = max(float(aim_point[2]) - self.ground_elevation, 1.0)
        tilt = min(float(np.degrees(np.arctan2(horizontal, altitude_agl))),
                   self.MAX_LAUNCH_TILT)

        return np.array([90.0 - tilt, heading])

    def select_target(self, rocket_state: np.ndarray, balloon_states: np.ndarray) -> int | None:
        """
        Reachability-aware target selection with forward chaining.

        Candidates are scored by the lateral acceleration an intercept from the
        current course would demand (a_req = 2*|ZEM| / t_go^2); the soonest
        affordable one wins, chaining along the stream after each pop.

        Parameters
        ----------
        rocket_state : np.ndarray
            Estimated state [pos(3), vel(3), acc(3), quat(4), gyro(3)].
        balloon_states : np.ndarray
            Shape (N, 6) predicted states; NaN position marks inactive balloons.

        Returns
        -------
        int or None
            Index of the selected balloon, or None if no active targets remain.
        """
        rocket_pos = rocket_state[0:3]
        rocket_vel = rocket_state[3:6]

        valid = ~np.isnan(balloon_states[:, 0:3]).any(axis=1)
        if not valid.any():
            self.current_target_idx = None
            return None

        distances = np.linalg.norm(balloon_states[:, 0:3] - rocket_pos, axis=1)
        distances[~valid] = np.inf

        # Course not established yet. On the pad, rank engageable balloons by
        # horizontal offset rather than range: closing the vertical gap is just
        # a matter of climbing, whereas the lateral scatter is what actually
        # goes unengaged. This also matches where get_launch_heading tips the
        # rail. Once moving, take the plain nearest so the tracked target is
        # what the rocket actually passes.
        speed = float(np.linalg.norm(rocket_vel))
        if speed < self.MIN_CHAIN_SPEED:
            on_pad = speed < 3.0
            climbing = float(np.max(balloon_states[valid, 5])) > 0.1
            agl = balloon_states[:, 2] - self.ground_elevation
            engageable = valid & (agl >= self.ENGAGE_FLOOR_AGL)
            if on_pad and climbing and engageable.any():
                offsets = np.hypot(balloon_states[:, 0] - rocket_pos[0],
                                   balloon_states[:, 1] - rocket_pos[1])
                self.current_target_idx = int(np.argmin(np.where(engageable, offsets, np.inf)))
            else:
                self.current_target_idx = int(np.argmin(distances))
            return self.current_target_idx

        blend = np.clip((rocket_pos[2] - self.ground_elevation - self.BUDGET_LOW_ALT)
                        / (self.BUDGET_HIGH_ALT - self.BUDGET_LOW_ALT), 0.0, 1.0)
        budget = (self.LATERAL_BUDGET_LOW
                  + blend * (self.LATERAL_BUDGET_HIGH - self.LATERAL_BUDGET_LOW))

        num = len(balloon_states)
        a_req = np.full(num, np.inf)
        t_go_all = np.full(num, np.inf)
        for i in np.flatnonzero(valid):
            rel = balloon_states[i, 0:3] - rocket_pos
            dist = float(np.linalg.norm(rel))
            if dist < 1e-6:
                a_req[i] = 0.0
                t_go_all[i] = 0.0
                continue
            v_rel = rocket_vel - balloon_states[i, 3:6]
            v_close = float(np.dot(v_rel, rel)) / dist
            if v_close <= 0.5:
                continue  # receding or barely closing
            t_go = dist / v_close
            if t_go > self.MAX_TIME_TO_GO:
                continue
            zem = rel - v_rel * t_go
            # Only the miss beyond the balloon radius needs steering out, and
            # only the time left after the attitude has swung round can do it.
            miss = max(float(np.linalg.norm(zem)) - self.balloon_radius, 0.0)
            t_eff = max(t_go - self.SLEW_MARGIN, 0.3)
            a_req[i] = 2.0 * miss / (t_eff * t_eff)
            t_go_all[i] = t_go

        # Hold the current target while it stays affordable (hysteresis).
        cur = self.current_target_idx
        cur_alive = cur is not None and cur < num and valid[cur]
        if cur_alive and a_req[cur] <= budget * self.SWITCH_HYSTERESIS:
            return cur

        feasible = a_req <= budget
        if feasible.any():
            self.current_target_idx = int(np.argmin(np.where(feasible, t_go_all, np.inf)))
        elif cur_alive:
            # Nothing affordable: keep it; the miss detector ends the episode.
            return cur
        elif np.isfinite(a_req).any():
            self.current_target_idx = int(np.argmin(a_req))
        else:
            self.current_target_idx = int(np.argmin(distances))

        return self.current_target_idx

    def get_target_state(self, balloon_states: np.ndarray, target_idx: int) -> np.ndarray:
        if target_idx is None:
            return np.full(6, np.nan)
        return balloon_states[target_idx]
