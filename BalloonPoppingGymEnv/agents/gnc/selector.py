import logging
import numpy as np
from BalloonPoppingGymEnv.utils.schema import Schema


class Selector:
    # Launch when the FIRST balloon climbs past this altitude: engaging the
    # fresh end of the stream minimizes wind displacement. (Measured closest
    # pass: gate 60 -> 4-17 m, gate 100 -> 19-24 m, gate 150 -> worse.)
    LAUNCH_GATE_AGL = 60.0  # (m)

    # On the pad, only balloons at/above this altitude qualify as the opening
    # target; freshly released ones at pad altitude are unreachable at T/W ~1.3.
    ENGAGE_FLOOR_AGL = 60.0  # (m)

    # ZEM targeting parameters (see select_target).
    MAX_LATERAL_ACCEL = 10.0  # (m/s^2) sustainable lateral authority
    SWITCH_HYSTERESIS = 1.5   # keep current target until this factor over budget
    MIN_CHAIN_SPEED = 25.0    # (m/s) below this the course is still forming
    MAX_TIME_TO_GO = 30.0     # (s) slower intercepts are irrelevant

    # Pad tilt toward the stream must stay small: larger tilts lawn-dart at
    # T/W ~1.25. (Measured closest pass: 0deg 13 m, 5deg 5 m, 10deg+ diverges.)
    MAX_LAUNCH_TILT = 5.0  # (deg)

    def __init__(self, given_parameters):
        self.logger = logging.getLogger(__name__)
        self.given_parameters = given_parameters

        env_cfg = given_parameters[Schema.Given.Section.ENVIRONMENT]
        self.ground_elevation = float(env_cfg[Schema.Given.Environment.ELEVATION])
        balloon_cfg = given_parameters[Schema.Given.Section.BALLOON]
        self.balloon_radius = float(balloon_cfg[Schema.Given.Balloon.RADIUS])

        self.current_target_idx = None

    def reset(self):
        pass

    def should_launch(self, observation: dict) -> bool:
        raw_balloons = np.asarray(observation["balloon_states"], dtype=float)
        if raw_balloons.size == 0 or len(raw_balloons) == 0:
            return True

        status = np.asarray(observation["balloon_status"], dtype=int).reshape(-1)
        airborne = status == 1
        if not airborne.any():
            return False

        # Static/descending targets cannot be waited out -- engage now.
        if float(np.max(raw_balloons[airborne, 5])) <= 0.1:
            return True

        highest_agl = float(np.max(raw_balloons[airborne, 2])) - self.ground_elevation
        return highest_agl >= self.LAUNCH_GATE_AGL

    def get_launch_heading(self, observation: dict) -> np.ndarray:
        """[inclination, heading] in degrees, tipping the rail toward the
        opening target (vertical fallback)."""
        vertical = np.array([90.0, 0.0])

        states = np.asarray(observation["balloon_states"], dtype=float)
        if states.size == 0:
            return vertical
        status = np.asarray(observation["balloon_status"], dtype=int).reshape(-1)
        airborne = status == 1
        if not airborne.any():
            return vertical

        # Aim where the opening target selection will aim: nearest airborne
        # balloon above the floor; centroid until one is that high.
        agl = states[:, 2] - self.ground_elevation
        climbing = float(np.max(states[airborne, 5])) > 0.1
        engageable = airborne & (agl >= self.ENGAGE_FLOOR_AGL)
        if climbing and engageable.any():
            horiz_dist = np.hypot(states[:, 0], states[:, 1])
            horiz_dist[~engageable] = np.inf
            aim_point = states[int(np.argmin(horiz_dist)), 0:3]
        else:
            aim_point = states[airborne, 0:3].mean(axis=0)

        east, north = float(aim_point[0]), float(aim_point[1])
        horizontal = float(np.hypot(east, north))
        if horizontal < 10.0:
            return vertical

        heading = float(np.degrees(np.arctan2(east, north))) % 360.0
        altitude_agl = max(float(aim_point[2]) - self.ground_elevation, 1.0)
        tilt = float(np.degrees(np.arctan2(horizontal, altitude_agl)))
        tilt = min(tilt, self.MAX_LAUNCH_TILT)

        return np.array([90.0 - tilt, heading])

    def select_target(self, balloon_states: np.ndarray, rocket_state: np.ndarray) -> int | None:
        """
        Reachability-aware target selection with forward chaining.

        Candidates are scored by the lateral acceleration an intercept from the
        current course would demand (a_req = 2*|ZEM| / t_go^2); the soonest
        affordable one wins, chaining along the stream after each pop.

        Parameters
        ----------
        balloon_states : np.ndarray
            Shape (N, 6) predicted states; NaN position marks inactive balloons.
        rocket_state : np.ndarray
            Estimated state [pos(3), vel(3), acc(3), quat(4), gyro(3)].

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

        def _nearest_valid():
            dists = np.linalg.norm(balloon_states[:, 0:3] - rocket_pos, axis=1)
            dists[~valid] = np.inf
            return int(np.argmin(dists))

        # Course not established yet: on the pad pick the nearest balloon above
        # the floor; once airborne (slow boost) track the plain nearest so the
        # tracked target is what the rocket actually passes.
        speed = float(np.linalg.norm(rocket_vel))
        if speed < self.MIN_CHAIN_SPEED:
            on_pad = speed < 3.0
            climbing = float(np.max(balloon_states[valid, 5])) > 0.1
            agl = balloon_states[:, 2] - self.ground_elevation
            engageable = valid & (agl >= self.ENGAGE_FLOOR_AGL)
            if on_pad and climbing and engageable.any():
                dists = np.linalg.norm(balloon_states[:, 0:3] - rocket_pos, axis=1)
                dists[~engageable] = np.inf
                self.current_target_idx = int(np.argmin(dists))
            else:
                self.current_target_idx = _nearest_valid()
            return self.current_target_idx

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
            # Only the miss beyond the balloon radius needs steering out.
            miss = max(float(np.linalg.norm(zem)) - self.balloon_radius, 0.0)
            a_req[i] = 2.0 * miss / (t_go * t_go)
            t_go_all[i] = t_go

        # Hold the current target while it stays affordable (hysteresis).
        cur = self.current_target_idx
        cur_alive = cur is not None and cur < num and valid[cur]
        if cur_alive and a_req[cur] <= self.MAX_LATERAL_ACCEL * self.SWITCH_HYSTERESIS:
            return cur

        feasible = a_req <= self.MAX_LATERAL_ACCEL
        if feasible.any():
            t_sel = np.where(feasible, t_go_all, np.inf)
            self.current_target_idx = int(np.argmin(t_sel))
        elif cur_alive:
            # Nothing affordable: keep it; the miss detector ends the episode.
            return cur
        elif np.isfinite(a_req).any():
            self.current_target_idx = int(np.argmin(a_req))
        else:
            self.current_target_idx = _nearest_valid()

        return self.current_target_idx
