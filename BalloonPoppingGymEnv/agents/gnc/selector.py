import logging
import numpy as np
from BalloonPoppingGymEnv.utils.schema import Schema


class Selector:
    # Launch gating for rising targets: hold the rocket on the pad until every
    # balloon is airborne and the LOWEST one has climbed past this altitude
    # above ground. The release sequence turns the balloons into a vertical
    # column (~release_interval * climb_rate spacing) almost directly overhead;
    # engaging that column from below with a pure climb is the vehicle's
    # strength (T/W ~1.3 gives almost no lateral authority near the ground, so
    # launching at a freshly released balloon at pad altitude is unwinnable).
    LAUNCH_GATE_AGL = 100.0  # (m) hand-tuned engagement altitude

    # Targeting (see select_target): candidates are scored by the lateral
    # acceleration their intercept would demand from the current course.
    MAX_LATERAL_ACCEL = 10.0  # (m/s^2) sustainable lateral authority at T/W ~1.3
    SWITCH_HYSTERESIS = 1.5   # keep the current target until 1.5x over budget
    MIN_CHAIN_SPEED = 5.0     # (m/s) below this, course-based scoring is undefined
    MAX_TIME_TO_GO = 30.0     # (s) slower intercepts than this are irrelevant

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

        # If no targets
        if raw_balloons.size == 0 or len(raw_balloons) == 0:
            return True

        status = np.asarray(observation["balloon_status"], dtype=int).reshape(-1)

        # Wait out the release sequence: a balloon still on the ground would
        # otherwise keep resetting the "lowest airborne" gate below.
        airborne = status == 1
        if (status == 0).any() or not airborne.any():
            return False

        # Static (or descending) targets cannot be waited out -- engage now.
        # This keeps the immediate-launch behavior for the static scenarios.
        if float(np.max(raw_balloons[airborne, 5])) <= 0.1:
            return True

        lowest_agl = float(np.min(raw_balloons[airborne, 2])) - self.ground_elevation
        return lowest_agl >= self.LAUNCH_GATE_AGL

    def get_launch_heading(self, observation: dict) -> np.ndarray:
        """
        Returns [inclination, heading] in degrees based on balloon positions.
        """
        return np.array([90.0, 0.0])

    def select_target(self, balloon_states: np.ndarray, rocket_state: np.ndarray) -> int | None:
        """
        Reachability-aware target selection with forward chaining.

        A low-T/W interceptor cannot turn back, so candidates are scored by the
        lateral acceleration an intercept from the current course would demand
        (zero-effort-miss guidance demand: a_req = 2*|ZEM| / t_go^2). Among the
        affordable candidates the soonest intercept wins, which naturally chains
        target-to-target up the rising balloon column after each pop. The
        current target is kept (hysteresis) until it costs 1.5x the budget, so
        the selection does not jitter between similar candidates.

        Parameters
        ----------
        balloon_states : np.ndarray
            Shape (N, 6) predicted states [x, y, z, vx, vy, vz]; NaN position
            marks inactive (unreleased or popped) balloons.
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

        # On the pad / just after liftoff there is no meaningful course yet.
        # Under the launch gate the nearest balloon is the lowest of the column
        # overhead -- exactly the right opening target.
        if float(np.linalg.norm(rocket_vel)) < self.MIN_CHAIN_SPEED:
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
                continue  # receding or barely closing: not reachable ahead
            t_go = dist / v_close
            if t_go > self.MAX_TIME_TO_GO:
                continue  # closing too slowly to matter this flight
            zem = rel - v_rel * t_go  # miss vector if the course is held
            # A pass within the balloon radius already pops it, so only the
            # miss beyond the radius needs to be steered out.
            miss = max(float(np.linalg.norm(zem)) - self.balloon_radius, 0.0)
            a_req[i] = 2.0 * miss / (t_go * t_go)
            t_go_all[i] = t_go

        # Hysteresis: hold the current target while it stays affordable.
        cur = self.current_target_idx
        cur_alive = cur is not None and cur < num and valid[cur]
        if cur_alive and a_req[cur] <= self.MAX_LATERAL_ACCEL * self.SWITCH_HYSTERESIS:
            return cur

        feasible = a_req <= self.MAX_LATERAL_ACCEL
        if feasible.any():
            # Soonest affordable intercept -> chain forward along the course.
            t_sel = np.where(feasible, t_go_all, np.inf)
            self.current_target_idx = int(np.argmin(t_sel))
        elif cur_alive:
            # Nothing affordable: keep the current one; the training env's miss
            # detector ends the episode if it keeps receding.
            return cur
        elif np.isfinite(a_req).any():
            # No current target either: take the least-demanding candidate.
            self.current_target_idx = int(np.argmin(a_req))
        else:
            self.current_target_idx = _nearest_valid()

        return self.current_target_idx
