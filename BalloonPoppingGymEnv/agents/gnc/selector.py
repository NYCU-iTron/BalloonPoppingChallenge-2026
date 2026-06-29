"""Universal target selector with multi-factor cost function.

Designed to work across all scenarios (0/1/2) by combining environmental
and vehicle state information into a single cost metric.  The selector
also handles launch-time and launch-heading decisions so that these
parameters are determined centrally rather than being hard-coded in the
agent loop.

Cost function design rationale
------------------------------
The cost is a weighted sum of features that capture:
  1. Geometric reachability (miss distance, range, alignment)
  2. Kinematic favourability (closing speed, time-to-go)
  3. Strategic value (cluster density, altitude advantage)
  4. Hard penalties (unreachable, behind rocket, below rocket)

Lower cost = better target.
"""

import logging
import math
from dataclasses import dataclass, field

import numpy as np

from BalloonPoppingGymEnv.utils.schema import Schema


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def _unit(v: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    n = _norm(v)
    if n > 1e-9:
        return np.asarray(v, dtype=float) / n
    return np.asarray(fallback if fallback is not None else [0.0, 0.0, 1.0], dtype=float)


def _clip(val: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, val)))


def _vector_to_inclination_heading(vector: np.ndarray) -> np.ndarray:
    """Convert ENU vector → [inclination, heading] in degrees."""
    east, north, up = np.asarray(vector, dtype=float)
    horizontal = math.hypot(east, north)
    inclination = math.degrees(math.atan2(max(up, 0.0), horizontal))
    heading = math.degrees(math.atan2(east, north)) % 360.0
    if horizontal < 1e-9:
        heading = 0.0
    return np.array([_clip(inclination, 0.0, 90.0), heading], dtype=float)


def _inclination_heading_to_vector(ih: np.ndarray) -> np.ndarray:
    """Convert [inclination, heading] in degrees → ENU unit vector."""
    inc = math.radians(float(ih[0]))
    hdg = math.radians(float(ih[1]))
    h = math.cos(inc)
    return _unit(np.array([h * math.sin(hdg), h * math.cos(hdg), math.sin(inc)], dtype=float))


# ---------------------------------------------------------------------------
# Feature / candidate data structures
# ---------------------------------------------------------------------------

@dataclass
class CandidateFeatures:
    """Raw features computed for one balloon candidate."""
    index: int
    miss_distance: float       # predicted closest approach (m)
    range_now: float           # current distance (m)
    forward_alignment: float   # dot(rocket_vel_hat, los_hat), [-1, 1]
    closing_speed: float       # -dot(rel_vel, los_hat) (m/s), positive = closing
    cluster_density: float     # count of neighbours within cluster_radius
    altitude_advantage: float  # target_z - rocket_z (m), positive = above
    t_go: float                # best estimated time-to-intercept (s)
    balloon_pos: np.ndarray = field(repr=False)
    balloon_vel: np.ndarray = field(repr=False)


# ---------------------------------------------------------------------------
# Cost function weights — the primary tunable knobs
# ---------------------------------------------------------------------------

@dataclass
class CostWeights:
    """Weights for the linear cost function.  Lower total = better target."""
    w_miss: float = 1.0        # miss distance penalty (per metre)
    w_range: float = 0.04      # range penalty (per metre)
    w_alignment: float = -8.0  # alignment reward (negative = reward)
    w_closing: float = -0.3    # closing speed reward
    w_density: float = -3.0    # cluster density reward
    w_altitude: float = 0.0    # altitude advantage reward (negative = reward up)
    w_tgo: float = 0.5         # time-to-go penalty
    # Hard penalty magnitudes
    penalty_behind: float = 80.0     # forward_alignment < behind_threshold
    penalty_below: float = 20.0      # altitude_advantage < below_threshold
    penalty_unreachable: float = 200.0  # miss too large or fuel insufficient
    # Thresholds for hard penalties
    behind_threshold: float = -0.10
    below_threshold: float = -10.0   # m
    unreachable_miss: float = 500.0  # m


# ---------------------------------------------------------------------------
# Main Selector
# ---------------------------------------------------------------------------

class Selector:
    """Universal multi-factor target selector.

    Integrates with the itron-dev GNC pipeline via:
      - select(balloon_states, rocket_state) → int | None
      - get_launch_time(observation) → float
      - get_launch_heading(observation) → np.ndarray
    """

    def __init__(self, given_parameters: dict, **kwargs):
        self.logger = logging.getLogger(__name__)
        self.given_parameters = given_parameters

        # --- Target selection state ---
        self.current_target_idx: int | None = None
        self._target_selected_at: float = 0.0
        self._target_switch_count: int = 0

        # --- Launch state ---
        self._launch_heading_cache: np.ndarray = np.array([90.0, 0.0], dtype=float)

        # --- Configuration from given_parameters ---
        self._balloon_radius: float = float(
            given_parameters.get(Schema.Given.Section.BALLOON, {})
            .get(Schema.Given.Balloon.RADIUS, 1.5)
        )
        self._elevation: float = float(
            given_parameters.get(Schema.Given.Section.ENVIRONMENT, {})
            .get(Schema.Given.Environment.ELEVATION, 20.0)
        )
        sampling_rate = float(
            given_parameters.get(Schema.Given.Section.ROCKET, {})
            .get(Schema.Given.Rocket.SENSORS, {})
            .get(Schema.Given.Sensors.SAMPLING_RATE, 100)
        )
        self._dt: float = 1.0 / sampling_rate

        # --- Tunable parameters (overridable via kwargs) ---
        self.weights = CostWeights(
            w_miss=float(kwargs.get("w_miss", 1.0)),
            w_range=float(kwargs.get("w_range", 0.04)),
            w_alignment=float(kwargs.get("w_alignment", -8.0)),
            w_closing=float(kwargs.get("w_closing", -0.3)),
            w_density=float(kwargs.get("w_density", -3.0)),
            w_altitude=float(kwargs.get("w_altitude", 0.0)),
            w_tgo=float(kwargs.get("w_tgo", 0.5)),
            penalty_behind=float(kwargs.get("penalty_behind", 80.0)),
            penalty_below=float(kwargs.get("penalty_below", 20.0)),
            penalty_unreachable=float(kwargs.get("penalty_unreachable", 200.0)),
            behind_threshold=float(kwargs.get("behind_threshold", -0.10)),
            below_threshold=float(kwargs.get("below_threshold", -10.0)),
            unreachable_miss=float(kwargs.get("unreachable_miss", 500.0)),
        )

        # Target selection hysteresis
        self._min_dwell: float = float(kwargs.get("min_dwell", 1.5))       # s
        self._switch_margin: float = float(kwargs.get("switch_margin", 0.35))  # fraction

        # Intercept speed estimate for t_go calculation
        self._intercept_speed_floor: float = float(kwargs.get("intercept_speed_floor", 60.0))
        self._max_t_go: float = float(kwargs.get("max_t_go", 12.0))

        # Cluster radius for density calculation
        self._cluster_radius: float = float(kwargs.get("cluster_radius", 25.0))

        # Lookahead grid for miss-distance search (seconds)
        self._lookahead_grid: np.ndarray = np.array(
            kwargs.get("lookahead_grid", [2.0, 4.0, 6.0, 8.0, 10.0]),
            dtype=float,
        )

        # Launch parameters
        self._min_launch_time: float = float(kwargs.get("min_launch_time", 1.0))
        self._max_launch_time: float = float(kwargs.get("max_launch_time", 6.0))
        self._min_released: int = int(kwargs.get("min_released", 8))
        self._min_launch_inclination: float = float(kwargs.get("min_launch_inclination", 85.0))
        self._launch_prediction_time: float = float(kwargs.get("launch_prediction_time", 12.0))
        self._launch_cluster_radius: float = float(kwargs.get("launch_cluster_radius", 25.0))

        # Internal time tracking
        self._current_time: float = 0.0

        self.logger.info(
            "Selector initialised (weights: miss=%.2f range=%.3f align=%.1f close=%.2f "
            "density=%.1f tgo=%.2f)",
            self.weights.w_miss, self.weights.w_range, self.weights.w_alignment,
            self.weights.w_closing, self.weights.w_density, self.weights.w_tgo,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset internal state for a new episode."""
        self.current_target_idx = None
        self._target_selected_at = 0.0
        self._target_switch_count = 0
        self._launch_heading_cache = np.array([90.0, 0.0], dtype=float)
        self._current_time = 0.0

    def get_launch_time(self, observation: dict) -> float:
        """Return desired launch time based on balloon release state.

        Logic: launch when enough balloons are released, or after max wait.
        """
        t = float(observation[Schema.Observation.SIMULATION_TIME])
        self._current_time = t

        if t < self._min_launch_time:
            # Return a time in the future to prevent launch
            return t + 1.0

        statuses = np.asarray(observation[Schema.Observation.BALLOON_STATUS], dtype=int).flatten()
        released = int(np.sum(statuses == 1))

        if released >= self._min_released or t >= self._max_launch_time:
            # Recompute heading at launch moment
            self._launch_heading_cache = self._compute_launch_heading(observation)
            return t  # launch now

        return t + 1.0  # not yet

    def get_launch_heading(self, observation: dict) -> np.ndarray:
        """Return [inclination, heading] in degrees for launch orientation.

        Uses cluster-based heading toward the densest group of released
        (or all) balloons, predicted forward by launch_prediction_time.
        """
        # Always recompute to keep heading fresh pre-launch
        self._launch_heading_cache = self._compute_launch_heading(observation)
        return self._launch_heading_cache.copy()

    def select(self, balloon_states: np.ndarray, rocket_state: np.ndarray) -> int | None:
        """Select best target balloon using multi-factor cost function.

        Parameters
        ----------
        balloon_states : np.ndarray, shape (N, 6)
            Predicted balloon states [x, y, z, vx, vy, vz].
            NaN rows indicate inactive/invalid balloons.
        rocket_state : np.ndarray, shape (16,)
            Estimated rocket state [pos(3), vel(3), acc(3), quat(4), gyro(3)].

        Returns
        -------
        int or None
            Index of selected target balloon, or None if no valid target.
        """
        self._current_time += self._dt  # approximate time tracking

        rocket_pos = rocket_state[0:3]
        rocket_vel = rocket_state[3:6]

        # Build candidates for all valid balloons
        candidates = self._build_candidates(balloon_states, rocket_pos, rocket_vel)

        if not candidates:
            self.current_target_idx = None
            return None

        # Score and sort
        scored = [(c, self._compute_cost(c)) for c in candidates]
        scored.sort(key=lambda x: x[1])
        best_candidate, best_cost = scored[0]

        # Hysteresis: keep current target if still valid and not much worse
        if self.current_target_idx is not None:
            current_entry = next(
                ((c, s) for c, s in scored if c.index == self.current_target_idx),
                None,
            )
            if current_entry is not None:
                current_candidate, current_cost = current_entry
                dwell = self._current_time - self._target_selected_at

                if dwell < self._min_dwell:
                    return self.current_target_idx

                improvement = current_cost - best_cost
                threshold = max(
                    abs(current_cost) * self._switch_margin,
                    self._switch_margin,
                )
                if improvement <= threshold:
                    return self.current_target_idx

        # Switch to new target
        if self.current_target_idx != best_candidate.index:
            self._target_switch_count += 1
            self._target_selected_at = self._current_time

        self.current_target_idx = best_candidate.index
        return self.current_target_idx

    # ------------------------------------------------------------------
    # Cost function
    # ------------------------------------------------------------------

    def _compute_cost(self, c: CandidateFeatures) -> float:
        """Evaluate the cost function for a single candidate."""
        w = self.weights

        cost = (
            w.w_miss * c.miss_distance
            + w.w_range * c.range_now
            + w.w_alignment * max(c.forward_alignment, 0.0)
            + w.w_closing * max(c.closing_speed, 0.0)
            + w.w_density * c.cluster_density
            + w.w_altitude * c.altitude_advantage
            + w.w_tgo * c.t_go
        )

        # Hard penalties
        if c.forward_alignment < w.behind_threshold:
            cost += w.penalty_behind
        if c.altitude_advantage < w.below_threshold:
            cost += w.penalty_below
        if c.miss_distance > w.unreachable_miss:
            cost += w.penalty_unreachable

        # Progressive range penalty: strongly penalise targets beyond
        # a reasonable intercept envelope.  This prevents the selector
        # from being lured by very far balloons that are geometrically
        # aligned but kinematically unreachable.
        if c.range_now > 150.0:
            cost += 0.1 * (c.range_now - 150.0)

        return float(cost)

    # ------------------------------------------------------------------
    # Candidate building
    # ------------------------------------------------------------------

    def _build_candidates(
        self,
        balloon_states: np.ndarray,
        rocket_pos: np.ndarray,
        rocket_vel: np.ndarray,
    ) -> list[CandidateFeatures]:
        """Compute features for all valid balloons."""
        n = balloon_states.shape[0]
        candidates: list[CandidateFeatures] = []

        # Identify valid balloons (not NaN)
        valid_mask = ~np.isnan(balloon_states[:, 0])

        if not np.any(valid_mask):
            return candidates

        # Pre-compute predicted positions at each lookahead time for density
        valid_indices = np.flatnonzero(valid_mask)
        positions_at_grid: dict[float, np.ndarray] = {}
        for t in self._lookahead_grid:
            positions_at_grid[float(t)] = (
                balloon_states[valid_indices, :3]
                + balloon_states[valid_indices, 3:6] * float(t)
            )

        # Fallback velocity for pre-launch or very slow rocket
        rocket_speed = _norm(rocket_vel)
        effective_vel = rocket_vel.copy()
        if rocket_speed < 1.0:
            # Use upward direction scaled by intercept speed floor
            effective_vel = np.array([0.0, 0.0, 1.0], dtype=float) * self._intercept_speed_floor

        for i in valid_indices:
            balloon_pos = balloon_states[i, :3]
            balloon_vel = balloon_states[i, 3:6]

            # Relative kinematics
            rel = balloon_pos - rocket_pos
            rel_vel = balloon_vel - effective_vel
            range_now = _norm(rel)
            los_hat = _unit(rel, np.array([0.0, 0.0, 1.0]))

            # Closest approach time estimate
            rel_vel_sq = max(float(np.dot(rel_vel, rel_vel)), 1e-6)
            t_ca = _clip(-float(np.dot(rel, rel_vel)) / rel_vel_sq, 0.0, self._max_t_go)
            t_range = _clip(range_now / self._intercept_speed_floor, 0.0, self._max_t_go)

            # Search over lookahead grid + analytic times for best miss
            candidate_times = np.unique(
                np.concatenate((self._lookahead_grid, np.array([t_ca, t_range])))
            )

            best_miss = float("inf")
            best_t = float(candidate_times[0])
            best_density = 0.0

            for t_go in candidate_times:
                target_pred = balloon_pos + balloon_vel * t_go
                rocket_pred = rocket_pos + effective_vel * t_go
                miss = _norm(target_pred - rocket_pred)

                # Cluster density at this time
                nearest_time = min(positions_at_grid.keys(), key=lambda t: abs(t - t_go))
                grid_positions = positions_at_grid[nearest_time]
                distances = np.linalg.norm(grid_positions - target_pred, axis=1)
                density = float(np.sum(distances <= self._cluster_radius))

                # Density-adjusted miss
                adjusted_miss = miss - 2.0 * density
                if adjusted_miss < best_miss:
                    best_miss = adjusted_miss
                    best_t = float(t_go)
                    best_density = density

            # Raw miss (undo density adjustment for the feature)
            raw_miss = max(0.0, best_miss + 2.0 * best_density)

            # Forward alignment
            vel_hat = _unit(effective_vel, np.array([0.0, 0.0, 1.0]))
            forward_alignment = float(np.dot(vel_hat, los_hat))

            # Closing speed
            closing_speed = -float(np.dot(rel_vel, los_hat))

            # Altitude advantage
            altitude_advantage = float(balloon_pos[2] - rocket_pos[2])

            candidates.append(CandidateFeatures(
                index=int(i),
                miss_distance=raw_miss,
                range_now=range_now,
                forward_alignment=forward_alignment,
                closing_speed=closing_speed,
                cluster_density=best_density,
                altitude_advantage=altitude_advantage,
                t_go=best_t,
                balloon_pos=balloon_pos.copy(),
                balloon_vel=balloon_vel.copy(),
            ))

        return candidates

    # ------------------------------------------------------------------
    # Launch heading computation
    # ------------------------------------------------------------------

    def _compute_launch_heading(self, observation: dict) -> np.ndarray:
        """Compute launch heading from balloon cluster analysis."""
        statuses = np.asarray(
            observation[Schema.Observation.BALLOON_STATUS], dtype=int
        ).flatten()
        states = np.asarray(
            observation[Schema.Observation.BALLOON_STATES], dtype=float
        )

        # Use released balloons if available, otherwise all
        released_indices = np.flatnonzero(statuses == 1)
        if len(released_indices) > 0:
            active_indices = released_indices
        else:
            active_indices = np.arange(states.shape[0])

        if len(active_indices) == 0:
            return np.array([90.0, 0.0], dtype=float)

        # Predict positions at launch_prediction_time
        positions = (
            states[active_indices, :3]
            + states[active_indices, 3:6] * self._launch_prediction_time
        )

        # Find densest cluster center
        rocket_pos = np.array([0.0, 0.0, self._elevation], dtype=float)
        best_center = positions[0]
        best_score = -1e9

        for pos in positions:
            distances = np.linalg.norm(positions - pos, axis=1)
            density = float(np.sum(distances <= max(self._launch_cluster_radius, 1.0)))
            horizontal = math.hypot(pos[0], pos[1])
            score = density - 0.01 * horizontal + 0.002 * (pos[2] - self._elevation)
            if score > best_score:
                best_score = score
                best_center = pos

        raw_vector = _unit(best_center - rocket_pos, np.array([0.0, 0.0, 1.0]))
        launch_angles = _vector_to_inclination_heading(raw_vector)

        # Clamp inclination to minimum
        launch_angles[0] = max(
            float(launch_angles[0]),
            _clip(self._min_launch_inclination, 0.0, 90.0),
        )

        return launch_angles

    # ------------------------------------------------------------------
    # Debug interface
    # ------------------------------------------------------------------

    def get_debug_stats(self) -> dict:
        """Return diagnostic statistics for MC evaluator."""
        return {
            "target_switch_count": self._target_switch_count,
            "current_target_idx": (
                int(self.current_target_idx) if self.current_target_idx is not None else -1
            ),
        }
