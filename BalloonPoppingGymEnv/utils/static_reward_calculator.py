import numpy as np


class StaticRewardCalculator:
    def __init__(self, given_parameters):
        # Sparse
        self.max_terminate_penalty = -200.0
        self.min_terminate_penalty = -100.0
        self.min_pop_reward = 500.0
        self.max_pop_reward = 800.0

        # Dense
        dense_budget = 0.2 * self.min_pop_reward
        self.base_approach_weight = 0.8 * dense_budget
        self.base_zem_weight = 0.2 * dense_budget
        self.worst_phi_dist = 2.0
        self.worst_phi_zem = 2.0
        self.phi_worst = -1.0

        # Curve tracking (Stage-0 only): a reference-path potential added
        # alongside phi_dist/phi_zem, same telescoping PBRS form. Weight kept
        # modest -- about half base_zem_weight, an eighth of
        # base_approach_weight -- so it nudges early learning without
        # competing with the sparse-pop-driven objective.
        self.curve_weight = 10.0
        self.worst_phi_curve = 2.0
        self.curve_ref_perp = 15.0  # (m) fixed normalization: NOT re-locked on
                                     # target switch like curr_ref_dist -- see
                                     # set_reference_trajectory()
        self.curve_window_back = 50
        self.curve_window_fwd = 200

        # Attitude stability (every step, not PBRS -- there's no natural
        # "goal state" to define a potential over, just a continuous safety
        # cost). sin_alpha/sin_beta are already clipped to [-1,1] by
        # RLObservator, so this term is naturally bounded per step
        # ([0,2]*weight) without needing its own clamp. Proposed/adjustable:
        # picked so a sustained, meaningfully unstable episode (~0.2-0.3
        # average sin^2 sum over a few hundred steps) accumulates a penalty
        # on the same order as the approach/zem dense budget, without a
        # near-zero-AoA episode ever accumulating much at all.
        self.stability_weight = 0.5

        # Safeguard lower bounds
        self.min_ref_dist = 5.0
        self.min_ref_zem_dist = 2.0

        # Dynamic normalization state
        self.curr_ref_dist = 100.0
        self.curr_ref_zem_dist = 30.0

        # PBRS state memory
        self.prev_target_idx = None
        self.prev_phi_dist = self.phi_worst
        self.prev_phi_zem = self.phi_worst
        self.prev_phi_curve = self.phi_worst

        self._ref_positions = None
        self._nearest_idx = 0

    def reset(self) -> None:
        self.curr_ref_dist = 100.0
        self.curr_ref_zem_dist = 30.0
        self.prev_target_idx = None
        self.prev_phi_dist = self.phi_worst
        self.prev_phi_zem = self.phi_worst
        self.prev_phi_curve = self.phi_worst
        self._ref_positions = None
        self._nearest_idx = 0

    def set_reference_trajectory(self, trajectory: np.ndarray) -> None:
        """trajectory: (N, 8) array from generate_reference_trajectory,
        columns [t, x, y, z, vx, vy, vz, s]. Call once per episode, after
        launch handoff and target sampling, before the first compute()."""
        self._ref_positions = trajectory[:, 1:4]
        self._nearest_idx = 0

    def compute(
        self,
        *,
        pop_count: float,
        rocket_state: np.ndarray,
        target_idx: int,
        target_state: np.ndarray,
        terminated: bool,
        info: dict,
        sin_alpha: float,
        sin_beta: float,
    ) -> tuple[float, dict]:

        # ---------------------------------- Sparse ---------------------------------- #
        popped_count = info.get("popped_count", 0)

        # Terminated
        terminated_penalty = 0.0
        if terminated:
            progress = 1.0 - np.exp(-0.25 * popped_count)
            terminated_penalty = self.max_terminate_penalty + (
                self.min_terminate_penalty - self.max_terminate_penalty
            ) * progress

        # Pop
        pop_reward = 0.0
        if pop_count > 0:
            delta = max(0, popped_count - 1)
            progress = 1.0 - np.exp(-0.2 * delta)
            unit_pop_reward = self.min_pop_reward + (
                self.max_pop_reward - self.min_pop_reward
            ) * progress
            pop_reward = pop_count * unit_pop_reward

        # ----------------------------------- Dense ---------------------------------- #
        is_invalid_target = (
            target_idx is None
            or target_state is None
            or np.isnan(target_state).any()
        )
        switched = target_idx != self.prev_target_idx

        if is_invalid_target:
            phi_dist = self.phi_worst
            phi_zem = self.phi_worst
            phi_curve = self.phi_worst
        else:
            rel_pos = target_state[0:3] - rocket_state[0:3]
            rel_vel = rocket_state[3:6] - target_state[3:6]

            dist = float(np.linalg.norm(rel_pos))
            unit_los = rel_pos / max(dist, 1e-9)

            # zem dist
            rel_speed_sq = float(np.dot(rel_vel, rel_vel))
            if rel_speed_sq > 1e-6:
                closing_vel = float(np.dot(rel_vel, unit_los))
                t_go = max(dist * closing_vel, 0.0) / rel_speed_sq
                zem_dist = float(np.linalg.norm(rel_pos - rel_vel * t_go))
            else:
                zem_dist = dist

            # Lock per-engagement reference on lock-on
            if switched:
                self.curr_ref_dist = max(dist, self.min_ref_dist)
                self.curr_ref_zem_dist = max(zem_dist, self.min_ref_zem_dist)

            # Distance potential (normalized to -1.0 -> 0.0)
            phi_dist = -min(dist / self.curr_ref_dist, self.worst_phi_dist)

            # ZEM potential
            phi_zem = -min(zem_dist / self.curr_ref_zem_dist, self.worst_phi_zem)

            # Curve-tracking potential: perpendicular distance to the nearest
            # point on the precomputed reference path, windowed around the
            # previously-matched index so the match tracks forward progress
            # along the path instead of jumping to a spatially-close but
            # temporally-distant point elsewhere on it (e.g. across the
            # turn-in arc) -- that would make the signal spiky/discontinuous
            # near turns even though it can't be net-farmed either way (see
            # module docstring / plan notes on PBRS policy-invariance).
            if self._ref_positions is not None and self._ref_positions.shape[0] > 0:
                lo = max(0, self._nearest_idx - self.curve_window_back)
                hi = min(self._ref_positions.shape[0], self._nearest_idx + self.curve_window_fwd)
                window = self._ref_positions[lo:hi]
                diffs = window - rocket_state[0:3]
                dist_sq = np.einsum("ij,ij->i", diffs, diffs)
                local_best = int(np.argmin(dist_sq))
                self._nearest_idx = lo + local_best
                perp_dist = float(np.sqrt(dist_sq[local_best]))
                phi_curve = -min(perp_dist / self.curve_ref_perp, self.worst_phi_curve)
            else:
                phi_curve = self.phi_worst

        if switched and pop_count > 0:
            approach_reward = 0.0
            zem_reward = 0.0
            curve_reward = 0.0
        else:
            approach_reward = self.base_approach_weight * (phi_dist - self.prev_phi_dist)
            zem_reward = self.base_zem_weight * (phi_zem - self.prev_phi_zem)
            curve_reward = self.curve_weight * (phi_curve - self.prev_phi_curve)

        # Update memory
        self.prev_phi_dist = phi_dist
        self.prev_phi_zem = phi_zem
        self.prev_phi_curve = phi_curve
        self.prev_target_idx = None if is_invalid_target else target_idx

        # Attitude stability -- independent of target lock-on, a continuous
        # per-step cost rather than a PBRS potential (see __init__ comment).
        stability_penalty = -self.stability_weight * (sin_alpha ** 2 + sin_beta ** 2)

        # ----------------------------------- Total ---------------------------------- #
        reward = terminated_penalty + pop_reward + approach_reward + zem_reward + curve_reward + stability_penalty
        reward_dict = {
            "terminated": terminated_penalty,
            "pop": pop_reward,
            "approach": approach_reward,
            "zem": zem_reward,
            "curve": curve_reward,
            "stability": stability_penalty,
        }

        return reward, reward_dict
