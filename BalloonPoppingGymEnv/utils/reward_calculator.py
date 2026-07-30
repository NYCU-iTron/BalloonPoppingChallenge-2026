import numpy as np


class RewardCalculator:
    def __init__(self, given_parameters):
        # Sparse
        self.max_terminate_penalty = -200.0
        self.min_terminate_penalty = -100.0
        self.min_pop_reward = 500.0
        self.max_pop_reward = 800.0

        # Dense
        dense_budget = 0.25 * self.min_pop_reward
        self.base_approach_weight = 0.5 * dense_budget
        self.base_zem_weight = 0.5 * dense_budget
        self.worst_phi_zem = 2.0

        # Safeguard lower bounds
        self.min_ref_dist = 5.0
        self.min_ref_zem_dist = 2.0

        # Dynamic normalization state
        self.curr_ref_dist = 100.0
        self.curr_ref_zem_dist = 30.0

        # PBRS state memory
        self.prev_target_idx = None
        self.prev_valid = False
        self.prev_phi_dist = None
        self.prev_phi_zem = None

    def reset(self) -> None:
        self.curr_ref_dist = 100.0
        self.curr_ref_zem_dist = 30.0
        self.prev_target_idx = None
        self.prev_valid = False
        self.prev_phi_dist = None
        self.prev_phi_zem = None

    def compute(
        self,
        *,
        observation: dict,
        pop_count: float,
        rocket_state: np.ndarray,
        target_idx: int,
        target_state: np.ndarray,
        desired_acc: np.ndarray,
        terminated: bool,
        info: dict,
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

        # -------------------------------- Guard Clauses ----------------------------- #
        is_invalid_target = (
            target_idx is None
            or target_state is None
            or np.isnan(target_state).any()
        )
        if is_invalid_target:
            self.prev_valid = False
            reward = terminated_penalty + pop_reward
            return reward, {
                "terminated": terminated_penalty,
                "pop": pop_reward,
                "approach": 0.0,
                "zem": 0.0,
            }

        rel_pos = target_state[0:3] - rocket_state[0:3]
        dist = float(np.linalg.norm(rel_pos))

        if dist <= 1e-6:
            self.prev_valid = False
            reward = terminated_penalty + pop_reward
            return reward, {
                "terminated": terminated_penalty,
                "pop": pop_reward,
                "approach": 0.0,
                "zem": 0.0,
            }

        # ----------------------------------- Dense ---------------------------------- #
        rel_vel = rocket_state[3:6] - target_state[3:6]
        unit_los = rel_pos / dist
        closing_vel = float(np.dot(rel_vel, unit_los))

        zem_dist = None
        if closing_vel > 0.5:
            t_go = dist / closing_vel
            zem_dist = float(np.linalg.norm(rel_pos - rel_vel * t_go))

        # Continuity check against previous step
        is_continuous = (
            self.prev_valid
            and self.prev_target_idx == target_idx
            and self.prev_phi_dist is not None
            and self.prev_phi_zem is not None
        )

        # Mark current state as valid for next step
        self.prev_valid = True

        # Lock per-engagement reference on lock-on
        if not is_continuous:
            self.curr_ref_dist = max(dist, self.min_ref_dist)
            self.curr_ref_zem_dist = (
                max(zem_dist, self.min_ref_zem_dist) if zem_dist is not None
                else max(0.3 * self.curr_ref_dist, self.min_ref_zem_dist)
            )

        # Distance potential (normalized to -1.0 -> 0.0)
        phi_dist = -dist / self.curr_ref_dist

        # ZEM potential
        phi_zem = (
            -min(zem_dist / self.curr_ref_zem_dist, self.worst_phi_zem)
            if zem_dist is not None else -self.worst_phi_zem
        )

        approach_reward = 0.0
        zem_reward = 0.0
        if is_continuous:
            approach_reward = self.base_approach_weight * (
                phi_dist - self.prev_phi_dist
            )
            zem_reward = self.base_zem_weight * (phi_zem - self.prev_phi_zem)

        # Update memory
        self.prev_phi_dist = phi_dist
        self.prev_phi_zem = phi_zem
        self.prev_target_idx = target_idx

        # ----------------------------------- Total ---------------------------------- #
        reward = terminated_penalty + pop_reward + approach_reward + zem_reward
        reward_dict = {
            "terminated": terminated_penalty,
            "pop": pop_reward,
            "approach": approach_reward,
            "zem": zem_reward,
        }

        return reward, reward_dict
