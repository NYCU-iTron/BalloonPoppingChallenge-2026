import numpy as np


class RewardCalculator:
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

    def reset(self) -> None:
        self.curr_ref_dist = 100.0
        self.curr_ref_zem_dist = 30.0
        self.prev_target_idx = None
        self.prev_phi_dist = self.phi_worst
        self.prev_phi_zem = self.phi_worst

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

        if switched and pop_count > 0:
            approach_reward = 0.0
            zem_reward = 0.0
        else:
            approach_reward = self.base_approach_weight * (phi_dist - self.prev_phi_dist)
            zem_reward = self.base_zem_weight * (phi_zem - self.prev_phi_zem)

        # Update memory
        self.prev_phi_dist = phi_dist
        self.prev_phi_zem = phi_zem
        self.prev_target_idx = None if is_invalid_target else target_idx

        # ----------------------------------- Total ---------------------------------- #
        reward = terminated_penalty + pop_reward + approach_reward + zem_reward
        reward_dict = {
            "terminated": terminated_penalty,
            "pop": pop_reward,
            "approach": approach_reward,
            "zem": zem_reward,
        }

        return reward, reward_dict
