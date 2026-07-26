import numpy as np


class RewardCalculator:
    def __init__(self, given_parameters):
        self.target_idx = None
        self.best_dist = None
        self.closest_dist = None

        self.balloon_radius = given_parameters["balloon"]["radius"]

    def reset(self) -> None:
        self.target_idx = None
        self.best_dist = None
        self.closest_dist = None

    def compute(self,
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

        pop_reward = 0.0
        progress_reward = 0.0
        failure = 0.0

        # Target changed
        if target_idx != self.target_idx:
            self.target_idx = target_idx
            self.best_dist = None

        balloon_pos = np.asarray(observation["balloon_states"], dtype=float)[target_idx, 0:3]
        curr_dist = float(np.linalg.norm(rocket_state[0:3] - balloon_pos))

        # Update closest distance
        if self.closest_dist is None:
            self.closest_dist = curr_dist
        else:
            self.closest_dist = min(self.closest_dist, curr_dist)

        # Failure
        if terminated and info["popped_count"] == 0:
            failure = -100 - min(self.closest_dist, 300)

        # Progress shaping
        if self.best_dist is None:
            self.best_dist = curr_dist
        elif curr_dist < self.best_dist:
            progress = (np.log(max(self.best_dist, self.balloon_radius))
                        - np.log(max(curr_dist, self.balloon_radius)))
            progress_reward = progress * 20.0
            self.best_dist = curr_dist

        # Pop
        if pop_count > 0:
            pop_reward = pop_count * 500

        reward = pop_reward + progress_reward + failure
        reward_dict = {
            "pop": pop_reward,
            "progress": progress_reward,
            "failure": failure,
        }

        return reward, reward_dict
