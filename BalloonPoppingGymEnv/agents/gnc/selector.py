import logging
import numpy as np


class Selector:
    def __init__(self, given_parameters):
        self.logger = logging.getLogger(__name__)
        self.given_parameters = given_parameters

        self.current_target_idx = None

    def reset(self):
        pass

    def should_launch(self, observation: dict) -> bool:
        raw_balloons = np.asarray(observation["balloon_states"], dtype=float)

        # If no targets
        if raw_balloons.size == 0 or len(raw_balloons) == 0:
            return True

        balloon_altitudes = raw_balloons[:, 2]
        launch_threshold = 20.0
        should_launch = bool(np.any(balloon_altitudes > launch_threshold))

        return should_launch

    def get_launch_heading(self, observation: dict) -> np.ndarray:
        """
        Returns [inclination, heading] in degrees based on balloon positions.
        """
        return np.array([90.0, 0.0])

    def select_target(self, balloon_states: np.ndarray, rocket_state: np.ndarray) -> int | None:
        """
        Selects the target tracking balloon from the active environment cluster.

        Parameters
        ----------
        observation : dict
            Current telemetry dictionary structured as follows:
            - simulation_time : float -> [s]
            - balloon_states : np.ndarray -> shape (N, 6), tracking states [x, y, z, vx, vy, vz] in [m, m/s]
            - rocket_sensors : np.ndarray -> shape (12,), [gyro(3), acc(3), pos(3), vel(3)] in [rad/s, m/s², m, m/s]

        Returns
        -------
        balloon_state : np.ndarray or None
            A 6-element array containing the full target state vector [x, y, z, vx, vy, vz]
            of the selected balloon, or None if no active targets remain.
        """
        rocket_pos = rocket_state[0:3]

        # Check validity for all balloons (returns a boolean array)
        not_valid_idx = np.isnan(balloon_states[:, 0]) | np.isnan(balloon_states[:, 1]) | np.isnan(balloon_states[:, 2])

        # Fix: Extract the specific scalar element using the current target index
        if self.current_target_idx is not None and self.current_target_idx < len(balloon_states):
            if not not_valid_idx[self.current_target_idx]:
                return self.current_target_idx

        min_dist = float("inf")
        best_target_idx = None

        # Greedy search for the closest active balloon
        for i in range(len(balloon_states)):
            if np.isnan(balloon_states[i, 0]):
                continue

            balloon_pos = balloon_states[i, 0:3]
            dist = np.linalg.norm(balloon_pos - rocket_pos)
            if dist < min_dist:
                min_dist = dist
                best_target_idx = i

        # Update current target index
        if best_target_idx is not None:
            self.current_target_idx = best_target_idx
            return best_target_idx

        self.current_target_idx = None
        return None
