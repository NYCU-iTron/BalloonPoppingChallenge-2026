from dataclasses import dataclass

import numpy as np


@dataclass
class BalloonSpawnConfig:
    initial_distance: float = 25.0
    distance_increment: float = 5.0
    distance_jitter: float = 0.15
    min_altitude: float = 5.0
    max_altitude: float = 400.0
    axis_weights: tuple[float, float, float] = (1.0, 1.0, 0.6)
    velocity_std: float = 0.0


class ProgressiveBalloonGenerator:
    """Generate one balloon around the rocket, moving farther after each hit."""

    def __init__(self, config: BalloonSpawnConfig | None = None, seed: int | None = None):
        self.config = config or BalloonSpawnConfig()
        self.rng = np.random.default_rng(seed)
        self.spawn_index = 0

    def reset(self):
        self.spawn_index = 0

    def current_distance(self) -> float:
        return (
            self.config.initial_distance
            + self.spawn_index * self.config.distance_increment
        )

    def next_balloon_state(self, rocket_position: np.ndarray) -> np.ndarray:
        distance = self.current_distance()
        jitter = self.rng.uniform(
            1.0 - self.config.distance_jitter,
            1.0 + self.config.distance_jitter,
        )
        offset = self._sample_direction() * distance * jitter
        position = np.asarray(rocket_position, dtype=float) + offset
        position[2] = np.clip(
            position[2],
            self.config.min_altitude,
            self.config.max_altitude,
        )

        velocity = self.rng.normal(0.0, self.config.velocity_std, size=3)
        self.spawn_index += 1
        return np.concatenate([position, velocity])

    def _sample_direction(self) -> np.ndarray:
        weights = np.asarray(self.config.axis_weights, dtype=float)
        if np.any(weights < 0) or np.all(weights == 0):
            raise ValueError("axis_weights must contain non-negative values and at least one positive value")

        direction = self.rng.normal(0.0, weights + 1e-12, size=3)
        norm = np.linalg.norm(direction)
        if norm < 1e-9:
            return np.array([1.0, 0.0, 0.0])
        return direction / norm
