from collections import deque
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class CurriculumCallback(BaseCallback):
    """Advance trajectory difficulty after sustained episode-level mastery."""

    def __init__(
        self,
        stages,
        num_balloons,
        eval_env=None,
        window_size=100,
        mean_popped_ratio_threshold=0.20,
        proximity_distance=5.0,
        mean_close_approach_ratio_threshold=0.40,
        check_every_episodes=25,
        required_consecutive_passes=3,
        verbose=1,
    ):
        super().__init__(verbose=verbose)
        if not stages:
            raise ValueError("At least one curriculum stage is required")
        if num_balloons <= 0:
            raise ValueError("num_balloons must be positive")
        if window_size <= 0 or check_every_episodes <= 0:
            raise ValueError("Episode window and check interval must be positive")
        if required_consecutive_passes <= 0:
            raise ValueError("required_consecutive_passes must be positive")
        if proximity_distance <= 0:
            raise ValueError("proximity_distance must be positive")
        for value, name in (
            (mean_popped_ratio_threshold, "mean_popped_ratio_threshold"),
            (
                mean_close_approach_ratio_threshold,
                "mean_close_approach_ratio_threshold",
            ),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

        self.num_balloons = int(num_balloons)
        self.eval_env = eval_env
        self.stages = []
        self.window_size = int(window_size)
        self.mean_popped_ratio_threshold = float(mean_popped_ratio_threshold)
        self.proximity_distance = float(proximity_distance)
        self.mean_close_approach_ratio_threshold = float(
            mean_close_approach_ratio_threshold
        )
        self.check_every_episodes = int(check_every_episodes)
        self.required_consecutive_passes = int(required_consecutive_passes)

        for name, pool_path in stages:
            path = Path(pool_path).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing curriculum pool '{name}': {path}. "
                    "Generate it with scripts/generate_balloon_pool.py first."
                )
            self.stages.append((str(name), path))

        self._current_stage_index = None
        self._popped_ratios = deque(maxlen=self.window_size)
        self._close_approach_ratios = deque(maxlen=self.window_size)
        self._mean_closest_distances = deque(maxlen=self.window_size)
        self._stage_episode_count = 0
        self._episodes_since_check = 0
        self._consecutive_passes = 0

    @property
    def _current_stage(self):
        return self.stages[self._current_stage_index]

    def _reset_stage_metrics(self):
        self._popped_ratios.clear()
        self._close_approach_ratios.clear()
        self._mean_closest_distances.clear()
        self._stage_episode_count = 0
        self._episodes_since_check = 0
        self._consecutive_passes = 0

    def _apply_stage(self, stage_index):
        name, pool_path = self.stages[stage_index]
        self.training_env.env_method("set_pool_path", str(pool_path))
        if self.eval_env is not None:
            self.eval_env.env_method("set_pool_path", str(pool_path))

        self._current_stage_index = stage_index
        self._reset_stage_metrics()
        if self.verbose:
            print(
                f"[Curriculum] timestep {self.num_timesteps}: level={name}, "
                f"pool={pool_path.name}"
            )

    def _on_training_start(self):
        self._apply_stage(0)

    def _record_completed_episodes(self):
        _, current_pool_path = self._current_stage
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if not done or info.get("curriculum_pool_path") != str(current_pool_path):
                continue

            popped_count = int(info.get("popped_count", 0))
            popped_ratio = min(max(popped_count / self.num_balloons, 0.0), 1.0)
            closest_distances = np.asarray(
                info.get("balloon_closest_distances", []), dtype=float
            ).reshape(-1)
            if closest_distances.size == self.num_balloons:
                close_count = int(
                    np.count_nonzero(closest_distances <= self.proximity_distance)
                )
                finite_distances = closest_distances[np.isfinite(closest_distances)]
                mean_closest_distance = (
                    float(np.mean(finite_distances))
                    if finite_distances.size
                    else 0.0
                )
            else:
                close_count = 0
                mean_closest_distance = 0.0
            # A popped balloon necessarily passed within the proximity radius,
            # even if its post-pop estimator state became NaN on that sim step.
            close_count = max(close_count, min(popped_count, self.num_balloons))
            close_approach_ratio = close_count / self.num_balloons

            self._popped_ratios.append(popped_ratio)
            self._close_approach_ratios.append(close_approach_ratio)
            self._mean_closest_distances.append(mean_closest_distance)
            self._stage_episode_count += 1
            self._episodes_since_check += 1

    def _maybe_advance(self):
        if self._current_stage_index >= len(self.stages) - 1:
            return
        # only if with enough popped_ratios, we can start checking for advancement
        if len(self._popped_ratios) < self.window_size:
            return
        if self._episodes_since_check < self.check_every_episodes:
            return

        self._episodes_since_check = 0
        mean_popped_ratio = sum(self._popped_ratios) / len(self._popped_ratios)
        mean_close_approach_ratio = sum(self._close_approach_ratios) / len(
            self._close_approach_ratios
        )
        tolerance = 1e-12
        passed = (
            mean_popped_ratio + tolerance >= self.mean_popped_ratio_threshold
            and mean_close_approach_ratio + tolerance
            >= self.mean_close_approach_ratio_threshold
        )
        self._consecutive_passes = self._consecutive_passes + 1 if passed else 0

        if self.verbose:
            status = "PASS" if passed else "WAIT"
            print(
                f"[Curriculum] {self._current_stage[0]} check: {status}, "
                f"mean_popped_ratio={mean_popped_ratio:.3f}, "
                f"mean_close_approach_ratio={mean_close_approach_ratio:.3f}, "
                f"consecutive={self._consecutive_passes}/"
                f"{self.required_consecutive_passes}"
            )

        if self._consecutive_passes >= self.required_consecutive_passes:
            self._apply_stage(self._current_stage_index + 1)

    def _on_step(self):
        self._record_completed_episodes()
        self._maybe_advance()

        mean_popped_ratio = (
            sum(self._popped_ratios) / len(self._popped_ratios)
            if self._popped_ratios
            else 0.0
        )
        mean_close_approach_ratio = (
            sum(self._close_approach_ratios) / len(self._close_approach_ratios)
            if self._close_approach_ratios
            else 0.0
        )
        mean_closest_distance = (
            sum(self._mean_closest_distances) / len(self._mean_closest_distances)
            if self._mean_closest_distances
            else 0.0
        )

        self.logger.record("curriculum/level", self._current_stage_index + 1)
        self.logger.record("curriculum/mean_popped_ratio", mean_popped_ratio)
        self.logger.record(
            "curriculum/mean_close_approach_ratio", mean_close_approach_ratio
        )
        self.logger.record(
            "curriculum/mean_close_approach_count",
            mean_close_approach_ratio * self.num_balloons,
        )
        self.logger.record(
            "curriculum/mean_closest_distance_m", mean_closest_distance
        )
        self.logger.record(
            "curriculum/proximity_distance_m", self.proximity_distance
        )
        self.logger.record("curriculum/stage_episodes", self._stage_episode_count)
        self.logger.record("curriculum/consecutive_passes", self._consecutive_passes)
        return True
