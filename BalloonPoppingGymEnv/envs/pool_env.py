"""Balloon-pool adapter for the production simulation environment."""

import numpy as np

from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv


class PoolEnv(BalloonPoppingEnv):
    """Run current rocket physics with externally supplied balloon tracks.

    ``BalloonPoppingEnv`` remains authoritative for actions, actuators, sensors,
    pop detection, rendering, and episode lifecycle. This adapter replaces only
    its Monte Carlo balloon generation step with trajectories sampled from the
    development pool.
    """

    state_dims = 6

    def __init__(self, render_mode, parameters):
        self._raw_source_trajectories = None
        super().__init__(render_mode=render_mode, parameters=parameters)

    def update_source_trajectories(self, raw_trajectories: np.ndarray) -> None:
        """Inject unshifted tracks before calling :meth:`reset`.

        Parameters
        ----------
        raw_trajectories
            Array shaped ``(scenario balloon count, 6, source timesteps)``.
        """
        trajectories = np.asarray(raw_trajectories)
        expected_prefix = (self.balloon_parameters["num"], self.state_dims)
        if trajectories.ndim != 3 or trajectories.shape[:2] != expected_prefix:
            raise ValueError(
                "Trajectory shape mismatch: expected "
                f"({expected_prefix[0]}, {expected_prefix[1]}, timesteps), "
                f"received {trajectories.shape}."
            )
        if trajectories.shape[2] < 1:
            raise ValueError(
                "Trajectory pool tracks must contain at least one timestep."
            )
        self._raw_source_trajectories = trajectories

    def _BalloonPoppingEnv__generate_balloon_flights(self) -> None:
        """Replace Monte Carlo generation while preserving release timing."""
        if self._raw_source_trajectories is None:
            raise RuntimeError(
                "No pool trajectories supplied. Call "
                "env.update_source_trajectories(tracks) before env.reset()."
            )

        source = self._raw_source_trajectories
        num_balloons, state_dims, source_steps = source.shape
        episode_steps = len(
            np.arange(
                0,
                self.simulation_parameters["max_time"],
                self.simulation_parameters["time_step"],
            )
        )

        release_steps = np.asarray(self._balloon_release_at_step, dtype=int)
        release_steps = np.clip(release_steps, 0, episode_steps)
        time_index = np.arange(episode_steps)
        source_index = np.clip(
            time_index[np.newaxis, :] - release_steps[:, np.newaxis],
            0,
            source_steps - 1,
        )

        balloon_index = np.arange(num_balloons)[:, None, None]
        state_index = np.arange(state_dims)[None, :, None]
        shifted = source[
            balloon_index,
            state_index,
            source_index[:, np.newaxis, :],
        ]

        before_release = time_index < release_steps[:, np.newaxis]
        self._balloon_flights = np.where(
            before_release[:, np.newaxis, :],
            source[:, :, :1],
            shifted,
        )
