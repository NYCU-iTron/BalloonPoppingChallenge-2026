"""Development environment with an exact on-disk balloon trajectory cache."""

import copy
import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path

import numpy as np

from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv


logger = logging.getLogger(__name__)

_CACHE_SCHEMA_VERSION = 1
_REQUIRED_CACHE_FIELDS = {
    "schema_version",
    "parameter_fingerprint",
    "random_seed",
    "release_steps",
    "balloon_flights",
}


def scenario_parameter_fingerprint(parameters: dict) -> str:
    """Return a stable physics fingerprint, with the reset seed kept separate."""
    normalized = copy.deepcopy(parameters)
    normalized.get("scenario", {}).pop("random_seed", None)
    encoded = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class CachedBalloonEnv(BalloonPoppingEnv):
    """Run production physics while caching Scenario Monte Carlo trajectories.

    The first ``reset(seed=...)`` uses the normal environment generator and
    writes its fully release-shifted balloon trajectories. Later resets with
    the same seed and scenario parameters load those arrays instead. Rocket
    simulation, observations, pop detection, and all stepping remain on the
    production environment path.

    This adapter is development tooling. Official evaluation deliberately
    continues to construct :class:`BalloonPoppingEnv` directly.
    """

    def __init__(self, render_mode, parameters, *, cache_dir):
        self.trajectory_cache_dir = Path(cache_dir)
        self.parameter_fingerprint = scenario_parameter_fingerprint(parameters)
        self.last_trajectory_cache_path: Path | None = None
        self.last_trajectory_cache_hit = False
        super().__init__(render_mode=render_mode, parameters=parameters)

    def trajectory_cache_path(self, seed: int) -> Path:
        """Path used for one scenario/seed/parameter combination."""
        scenario = int(self.scenario_parameters["number"])
        fingerprint = self.parameter_fingerprint[:16]
        return self.trajectory_cache_dir / (
            f"scenario_{scenario}_seed_{int(seed)}_{fingerprint}.npz"
        )

    def _BalloonPoppingEnv__generate_balloon_flights(self) -> None:
        """Load a matching cache entry or run and preserve Monte Carlo once."""
        seed = int(self.np_random_seed)
        cache_path = self.trajectory_cache_path(seed)
        self.last_trajectory_cache_path = cache_path

        if cache_path.is_file():
            self._load_trajectory_cache(cache_path, seed)
            self.last_trajectory_cache_hit = True
            logger.info("Loaded balloon trajectory cache: %s", cache_path)
            return

        self.last_trajectory_cache_hit = False
        BalloonPoppingEnv._BalloonPoppingEnv__generate_balloon_flights(self)
        self._write_trajectory_cache(cache_path, seed)
        logger.info("Stored balloon trajectory cache: %s", cache_path)

    def _load_trajectory_cache(self, cache_path: Path, seed: int) -> None:
        try:
            with np.load(cache_path, allow_pickle=False) as cached:
                missing = _REQUIRED_CACHE_FIELDS.difference(cached.files)
                if missing:
                    raise ValueError(f"missing fields: {sorted(missing)}")

                schema_version = int(cached["schema_version"].item())
                fingerprint = str(cached["parameter_fingerprint"].item())
                cached_seed = int(str(cached["random_seed"].item()))
                release_steps = np.asarray(cached["release_steps"], dtype=int)
                flights = np.asarray(cached["balloon_flights"])
        except (OSError, ValueError, KeyError) as exc:
            raise ValueError(
                f"Invalid balloon trajectory cache {cache_path}; delete it and "
                "generate it again"
            ) from exc

        if schema_version != _CACHE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported trajectory cache schema {schema_version} in "
                f"{cache_path}; expected {_CACHE_SCHEMA_VERSION}"
            )
        if fingerprint != self.parameter_fingerprint or cached_seed != seed:
            raise ValueError(f"Trajectory cache metadata does not match {cache_path}")

        expected_shape = self._expected_trajectory_shape()
        if flights.shape != expected_shape:
            raise ValueError(
                f"Trajectory cache shape mismatch in {cache_path}: expected "
                f"{expected_shape}, received {flights.shape}"
            )

        current_release_steps = np.asarray(self._balloon_release_at_step, dtype=int)
        if not np.array_equal(release_steps, current_release_steps):
            raise ValueError(
                f"Trajectory cache release schedule does not match seed {seed}: "
                f"{cache_path}"
            )

        self._balloon_flights = flights

    def _write_trajectory_cache(self, cache_path: Path, seed: int) -> None:
        flights = np.asarray(self._balloon_flights)
        expected_shape = self._expected_trajectory_shape()
        if flights.shape != expected_shape:
            raise ValueError(
                f"Refusing to cache trajectories shaped {flights.shape}; "
                f"expected {expected_shape}"
            )

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{cache_path.stem}_",
            suffix=".tmp",
            dir=cache_path.parent,
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as temporary_file:
                np.savez_compressed(
                    temporary_file,
                    schema_version=np.array(_CACHE_SCHEMA_VERSION, dtype=np.int64),
                    parameter_fingerprint=np.array(self.parameter_fingerprint),
                    random_seed=np.array(str(seed)),
                    release_steps=np.asarray(
                        self._balloon_release_at_step, dtype=np.int64
                    ),
                    balloon_flights=flights,
                )
            os.replace(temporary_path, cache_path)
        finally:
            temporary_path.unlink(missing_ok=True)

    def _expected_trajectory_shape(self) -> tuple[int, int, int]:
        num_timesteps = len(
            np.arange(
                0,
                self.simulation_parameters["max_time"],
                self.simulation_parameters["time_step"],
            )
        )
        return (int(self.balloon_parameters["num"]), 6, num_timesteps)
