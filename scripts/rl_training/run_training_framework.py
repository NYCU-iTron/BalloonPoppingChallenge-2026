import copy
import argparse
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters
from BalloonPoppingGymEnv.utils.setup_logging import setup_logging
from BalloonPoppingGymEnv.balloon_generator import (
    BalloonSpawnConfig,
    ProgressiveBalloonGenerator,
)


@dataclass
class RLTrainingConfig:
    scenario_number: int = 1
    no_hit_timeout: float = 5.0
    max_episode_time: float = 120.0
    launch_time: float = 0.05
    render_mode: str | None = None
    seed: int = 0

    initial_balloon_distance: float = 25.0
    balloon_distance_increment: float = 5.0
    balloon_distance_jitter: float = 0.15
    balloon_axis_weights: tuple[float, float, float] = (1.0, 1.0, 0.6)


class BalloonPoppingRLWrapper:
    """Small RL-oriented wrapper around BalloonPoppingEnv.

    The wrapped environment always tracks one active balloon. When the rocket
    pops it, this wrapper returns reward=1 and respawns the next balloon around
    the current rocket position. An episode ends if no balloon is hit for
    no_hit_timeout seconds.
    """

    def __init__(self, scenario_parameters: dict, config: RLTrainingConfig):
        self.config = config
        self.scenario_parameters = self._make_single_balloon_scenario(scenario_parameters)
        self.generator = ProgressiveBalloonGenerator(
            BalloonSpawnConfig(
                initial_distance=config.initial_balloon_distance,
                distance_increment=config.balloon_distance_increment,
                distance_jitter=config.balloon_distance_jitter,
                axis_weights=config.balloon_axis_weights,
            ),
            seed=config.seed,
        )
        self.env = BalloonPoppingEnv(
            render_mode=config.render_mode,
            parameters=self.scenario_parameters,
        )
        self.last_hit_time = 0.0
        self.total_hits = 0

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

    def reset(self):
        self.generator.reset()
        self.last_hit_time = 0.0
        self.total_hits = 0
        observation, info = self.env.reset(seed=self.scenario_parameters["scenario"]["random_seed"])
        self._spawn_next_balloon(np.array([0.0, 0.0, self.scenario_parameters["environment"]["elevation"]]))
        return self.env._get_obs(), self._augment_info(info)

    def step(self, action):
        observation, _, base_terminated, truncated, info = self.env.step(action)
        sim_time = observation["simulation_time"]
        hit = int(np.any(self.env._balloon_status[:, 0] == 2))
        reward = float(hit)

        if hit:
            self.total_hits += hit
            self.last_hit_time = sim_time
            rocket_position = self._rocket_position_or_origin()
            self._spawn_next_balloon(rocket_position)
            observation = self.env._get_obs()

        timed_out = (sim_time - self.last_hit_time) >= self.config.no_hit_timeout
        max_time_reached = sim_time >= self.config.max_episode_time
        terminated = base_terminated or timed_out or max_time_reached

        info = self._augment_info(info)
        info["hit"] = bool(hit)
        info["no_hit_timeout"] = timed_out
        info["total_hits"] = self.total_hits
        info["target_distance"] = self.generator.current_distance()
        return observation, reward, terminated, truncated, info

    def _spawn_next_balloon(self, rocket_position):
        balloon_state = self.generator.next_balloon_state(rocket_position)
        current_step = min(self.env.current_step, self.env.num_timesteps - 1)

        self.env._balloon_flights[0, :, current_step:] = balloon_state[:, None]
        self.env._balloon_states[0, :] = balloon_state
        self.env._balloon_status[0, 0] = 1
        self.env._popped_count = 0
        self.env._balloon_release_at_step[:] = current_step

    def _rocket_position_or_origin(self):
        rocket_position = self.env._rocket_states[:3]
        if np.isfinite(rocket_position).all():
            return rocket_position
        return np.array([0.0, 0.0, self.scenario_parameters["environment"]["elevation"]])

    def _augment_info(self, info):
        out = dict(info)
        out["total_hits"] = self.total_hits
        out["spawn_index"] = self.generator.spawn_index
        out["target_distance"] = self.generator.current_distance()
        out["no_hit_timeout"] = False
        return out

    def _make_single_balloon_scenario(self, scenario_parameters):
        scenario = copy.deepcopy(scenario_parameters)
        # Use BalloonPoppingEnv's scenario-0 static balloon path to create the
        # single-balloon buffers in balloon_world.py. The RL wrapper then
        # rewrites that one live balloon on reset and after each hit.
        scenario["scenario"]["number"] = 0
        scenario["scenario"]["random_seed"] = self.config.seed
        scenario["simulation"]["max_time"] = max(
            scenario["simulation"]["max_time"],
            self.config.max_episode_time,
        )
        scenario["balloon"]["num"] = 1
        scenario["balloon"]["release_interval"] = 0.0
        scenario["balloon"].pop("training_points", None)
        return scenario


class MatplotlibRLTrainingRenderer:
    def __init__(self):
        plt.ion()
        self.fig = plt.figure(figsize=(12, 7), constrained_layout=True)
        layout = self.fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0])
        self.scene_ax = self.fig.add_subplot(layout[:, 0], projection="3d")
        self.reward_ax = self.fig.add_subplot(layout[0, 1])
        self.distance_ax = self.fig.add_subplot(layout[1, 1])

        self.time = []
        self.reward = []
        self.hits = []
        self.distance = []
        self.rocket_trail = []

    def reset(self, episode):
        self.time.clear()
        self.reward.clear()
        self.hits.clear()
        self.distance.clear()
        self.rocket_trail.clear()
        self.episode = episode

    def update(self, observation, info, episode_reward):
        sim_time = float(observation["simulation_time"])
        rocket_state = np.asarray(info["rocket_states"], dtype=float)
        balloon_state = np.asarray(observation["balloon_states"][0], dtype=float)
        rocket_position = rocket_state[:3]
        balloon_position = balloon_state[:3]

        self.time.append(sim_time)
        self.reward.append(episode_reward)
        self.hits.append(info["total_hits"])
        self.distance.append(info["target_distance"])

        if np.isfinite(rocket_position).all():
            self.rocket_trail.append(rocket_position.copy())

        self._draw_scene(rocket_position, balloon_position, info)
        self._draw_metrics()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def show_final(self):
        plt.ioff()
        plt.show()

    def _draw_scene(self, rocket_position, balloon_position, info):
        self.scene_ax.clear()
        self.scene_ax.set_title(
            f"Episode {self.episode} | hits={info['total_hits']} | "
            f"next distance={info['target_distance']:.1f} m"
        )
        self.scene_ax.set_xlabel("X (m)")
        self.scene_ax.set_ylabel("Y (m)")
        self.scene_ax.set_zlabel("Z (m)")

        points = []
        if np.isfinite(balloon_position).all():
            self.scene_ax.scatter(
                [balloon_position[0]],
                [balloon_position[1]],
                [balloon_position[2]],
                facecolors="none",
                edgecolors="red",
                marker="o",
                s=160,
                linewidths=2.0,
                label="Target balloon",
            )
            points.append(balloon_position)

        if np.isfinite(rocket_position).all():
            self.scene_ax.scatter(
                [rocket_position[0]],
                [rocket_position[1]],
                [rocket_position[2]],
                c="blue",
                marker="^",
                s=90,
                label="Rocket",
            )
            points.append(rocket_position)
            if np.isfinite(balloon_position).all():
                self.scene_ax.plot(
                    [rocket_position[0], balloon_position[0]],
                    [rocket_position[1], balloon_position[1]],
                    [rocket_position[2], balloon_position[2]],
                    color="gold",
                    linestyle="--",
                    linewidth=1.5,
                    label="Line of sight",
                )

        if self.rocket_trail:
            trail = np.asarray(self.rocket_trail)
            self.scene_ax.plot(
                trail[:, 0],
                trail[:, 1],
                trail[:, 2],
                color="navy",
                linewidth=1.0,
                alpha=0.6,
                label="Rocket trail",
            )
            points.append(trail[-1])

        self._set_equal_scene_limits(points)
        self.scene_ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.0), fontsize=8)

    def _set_equal_scene_limits(self, points):
        if not points:
            center = np.array([0.0, 0.0, 20.0])
            half = 50.0
        else:
            arr = np.asarray(points, dtype=float).reshape(-1, 3)
            center = (arr.min(axis=0) + arr.max(axis=0)) / 2.0
            half = max((arr.max(axis=0) - arr.min(axis=0)).max() / 2.0 + 20.0, 30.0)
        self.scene_ax.set_xlim(center[0] - half, center[0] + half)
        self.scene_ax.set_ylim(center[1] - half, center[1] + half)
        self.scene_ax.set_zlim(max(0.0, center[2] - half), center[2] + half)
        self.scene_ax.set_box_aspect((1, 1, 1))

    def _draw_metrics(self):
        self.reward_ax.clear()
        self.reward_ax.plot(self.time, self.reward, "g-", label="episode_reward")
        self.reward_ax.plot(self.time, self.hits, "b--", label="hits")
        self.reward_ax.set_xlabel("Time (s)")
        self.reward_ax.set_ylabel("Count")
        self.reward_ax.grid(True, alpha=0.3)
        self.reward_ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.0,
            frameon=True,
        )

        self.distance_ax.clear()
        self.distance_ax.plot(self.time, self.distance, "r-", label="next_target_distance")
        self.distance_ax.set_xlabel("Time (s)")
        self.distance_ax.set_ylabel("Distance (m)")
        self.distance_ax.grid(True, alpha=0.3)
        self.distance_ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.0,
            frameon=True,
        )


def random_policy(observation, action_space, config: RLTrainingConfig):
    t = observation["simulation_time"]
    launch = t >= config.launch_time
    spaces = action_space.spaces
    return {
        "launch": launch,
        "launch_inclination_heading": np.array([90.0, 0.0]),
        "tvc": spaces["tvc"].sample(),
        "roll": float(spaces["roll"].sample()),
        "throttle": 1.0,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run a lightweight RL training scaffold.")
    parser.add_argument("--scenario-number", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-hit-timeout", type=float, default=5.0)
    parser.add_argument("--max-episode-time", type=float, default=120.0)
    parser.add_argument("--initial-distance", type=float, default=25.0)
    parser.add_argument("--distance-increment", type=float, default=5.0)
    parser.add_argument("--distance-jitter", type=float, default=0.15)
    parser.add_argument("--axis-weights", type=float, nargs=3, default=(1.0, 1.0, 0.6))
    parser.add_argument("--render-mode", choices=("matplotlib", "vpython"), default=None)
    parser.add_argument("--no-plot", action="store_true", help="Disable live matplotlib training plots.")
    return parser.parse_args()


def run_training_demo():
    args = parse_args()
    config = RLTrainingConfig(
        scenario_number=args.scenario_number,
        no_hit_timeout=args.no_hit_timeout,
        max_episode_time=args.max_episode_time,
        render_mode=args.render_mode,
        seed=args.seed,
        initial_balloon_distance=args.initial_distance,
        balloon_distance_increment=args.distance_increment,
        balloon_distance_jitter=args.distance_jitter,
        balloon_axis_weights=tuple(args.axis_weights),
    )
    scenario_parameters, _ = load_scenario_parameters(config.scenario_number)
    env = BalloonPoppingRLWrapper(scenario_parameters, config)
    renderer = None if args.no_plot else MatplotlibRLTrainingRenderer()

    for episode in range(args.episodes):
        observation, info = env.reset()
        terminated = False
        episode_reward = 0.0
        if renderer is not None:
            renderer.reset(episode)
            renderer.update(observation, info, episode_reward)

        while not terminated:
            action = random_policy(observation, env.action_space, config)
            observation, reward, terminated, _, info = env.step(action)
            episode_reward += reward
            if renderer is not None:
                renderer.update(observation, info, episode_reward)

            print(
                "episode: "
                f"{episode} time: {observation['simulation_time']:.2f}s "
                f"reward: {episode_reward:.0f} "
                f"hits: {info['total_hits']} "
                f"next_distance: {info['target_distance']:.1f}m",
                end="\r",
            )

        print(
            "\nEpisode finished: "
            f"reward={episode_reward:.0f}, "
            f"hits={info['total_hits']}, "
            f"no_hit_timeout={info['no_hit_timeout']}"
        )
    if renderer is not None:
        renderer.show_final()


if __name__ == "__main__":
    setup_logging()
    run_training_demo()
