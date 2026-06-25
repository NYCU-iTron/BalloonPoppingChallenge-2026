"""Collect per-step states during a rollout and draw the 3D trajectories.

Usage
-----
    from render_scene import RenderScene

    render_scene = RenderScene()
    observation, info = env.reset(seed=...)
    render_scene.get_ob(observation, info)        # optional: capture t = 0

    terminated = False
    while not terminated:
        action = agent.get_action(observation)
        observation, reward, terminated, _, info = env.step(action)
        render_scene.get_ob(observation, info)    # store each step

    render_scene.draw()                           # plot after the loop

Balloon positions/status come from ``observation`` (``balloon_states`` /
``balloon_status``); the rocket trajectory comes from ``info["rocket_states"]``.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


class RenderScene:
    """Accumulate observation/info frames and render them as 3D trajectories."""

    # status code -> colour, matching the env's render conventions
    STATUS_COLORS = {0: "grey", 1: "magenta", 2: "red"}
    STATUS_LABELS = {0: "Balloon (ground)", 1: "Balloon (released)", 2: "Balloon (popped)"}

    def __init__(self):
        # one entry appended per get_ob() call
        self.times = []              # list of float
        self.balloon_positions = []  # list of (num, 3) arrays
        self.balloon_status = []     # list of (num,) int arrays
        self.rocket_positions = []   # list of (3,) arrays (NaN before launch)

    def get_ob(self, observation, info=None):
        """Store one frame of balloon (from observation) and rocket (from info) state.

        Parameters
        ----------
        observation : dict
            Env observation. Uses ``balloon_states`` (num, 6) and
            ``balloon_status`` (num, 1); optionally ``simulation_time``.
        info : dict, optional
            Env info. Uses ``rocket_states`` (13,) whose first three entries are
            the rocket position. If omitted, the rocket position is recorded as
            NaN for this frame.
        """
        # simulation time (fall back to a running index if not present)
        sim_time = observation.get("simulation_time", len(self.times))
        self.times.append(float(np.asarray(sim_time).item()))

        # balloon positions: (num, 6) -> keep x, y, z
        balloon_states = np.asarray(observation["balloon_states"], dtype=float)
        self.balloon_positions.append(balloon_states[:, :3].copy())

        # balloon status: (num, 1) -> (num,)
        status = np.asarray(observation["balloon_status"], dtype=int).reshape(-1)
        self.balloon_status.append(status.copy())

        # rocket position from info["rocket_states"][:3]; NaN if unavailable
        rocket_pos = np.full(3, np.nan)
        if info is not None and info.get("rocket_states") is not None:
            rocket_states = np.asarray(info["rocket_states"], dtype=float)
            if rocket_states.size >= 3:
                rocket_pos = rocket_states[:3].copy()
        self.rocket_positions.append(rocket_pos)

    def draw(self, show=True, save_path=None, ax=None, equal_aspect=False):
        """Plot the collected balloon and rocket trajectories in 3D.

        Parameters
        ----------
        show : bool
            Call ``plt.show()`` when done.
        save_path : str, optional
            If given, save the figure to this path.
        ax : mpl_toolkits.mplot3d.axes3d.Axes3D, optional
            Draw into an existing 3D axis instead of creating a new figure.
        equal_aspect : bool
            If True, scale the three axes to the true data ranges (1:1:1). This
            is physically faithful but, when altitude dwarfs the horizontal
            spread, produces a thin sliver. Default False lets matplotlib
            auto-scale each axis for readability.

        Returns
        -------
        ax : the 3D axis the scene was drawn on.
        """
        if not self.times:
            raise RuntimeError(
                "No frames recorded. Call get_ob() inside the loop before draw()."
            )

        balloon_traj = np.stack(self.balloon_positions, axis=0)  # (T, num, 3)
        status_traj = np.stack(self.balloon_status, axis=0)      # (T, num)
        rocket_traj = np.stack(self.rocket_positions, axis=0)    # (T, 3)
        _, num_balloons, _ = balloon_traj.shape

        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(projection="3d")

        # --- balloons: faint path + final position coloured by final status ---
        for b in range(num_balloons):
            xs, ys, zs = balloon_traj[:, b, 0], balloon_traj[:, b, 1], balloon_traj[:, b, 2]
            # path is only meaningful if the balloon actually moves
            if np.ptp(xs) + np.ptp(ys) + np.ptp(zs) > 1e-6:
                ax.plot(xs, ys, zs, color="orchid", linewidth=1.0, alpha=0.6)
            final_status = int(status_traj[-1, b])
            ax.scatter(
                xs[-1], ys[-1], zs[-1],
                color=self.STATUS_COLORS.get(final_status, "magenta"),
                s=45, edgecolor="black", linewidth=0.4, depthshade=True,
            )

        # --- rocket: drop pre-launch NaN frames, then draw the path ---
        valid = ~np.isnan(rocket_traj).any(axis=1)
        if valid.any():
            rx, ry, rz = rocket_traj[valid, 0], rocket_traj[valid, 1], rocket_traj[valid, 2]
            ax.plot(rx, ry, rz, color="blue", linewidth=2.0)
            ax.scatter(rx[0], ry[0], rz[0], color="green", marker="^", s=70,
                       edgecolor="black", linewidth=0.4)
            ax.scatter(rx[-1], ry[-1], rz[-1], color="navy", marker="s", s=60,
                       edgecolor="black", linewidth=0.4)

        # --- optional 1:1:1 scaling using data ranges ---
        if equal_aspect:
            all_pts = balloon_traj.reshape(-1, 3)
            if valid.any():
                all_pts = np.vstack([all_pts, rocket_traj[valid]])
            finite = all_pts[np.isfinite(all_pts).all(axis=1)]
            if finite.size:
                ranges = np.ptp(finite, axis=0)
                ranges = np.where(ranges < 1e-6, 1.0, ranges)
                ax.set_box_aspect(ranges)

        ax.set_xlabel("X / East (m)")
        ax.set_ylabel("Y / North (m)")
        ax.set_zlabel("Z / Up (m)")
        ax.set_title("Balloon & Rocket Trajectories")

        # --- legend (proxy handles for status colours + rocket markers) ---
        present_status = sorted(set(int(s) for s in status_traj[-1]))
        legend_handles = [Line2D([0], [0], color="orchid", lw=1.5, label="Balloon path")]
        for s in present_status:
            legend_handles.append(
                Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=self.STATUS_COLORS.get(s, "magenta"),
                       markeredgecolor="black", markersize=8,
                       label=self.STATUS_LABELS.get(s, f"Balloon ({s})"))
            )
        if valid.any():
            legend_handles += [
                Line2D([0], [0], color="blue", lw=2, label="Rocket path"),
                Line2D([0], [0], marker="^", color="w", markerfacecolor="green",
                       markeredgecolor="black", markersize=10, label="Launch"),
                Line2D([0], [0], marker="s", color="w", markerfacecolor="navy",
                       markeredgecolor="black", markersize=9, label="Rocket end"),
            ]
        ax.legend(handles=legend_handles, loc="upper left", fontsize=8)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        return ax
