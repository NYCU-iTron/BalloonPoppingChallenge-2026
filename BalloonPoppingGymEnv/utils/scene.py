import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


class Scene:

    STATUS_COLORS = {0: "grey", 1: "magenta", 2: "red"}
    STATUS_LABELS = {0: "Balloon (ground)", 1: "Balloon (released)", 2: "Balloon (popped)"}

    def __init__(self):
        self.times = []
        self.balloon_positions = []
        self.balloon_status = []
        self.rocket_positions = []

    def update(self, observation, info):
        sim_time = observation.get("simulation_time", len(self.times))
        self.times.append(float(np.asarray(sim_time).item()))

        balloon_states = np.asarray(observation["balloon_states"], dtype=float)
        self.balloon_positions.append(balloon_states[:, :3].copy())

        status = np.asarray(observation["balloon_status"], dtype=int).reshape(-1)
        self.balloon_status.append(status.copy())

        rocket_states = np.asarray(info["rocket_states"], dtype=float)

        rocket_pos = np.full(3, np.nan)
        if rocket_states.size >= 3:
            rocket_pos = rocket_states[:3].copy()
        self.rocket_positions.append(rocket_pos)

    def draw(self, ax=None):
        if not self.times:
            print("[Warning] No simulation data recorded yet. Skipping plot.")
            return ax

        balloon_traj = np.stack(self.balloon_positions, axis=0)
        status_traj = np.stack(self.balloon_status, axis=0)
        rocket_traj = np.stack(self.rocket_positions, axis=0)
        _, num_balloons, _ = balloon_traj.shape

        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(projection="3d")

        # Draw balloons
        for b in range(num_balloons):
            xs, ys, zs = balloon_traj[:, b, 0], balloon_traj[:, b, 1], balloon_traj[:, b, 2]
            if np.ptp(xs) + np.ptp(ys) + np.ptp(zs) > 1e-6:
                ax.plot(xs, ys, zs, color="orchid", linewidth=1.0, alpha=0.6)
            final_status = int(status_traj[-1, b])
            ax.scatter(
                xs[-1], ys[-1], zs[-1],
                color=self.STATUS_COLORS.get(final_status, "magenta"),
                s=45, edgecolor="black", linewidth=0.4, depthshade=True,
            )

        # Draw rocket
        valid = ~np.isnan(rocket_traj).any(axis=1)
        if valid.any():
            rx, ry, rz = rocket_traj[valid, 0], rocket_traj[valid, 1], rocket_traj[valid, 2]
            ax.plot(rx, ry, rz, color="blue", linewidth=2.0)
            ax.scatter(rx[0], ry[0], rz[0], color="green", marker="^", s=70,
                       edgecolor="black", linewidth=0.4)
            ax.scatter(rx[-1], ry[-1], rz[-1], color="navy", marker="s", s=60,
                       edgecolor="black", linewidth=0.4)

        # Bounds and scaling logic
        all_pts = balloon_traj.reshape(-1, 3)
        if valid.any():
            all_pts = np.vstack([all_pts, rocket_traj[valid]])

        finite = all_pts[np.isfinite(all_pts).all(axis=1)]
        if finite.size:
            mins = np.min(finite, axis=0)
            maxs = np.max(finite, axis=0)
            mids = (mins + maxs) / 2.0

            ranges = maxs - mins
            max_range = np.max(ranges)
            max_range = max(max_range, 20.0)

            half_range = max_range / 2.0

            # Define the exact wall boundaries for projections
            x_min_bound = mids[0] - half_range
            y_min_bound = mids[1] - half_range
            z_min_bound = 0

            ax.set_xlim(x_min_bound, mids[0] + half_range)
            ax.set_ylim(y_min_bound, mids[1] + half_range)
            ax.set_zlim(z_min_bound, mids[2] + half_range)
            ax.set_box_aspect((1, 1, 1))

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title("Balloon & Rocket Trajectories")

        # Legend building
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

        plt.show()
        return ax
