from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    target_file = script_dir / "pool_scenario_1.npy"

    num_tracks = 30

    tracks = np.load(str(target_file), mmap_mode='r')
    total_tracks = tracks.shape[0]

    # Cap requested tracks if pool is smaller
    num_tracks = min(num_tracks, total_tracks)

    # Randomly select tracks to visualize
    sampled_indices = np.random.choice(total_tracks, size=num_tracks, replace=False)

    # Set up 3D plotting canvas
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    all_points = []

    for idx in sampled_indices:
        # Extract kinematic positional trajectories
        x = tracks[idx, 0, :]
        y = tracks[idx, 1, :]
        z = tracks[idx, 2, :]

        # Plot continuous flight path
        ax.plot(x, y, z, linewidth=1.5, alpha=0.8, label=f"Track #{idx}")

        # Mark Launch Point
        ax.scatter(x[0], y[0], z[0], color='green', marker='o', s=30, alpha=0.9)

        # Mark Final Point
        ax.scatter(x[-1], y[-1], z[-1], color='red', marker='x', s=30, alpha=0.9)

        # Collect points for bounds calculation
        all_points.append(np.column_stack((x, y, z)))

    # Set equal aspect ratio
    all_pts_array = np.vstack(all_points)
    finite = all_pts_array[np.isfinite(all_pts_array).all(axis=1)]
    if finite.size > 0:
        mins = np.min(finite, axis=0)
        maxs = np.max(finite, axis=0)
        mids = (mins + maxs) / 2.0

        ranges = maxs - mins
        max_range = np.max(ranges)
        max_range = max(max_range, 20.0)

        half_range = max_range / 2.0

        ax.set_xlim(mids[0] - half_range, mids[0] + half_range)
        ax.set_ylim(mids[1] - half_range, mids[1] + half_range)
        ax.set_zlim(mids[2] - half_range, mids[2] + half_range)
        ax.set_box_aspect((1, 1, 1))

    # Configure axes labels and layout details
    ax.set_xlabel('X (m)', labelpad=10)
    ax.set_ylabel('Y (m)', labelpad=10)
    ax.set_zlabel('Z (m)', labelpad=10)
    ax.set_title(f"Balloon Flight Trajectories", fontsize=14, pad=15)

    ax.scatter([], [], [], color='green', marker='o', label='Launch Origin')
    ax.scatter([], [], [], color='red', marker='x', label='Terminal Point')

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()
