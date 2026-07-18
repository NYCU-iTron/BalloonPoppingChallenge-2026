import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_balloon_tracks(file_path, num_tracks_to_plot=5):
    """
    Loads a memory-mapped trajectory array and plots a subset of tracks in 3D.
    Expected array shape: (total_tracks, 6, num_timesteps)
    State mapping: 0:X, 1:Y, 2:Z, 3:Vx, 4:Vy, 5:Vz
    """
    print(f"Loading dataset from '{file_path}'...")
    # Use mmap_mode='r' to prevent loading the entire multi-GB file into RAM
    tracks = np.load(file_path, mmap_mode='r')
    total_tracks = tracks.shape[0]

    print(f"Dataset successfully mapped. Shape: {tracks.shape}")

    # Cap requested tracks if pool is smaller
    num_tracks_to_plot = min(num_tracks_to_plot, total_tracks)

    # Randomly select tracks to visualize
    sampled_indices = np.random.choice(total_tracks, size=num_tracks_to_plot, replace=False)
    print(f"Selected track indices for visualization: {sampled_indices}")

    # Set up 3D plotting canvas
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    for idx in sampled_indices:
        # Extract kinematic positional trajectories
        x = tracks[idx, 0, :]
        y = tracks[idx, 1, :]
        z = tracks[idx, 2, :]

        # Plot continuous flight path
        line = ax.plot(x, y, z, linewidth=1.5, alpha=0.8, label=f"Track #{idx}")

        # Mark Launch Point (Green dot)
        ax.scatter(x[0], y[0], z[0], color='green', marker='o', s=30, alpha=0.9)
        # Mark Final Point (Red cross)
        ax.scatter(x[-1], y[-1], z[-1], color='red', marker='x', s=30, alpha=0.9)

    # Configure axes labels and layout details
    ax.set_xlabel('X Position (meters)', labelpad=10)
    ax.set_ylabel('Y Position (meters)', labelpad=10)
    ax.set_zlabel('Z Position (meters)', labelpad=10)
    ax.set_title(f"3D Kinematic Flight Trajectories\nSource: {file_path}", fontsize=14, pad=15)

    # Add aesthetic reference marker legends
    ax.scatter([], [], [], color='green', marker='o', label='Launch Origin')
    ax.scatter([], [], [], color='red', marker='x', label='Terminal Point')
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))

    plt.grid(True, linestyle='--', alpha=0.5)

    # Export high-res visualization plot onto disk
    # output_img = file_path.replace('.npy', '_plot.png')
    # plt.savefig(output_img, dpi=300, bbox_inches='tight')
    # print(f"[SUCCESS] Visualization figure compiled at: '{output_img}'")
    plt.show()

if __name__ == "__main__":
    script_dir = Path(__file__).parent
    target_file = str(script_dir / "pool_level_1_easy.npy")

    if not os.path.exists(target_file):
        print(f"[ERROR] Target file '{target_file}' not found. Please run your generator script first.")
    else:
        plot_balloon_tracks(target_file, num_tracks_to_plot=5)
