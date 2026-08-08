"""Plot the airborne balloons in 2D and 3D at a given sim time,
including the fitted main axis line and the target path selected by Selector.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from BalloonPoppingGymEnv.agents.gnc.selector import Selector
from BalloonPoppingGymEnv.evaluation.evaluate import load_pool_parameters

SCRIPT_DIR = Path(__file__).resolve().parent

POOL_PATH = SCRIPT_DIR / "pool_scenario_1.npy"
EVAL_TIME = 60.0  # (sec) simulation time to slice at
SEED = 0  # seed for track sampling and release order

# Sequential blue ramp (light -> dark), keyed to time since release.
AGE_CMAP = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)
INK = "#0b0b0b"
INK_MUTED = "#52514e"


def sample_balloons(pool, eval_time, time_step, release_interval, num_balloons, seed):
    """Radius, altitude, age, full 6D states, and raw pool ID index map."""
    rng = np.random.default_rng(seed)
    track_idx = rng.choice(pool.shape[0], size=num_balloons, replace=False)
    release_steps = np.arange(num_balloons) * int(release_interval / time_step)
    rng.shuffle(release_steps)

    eval_step = int(round(eval_time / time_step))
    airborne = eval_step >= release_steps
    if not np.any(airborne):
        raise ValueError(f"No balloon has been released yet at t = {eval_time} sec.")

    source_idx = np.clip(eval_step - release_steps, 0, pool.shape[2] - 1)

    # 建立 (100, 6) 全局矩陣，未釋放者填入 NaN
    balloon_states = np.full((num_balloons, 6), np.nan)
    for i in range(num_balloons):
        if airborne[i]:
            balloon_states[i] = pool[track_idx[i], :6, source_idx[i]]

    airborne_indices = np.where(airborne)[0]
    states = balloon_states[airborne_indices]

    radius = np.hypot(states[:, 0], states[:, 1])
    altitude = states[:, 2]
    age = source_idx[airborne] * time_step

    return radius, altitude, age, balloon_states, airborne_indices


def get_fitted_slope(balloon_states):
    """計算 (r, z) 平面的擬合斜率"""
    valid_mask = ~np.isnan(balloon_states[:, 0])
    pos = balloon_states[valid_mask, :3]
    r = np.hypot(pos[:, 0], pos[:, 1])
    z = pos[:, 2]
    sum_r2 = np.sum(r**2)
    return np.sum(r * z) / sum_r2 if sum_r2 > 1e-6 else 1.0


def plot_balloons_2d(radius, altitude, age, eval_time, num_balloons, seed, slope, balloon_states, target_ids):
    fig, ax = plt.subplots(figsize=(8, 6.5))

    # 1. 畫所有氣球
    scatter = ax.scatter(
        radius, altitude, c=age, cmap=AGE_CMAP, s=70,
        edgecolors="white", linewidths=1.2, zorder=3, label="Airborne Balloons"
    )

    # 2. 畫 (r, z) 擬合斜直線
    r_max = np.max(radius) * 1.1
    r_line = np.linspace(0, r_max, 100)
    z_line = slope * r_line
    ax.plot(r_line, z_line, color="#e63946", linestyle="--", linewidth=2, zorder=4, label=f"Fitted Axis (slope={slope:.2f})")

    # 3. 標註 Selected Targets 與目標間的連線 (2D: r vs z)
    if target_ids is not None:
        target_positions = balloon_states[target_ids, :3]
        target_r = np.hypot(target_positions[:, 0], target_positions[:, 1])
        target_z = target_positions[:, 2]

        # 畫目標間連線
        ax.plot(target_r, target_z, color="#2a9d8f", linestyle="-", linewidth=2, zorder=5, label="Selected Path")

        # 標註目標點與順序編號
        ax.scatter(target_r, target_z, c="#2a9d8f", s=90, edgecolors="black", linewidths=1.2, zorder=6)
        for rank, (tr, tz) in enumerate(zip(target_r, target_z), 1):
            ax.annotate(f"T{rank}", (tr, tz), textcoords="offset points", xytext=(0, 6),
                        ha='center', fontsize=8, fontweight='bold', color="#2a9d8f", zorder=7)

    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("time since release (s)", color=INK_MUTED, fontsize=9)
    colorbar.outline.set_visible(False)

    ax.set_xlabel(r"horizontal range  $r=\sqrt{x^2+y^2}$  (m)", color=INK)
    ax.set_ylabel("altitude  z  (m)", color=INK)
    ax.set_title(
        f"2D Balloons & Path at t = {eval_time:g} s "
        f"({radius.size} of {num_balloons} airborne, seed {seed})",
        fontsize=12, color=INK, loc="left", pad=10,
    )
    ax.grid(True, color="#e6e5e2", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_balloons_3d(balloon_states, eval_time, seed, target_ids):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    valid_mask = ~np.isnan(balloon_states[:, 0])
    pos = balloon_states[valid_mask, :3]

    # 1. 繪製所有 active 氣球
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c="#3987e5", s=40, alpha=0.6, edgecolors="white", linewidths=0.5, label="Airborne Balloons")

    # 2. 繪製原點
    ax.scatter([0], [0], [0], c="black", s=100, marker="^", label="Origin")

    # 3. 計算並繪製 3D 擬合斜直線 (無 Projection 輔助線/點)
    vel = balloon_states[valid_mask, 3:]
    r = np.hypot(pos[:, 0], pos[:, 1])
    z = pos[:, 2]
    slope = np.sum(r * z) / np.sum(r**2)

    mean_vel_xy = np.mean(vel[:, :2], axis=0)
    dir_xy = mean_vel_xy / np.linalg.norm(mean_vel_xy)
    u = np.array([dir_xy[0], dir_xy[1], slope])
    u /= np.linalg.norm(u)

    max_s = np.max(np.dot(pos, u)) * 1.1
    line_s = np.linspace(0, max_s, 100)
    axis_line = np.outer(line_s, u)
    ax.plot(axis_line[:, 0], axis_line[:, 1], axis_line[:, 2], color="#e63946", linestyle="--", linewidth=2, label="3D Main Axis")

    # 4. 繪製選出的 10 顆目標與之間的連線 (原點 -> T1 -> T2 ... -> T10)
    if target_ids is not None:
        target_pts = balloon_states[target_ids, :3]
        path_pts = np.vstack([np.array([0, 0, 0]), target_pts])  # 含原點的打擊路線

        # 畫路徑連線
        ax.plot(path_pts[:, 0], path_pts[:, 1], path_pts[:, 2], color="#2a9d8f", linestyle="-", linewidth=2.5, zorder=5, label="Target Path")

        # 高亮目標氣球
        ax.scatter(target_pts[:, 0], target_pts[:, 1], target_pts[:, 2], c="#e63946", s=80, edgecolors="black", linewidths=1.2, zorder=6, label="Targets (T1-T10)")

        # 標註 T1 ~ T10 文字
        for rank, pt in enumerate(target_pts, 1):
            ax.text(pt[0], pt[1], pt[2] + 2, f"T{rank}", color="#1d3557", fontsize=9, fontweight='bold')

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"3D Selected Targets & Path (t = {eval_time:g} s, seed {seed})", pad=15)
    ax.legend(loc="upper left")

    ax.view_init(elev=20, azim=-60)


def main():
    scenario_parameters, given_parameters = load_pool_parameters()
    simulation = scenario_parameters["simulation"]
    balloon = scenario_parameters["balloon"]

    if not 0.0 <= EVAL_TIME <= simulation["max_time"]:
        raise ValueError(f"EVAL_TIME must lie in [0, {simulation['max_time']}] sec.")
    if not POOL_PATH.exists():
        raise FileNotFoundError(f"Trajectory pool '{POOL_PATH}' not found.")

    pool = np.load(POOL_PATH, mmap_mode="r")
    radius, altitude, age, balloon_states, airborne_indices = sample_balloons(
        pool, EVAL_TIME, simulation["time_step"], balloon["release_interval"], balloon["num"], SEED
    )

    # 1. 實例化 Selector 並選擇目標
    selector = Selector(given_parameters)
    target_ids = selector.select_targets(balloon_states)

    if target_ids is not None:
        print(f"[{EVAL_TIME:g}s EVAL] Selected Target IDs (T1 -> T10): {target_ids}")
    else:
        print(f"[{EVAL_TIME:g}s EVAL] Failed to select 10 targets.")

    slope = get_fitted_slope(balloon_states)

    # 2. 繪製 2D 圖與 3D 圖
    plot_balloons_2d(radius, altitude, age, EVAL_TIME, balloon["num"], SEED, slope, balloon_states, target_ids)
    plot_balloons_3d(balloon_states, EVAL_TIME, SEED, target_ids)

    plt.show()


if __name__ == "__main__":
    main()
