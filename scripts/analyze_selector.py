"""Track the 10 selected target balloons over time.

1. Select 10 targets at T_SELECT = 60.0s using Selector.
2. Animate the 3D scene over sim time so the targets' positions can be watched
   as they drift: all balloons, the 10 targets with trails, the origin -> T1 ->
   ... -> T10 chain, and the fitted main axis.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from BalloonPoppingGymEnv.agents.gnc.selector import Selector
from BalloonPoppingGymEnv.evaluation.evaluate import load_pool_parameters

SCRIPT_DIR = Path(__file__).resolve().parent

POOL_PATH = SCRIPT_DIR / "pool_scenario_1.npy"
T_SELECT = 60.0  # (sec) time to run Selector and pick targets
SEED = 0  # seed for track sampling and release order

ANIM_START = 0.0  # (sec) 動畫起始時間
ANIM_END = None  # (sec) 動畫結束時間，None = max_time
ANIM_STRIDE = 0.5  # (sec) 動畫每幀前進的模擬時間
ANIM_FPS = 20  # 動畫播放幀率
TRAIL_SECONDS = 15.0  # (sec) 動畫中目標尾跡長度
SAVE_ANIM = None  # 例如 SCRIPT_DIR / "selector_targets.gif"，None = 不存檔

INK = "#0b0b0b"
ORIGIN = np.zeros(3)


def sample_full_tracks(pool, time_step, release_interval, num_balloons, seed):
    """Sample balloon trajectories and their (shuffled) release steps."""
    rng = np.random.default_rng(seed)
    track_idx = rng.choice(pool.shape[0], size=num_balloons, replace=False)
    release_steps = np.arange(num_balloons) * int(release_interval / time_step)
    rng.shuffle(release_steps)
    return np.asarray(pool[track_idx]), release_steps


def get_balloon_states_at_time(sampled_pool, release_steps, eval_time, time_step):
    """Extract (N, 6) balloon states at a specific evaluation time."""
    num_balloons = sampled_pool.shape[0]
    eval_step = int(round(eval_time / time_step))
    airborne = eval_step >= release_steps

    balloon_states = np.full((num_balloons, 6), np.nan)
    if not np.any(airborne):
        return balloon_states

    source_idx = np.clip(eval_step - release_steps, 0, sampled_pool.shape[2] - 1)
    for i in range(num_balloons):
        if airborne[i]:
            balloon_states[i] = sampled_pool[i, :6, source_idx[i]]

    return balloon_states


def get_positions_at_steps(sampled_pool, release_steps, frame_steps):
    """(F, N, 3) positions of every balloon at the given sim steps; NaN if not airborne."""
    num_balloons, _, total_steps = sampled_pool.shape
    positions = np.full((len(frame_steps), num_balloons, 3), np.nan)

    for i in range(num_balloons):
        source_idx = frame_steps - release_steps[i]
        active = (source_idx >= 0) & (source_idx < total_steps)
        if not np.any(active):
            continue
        # 進階索引 (scalar, slice, array) -> 結果形狀為 (m, 3)
        positions[active, i] = sampled_pool[i, :3, source_idx[active]]

    return positions


def get_target_trajectories(sampled_pool, release_steps, target_ids):
    """(K, total_steps, 3) target positions on the global sim clock; NaN before release."""
    total_steps = sampled_pool.shape[2]
    trajectories = np.full((len(target_ids), total_steps, 3), np.nan)

    for k, target_id in enumerate(target_ids):
        rel_step = int(release_steps[target_id])
        if rel_step >= total_steps:
            continue
        num_active = total_steps - rel_step
        trajectories[k, rel_step:] = sampled_pool[target_id, :3, :num_active].T

    return trajectories


def fit_main_axis(balloon_states):
    """3D main axis unit vector u from the (r, z) slope + mean horizontal velocity."""
    valid_mask = ~np.isnan(balloon_states[:, 0])
    if np.count_nonzero(valid_mask) < 2:
        return None

    pos = balloon_states[valid_mask, :3]
    vel = balloon_states[valid_mask, 3:]

    r = np.hypot(pos[:, 0], pos[:, 1])
    sum_r2 = np.sum(r**2)
    slope = np.sum(r * pos[:, 2]) / sum_r2 if sum_r2 > 1e-6 else 1.0

    mean_vel_xy = np.mean(vel[:, :2], axis=0)
    vel_norm = np.linalg.norm(mean_vel_xy)
    dir_xy = mean_vel_xy / vel_norm if vel_norm > 1e-5 else np.array([1.0, 0.0])

    axis = np.array([dir_xy[0], dir_xy[1], slope])
    return axis / np.linalg.norm(axis)


def set_equal_3d_box(ax, positions, margin=0.05):
    """Lock the axes to the data bounds; X and Y share one common span.

    X and Y are both widened to the larger of the two spans (each kept centred on
    its own data), so horizontal distances read the same on both axes.
    """
    flat = positions.reshape(-1, 3)
    flat = flat[~np.isnan(flat[:, 0])]
    lo = np.minimum(flat.min(axis=0), ORIGIN)
    hi = np.maximum(flat.max(axis=0), ORIGIN)

    span = np.maximum(hi - lo, 1.0) * (1.0 + 2.0 * margin)
    span[:2] = span[:2].max()  # X、Y 使用一致的範圍
    center = (lo + hi) / 2.0
    lo, hi = center - span / 2.0, center + span / 2.0

    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi - lo)


def animate_targets_3d(
    sampled_pool, release_steps, trajectories, target_ids, t_select, time_step, max_time
):
    """Animate the 3D scene: all balloons, the 10 targets, and their chain over time."""
    anim_end = max_time if ANIM_END is None else ANIM_END
    start_step = int(round(ANIM_START / time_step))
    end_step = min(int(round(anim_end / time_step)), sampled_pool.shape[2] - 1)
    stride = max(1, int(round(ANIM_STRIDE / time_step)))
    frame_steps = np.arange(start_step, end_step + 1, stride)

    all_positions = get_positions_at_steps(sampled_pool, release_steps, frame_steps)
    trail_steps = int(round(TRAIL_SECONDS / time_step))

    fig = plt.figure(figsize=(9.5, 7))
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.get_cmap("tab10")

    ax.scatter(*ORIGIN, c="black", s=100, marker="^", label="Origin")

    others = ax.scatter([], [], [], c="#b8c6d6", s=18, alpha=0.5, label="Other balloons")
    axis_line, = ax.plot([], [], [], color="#e63946", linestyle="--", linewidth=1.8, label="Main axis")
    chain_line, = ax.plot([], [], [], color="#2a9d8f", linewidth=2.2, label="Target chain")
    target_dots = ax.scatter([], [], [], c="#1d3557", s=70, edgecolors="black", linewidths=1.0, zorder=6, label="Targets T1-T10")

    trails = [
        ax.plot([], [], [], color=cmap(rank % 10), linewidth=1.5, alpha=0.9)[0]
        for rank in range(len(target_ids))
    ]
    labels = [
        ax.text(0, 0, 0, "", color=cmap(rank % 10), fontsize=8, fontweight="bold")
        for rank in range(len(target_ids))
    ]
    time_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes, fontsize=11, color=INK)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"Selected Targets in 3D over Time (selected at t = {t_select:g} s)", pad=15)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    set_equal_3d_box(ax, np.concatenate([all_positions.reshape(-1, 3), trajectories.reshape(-1, 3)]))
    ax.view_init(elev=20, azim=-60)
    fig.tight_layout()

    def update(frame):
        step = frame_steps[frame]
        sim_time = step * time_step

        # 1. 非目標氣球
        frame_pos = all_positions[frame]
        mask = ~np.isnan(frame_pos[:, 0])
        mask[list(target_ids)] = False
        other_pts = frame_pos[mask]
        others._offsets3d = (other_pts[:, 0], other_pts[:, 1], other_pts[:, 2])

        # 2. 目前的主軸方向 (由當下所有氣球擬合)
        states = get_balloon_states_at_time(sampled_pool, release_steps, sim_time, time_step)
        u = fit_main_axis(states)
        if u is not None:
            valid = ~np.isnan(states[:, 0])
            max_s = max(np.max(states[valid, :3] @ u) * 1.05, 1.0)
            seg = np.outer([0.0, max_s], u)
            axis_line.set_data(seg[:, 0], seg[:, 1])
            axis_line.set_3d_properties(seg[:, 2])

        # 3. 目標尾跡、目前位置與打擊鏈
        pts = trajectories[:, step, :]
        for rank, trail in enumerate(trails):
            lo = max(0, step - trail_steps)
            seg = trajectories[rank, lo : step + 1]
            seg = seg[~np.isnan(seg[:, 0])]
            trail.set_data(seg[:, 0], seg[:, 1])
            trail.set_3d_properties(seg[:, 2])

            pt = pts[rank]
            if np.isnan(pt[0]):
                labels[rank].set_text("")
            else:
                labels[rank].set_position_3d((pt[0], pt[1], pt[2] + 3))
                labels[rank].set_text(f"T{rank + 1}")

        live = pts[~np.isnan(pts[:, 0])]
        target_dots._offsets3d = (live[:, 0], live[:, 1], live[:, 2])

        chain = np.vstack([ORIGIN, pts]) if not np.any(np.isnan(pts[:, 0])) else np.empty((0, 3))
        chain_line.set_data(chain[:, 0], chain[:, 1])
        chain_line.set_3d_properties(chain[:, 2])

        airborne = int(np.count_nonzero(~np.isnan(frame_pos[:, 0])))
        time_text.set_text(f"t = {sim_time:6.2f} s   ({airborne} airborne)")

        return [others, axis_line, chain_line, target_dots, time_text, *trails, *labels]

    anim = FuncAnimation(
        fig, update, frames=len(frame_steps), interval=1000 / ANIM_FPS, blit=False, repeat=True
    )

    if SAVE_ANIM is not None:
        anim.save(str(SAVE_ANIM), fps=ANIM_FPS)
        print(f"Animation saved to {SAVE_ANIM}")

    return anim


def main():
    scenario_parameters, _ = load_pool_parameters()
    simulation = scenario_parameters["simulation"]
    balloon = scenario_parameters["balloon"]

    time_step = simulation["time_step"]
    max_time = simulation["max_time"]

    if not 0.0 <= T_SELECT <= max_time:
        raise ValueError(f"T_SELECT ({T_SELECT}) must lie in [0, {max_time}] sec.")
    if not POOL_PATH.exists():
        raise FileNotFoundError(f"Trajectory pool '{POOL_PATH}' not found.")

    pool = np.load(POOL_PATH, mmap_mode="r")

    # 1. 採樣整場 episode 的氣球軌跡數據
    sampled_pool, release_steps = sample_full_tracks(
        pool, time_step, balloon["release_interval"], balloon["num"], SEED
    )

    # 2. 在 t = T_SELECT 時獲取氣球狀態並執行 Selector 鎖定 10 顆目標
    select_states = get_balloon_states_at_time(sampled_pool, release_steps, T_SELECT, time_step)

    selector = Selector()
    target_ids = selector.select_targets(select_states)

    if target_ids is None:
        print(f"[{T_SELECT:g}s] Failed to select 10 targets.")
        return

    print(f"[{T_SELECT:g}s] Successfully selected 10 target IDs (T1 -> T10): {target_ids}")

    # 3. 目標在全域模擬時鐘上的 3D 軌跡 (K, total_steps, 3)
    trajectories = get_target_trajectories(sampled_pool, release_steps, target_ids)

    # 4. 3D 時間動畫
    anim = animate_targets_3d(  # noqa: F841 - 需保留參考，否則動畫會被 GC
        sampled_pool, release_steps, trajectories, target_ids, T_SELECT, time_step, max_time
    )

    plt.show()


if __name__ == "__main__":
    main()
