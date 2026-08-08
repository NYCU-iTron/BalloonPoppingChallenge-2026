import numpy as np
from BalloonPoppingGymEnv.utils.schema import Schema


class Selector:
    def __init__(self, given_parameters):
        # Balloon and rocket positions are reported as altitude above sea level,
        # so the launch pad sits at z = elevation, not at the coordinate origin.
        elevation = given_parameters[Schema.Given.Section.ENVIRONMENT][
            Schema.Given.Environment.ELEVATION
        ]
        self.pad_origin = np.array([0.0, 0.0, float(elevation)])

    def reset(self):
        pass

    def should_launch(self, observation: dict) -> bool:
        launch_time = 70
        should_launch = observation[Schema.Observation.SIMULATION_TIME] >= launch_time
        return should_launch

    def get_launch_heading(self, observation: dict) -> np.ndarray:
        """
        Returns
        -------
        heading: np.ndarray
            [inclination, heading] in degrees based on balloon positions.
        """
        balloon_states = np.array(observation[Schema.Observation.BALLOON_STATES], dtype=float)
        valid_mask = ~np.isnan(balloon_states[:, 0])
        valid_indices = np.where(valid_mask)[0]
        velocities = balloon_states[valid_indices, 3:]

        mean_vel_xy = np.mean(velocities[:, :2], axis=0)
        vel_norm = np.linalg.norm(mean_vel_xy)
        dir_xy = (
            mean_vel_xy / vel_norm
            if vel_norm > 1e-5
            else np.array([1.0, 0.0])
        )

        # The simulator's heading is a compass bearing (0 = North, 90 = East,
        # clockwise), while dir_xy is an ENU vector -- so the East component
        # goes first in the arctan2.
        heading_deg = np.degrees(np.arctan2(dir_xy[0], dir_xy[1]))
        heading_deg = heading_deg % 360.0

        heading = np.array([90.0, heading_deg])
        return heading

    def select_targets(self, balloon_states: np.ndarray) -> list[int] | None:
        """
        Parameters
        ----------
        balloon_states : np.ndarray
            Shape (N, 6) predicted states [x, y, z, vx, vy, vz]; NaN position
            marks inactive (unreleased or popped) balloons.

        Returns
        -------
        list[int] | None
            A list of 10 target balloon IDs ordered from T1 to T10,
            or None if invalid.
        """
        # --- 1. 權重與門檻參數設定 ---
        dist_weight = 10.0  # 離主軸距離 (d_i) 的懲罰權重
        angle_weight = 20.0  # 氣球之間轉向折角的懲罰權重

        min_dist = 20.0  # 期望的兩氣球間最短距離 (單位: 公尺)
        too_close_weight = 50.0  # 低於 min_dist 時的平方懲罰權重
        max_dist = 100.0
        too_far_weight = 5.0

        # --- 2. 過濾 Valid 氣球 ---
        valid_mask = ~np.isnan(balloon_states[:, 0])
        valid_indices = np.where(valid_mask)[0]

        K = 8  # 目標數量
        if len(valid_indices) < K:
            return None

        # 一律換算成「相對發射台」座標：主軸擬合、投影 s、離軸距離 d 與發射夾角
        # 都應該以發射台為原點，而不是海平面。
        positions = balloon_states[valid_indices, :3] - self.pad_origin
        velocities = balloon_states[valid_indices, 3:]

        # --- 3. (r, z) 擬合斜率 + 速度對齊生成 3D 主軸向量 u ---
        r = np.hypot(positions[:, 0], positions[:, 1])
        z = positions[:, 2]

        sum_r2 = np.sum(r**2)
        slope = np.sum(r * z) / sum_r2 if sum_r2 > 1e-6 else 1.0

        mean_vel_xy = np.mean(velocities[:, :2], axis=0)
        vel_norm = np.linalg.norm(mean_vel_xy)
        dir_xy = (
            mean_vel_xy / vel_norm
            if vel_norm > 1e-5
            else np.array([1.0, 0.0])
        )

        main_axis = np.array([dir_xy[0], dir_xy[1], slope])
        u = main_axis / np.linalg.norm(main_axis)

        # --- 4. 計算投影高度 s 與垂直離軸距離 d，並按 s 排序 ---
        s_vals = np.dot(positions, u)
        valid_balloons = []

        for idx, pos, s_val in zip(valid_indices, positions, s_vals):
            if s_val > 0:  # 只考慮原點前方的氣球
                d_val = np.linalg.norm(pos - s_val * u)
                valid_balloons.append(
                    {"id": int(idx), "pos": pos, "s": s_val, "d": d_val}
                )

        if len(valid_balloons) < K:
            return None

        # 依主軸進度 s 由低到高排序 (天然保證單向推進)
        sorted_balloons = sorted(valid_balloons, key=lambda b: b["s"])
        N = len(sorted_balloons)

        # --- 5. DP 演算法實作 ---
        dp = np.full((N, K + 1), float("inf"))
        parent = np.full((N, K + 1), -1, dtype=int)
        origin = np.array([0.0, 0.0, 0.0])  # 發射台，因 positions 已是相對座標

        # Base Case: k = 1 (第一顆目標：不對原點算距離過近懲罰，只算離軸與發射夾角)
        for i in range(N):
            pos_i = sorted_balloons[i]["pos"]

            dir_origin = pos_i - origin
            norm_orig = np.linalg.norm(dir_origin)
            cos_a = (
                np.clip(np.dot(u, dir_origin) / norm_orig, -1.0, 1.0)
                if norm_orig > 1e-5
                else 1.0
            )
            init_angle = np.degrees(np.arccos(cos_a))

            # Base Cost：只考慮離主軸距離與發射角
            dp[i][1] = (
                dist_weight * sorted_balloons[i]["d"]
                + angle_weight * init_angle
            )
            parent[i][1] = -1

        # DP State Transition: k = 2 -> K
        for k in range(2, K + 1):
            for i in range(N):
                pos_i = sorted_balloons[i]["pos"]

                for j in range(i):  # j < i 確保單向推進 (s_j < s_i)
                    if dp[j][k - 1] == float("inf"):
                        continue

                    pos_j = sorted_balloons[j]["pos"]

                    # 1) 【只在此處計算】兩顆氣球之間 (P_j -> P_i) 的距離與近距離懲罰
                    segment_dist = np.linalg.norm(pos_i - pos_j)
                    too_close_penalty = 0.0
                    if segment_dist < min_dist:
                        too_close_penalty = (
                            min_dist - segment_dist
                        ) ** 2 * too_close_weight

                    too_far_penalty = 0.0
                    if segment_dist > max_dist:
                        too_far_penalty = (
                            segment_dist - max_dist
                        ) ** 2 * too_far_weight

                    # 2) 計算兩氣球之間的轉向折角 (P_prev -> P_j 與 P_j -> P_i)
                    prev_idx = parent[j][k - 1]
                    prev_pos = (
                        origin
                        if prev_idx == -1
                        else sorted_balloons[prev_idx]["pos"]
                    )

                    v1 = pos_j - prev_pos
                    v2 = pos_i - pos_j

                    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                    turn_angle = (
                        np.degrees(
                            np.arccos(
                                np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
                            )
                        )
                        if (n1 > 1e-5 and n2 > 1e-5)
                        else 0.0
                    )

                    # 總代價 = 前一步 Cost + 離軸懲罰 + 轉向角懲罰 + 兩氣球間過近懲罰
                    cost = (
                        dp[j][k - 1]
                        + dist_weight * sorted_balloons[i]["d"]
                        + angle_weight * turn_angle
                        + too_close_penalty
                        + too_far_penalty
                    )

                    if cost < dp[i][k]:
                        dp[i][k] = cost
                        parent[i][k] = j

        # --- 6. 回溯找出最佳序列 (T1 -> T10) ---
        best_last_idx = int(np.argmin(dp[:, K]))
        if dp[best_last_idx, K] == float("inf"):
            return None

        curr = best_last_idx
        target_ids = []
        for k in range(K, 0, -1):
            target_ids.append(sorted_balloons[curr]["id"])
            curr = parent[curr][k]

        target_ids.reverse()
        return target_ids

    def check_target_popped(self, target_idx: int, observation: dict) -> bool:
        balloon_status = np.array(observation[Schema.Observation.BALLOON_STATUS], dtype=int).flatten()
        target_status = balloon_status[target_idx]
        return target_status == 2

    def get_target_state(self, target_idx: int, observation: dict) -> np.ndarray:
        balloon_states = observation[Schema.Observation.BALLOON_STATES]
        target_state = balloon_states[target_idx]
        return target_state

