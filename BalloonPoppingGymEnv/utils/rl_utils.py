import numpy as np
from pathlib import Path

RL_FRAME_SKIP = 5

def make_custom_env(scenario_params, given_params, pool_path_list: list[Path]):
    def _init():
        from BalloonPoppingGymEnv.envs.rl_navigator_env import RLNavigatorEnv
        from BalloonPoppingGymEnv.envs.pool_env import PoolEnv

        raw_env = PoolEnv(render_mode=None, parameters=scenario_params)

        return RLNavigatorEnv(raw_env, given_params, pool_path_list)
    return _init

def scale_rl_action(normalized_action: np.ndarray) -> np.ndarray:
    action = np.asarray(normalized_action, dtype=np.float32).reshape(-1)

    acc_limit_xy = 8.0
    acc_limit_z_low = -10.0
    acc_limit_z_high = 5.0

    a_cmd = np.zeros(3, dtype=np.float32)
    a_cmd[0] = action[0] * acc_limit_xy
    a_cmd[1] = action[1] * acc_limit_xy

    # Linear interpolation for Z axis: [-1, 1] -> [acc_limit_z_low, acc_limit_z_high]
    z_norm = (action[2] + 1.0) * 0.5
    a_cmd[2] = acc_limit_z_low + z_norm * (acc_limit_z_high - acc_limit_z_low)

    return a_cmd

def compute_rl_observation(rocket_state, target_state):
    rel_pos = target_state[0:3] - rocket_state[0:3]
    rel_vel = target_state[3:6] - rocket_state[3:6]
    rocket_vel = rocket_state[3:6]

    rocket_z = rocket_state[2]
    target_z = target_state[2]

    rl_obs = np.concatenate([rel_pos, rel_vel, rocket_vel, [rocket_z, target_z]]).astype(np.float32)
    rl_obs = np.nan_to_num(rl_obs, nan=0.0, posinf=1e4, neginf=-1e4)

    return rl_obs

def linear_schedule(initial_value, final_value):
    def schedule(progress_remaining):
        return final_value + progress_remaining * (initial_value - final_value)
    return schedule
