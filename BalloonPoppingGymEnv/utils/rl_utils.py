import numpy as np

from BalloonPoppingGymEnv.envs.pool_env import PoolEnv
from BalloonPoppingGymEnv.envs.rl_navigator_env import RLNavigatorEnv

RL_FRAME_SKIP = 5
MAX_ACC = 20.0

def make_custom_env(scenario_params, given_params, pool_path):
    def _init():
        raw_env = PoolEnv(render_mode=None, parameters=scenario_params)
        return RLNavigatorEnv(raw_env, given_params, pool_path)
    return _init

def scale_rl_action(normalized_action: np.ndarray) -> np.ndarray:
    action = np.asarray(normalized_action, dtype=np.float64).reshape(-1) * MAX_ACC
    return action

def compute_rl_observation(rocket_state, target_state):
    rel_pos = target_state[0:3] - rocket_state[0:3]
    rel_vel = (target_state[3:6] if target_state.size >= 6 else np.zeros(3)) - rocket_state[3:6]

    rocket_vel = rocket_state[3:6]
    z = rocket_state[2]

    rl_obs = np.concatenate([rel_pos, rel_vel, rocket_vel, [z]]).astype(np.float32)

    rl_obs = np.nan_to_num(rl_obs, nan=0.0, posinf=1e4, neginf=-1e4)

    return rl_obs

def linear_schedule(initial_value, final_value):
    def schedule(progress_remaining):
        return final_value + progress_remaining * (initial_value - final_value)
    return schedule
