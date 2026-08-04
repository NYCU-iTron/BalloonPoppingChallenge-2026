from gymnasium import spaces
import numpy as np

from BalloonPoppingGymEnv.utils.schema import Schema


# roll (1)
# tvc (2)
# throttle (1)
action_space = spaces.Box(
    low=-1,
    high=1,
    shape=(4,),
    dtype=np.float32
)

# aim angle (1)
# relative dist (1)
# relative body pos (3)
# relative body vel (3)
# rocket z (1)
# rocket body vx (1)
# rocket body vy (1)
# rocket body vz (1)
# rocket world vz (1)
# rocket acc (3)
# rocket quat (4)
# rocket body rates (3)
# sin alpha (1)
# sin beta (1)
# prev tvc (2)
# prev roll (1)
# prev throttle (1)
observation_space = spaces.Box(
    low=-np.inf,
    high=np.inf,
    shape=(29,),
    dtype=np.float32
)

def compute_rl_observation(rocket_state: np.ndarray, target_state: np.ndarray) -> np.ndarray:
    ...


def scale_rl_action(normalized_action: np.ndarray) -> np.ndarray:
    ...
