"""A package for designing RL ANM tasks in power grids."""

from gym.envs.registration import register

from .agents import MPCAgentPerfect, MPCAgentConstant
from .envs import ANMEnv, anm4_reward

register(
    id='ANM6Easy-v0',
    entry_point='gym_anm.envs:ANM6Easy',
)

register(
    id='ANM4Easier-v0',
    entry_point='gym_anm.envs:ANM4Easier',
)
