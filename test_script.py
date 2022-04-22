import gym
import time
import numpy as np
from gym_anm import MPCAgentConstant, MPCAgentPerfect
from gym_anm.envs.anm4_env.anm4_easier import anm4_reward
from tqdm import trange

def run():
    env = gym.make('gym_anm:ANM4Easier-v0')
    o = env.reset()

    '''
    agent = MPCAgentConstant(env.simulator, env.action_space, env.gamma,
                              safety_margin=0.96, planning_steps=10)
    '''
    agent = MPCAgentPerfect(env.simulator, env.action_space, env.gamma,
                              safety_margin=0.96, planning_steps=10)

    rewards = 0.
    for i in trange(100):
        a = agent.act(env)
        # a = env.action_space.sample()
        no, r, done, info = env.step(a)
        # env.render()
        # time.sleep(0.5)  # otherwise the rendering is too fast for the human eye.
        rewards += r
        x = np.concatenate([o, a])
        r_hat = anm4_reward(x, no)
        assert np.allclose(r, r_hat, atol=0.6), f"{r=}, {r_hat=}"
        o = no
    print(f"{rewards=}")

    env.close()

if __name__ == '__main__':
    run()
