import gym
import time
from gym_anm import MPCAgentConstant, MPCAgentPerfect
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
        o, r, done, info = env.step(a)
        # env.render()
        # time.sleep(0.5)  # otherwise the rendering is too fast for the human eye.
        rewards += r
    print(f"{rewards=}")

    env.close()

if __name__ == '__main__':
    run()
