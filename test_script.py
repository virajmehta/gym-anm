import gym
import time
import numpy as np
from gym_anm import MPCAgentConstant, MPCAgentPerfect
from gym_anm.envs.anm4_env.anm4_easier import anm4_reward, unconstrained_anm4_reward
from tqdm import trange
from argparse import Namespace
from barl.envs.wrappers import NormalizedEnv
from barl.models.gpflow_gp import get_gpflow_hypers_from_data
from barl.models.gpfs_gp import BatchMultiGpfsGp
from sklearn.metrics import explained_variance_score


n_trials = 2
TRAIN_FRAC = 0.7
FIT_TEST_SET = False
easiest = True

def run_trial(env, rew_fn, agent=None):
    obs, action, rew = [], [], []
    o = env.reset()
    obs.append(o)
    rewards = 0.
    for i in trange(env.horizon):
        if agent:
            a = agent.act(env)
        else:
            a = env.action_space.sample()
        action.append(a)
        no, r, done, info = env.step(a)
        obs.append(no)
        # env.render()
        # time.sleep(0.5)  # otherwise the rendering is too fast for the human eye.
        rewards += r
        x = np.concatenate([o, a])
        r_hat = rew_fn(x, no)
        assert np.allclose(r, r_hat, atol=0.1), f"{r=}, {r_hat=}"
        o = no
    print(f"{rewards=}")
    return obs, action, rew

def main():
    # env = NormalizedEnv(gym.make('gym_anm:ANM4Easier-v0'))
    if easiest:
        env = gym.make('gym_anm:ANM4Easiest-v0')
        rew_fn = unconstrained_anm4_reward
    else:
        env = gym.make('gym_anm:ANM4Easier-v0')
        rew_fn = anm4_reward
    norm_env = NormalizedEnv(env)
    agents = [
        MPCAgentConstant(env.simulator, env.action_space, env.gamma,
                              safety_margin=0.96, planning_steps=10),
        MPCAgentPerfect(env.simulator, env.action_space, env.gamma,
                              safety_margin=0.96, planning_steps=10),
        None
        ]
    Xs = []
    ys = []
    for agent in agents:
        for i in range(n_trials):
            obs, actions, rew = run_trial(env, rew_fn, agent)
            obs = norm_env.normalize_obs(np.array(obs))
            actions = norm_env.normalize_action(np.array(actions))
            X = np.concatenate([obs[:-1, :], np.array(actions)], axis=-1)
            y = obs[1:, :] - obs[:-1, :]
            Xs.append(X)
            ys.append(y)
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    print(f"{X.shape=}")
    print(f"{np.unique(X, axis=0).shape=}")
    X, unique_indices = np.unique(X, axis=0, return_index=True)
    y = y[unique_indices, ...]
    if FIT_TEST_SET:
        X_y = np.concatenate([X, y], axis=1)
        np.random.shuffle(X_y)
        X = X_y[:, :X.shape[1]]
        y = X_y[:, X.shape[1]:]
        train_set_size = int(X.shape[0] * TRAIN_FRAC)
        X_train = X[:train_set_size, :]
        y_train = y[:train_set_size, :]
        X_test = X[train_set_size:, :]
        y_test = y[train_set_size:, :]
    else:
        X_train = X
        y_train = y

    time.sleep(1)
    gp_params_list = []
    for i in range(y.shape[1]):
        data = Namespace(x=X_train, y=y_train[:, i])
        test_data = Namespace(x=X_test, y=y_test[:, i]) if FIT_TEST_SET else None
        gp_params = get_gpflow_hypers_from_data(data, print_fit_hypers=False, opt_max_iter=1000, test_data=test_data, retries=30)
        print(f"Output dimension {i}")
        print(gp_params)
        gp_params_list.append(gp_params)
    env.close()

    Xs = []
    ys = []
    for agent in agents:
        for i in range(n_trials):
            obs, actions, rew = run_trial(env, rew_fn, agent)
            obs = norm_env.normalize_obs(np.array(obs))
            actions = norm_env.normalize_action(np.array(actions))
            X = np.concatenate([obs[:-1, :], np.array(actions)], axis=-1)
            y = obs[1:, :] - obs[:-1, :]
            Xs.append(X)
            ys.append(y)
    new_X = np.concatenate(Xs, axis=0)
    new_y = np.concatenate(ys, axis=0)
    X_y = np.concatenate([new_X, new_y], axis=1)
    np.random.shuffle(X_y)
    obs_dim = env.observation_space.low.shape[0]
    action_dim = env.action_space.low.shape[0]
    new_X = X_y[:200, :-obs_dim]
    new_Y = X_y[:200, -obs_dim:]
    gp_params = {
            'ls': [gpp['ls'] for gpp in gp_params_list],
            'alpha': [gpp['alpha'] for gpp in gp_params_list],
            'sigma': 0.01,
            'n_dimx': obs_dim + action_dim
            }
    gp_model_params = {
            'n_dimy': obs_dim,
            'gp_params': gp_params,
            }
    data = Namespace(x=list(X_train), y=list(y_train))
    model = BatchMultiGpfsGp(gp_model_params, data)
    mu_list, covs = model.get_post_mu_cov(list(new_X))
    yhat = np.array(mu_list).T
    residuals = new_Y - yhat
    print(f"{np.mean(np.abs(residuals))=}")
    ev = explained_variance_score(new_Y, yhat)
    print(f"{ev=}")
    breakpoint()


if __name__ == '__main__':
    main()
