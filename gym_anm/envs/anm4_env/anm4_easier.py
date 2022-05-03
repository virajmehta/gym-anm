"""The :code:`ANM4Easier-v0` task."""

import numpy as np
from scipy.sparse import csr_matrix

from .anm4 import ANM4
from ...simulator.solve_load_flow import _construct_v_from_guess, _newton_raphson_sparse


class ANM4Easier(ANM4):
    """The :code:`ANM4Easier-v0` task."""

    def __init__(self, remove_constraints=False):
        observation = 'state'  # fully observable environment
        K = 1
        delta_t = 0.25         # 15 minutes between timesteps
        gamma = 0.995
        lamb = 100 if not remove_constraints else 0.
        aux_bounds = np.array([[0, 24 / delta_t - 1]])
        costs_clipping = (1, 100)
        super().__init__(observation, K, delta_t, gamma, lamb, aux_bounds,
                         costs_clipping)

        # Consumption and maximum generation 24-hour time series.
        self.P_loads = _get_load_time_series()
        self.P_maxs = _get_gen_time_series()
        self.periodic_dimensions = [12]
        self.horizon = 96

    def init_state(self):
        n_dev, n_gen, n_des = 5, 1, 1

        state = np.zeros(2 * n_dev + n_des + n_gen + self.K)

        t_0 = self.np_random.randint(0, int(24 / self.delta_t))
        state[-1] = t_0

        # Load (P, Q) injections.
        for dev_id, p_load in zip([1, 3], self.P_loads):
            state[dev_id] = p_load[t_0]
            state[n_dev + dev_id] = \
                p_load[t_0] * self.simulator.devices[dev_id].qp_ratio

        # Non-slack generator (P, Q) injections.
        for idx, (dev_id, p_max) in enumerate(zip([2], self.P_maxs)):
            state[2 * n_dev + n_des + idx] = p_max[t_0]
            state[dev_id] = p_max[t_0]
            state[n_dev + dev_id] = \
                self.np_random.uniform(self.simulator.devices[dev_id].q_min,
                                       self.simulator.devices[dev_id].q_max)

        # Energy storage unit.
        for idx, dev_id in enumerate([4]):
            state[2 * n_dev + idx] = \
                self.np_random.uniform(self.simulator.devices[dev_id].soc_min,
                                       self.simulator.devices[dev_id].soc_max)

        return state

    def next_vars(self, s_t):
        aux = int((s_t[-1] + 1) % (24 / self.delta_t))

        vars = []
        for p_load in self.P_loads:
            vars.append(p_load[aux])
        for p_max in self.P_maxs:
            vars.append(p_max[aux])

        vars.append(aux)

        return np.array(vars)

    def reset(self, date_init=None):
        obs = super().reset()

        # Reset the time of the day based on the auxiliary variable.
        date = self.date
        new_date = self.date + self.state[-1] * self.timestep_length
        super().reset_date(new_date)

        return obs


def _get_load_time_series():
    """Return the fixed 24-hour time-series for the load injections."""

    '''
    NO more residential loads
    # Device 1 (residential load).
    s1 = - np.ones(25)
    s12 = np.linspace(-1.5, -4.5, 7)
    s2 = - 5 * np.ones(13)
    s23 = np.linspace(-4.625, -2.375, 7)
    s3 = - 2 * np.ones(13)
    P1 = np.concatenate((s1, s12, s2, s23, s3, s23[::-1], s2, s12[::-1],
                         s1[:4]))
    '''

    # Device 3 (industrial load).
    s1 = -4 * np.ones(25)
    s12 = np.linspace(-4.75, -9.25, 7)
    s2 = - 10 * np.ones(13)
    s23 = np.linspace(-11.25, -18.75, 7)
    s3 = - 20 * np.ones(13)
    P3 = np.concatenate((s1, s12, s2, s23, s3, s23[::-1], s2, s12[::-1],
                         s1[:4]))

    # Device 5 (EV charging station load).
    s1 = np.zeros(25)
    s12 = np.linspace(-3.125, -21.875, 7)
    s2 = - 25 * np.ones(13)
    s23 = np.linspace(-21.875, -3.125, 7)
    s3 = np.zeros(13)
    P5 = np.concatenate((s1, s12, s2, s23, s3, s23[::-1], s2, s12[::-1],
                         s1[:4]))

    P_loads = np.vstack((P3, P5))
    assert P_loads.shape == (2, 96)

    return P_loads


def _get_gen_time_series():
    """Return the fixed 24-hour time-series for the generator maximum production."""

    '''
    # Device 2 (residential PV aggregation).
    s1 = np.zeros(25)
    s12 = np.linspace(0.5, 3.5, 7)
    s2 = 4 * np.ones(13)
    s23 = np.linspace(7.25, 36.75, 7)
    s3 = 30 * np.ones(13)
    P2 = np.concatenate((s1, s12, s2, s23, s3, s23[::-1], s2, s12[::-1],
                         s1[:4]))
    '''

    # Device 2 (wind farm).
    s1 = 40 * np.ones(25)
    s12 = np.linspace(36.375, 14.625, 7)
    s2 = 11 * np.ones(13)
    s23 = np.linspace(14.725, 36.375, 7)
    s3 = 40 * np.ones(13)
    P4 = np.concatenate((s1, s12, s2, s23, s3, s23[::-1], s2, s12[::-1],
                         s1[:4]))

    P_maxs = np.vstack((P4,))
    assert P_maxs.shape == (1, 96)

    return P_maxs

def _anm4_reward(x, next_obs):
    # state is
    #       dev_p for 5 devices
    #       dev_q for 5 devices
    #       des_soc for device 4
    #       gen_p_max for device 2
    #       aux
    obs = x[next_obs.shape[0]:]
    Y = csr_matrix(np.matrix([[ 0.10698836 -5.45046261j, -0.10698836 +5.45046261j,
          0.         +0.j        ,  0.         +0.j        ],
        [-0.10698836 +5.45046261j, 12.35546861-36.69069179j,
         -6.51430783+13.13611248j, -5.73417242+18.1041167j ],
        [ 0.         +0.j        , -6.51430783+13.13611248j,
          6.51430783-13.13611248j,  0.         +0.j        ],
        [ 0.         +0.j        , -5.73417242+18.1041167j ,
          0.         +0.j        ,  5.73417242-18.1041167j ]]))
    baseMVA = 100.
    xtol = 1e-5
    delta_t = 0.25
    lamb = 100
    bus_v_max = [1, 1.1, 1.1, 1.1]
    bus_v_min = [1, 0.9, 0.9, 0.9]

    # Device 2 (wind farm) max powers
    s1 = 40 * np.ones(25)
    s12 = np.linspace(36.375, 14.625, 7)
    s2 = 11 * np.ones(13)
    s23 = np.linspace(14.725, 36.375, 7)
    s3 = 40 * np.ones(13)
    P4 = np.concatenate((s1, s12, s2, s23, s3, s23[::-1], s2, s12[::-1],
                         s1[:4]))
    aux = int((next_obs[-1]) % (24 / delta_t))
    p_pot_2 = P4[aux] / baseMVA

    # do this as a scalar, can vectorize later
    bus_p = np.zeros(4)
    bus_q = np.zeros(4)
    bus_p[0] = next_obs[0]
    bus_q[0] = next_obs[5]
    bus_p[2] = next_obs[1] + next_obs[2]
    bus_q[2] = next_obs[6] + next_obs[7]
    bus_p[3] = next_obs[3] + next_obs[4]
    bus_q[3] = next_obs[8] + next_obs[9]
    bus_p /= baseMVA
    bus_q /= baseMVA

    # Construct initial guess for nodal V.
    v_guess = np.array([0, 0, 0, 1, 1, 1])
    p_system = bus_p[1:]
    q_system = bus_q[1:]
    v, n_iter, diff, converged = \
        _newton_raphson_sparse(v_guess, p_system, q_system, Y, x_tol=xtol)
    V = _construct_v_from_guess(v)
    I = np.dot(Y.toarray(), V)

    # start with the energy loss term
    e_loss = np.sum(next_obs[:4]) / baseMVA
    curtailment = np.maximum(0, p_pot_2 - next_obs[2] / baseMVA)
    e_loss += curtailment
    e_loss *= delta_t

    penalty = 0

    # bus constraints
    for v, vmin, vmax in zip(V, bus_v_min, bus_v_max):
        v_magn = np.abs(v)
        penalty += np.maximum(0, v_magn - vmax) + np.maximum(0, vmin - v_magn)

    # branch constraints
    branches = [(0, 1), (1, 2), (1, 3)]
    branch_caps = [0.32, 0.18, 0.18]
    for branch, cap in zip(branches, branch_caps):
        sfrom = V[branch[0]] * np.conj(I[branch[0]])
        sto = V[branch[1]] * np.conj(I[branch[1]])
        smax_norm = np.maximum(np.abs(sfrom), np.abs(sto))
        penalty += np.maximum(0, smax_norm - cap)
    penalty *= delta_t * lamb
    reward = - (e_loss + penalty)
    return reward

def anm4_reward(x, next_obs):
    if x.ndim == 2:
        rews = []
        for i in range(x.shape[0]):
            xi = x[i, :]
            noi = next_obs[i, :]
            rews.append(_anm4_reward(xi, noi))
        return np.array(rews)
    elif x.ndim == 1:
        return _anm4_reward(x, next_obs)
    else:
        raise NotImplementedError()

def unconstrained_anm4_reward(x, next_obs):
    baseMVA = 100.
    delta_t = 0.25

    # Device 2 (wind farm) max powers
    s1 = 40 * np.ones(25)
    s12 = np.linspace(36.375, 14.625, 7)
    s2 = 11 * np.ones(13)
    s23 = np.linspace(14.725, 36.375, 7)
    s3 = 40 * np.ones(13)
    P4 = np.concatenate((s1, s12, s2, s23, s3, s23[::-1], s2, s12[::-1],
                         s1[:4]))
    aux = int((next_obs[-1]) % (24 / delta_t))
    p_pot_2 = P4[aux] / baseMVA

    e_loss = np.sum(next_obs[..., :4]) / baseMVA
    curtailment = np.maximum(0, p_pot_2 - next_obs[..., 2] / baseMVA)
    e_loss += curtailment
    e_loss *= delta_t
    return -e_loss



if __name__ == '__main__':
    import time

    env = ANM4Easier()
    env.reset()
    print('Environment reset and ready.')

    T = 50
    start = time.time()
    for i in range(T):
        print(i)
        a = env.action_space.sample()
        o, r, _, _ = env.step(a)
        env.render()
        time.sleep(0.5)

    print('Done with {} steps in {} seconds!'.format(T, time.time() - start))
