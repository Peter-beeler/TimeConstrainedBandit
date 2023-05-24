import numpy as np
from decimal import *
import time

import pylab as p
from numba import jit
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

getcontext().prec = 100
delta = Decimal(1e-100)


# Initialize Markov Decision Process model
# def value_iteration(m, num_states, P0, P1, rewards, discount=1, first_action=0):
#     actions = (0, 1)  # actions (0=passive, 1=active)
#     gamma = Decimal(discount)  # discount factor
#     m = Decimal(m)
#     states = [i for i in range(num_states)]
#     # Set value iteration parameters【
#     max_iter = 10000 # Maximum number of iterations
#     V = [Decimal(0)] * num_states  # Initialize values
#     pi = [None] * num_states  # Initialize policy
#     probs = [P0, P1]
#
#     # print(probs)
#     print(rewards)
#     #
#     # Start value iteration
#     print(P0)
#     print(P1)
#     for i in range(max_iter):
#         max_diff = Decimal(0)  # Initialize max difference
#         V_new = [Decimal(0)] * num_states  # Initialize values
#         for s in states:
#             if i == 0:
#                 direct_re = Decimal(float(rewards[s]))
#                 for s_next in states:
#                     if abs(probs[first_action][s][s_next]) > 1e-6:
#                         val = Decimal(float(rewards[s_next])) + (1 - first_action) * m + gamma * V[s_next]
#                       # Add discounted downstream values
#                 V_new[s] = val
#             else:
#                 max_val = Decimal(0)
#                 for a in actions:
#                     # Compute state value
#                     # print("Iteration :" + str(i))
#                     direct_re = Decimal(float(rewards[s]))  # Get direct reward
#                     val = 0
#                     for s_next in states:
#                         if abs(probs[a][s][s_next]) > 1e-6:
#                             val = Decimal(float(rewards[s_next])) + (1 - a) * m + gamma * V[s_next]
#                             # print(direct_re)
#                             # print(gamma)
#                             # print(V[s_next])
#                             # print(val)
#                     # Store value best action so far
#                     max_val = max(max_val, val)
#
#                     # Update best policy
#                     if V[s] < val:
#                         pi[s] = actions[a]  # Store action with highest value
#
#                 V_new[s] = max_val  # Update value with highest value
#
#             # Update maximum difference
#             max_diff = max(max_diff, abs(V[s] - V_new[s]))
#
#         # Update value functions
#         V = V_new
#         # print(V)
#         # print(str(i) + ": ", end=" ")
#         # print(i)
#         # If diff smaller than threshold delta for all states, algorithm terminates
#         if max_diff < delta:
#             print("convergence!")
#             break
#     # print("Delta" )
#     print(V)
#     return V


# def is_inWin(time_win, t):
#     for times in time_win:
#         left = time_win[0]
#         right = time_win[1]
#         if(left <= t <= right):
#             return True
#     else:
#         return False

@jit
def value_penalized(m, num_states, P0, P1, traj_passive, traj_active, rewards, discount=1):
    states = [i for i in range(num_states)]
    max_iter = 10000  # Maximum number of iterations
    V = np.zeros(num_states)
    H0 = np.zeros(num_states)
    H1 = np.zeros(num_states)

    for i in range(max_iter):
        for s in states:
            h0 = 0
            h1 = 0
            next_state = traj_passive[s]
            h0 += (1 - discount) * (-1 * rewards[next_state])
            h0 += discount * V[next_state]
            next_state = traj_active[s]
            h1 += (1 - discount) * (-1 * rewards[next_state] + m)
            h1 += discount * V[next_state]
            V[s] = min(h0, h1)
    for i in range(num_states):
        s = states[i]
        h0 = 0
        h1 = 0
        next_state = traj_passive[s]
        h0 += (1 - discount) * (-1 * rewards[next_state])
        h0 += discount * V[next_state]
        next_state = traj_active[s]
        h1 += (1 - discount) * (-1 * rewards[next_state] + m)
        h1 += discount * V[next_state]
        H0[i] = h0
        H1[i] = h1
    return np.subtract(H0, H1)

@jit
def binary_search(num_states, P0, P1,  traj_passive, traj_active, rewards, discount):
    range_left = 0.0
    range_right = 1.0
    indexes = []
    start_time = time.time()
    m = 0.0
    for i in range(num_states):
        left = m
        right = range_right
        while (1):
            if left >= right or abs(left - right) < 1e-6:
                break
            m = (left + right) / 2
            # time_fun = time.time()
            # print(P0)
            # print(P1)
            # print(rewards)
            diff = value_penalized(m, num_states, P0, P1, traj_passive, traj_active,  rewards, discount)
            # print("One Iter Time: " + str(time.time() - time_fun))
            if abs(diff[i]) < 1e-6:
                break
            if diff[i] > 0:
                left = m
            else:
                right = m
        indexes.append(m)
    print("Whittle Index Computing Time: " + str(time.time() - start_time))
    print(indexes)
    return indexes

import numpy as np
import scipy.linalg
from numba import jit

NON_INDEXABLE = False
INDEXABLE_BUT_NOT_STRONGLY = 1
STRONGLY_INDEXABLE = 2
MULTICHAIN = -1

# make numpy raise division by zero and 0/0 error
np.seterr(divide='raise', invalid='raise')

def initialize_X_from_update(beta_P0, beta_P1, beta, X, pi, atol):
    """
    Compute Delta*A_inv as defined in Algorithm 2 of the paper
    and store the product in matrix X
    """
    dim = beta_P0.shape[0]
    i0 = 0 # state having null bias in non-discounted case
    mat_pol = np.copy(beta_P1)
    for i, a in enumerate(pi):
        if a: continue
        else: mat_pol[i, :] = beta_P0[i, :]
    Delta = beta_P1 - beta_P0
    if abs(1.0 - beta) < atol:
        mat_pol[:, i0] = -1.0
        mat_pol[i0, i0] = 0.0
        Delta[:, i0] = 0.0
    A = np.eye(dim, dtype=np.double) - mat_pol
    X[:,:] = scipy.linalg.solve(A.transpose(), Delta.transpose(), overwrite_a=True, overwrite_b=True, check_finite=False).transpose()


def find_mu_min(y, z, current_mu, atol):
    """
    Find the smallest mu_i^k
    """
    try:
        mu_i_k = current_mu + z/(1.0-y)
    except FloatingPointError:
        nb_elems = z.shape[0]
        mu_i_k = np.empty(nb_elems)
        for i in range(z.shape[0]):
            if abs(z[i]) < atol: mu_i_k[i] = current_mu
            elif 1.0-y[i] > atol: mu_i_k[i] = current_mu + z[i]/(1.0-y[i])
            else: mu_i_k[i] = np.inf
    valid_idx = np.where( (mu_i_k > current_mu + atol) ) [0]
    if len(valid_idx)>0:
        argmin = mu_i_k[valid_idx].argmin()
        return valid_idx[argmin], mu_i_k[valid_idx[argmin]]
    else:
        return -1, current_mu

@jit
def update_W(W, sigma, X, k, atol, check_indexability=True, k0=0):
    n = X.shape[0]
    V = np.copy(X[:, sigma])
    if check_indexability:
        for l in range(k0+1, k):
            c = V[n-l]
            for i in range(n):
                V[i] = V[i] - c * W[l-1, i]
        c = 1.0 + V[n-k]
        if abs(c) < atol: raise ZeroDivisionError
        for i in range(n):
            W[k-1, i] = V[i] / c
    else:
        for l in range(k0+1, k):
            c = V[n-l]
            for i in range(n-l+1):
                V[i] = V[i] - c * W[l-1, i]
        c = 1.0 + V[n-k]
        if abs(c) < atol: raise ZeroDivisionError
        for i in range(n-k):
            W[k-1, i] = V[i] / c

def compute_whittle_indices(P0, P1, R0, R1, beta=1, check_indexability=True, verbose=False, atol=1e-12, number_of_updates='2n**0.1'):
    """
    Implementation of Algorithm 2 of the paper
    Test whether the problem is indexable
    and compute Whittle indices when the problem is indexable
    The indices are computed in increasing order

    Args:
    - P0, P1: transition matrix for rest and activate actions respectively
    - R0, R1: reward vector for rest and activate actions respectively
    - beta: discount factor
    - check_indexability: if True check whether the problem is indexable or not
    - number_of_updates: (default = '2n**0.1'): number of time that X^{k} is recomputed from scratch.
    """
    dim = P0.shape[0]
    assert P0.shape == P1.shape
    assert R0.shape == R1.shape
    assert R0.shape[0] == dim

    is_indexable = STRONGLY_INDEXABLE
    pi = np.ones(dim, dtype=np.double)
    sorted_sigmas = np.arange(dim)
    idx_in_sorted = np.arange(dim)
    whittle_idx = np.empty(dim, dtype=np.double)
    whittle_idx.fill(np.nan)
    X = np.empty((dim, dim), dtype=np.double, order='C')
    sorted_X = np.empty((dim, dim), dtype=np.double, order='C')
    beta_P0 = beta*P0
    beta_P1 = beta*P1
    W = np.empty((dim-1,dim), dtype=np.double, order='C')
    k0 = 0
    if number_of_updates == '2n**0.1':
        number_of_updates = int(2*dim**0.1)
    frequency_of_update = int(dim / max(1, number_of_updates))

    try:
        initialize_X_from_update(beta_P0, beta_P1, beta, X, pi, atol)
    except np.linalg.LinAlgError as e:
        if 'Matrix is singular' in str(e):
            print("The arm is multichain!")
            return MULTICHAIN, whittle_idx
        else: raise e
    y = np.zeros(dim)
    z = R1 - R0 + X.dot(R1)
    argmin = np.argmin(z)
    sigma = sorted_sigmas[argmin]
    whittle_idx[sigma] = z[sigma]
    z -= whittle_idx[sigma]

    if verbose: print('       ', end='')
    for k in range(1, dim):
        if verbose: print('\b\b\b\b\b\b\b{:7}'.format(k), end='', flush=True)
        """
        1. We sort the states so that the 'non visited' states are the first "dim-k"
           To do so, we exchange only one column of all matrices.
        """
        tmp_s, idx_sigma = sorted_sigmas[dim-k], idx_in_sorted[sigma]
        idx_in_sorted[tmp_s], idx_in_sorted[sigma] = idx_in_sorted[sigma], dim-k
        sorted_sigmas[dim-k], sorted_sigmas[idx_sigma] = sigma, sorted_sigmas[dim-k]

        X[dim-k, :], X[idx_sigma, :] = X[idx_sigma, :], np.copy(X[dim-k, :])
        W[:k-1, dim-k], W[:k-1, idx_sigma] = W[:k-1, idx_sigma], np.copy(W[:k-1, dim-k])

        y[dim-k], y[idx_sigma] = y[idx_sigma], y[dim-k]
        z[dim-k], z[idx_sigma] = z[idx_sigma], z[dim-k]

        """
        2. If needed, we re-compute the matrix "beta times X". This should not be done too often.
        """
        if k > k0 + frequency_of_update:
            try:
                initialize_X_from_update(beta_P0, beta_P1, beta, X, pi, atol)
            except np.linalg.LinAlgError as e:
                if 'Matrix is singular' in str(e):
                    print("The arm is multichain!")
                    return MULTICHAIN, whittle_idx
                else: raise e
            for i in range(dim):
                sorted_X[i] = np.copy(X[sorted_sigmas[i]])
            X = np.copy(sorted_X)
            k0 = k-1
        pi[sigma] = 0

        """
        3. We perform the recursive operations to compute beta*X, beta*y and beta*z.
        """
        try:
            update_W(W, sigma, X, k, atol, check_indexability, k0)
        except ZeroDivisionError:
            print("The arm is multichain!")
            return MULTICHAIN, whittle_idx
        y += (1.0 - y[dim-k])*W[k-1]
        argmin, mu_min_k = find_mu_min(y[0:dim-k], z[0:dim-k], whittle_idx[sigma], atol)
        if np.isinf(mu_min_k):
            one_minus_y = 1.0 - y[dim-k:]
            if (one_minus_y < -atol).any() or (np.nonzero(abs(one_minus_y)<atol)[0] == np.nonzero(abs(z[dim-k:])<atol)[0]).any():
                is_indexable = NON_INDEXABLE
                print("Not indexable!")
                return is_indexable, whittle_idx
            else:
                for active_state in sorted_sigmas[0:dim-k]:
                    whittle_idx[active_state] = np.inf
                return is_indexable, whittle_idx
        next_sigma = sorted_sigmas[argmin]
        whittle_idx[next_sigma] = mu_min_k
        if next_sigma == 0:
            print("The first " + str(mu_min_k))
        if next_sigma == dim - 2:
            print("The last second " + str(mu_min_k))
        if next_sigma == dim:
            print("The last second " + str(mu_min_k))
        z -= (mu_min_k - whittle_idx[sigma])*(1.0-y)

        """
        4. If needed, we test if we violate the indexability condition.
        """
        if check_indexability and is_indexable:
            if (whittle_idx[sigma] + atol < mu_min_k) and ( z[dim-k:] > -atol ).any():
                is_indexable = NON_INDEXABLE
                print("Not indexable!")
                return is_indexable, whittle_idx
            elif ( y > 1.0 ).any():
                is_indexable = INDEXABLE_BUT_NOT_STRONGLY
        sigma = next_sigma
    if verbose: print('\b\b\b\b\b\b\b', end='')
    return is_indexable, whittle_idx
