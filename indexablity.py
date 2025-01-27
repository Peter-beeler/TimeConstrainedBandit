import numpy as np
from arm import Arm
from numba import jit
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning

granularity = 0.001


@jit
def value_penalized(m, num_states, traj_passive, traj_active, rewards, discount=1):
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
    return H0, H1


@jit
def optimal_policy(m, num_states, traj_passive, traj_active, rewards, discount=1):
    pi = []
    H0, H1 = value_penalized(m, num_states, traj_passive, traj_active, rewards, discount)
    for state in range(num_states):
        if H0[state] > H1[state]:
            pi.append(state)

    return pi, H0, H1


def is_p1_in_p2(p1, p2):
    for x in p1:
        if x not in p2:
            return False
    return True


if __name__ == '__main__':
    # matrix_no_act = np.array([[0.8, 0.2], [0.6, 0.4]])
    matrix_no_act = np.array([[0.16203461, 0.83796539],
    [0.77762515, 0.22237485]])
    matrix_act = np.array([[1, 0], [1, 0]])
    arm = Arm(matrix_no_act, matrix_act, [3, 4])
    states = [i for i in range(arm.num_states)]
    rel = True
    for i in range(int(1 / granularity)):
        p1, H0, H1 = optimal_policy(i * granularity, arm.num_states, arm.passive_traj, arm.active_traj, arm.rewards, arm.discount)
        p2, H0, H1 = optimal_policy((i + 1) * granularity, arm.num_states, arm.passive_traj, arm.active_traj, arm.rewards,
                            arm.discount)
        print(p1)
        rel = rel and is_p1_in_p2(p2, p1)
    print(rel)
