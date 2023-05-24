import numpy as np
import markovianbandit as bandit
import mdptoolbox
import whittle

class Arm:
    # def __init__(self, no_action_trans, action_trans, time_win, init_state=np.array([1, 0]), discount=1):
    def __init__(self, no_action_trans, action_trans, init_state=np.array([1, 0]), discount=1):
        self.state = init_state
        self.action_trans = action_trans
        self.no_action_trans = no_action_trans
        self.discount = 0.99
        self.mix_states = [init_state]
        while (1):
            new_state = np.dot(self.mix_states[-1], self.no_action_trans)
            if (abs(new_state - self.mix_states[-1]) < 1e-3).all():
                break
            self.mix_states.append(new_state)
        # print(self.mix_states)
        # for i in range(10):
        #     new_state = np.dot(self.mix_states[-1], self.no_action_trans)
        #     self.mix_states.append(new_state)
        #     # print(new_state)

        self.num_states = len(self.mix_states)
        self.P1 = [[0.0 for i in range(self.num_states)] for j in range(self.num_states)]
        self.P0 = [[0.0 for i in range(self.num_states)] for j in range(self.num_states)]
        for i in range(self.num_states):
            for j in range(self.num_states):
                if j == 0:
                    self.P1[i][j] = 1.0
                if j - i == 1:
                    self.P0[i][j] = 1.0
        self.P0[-1][-1] = 1.0
        self.rewards = []
        self.traj_active  = [0] * self.num_states
        self.traj_passive = [i + 1 for i in range(self.num_states)]
        self.traj_passive[-1] = self.num_states - 1
        # print(self.traj_passive)
        # print(self.traj_active)
        for states in self.mix_states:
            self.rewards.append(float(states[0]))
        self.current_state = 0
        P0 = np.array(self.P0)
        P1 = np.array(self.P1)
        R = np.array(self.rewards)
        # _, self.whittle_indices = whittle.compute_whittle_indices(P0, P1, R, R, beta=0.99)
        # print(self.whittle_indices)
        #
        # import pandas as pd
        # series = pd.Series(self.whittle_indices)
        # interpolated = series.interpolate(method="index")
        # self.whittle_indices = interpolated.tolist()
        # self.whittle_indices = [0 if x != x else x for x in self.whittle_indices]

        # self.time_win = []
        # for i in range(int(len(time_win)/2)):
        #     self.time_win.append([time_win[i], time_win[i+1]])
        # self.value_iter()
        # print(self.V)
        # value_passive = whittle.value_iteration(0.0, self.num_states, self.P0, self.P1, self.rewards, self.discount, 0)
        # self.whittle_index = whittle.binary_search(self.num_states, self.P0, self.P1, self.rewards, self.discount)
        # print(value_passive)
        # print(value_active)
        # print(value_passive > value_active)
        # print("Package Result:")
        # print(self.whittle_indices)
        self.whittle_indices =  whittle.binary_search(self.num_states, self.P0, self.P1, self.traj_passive, self.traj_active, self.rewards, self.discount)
        # print(self.whittle_indices)
        self.act_window_len = 3
        self.noact_len = 9
    def state_step(self, action):
        if not action:
            self.current_state = (self.current_state + 1) % self.num_states
        else:
            self.current_state = 0

    def get_state(self):
        return self.current_state

    def is_in_action_win(self, t):
        for x in self.time_win:
            if x[0] <= t <= x[1]:
                return True
        return False
    def sampling(self):
        x = np.random.rand()
        # print(x)
        if x < self.mix_states[self.current_state]:
            return 0
        else:
            return 1

    def get_index(self, t):

        return self.whittle_indices[self.current_state]


    def reward_whittle(self):
        return self.rewards[self.current_state]

    def debug(self):
        # print(self.mix_states)
        # print(self.P0)
        # print(self.P1)
        print(self.rewards)
        # print(self.whittle_indices)
        # print(self.nan_count())

    def nan_count(self):
        tmp = np.array(self.whittle_indices)
        return np.sum(np.isnan(tmp))

    def value_iter(self):
        vi = mdptoolbox.mdp.ValueIteration(np.array([self.P0, self.P1]), np.transpose(np.array([self.rewards, self.rewards])), self.discount)
        vi.run()
        self.V = vi.V
        q = mdptoolbox.mdp.QLearning(np.array([self.P0, self.P1]), np.transpose(np.array([self.rewards, self.rewards])), self.discount)
        q.run()
        self.Q = q.Q

    def whittle_index(self, range, start_state_idx):
        left = range[0]
        right = range[1]
        last_m = 9999999
        m = (left + right) / 2
        while(1):
            if abs(last_m - m) < 1e-3:
                break
            Val_passive = m + self.rewards[start_state_idx] + self.Q[start_state_idx][0]
            Val_active = self.rewards[start_state_idx] + self.discount * self.Q[start_state_idx][1]
            if Val_passive >= Val_active:
                right = m
            else:
                left = m
            m = (left + right) / 2
            print("m is " + str(m))
            print("passive is " + str(Val_passive))
            print("active is " + str(Val_active))

        print("The whittle index is :" + str(m))



if __name__ == '__main__':
    matrix_no_act = np.array([[0.8, 0.2], [0.6, 0.4]])
    matrix_act = np.array([[1, 0], [1, 0]])
    arm = Arm(matrix_no_act, matrix_act)
    # arm.debug()
    # arm.debug()
