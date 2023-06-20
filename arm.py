import numpy as np
import markovianbandit as bandit
import mdptoolbox
import whittle


class Arm:
    # def __init__(self, no_action_trans, action_trans, time_win, init_state=np.array([1, 0]), discount=1):
    def __init__(self, no_action_trans, action_trans, act_win, init_state=np.array([1, 0]), discount=1):
        self.state = init_state
        self.action_trans = action_trans
        self.no_action_trans = no_action_trans
        self.discount = 0.99
        self.time_period_len = 12
        self.action_window = act_win
        self.mix_states = [init_state]
        self.current_state = 0
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

        self.num_mix_states = len(self.mix_states)
        self.time_ext_states = []
        self.rewards = []
        for i in range(len(self.mix_states)):
            for t in range(self.time_period_len):
                self.time_ext_states.append([i, t, 0])
                if t in self.action_window:
                    self.time_ext_states.append([i, t, 1])

        self.num_states = len(self.time_ext_states)
        self.mix_traj_active = [0] * self.num_mix_states
        self.mix_traj_passive = [i + 1 for i in range(self.num_mix_states)]
        self.mix_traj_passive[-1] = self.num_mix_states - 1
        self.mix_rewards = []
        for state in self.mix_states:
            self.mix_rewards.append(float(state[0]))

        # self.mix_whittle = whittle.binary_search(self.num_mix_states, self.mix_traj_passive, self.mix_traj_active, self.mix_rewards, self.discount)
        # print(self.mix_whittle)
        # print(len(self.time_ext_states))
        # print(len(self.rewards))
        # print(self.time_ext_states)
        # print(self.rewards)
        # print(self.mix_traj_passive)
        # print(self.mix_traj_active)

        self.passive_traj = []
        self.active_traj = []
        self.rewards = []
        for i in range(self.num_states):
            # passive transition
            s = self.time_ext_states[i]
            s_prob = s[0]
            t = s[1]
            is_inspected = s[2]
            if t in self.action_window and t != self.action_window[-1]:
                next_state = [self.mix_traj_passive[s_prob], (t + 1) % self.time_period_len, is_inspected]
            else:
                next_state = [self.mix_traj_passive[s_prob], (t + 1) % self.time_period_len, 0]
            # self.passive_traj.append(self.time_ext_states.index(next_state))
            self.passive_traj.append(self.time_ext_states.index(next_state))
            # active transition
            if not t in self.action_window:
                next_state = next_state  # not in action, same as passive trans
            elif t in self.action_window and t != self.action_window[-1]:
                if is_inspected:
                    next_state = [self.mix_traj_passive[s_prob], (t + 1) % self.time_period_len, 1]
                else:
                    next_state = [self.mix_traj_active[s_prob], (t + 1) % self.time_period_len, 1]
            else:
                if is_inspected:
                    next_state = [self.mix_traj_passive[s_prob], (t + 1) % self.time_period_len, 0]
                else:
                    next_state = [self.mix_traj_active[s_prob], (t + 1) % self.time_period_len, 0]
            self.active_traj.append(self.time_ext_states.index(next_state))
            # rewards for each states
            self.rewards.append(float(self.mix_states[s_prob][0]))
        # print(self.time_ext_states)
        # print(self.passive_traj)
        # print(self.active_traj)
        # for i in range(len(self.passive_traj)):
        #     print([i, self.passive_traj[i]], end= " ")
        # print()
        # for i in range(len(self.active_traj)):
        #     print([i, self.active_traj[i]], end= " ")
        # print()
        # print(self.rewards)
        self.P1 = [[0.0 for i in range(self.num_mix_states)] for j in range(self.num_mix_states)]
        self.P0 = [[0.0 for i in range(self.num_mix_states)] for j in range(self.num_mix_states)]
        for i in range(self.num_mix_states):
            for j in range(self.num_mix_states):
                if j == 0:
                    self.P1[i][j] = 1.0
                if j - i == 1:
                    self.P0[i][j] = 1.0
        self.P0[-1][-1] = 1.0
        self.rewards_P0P1 = []
        self.P0 = np.array(self.P0)
        self.P1 = np.array(self.P1)

        for state in self.mix_states:
            self.rewards_P0P1.append(state[0])
        self.rewards_P0P1 = np.array(self.rewards_P0P1)
        # print(self.P0)
        # print(self.P1)
        # print(self.rewards_P0P1)
        # self.traj_active  = [0] * self.num_states
        # self.traj_passive = [i + 1 for i in range(self.num_states)]
        # self.traj_passive[-1] = self.num_states - 1
        # print(self.traj_passive)
        # print(self.traj_active)
        # for states in self.mix_states:
        #     self.rewards.append(float(states[0]))
        # self.current_state = 0
        # P0 = np.array(self.P0)
        # P1 = np.array(self.P1)
        # R = np.array(self.rewards)
        # _, self.whittle_indices = whittle.compute_whittle_indices(self.P0, self.P1, self.rewards_P0P1, self.rewards_P0P1, beta=0.99)
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

        self.whittle_indices = whittle.binary_search(self.num_states, self.passive_traj, self.active_traj, self.rewards,
                                                     self.discount)
        # print(self.whittle_indices)
        # self.act_window_len = 3
        # self.noact_len = 9

    def state_step(self, action):
        if not action:
            self.current_state = self.passive_traj[self.current_state]
        else:
            self.current_state = self.active_traj[self.current_state]

    def mix_state_step(self, action):
        if not action:
            self.current_state = self.mix_traj_passive[self.current_state]
        else:
            self.current_state = self.mix_traj_active[self.current_state]

    def get_state(self):
        mix_idx = self.time_ext_states[self.current_state][0]
        return self.current_state, self.time_ext_states[self.current_state], self.mix_states[mix_idx]

    def is_in_action_win(self, t):
        return t in self.action_window

    def sampling(self):
        x = np.random.rand()
        # print(x)
        if x < self.mix_states[self.current_state]:
            return 0
        else:
            return 1

    def get_index(self, t):

        return self.whittle_indices[self.current_state]

    def get_mix_index(self, t):
        return self.mix_whittle[self.current_state]

    def reward_whittle(self):
        return self.rewards[self.current_state]

    def reward_mix_whittle(self):
        return self.mix_rewards[self.current_state]

    def debug(self):
        print("Arm mix states:" , end="" )
        print(self.mix_states)
        print("Rewards: ", end= " ")
        print(self.mix_rewards)
        print("Action Window", end=" ")
        print(self.action_window)
        print()
        print()


    def nan_count(self):
        tmp = np.array(self.whittle_indices)
        return np.sum(np.isnan(tmp))

    def value_iter(self):
        vi = mdptoolbox.mdp.ValueIteration(np.array([self.P0, self.P1]),
                                           np.transpose(np.array([self.rewards, self.rewards])), self.discount)
        vi.run()
        self.V = vi.V
        q = mdptoolbox.mdp.QLearning(np.array([self.P0, self.P1]), np.transpose(np.array([self.rewards, self.rewards])),
                                     self.discount)
        q.run()
        self.Q = q.Q

    def whittle_index(self, range, start_state_idx):
        left = range[0]
        right = range[1]
        last_m = 9999999
        m = (left + right) / 2
        while (1):
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


class FullObservedArm:
    # def __init__(self, no_action_trans, action_trans, time_win, init_state=np.array([1, 0]), discount=1):
    def __init__(self, no_action_trans, action_trans, act_win, init_state=0, discount=1):
        self.state = init_state
        self.action_trans = action_trans
        self.no_action_trans = no_action_trans
        self.orgin_trans = [self.no_action_trans, self.action_trans]
        self.discount = 0.99
        self.time_period_len = 12
        self.action_window = act_win
        self.time_ext_states = []
        self.current_state = 0
        self.time_ext_states = []
        self.rewards = []
        for i in range(2):
            for t in range(self.time_period_len):
                self.time_ext_states.append([i, t, 0])
                self.rewards.append(i)
                if t in self.action_window:
                    self.time_ext_states.append([i, t, 1])
                    self.rewards.append(i)
        # print(self.time_ext_states)
        self.num_states = len(self.time_ext_states)
        self.passive_trans = np.zeros([self.num_states, self.num_states])
        self.active_trans = np.zeros([self.num_states, self.num_states])
        for i in range(self.num_states):
            # passive transtion
            state = self.time_ext_states[i]
            s = state[0]
            t = state[1]
            has_inspected = state[2]
            if t in self.action_window and t != self.action_window[-1]:
                next_state0 = [0, (t + 1) % self.time_period_len, has_inspected]
                next_state1 = [1, (t + 1) % self.time_period_len, has_inspected]
                index0 = self.time_ext_states.index(next_state0)
                index1 = self.time_ext_states.index(next_state1)
                self.passive_trans[i][index0] = self.no_action_trans[s][0]
                self.passive_trans[i][index1] = self.no_action_trans[s][1]
            else:
                next_state0 = [0, (t + 1) % self.time_period_len, 0]
                next_state1 = [1, (t + 1) % self.time_period_len, 0]
                index0 = self.time_ext_states.index(next_state0)
                index1 = self.time_ext_states.index(next_state1)
                self.passive_trans[i][index0] = self.no_action_trans[s][0]
                self.passive_trans[i][index1] = self.no_action_trans[s][1]
            # active transition
            if not t in self.action_window:
                next_state0 = [0, (t + 1) % self.time_period_len, 0]
                next_state1 = [1, (t + 1) % self.time_period_len, 0]
                index0 = self.time_ext_states.index(next_state0)
                index1 = self.time_ext_states.index(next_state1)
                self.active_trans[i][index0] = self.no_action_trans[s][0]
                self.active_trans[i][index1] = self.no_action_trans[s][1]
            elif t in self.action_window and t != self.action_window[-1]:
                if has_inspected:
                    next_state0 = [0, (t + 1) % self.time_period_len, 1]
                    next_state1 = [1, (t + 1) % self.time_period_len, 1]
                    index0 = self.time_ext_states.index(next_state0)
                    index1 = self.time_ext_states.index(next_state1)
                    self.active_trans[i][index0] = self.no_action_trans[s][0]
                    self.active_trans[i][index1] = self.no_action_trans[s][1]
                else:
                    next_state = [0, (t + 1) % self.time_period_len, 1]
                    index = self.time_ext_states.index(next_state)
                    self.active_trans[i][index] = self.action_trans[s][0]
            else:
                if has_inspected:
                    next_state0 = [0, (t + 1) % self.time_period_len, 0]
                    next_state1 = [1, (t + 1) % self.time_period_len, 0]
                    index0 = self.time_ext_states.index(next_state0)
                    index1 = self.time_ext_states.index(next_state1)
                    self.active_trans[i][index0] = self.no_action_trans[s][0]
                    self.active_trans[i][index1] = self.no_action_trans[s][1]
                else:
                    next_state = [0, (t + 1) % self.time_period_len, 0]
                    index = self.time_ext_states.index(next_state)
                    self.active_trans[i][index] = self.action_trans[s][0]
        self.whittle_indices = whittle.binary_search(self.num_states, self.passive_trans, self.active_trans, self.rewards, self.discount)
        # print(self.passive_trans)
        # print(self.active_trans)



    def state_step(self, action):
        if not action:
            self.current_state = self.passive_traj[self.current_state]
        else:
            self.current_state = self.active_traj[self.current_state]

    def get_state(self):
        mix_idx = self.time_ext_states[self.current_state][0]
        return self.current_state, self.time_ext_states[self.current_state], self.mix_states[mix_idx]

    def is_in_action_win(self, t):
        return t in self.action_window

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
        vi = mdptoolbox.mdp.ValueIteration(np.array([self.P0, self.P1]),
                                           np.transpose(np.array([self.rewards, self.rewards])), self.discount)
        vi.run()
        self.V = vi.V
        q = mdptoolbox.mdp.QLearning(np.array([self.P0, self.P1]), np.transpose(np.array([self.rewards, self.rewards])),
                                     self.discount)
        q.run()
        self.Q = q.Q

    def whittle_index(self, range, start_state_idx):
        left = range[0]
        right = range[1]
        last_m = 9999999
        m = (left + right) / 2
        while (1):
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
    arm = Arm(matrix_no_act, matrix_act, [3, 4])
    print(arm.rewards)
    # reward_total = 0
    # print(arm.time_ext_states)
    # print(arm.whittle_indices)
    # acts = [0] * 60
    # acts[3] = 1
    # acts[15] = 1
    # print(arm.time_ext_states)
    # for i in range(60):
    #     if acts[i]:
    #         print("Action")
    #     else:
    #         print("No Action")
    #     arm.state_step(acts[i])
    #     print(arm.get_state())
    #     reward = arm.reward_whittle()
    #     reward_total += reward
    #     print(reward_total)
    #     print()
    #     # self.show_all_states()
    # print("Total Reward is: " + str(reward_total))
    # arm.debug()
    # arm.debug()
