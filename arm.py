import random
import time

import numpy as np
# import markovianbandit as bandit
# import mdptoolbox
import whittle
import os
from whittle import Q_final
import json
from scipy.linalg import fractional_matrix_power

class Arm_MultiPull:
    # def __init__(self, no_action_trans, action_trans, time_win, init_state=np.array([1, 0]), discount=1):
    def __init__(self, no_action_trans, action_trans, act_win, win_len, period_len, whittle_index, multi=[1],
                 init_state=np.array([1, 0]),
                 discount=1,
                 require_whittle_idx=True, need_trans = True):
        self.state = init_state
        self.action_trans = action_trans
        self.no_action_trans = no_action_trans
        self.discount = 0.99
        self.time_period_len = period_len
        # multi ins in one action window
        # self.multi = act_win[1]
        self.multi = [1]
        day1 = act_win
        # day2 = act_win[0][1]
        # day3 = act_win[0][2]
        # self.action_window = [[i for i in range(day1, day1 + 5)]]
        # self.action_window = [act_win]
        self.action_window = [[i for i in range(day1, day1 + win_len)]]
        self.mix_states = [init_state]
        self.current_state = 0
        self.days_in_good_state = 0
        # self.inspect_cnt = [0] * len(self.action_window)
        self.not_inspected = True
        self.cnt = 0
        self.ins_record = []
        while (1):
            new_state = np.dot(self.mix_states[-1], self.no_action_trans)
            if (abs(new_state - self.mix_states[-1]) < 1e-2).all():
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
                self.time_ext_states.append([i, t])


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
            
            next_state = [self.mix_traj_passive[s_prob], (t + 1) % self.time_period_len]
            # self.passive_traj.append(self.time_ext_states.index(next_state))
            self.passive_traj.append(self.time_ext_states.index(next_state))
            # active transition
            if not self.isInWindow(t):
                next_state = next_state  # not in action, same as passive trans
            else:
                next_state = [self.mix_traj_active[s_prob], (t + 1) % self.time_period_len]
            self.active_traj.append(self.time_ext_states.index(next_state))
            # rewards for each states
            self.rewards.append(float(self.mix_states[s_prob][0]))
            # self.rewards.append(float(1 - s_prob / len(self.mix_states)))
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
        # self.P1 = [[0.0 for i in range(self.num_mix_states)] for j in range(self.num_mix_states)]
        # self.P0 = [[0.0 for i in range(self.num_mix_states)] for j in range(self.num_mix_states)]
        # for i in range(self.num_mix_states):
        #     for j in range(self.num_mix_states):
        #         if j == 0:
        #             self.P1[i][j] = 1.0
        #         if j - i == 1:
        #             self.P0[i][j] = 1.0
        # self.P0[-1][-1] = 1.0
        # self.rewards_P0P1 = []
        # self.P0 = np.array(self.P0)
        # self.P1 = np.array(self.P1)
        #
        # for state in self.mix_states:
        #     self.rewards_P0P1.append(state[0])
        # self.rewards_P0P1 = np.array(self.rewards_P0P1)
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
        #
        # self.passive_trans = np.zeros([self.num_states, self.num_states])
        # self.active_trans = np.zeros([self.num_states, self.num_states])
        # for i in range(self.num_states):
        #     self.passive_trans[i][self.passive_traj[i]] = 1.0
        #     self.active_trans[i][self.active_traj[i]] = 1.0
        # mb = bandit.restless_bandit_from_P0P1_R0R1(self.passive_trans, self.active_trans, self.rewards, self.rewards)
        # self.mbi = mb.whittle_indices(discount=0.99)
        #
        # if np.isnan(np.sum(self.mbi)):
        #     print(self.no_action_trans)
        #     print(self.mix_states)
        # print(mbi)

        if require_whittle_idx:
            if whittle_index is not None:
                if need_trans:
                    self.whittle_indices = whittle_index
                else:
                    self.mix_whittle_indices = whittle_index
            else:
                if need_trans:

                    self.whittle_indices = whittle.binary_search(self.num_states, self.passive_traj, self.active_traj,
                                                                 self.rewards, self.discount)
                else:
                    self.mix_whittle_indices = whittle.WhittleIndex_PKG(self.num_mix_states, self.mix_traj_passive,
                                                            self.mix_traj_active,
                                                            self.mix_rewards, 0.99999999)
        else:
            self.whittle_indices = None

        # if require_whittle_idx:
        #     start = time.time()
        #     self.whittle_indices = whittle.binary_search(self.num_states, self.passive_traj, self.active_traj, self.rewards, self.discount)
        #     print(self.num_states)
        #     print(len(self.whittle_indices))
        #     self.computation_time = time.time() - start
        #     self.Whittle_for_LP()
        # print(self.whittle_indices)
        # self.act_window_len = 3
        # self.noact_len = 9
    def isLastDay(self, t):
        return (t % self.time_period_len) == self.time_period_len - 1

    def reset_all(self):
        self.not_inspected = True
        self.current_state = 0
    def reset(self):
        # self.current_state = 0
        # self.days_in_good_state = 0
        # self.inspect_cnt = [0] * len(self.action_window)
        self.not_inspected = True

    def isInWindow(self, t):
        for win in self.action_window:
            if t in win:
                return True
        return False

    def searchMulIdx(self, t):
        idx = -1
        for i in range(len(self.action_window)):
            if t in self.action_window[i]:
                idx = i
        if idx >= 0:
            return idx
        else:
            return -1

    def isLastEntry(self, t):
        for win in self.action_window:
            if t == win[-1]:
                return True
        return False

    def need_action(self, t):
        # mulidx = self.searchMulIdx(t)
        #
        # return self.inspect_cnt[mulidx] < self.multi[mulidx]
        return self.not_inspected

    def get_risk(self):
        mix_idx = self.time_ext_states[self.current_state][0]
        return self.mix_states[mix_idx][1]

    def state_step(self, action, t):
        if not action:
            self.ins_record.append(0)
            self.current_state = self.passive_traj[self.current_state]
        else:
            self.ins_record.append(1)
            self.current_state = self.active_traj[self.current_state]
        if self.sampling() == 0:
            self.days_in_good_state += 1

    def reset_ins_record(self):
        self.ins_record = []
        self.days_in_good_state = 0
        self.not_inspected = True

    def isCovered(self):
        total_ins = 0
        num_ins_inWin = 0
        num_ins_outWin = 0
        repeat = int(len(self.ins_record) / self.time_period_len)
        for i in range(repeat):
            num_ins_inWin = 0
            num_ins_outWin = 0
            for j in range(self.time_period_len):
                if self.ins_record[i * self.time_period_len + j]:
                    total_ins += 1
                    if self.isInWindow(j % self.time_period_len):
                        num_ins_inWin += 1
                    else:
                        num_ins_outWin += 1
            if sum(self.multi) > num_ins_inWin:
                return -1
        return 0
        # repeat = int(len(self.ins_record) / self.time_period_len)
        # total_ins_req = repeat * sum(self.multi)

        # if total_ins_req == total_ins:
        #     if num_ins_outWin == 0:
        #         return 0
        #     else:
        #         return 1
        # elif total_ins_req > total_ins:
        #     return 2
        # else:
        #     return 3
        # print(total_ins, num_ins_inWin, num_ins_outWin)
        #
        # if total_ins_req > num_ins_inWin:
        #     return -1
        # elif total_ins_req == num_ins_inWin:
        #     return 0
        # else:
        #     return 1

    def mix_state_step(self, action):
        if not action:
            self.current_state = self.mix_traj_passive[self.current_state]
        else:
            self.current_state = self.mix_traj_active[self.current_state]

    def get_state(self):
        mix_idx = self.time_ext_states[self.current_state][0]
        # return self.current_state, self.time_ext_states[self.current_state], self.mix_states[mix_idx]
        return mix_idx

    def is_in_action_win(self, t):
        return self.isInWindow(t)

    def sampling(self):
        mix_idx = self.time_ext_states[self.current_state][0]

        # print(x)
        rel = 0
        for i in range(20):
            x = np.random.rand()
            if x < self.mix_states[mix_idx][0]:
                rel += 1
            else:
                rel -= 1
        if rel >= 0:
            return 0
        else:
            return 1

    def get_index(self, t):
        assert self.current_state < len(self.whittle_indices)
        return self.whittle_indices[self.current_state]

    def get_mix_index(self, t):
        mix_idx = self.time_ext_states[self.current_state][0]
        return self.mix_whittle_indices[mix_idx]

    def reward_whittle(self):
        return self.rewards[self.current_state]

    def reward_mix_whittle(self):
        return self.mix_rewards[self.current_state]

    # def Whittle_for_LP(self):
    #     # plan for 12mon for now
    #     self.whittle_indices_lp = np.zeros((self.time_period_len, self.time_period_len, 2))
    #     for t in range(self.time_period_len):
    #         for last_t in range(self.time_period_len):
    #             s = last_t % self.num_mix_states
    #             idx = self.time_ext_states.index([s,t,0])
    #             self.whittle_indices_lp[t,last_t,0] = self.whittle_indices[idx]
    #             if t in self.action_window:
    #                 idx1 = self.time_ext_states.index([s,t,1])
    #                 self.whittle_indices_lp[t,last_t,1] = self.whittle_indices[idx1]
    #             else:
    #                 self.whittle_indices_lp[t,last_t,1] = self.whittle_indices[idx]

    def Whittle_for_LP(self):
        whittle_indices_lp = [-1] * self.time_period_len
        
        current_state = self.time_ext_states[self.current_state][0]
        if self.whittle_indices is not None:
            for t in range(self.time_period_len):

                time_step = t % self.time_period_len
                if not self.is_in_action_win(time_step):
                    continue
                else:

                    state = (current_state + t) % self.num_mix_states
                    idx = self.time_ext_states.index([state, t, 0])

                    whittle_indices_lp[t] = self.whittle_indices[idx]

            # print(self.time_ext_states[idx])


        return whittle_indices_lp

    def Whittle_for_LP_multi(self):
        # plan for 12mon for now and only
        whittle_indices_lp = [0] * self.time_period_len * 5
        last_t = 0
        for t in range(self.time_period_len * 5):
            # for t in range(self.time_period_len):
            time_step = t % self.time_period_len
            if not self.is_in_action_win(time_step):
                if (t - last_t) >= self.num_mix_states - 1:
                    s = self.num_mix_states - 1
                else:

                    s = t - last_t
                idx = self.time_ext_states.index([s, time_step, 0])
            elif time_step in self.action_window[0] or time_step in self.action_window[1]:
                if time_step == self.action_window[0][1] or time_step == self.action_window[1][1]:
                    if (t - last_t) >= self.num_mix_states - 1:
                        s = self.num_mix_states - 1
                    else:

                        s = t - last_t
                    idx = self.time_ext_states.index([s, time_step, 0])
                    last_t = t
                else:
                    if (t - last_t) >= self.num_mix_states - 1:
                        s = self.num_mix_states - 1
                    else:

                        s = t - last_t
                    idx = self.time_ext_states.index([s, time_step, 0])

            whittle_indices_lp[t] = self.whittle_indices[idx]

            print(self.time_ext_states[idx])
        return whittle_indices_lp

    def debug(self):
        print("Arm mix states:", end="")
        print(self.mix_states)
        print("Rewards: ", end=" ")
        print(self.mix_rewards)
        print("Action Window", end=" ")
        print(self.action_window)
        print()
        print()

    def nan_count(self):
        tmp = np.array(self.whittle_indices)
        return np.sum(np.isnan(tmp))

    # def value_iter(self):
    #     vi = mdptoolbox.mdp.ValueIteration(np.array([self.P0, self.P1]),
    #                                        np.transpose(np.array([self.rewards, self.rewards])), self.discount)
    #     vi.run()
    #     self.V = vi.V
    #     q = mdptoolbox.mdp.QLearning(np.array([self.P0, self.P1]), np.transpose(np.array([self.rewards, self.rewards])),
    #                                  self.discount)
    #     q.run()
    #     self.Q = q.Q

    # def whittle_index(self, range, start_state_idx):
    #     left = range[0]
    #     right = range[1]
    #     last_m = 9999999
    #     m = (left + right) / 2
    #     while (1):
    #         if abs(last_m - m) < 1e-3:
    #             break
    #         Val_passive = m + self.rewards[start_state_idx] + self.Q[start_state_idx][0]
    #         Val_active = self.rewards[start_state_idx] + self.discount * self.Q[start_state_idx][1]
    #         if Val_passive >= Val_active:
    #             right = m
    #         else:
    #             left = m
    #         m = (left + right) / 2
    #         print("m is " + str(m))
    #         print("passive is " + str(Val_passive))
    #         print("active is " + str(Val_active))

    # print("The whittle index is :" + str(m))


class WIP:
    # def __init__(self, no_action_trans, action_trans, time_win, init_state=np.array([1, 0]), discount=1):
    def __init__(self, no_action_trans, action_trans, act_win, init_state=np.array([1, 0]), discount=1):
        self.state = init_state
        self.action_trans = action_trans
        self.no_action_trans = no_action_trans
        self.discount = 0.99
        self.time_period_len = 12
        self.action_window = [act_win, act_win + 1]
        self.states = [init_state]
        self.current_state = 0
        self.days_in_good_state = 0
        self.inspect_cnt = 0
        self.not_inspected = True
        self.fw = False
        self.sw = False

        self.cnt = 0
        while (1):
            new_state = np.dot(self.states[-1], self.no_action_trans)
            if (abs(new_state - self.states[-1]) < 1e-2).all():
                break
            self.states.append(new_state)
        # print(self.mix_states)
        # for i in range(10):
        #     new_state = np.dot(self.mix_states[-1], self.no_action_trans)
        #     self.mix_states.append(new_state)
        #     # print(new_state)

        self.num_states = len(self.states)

        self.traj_active = [0] * self.num_states
        self.traj_passive = [i + 1 for i in range(self.num_states)]
        self.traj_passive[-1] = self.num_states - 1
        self.rewards = []
        for state in self.states:
            self.rewards.append(float(state[0]))

        self.whittle_indices = whittle.WhittleIndex_PKG(self.num_states, self.traj_passive, self.traj_active, self.rewards,
                                                     self.discount)
        # self.computation_time = time.time() - start

        # self.Whittle_for_LP()
        # print(self.whittle_indices)
        # self.act_window_len = 3
        # self.noact_len = 9

    def reset(self):
        pass

    def reset_all(self):
        self.current_state = 0

    def state_step(self, action, t):
        if action and self.is_in_action_win(t):
            self.current_state = self.traj_active[self.current_state]
        else:
            self.current_state = self.traj_passive[self.current_state]
        

       

    def mix_state_step(self, action):
        if not action:
            self.current_state = self.mix_sraj_passive[self.current_state]
        else:
            self.current_state = self.mix_traj_active[self.current_state]

    def get_state(self):
        mix_idx = self.time_ext_states[self.current_state][0]
        return self.current_state, self.time_ext_states[self.current_state], self.mix_states[mix_idx]

    def is_in_action_win(self, t):
        return t in self.action_window

    def sampling(self):
        mix_idx = self.states[self.current_state][0]
        x = np.random.rand()
        # print(x)
        if x < mix_idx:
            return 0
        else:
            return 1

    def get_index(self, t):
        assert self.current_state < len(self.whittle_indices)
        return self.whittle_indices[self.current_state]

    def get_mix_index(self, t):
        return self.mix_whittle[self.current_state]

    def reward_whittle(self):
        return self.rewards[self.current_state]

    def reward_mix_whittle(self):
        return self.mix_rewards[self.current_state]

    # def Whittle_for_LP(self):
    #     # plan for 12mon for now 
    #     self.whittle_indices_lp = np.zeros((self.time_period_len, self.time_period_len, 2))
    #     for t in range(self.time_period_len):
    #         for last_t in range(self.time_period_len):
    #             s = last_t % self.num_mix_states
    #             idx = self.time_ext_states.index([s,t,0])
    #             self.whittle_indices_lp[t,last_t,0] = self.whittle_indices[idx]
    #             if t in self.action_window:
    #                 idx1 = self.time_ext_states.index([s,t,1])
    #                 self.whittle_indices_lp[t,last_t,1] = self.whittle_indices[idx1]
    #             else:
    #                 self.whittle_indices_lp[t,last_t,1] = self.whittle_indices[idx]

    # def Whittle_for_LP(self):
    #     # plan for 12mon for now and only
    #     self.whittle_indices_lp = np.zeros((self.time_period_len, self.time_period_len))
    #     for last_t in range(self.time_period_len):
    #         for t in range(self.time_period_len):
    #             s = last_t % self.num_mix_states
    #             idx = self.time_ext_states.index([s,t,0])
    #             self.whittle_indices_lp[last_t, t] = self.whittle_indices[idx]

    def debug(self):
        print("Arm mix states:", end="")
        print(self.mix_states)
        print("Rewards: ", end=" ")
        print(self.mix_rewards)
        print("Action Window", end=" ")
        print(self.action_window)
        print()
        print()

    def nan_count(self):
        tmp = np.array(self.whittle_indices)
        return np.sum(np.isnan(tmp))

    # def value_iter(self):
    #     vi = mdptoolbox.mdp.ValueIteration(np.array([self.P0, self.P1]),
    #                                        np.transpose(np.array([self.rewards, self.rewards])), self.discount)
    #     vi.run()
    #     self.V = vi.V
    #     q = mdptoolbox.mdp.QLearning(np.array([self.P0, self.P1]), np.transpose(np.array([self.rewards, self.rewards])),
    #                                  self.discount)
    #     q.run()
    #     self.Q = q.Q

    # def whittle_index(self, range, start_state_idx):
    #     left = range[0]
    #     right = range[1]
    #     last_m = 9999999
    #     m = (left + right) / 2
    #     while (1):
    #         if abs(last_m - m) < 1e-3:
    #             break
    #         Val_passive = m + self.rewards[start_state_idx] + self.Q[start_state_idx][0]
    #         Val_active = self.rewards[start_state_idx] + self.discount * self.Q[start_state_idx][1]
    #         if Val_passive >= Val_active:
    #             right = m
    #         else:
    #             left = m
    #         m = (left + right) / 2
    #         print("m is " + str(m))
    #         print("passive is " + str(Val_passive))
    #         print("active is " + str(Val_active))

    # print("The whittle index is :" + str(m))


class WIP_Arm:
    # def __init__(self, no_action_trans, action_trans, time_win, init_state=np.array([1, 0]), discount=1):
    def __init__(self, no_action_trans, action_trans, multi=[1], init_state=np.array([1, 0]),
                 discount=1,
                 require_whittle_idx=True):
        self.state = init_state
        self.action_trans = action_trans
        self.no_action_trans = no_action_trans
        self.discount = 0.99
        self.time_period_len = 12
        # multi ins in one action window

        self.mix_states = [init_state]
        self.current_state = 0
        self.days_in_good_state = 0
        self.not_inspected = True
        self.cnt = 0
        self.ins_record = []
        while (1):
            new_state = np.dot(self.mix_states[-1], self.no_action_trans)
            if (abs(new_state - self.mix_states[-1]) < 1e-2).all():
                break
            self.mix_states.append(new_state)
        # print(self.mix_states)
        # for i in range(10):
        #     new_state = np.dot(self.mix_states[-1], self.no_action_trans)
        #     self.mix_states.append(new_state)
        #     # print(new_state)

        self.num_states = len(self.mix_states)
        self.rewards = []
        for state in self.mix_states:
            self.rewards.append(float(state[0]))
        self.traj_passive = [i + 1 for i in range(self.num_states)]
        self.traj_passive[-1] = self.num_states - 1
        self.traj_active = [0] * self.num_states
        self.whittle_indices = whittle.WhittleIndex_PKG(self.num_states, self.traj_passive, self.traj_active, self.rewards,
                                                     self.discount)
        # print(self.mix_whittle)
        # print(len(self.time_ext_states))
        # print(len(self.rewards))
        # print(self.time_ext_states)
        # print(self.rewards)
        # print(self.mix_traj_passive)
        # print(self.mix_traj_active)

        # print(self.rewards)
        # self.P1 = [[0.0 for i in range(self.num_mix_states)] for j in range(self.num_mix_states)]
        # self.P0 = [[0.0 for i in range(self.num_mix_states)] for j in range(self.num_mix_states)]
        # for i in range(self.num_mix_states):
        #     for j in range(self.num_mix_states):
        #         if j == 0:
        #             self.P1[i][j] = 1.0
        #         if j - i == 1:
        #             self.P0[i][j] = 1.0
        # self.P0[-1][-1] = 1.0
        # self.rewards_P0P1 = []
        # self.P0 = np.array(self.P0)
        # self.P1 = np.array(self.P1)
        #
        # for state in self.mix_states:
        #     self.rewards_P0P1.append(state[0])
        # self.rewards_P0P1 = np.array(self.rewards_P0P1)
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
        #
        # self.passive_trans = np.zeros([self.num_states, self.num_states])
        # self.active_trans = np.zeros([self.num_states, self.num_states])
        # for i in range(self.num_states):
        #     self.passive_trans[i][self.passive_traj[i]] = 1.0
        #     self.active_trans[i][self.active_traj[i]] = 1.0
        # mb = bandit.restless_bandit_from_P0P1_R0R1(self.passive_trans, self.active_trans, self.rewards, self.rewards)
        # self.mbi = mb.whittle_indices(discount=0.99)
        #
        # if np.isnan(np.sum(self.mbi)):
        #     print(self.no_action_trans)
        #     print(self.mix_states)
        # print(mbi)

        # if require_whittle_idx:
        #     start = time.time()
        #     self.whittle_indices = whittle.binary_search(self.num_states, self.passive_traj, self.active_traj, self.rewards, self.discount)
        #     print(self.num_states)
        #     print(len(self.whittle_indices))
        #     self.computation_time = time.time() - start
        #     self.Whittle_for_LP()
        # print(self.whittle_indices)
        # self.act_window_len = 3
        # self.noact_len = 9

    def reset_all(self):
        self.current_state = 0
        self.days_in_good_state = 0
        # self.inspect_cnt = [0] * len(self.action_window)
        pass

    def reset(self):
        # self.current_state = 0
        # self.days_in_good_state = 0
        # self.inspect_cnt = [0] * len(self.action_window)
        self.not_inspected = True
        pass



    def need_action(self, t):
        mulidx = self.searchMulIdx(t)

        return self.inspect_cnt[mulidx] < self.multi[mulidx]

    def get_risk(self):
        mix_idx = self.time_ext_states[self.current_state][0]
        return self.mix_states[mix_idx][1]

    def state_step(self, action, t):
        if  action:

            self.ins_record.append(1)
            self.current_state = self.traj_active[self.current_state]

        else:
            self.ins_record.append(0)
            self.current_state = self.traj_passive[self.current_state]



    def reset_ins_record(self):
        self.ins_record = []
        self.days_in_good_state = 0

    def isCovered(self):
        total_ins = 0
        num_ins_inWin = 0
        num_ins_outWin = 0
        repeat = int(len(self.ins_record) / self.time_period_len)
        for i in range(repeat):
            num_ins_inWin = 0
            num_ins_outWin = 0
            for j in range(self.time_period_len):
                if self.ins_record[i * self.time_period_len + j]:
                    total_ins += 1
                    if self.isInWindow(j % self.time_period_len):
                        num_ins_inWin += 1
                    else:
                        num_ins_outWin += 1
            if sum(self.multi) > num_ins_inWin:
                return -1
        return 0
        # repeat = int(len(self.ins_record) / self.time_period_len)
        # total_ins_req = repeat * sum(self.multi)

        # if total_ins_req == total_ins:
        #     if num_ins_outWin == 0:
        #         return 0
        #     else:
        #         return 1
        # elif total_ins_req > total_ins:
        #     return 2
        # else:
        #     return 3
        # print(total_ins, num_ins_inWin, num_ins_outWin)
        #
        # if total_ins_req > num_ins_inWin:
        #     return -1
        # elif total_ins_req == num_ins_inWin:
        #     return 0
        # else:
        #     return 1

    def mix_state_step(self, action):
        if not action:
            self.current_state = self.mix_traj_passive[self.current_state]
        else:
            self.current_state = self.mix_traj_active[self.current_state]

    def get_state(self):

        # return self.current_state, self.time_ext_states[self.current_state], self.mix_states[mix_idx]
        return self.current_state

    def is_in_action_win(self, t):
        return self.isInWindow(t)

    def sampling(self):
        mix_idx = self.current_state

        # print(x)
        rel = 0
        for i in range(20):
            x = np.random.rand()
            if x < self.mix_states[mix_idx][0]:
                rel += 1
            else:
                rel -= 1
        if rel >= 0:
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

    # def Whittle_for_LP(self):
    #     # plan for 12mon for now
    #     self.whittle_indices_lp = np.zeros((self.time_period_len, self.time_period_len, 2))
    #     for t in range(self.time_period_len):
    #         for last_t in range(self.time_period_len):
    #             s = last_t % self.num_mix_states
    #             idx = self.time_ext_states.index([s,t,0])
    #             self.whittle_indices_lp[t,last_t,0] = self.whittle_indices[idx]
    #             if t in self.action_window:
    #                 idx1 = self.time_ext_states.index([s,t,1])
    #                 self.whittle_indices_lp[t,last_t,1] = self.whittle_indices[idx1]
    #             else:
    #                 self.whittle_indices_lp[t,last_t,1] = self.whittle_indices[idx]

    def Whittle_for_LP(self):
        # plan for 12mon for now and only
        s0 = self.time_ext_states[self.current_state][0]
        # whittle_indices_lp = np.zeros((self.time_period_len, self.time_period_len))
        whittle_indices_lp = [0] * self.time_period_len
        for last_t in range(self.time_period_len):
            # for t in range(self.time_period_len):
            if s0 + last_t >= self.num_mix_states - 1:
                s = self.num_mix_states - 1
            else:

                s = s0 + last_t
            idx = self.time_ext_states.index([s, last_t, 0])
            whittle_indices_lp[last_t] = self.whittle_indices[idx]

        return whittle_indices_lp

    def Whittle_for_LP_multi(self):
        # plan for 12mon for now and only
        whittle_indices_lp = [0] * self.time_period_len * 5
        # for last_t in range(self.time_period_len):
        #     # for t in range(self.time_period_len):
        #     if last_t  <= self.action_window[0][1]:
        #         if last_t >= self.num_mix_states - 1:
        #             s = self.num_mix_states - 1
        #         else:
        #
        #             s = last_t
        #         idx = self.time_ext_states.index([s, last_t, 0])
        #         whittle_indices_lp[last_t] = self.whittle_indices[idx]
        #     elif last_t < self.action_window[1][1]:
        #         t = last_t - self.action_window[0][1]
        #         if t >= self.num_mix_states - 1:
        #             s = self.num_mix_states - 1
        #         else:
        #
        #             s = t
        #         idx = self.time_ext_states.index([s, t, 0])
        #         whittle_indices_lp[last_t] = self.whittle_indices[idx]
        #     else:
        #         t = last_t - self.action_window[1][1]
        #         if t >= self.num_mix_states - 1:
        #             s = self.num_mix_states - 1
        #         else:
        #
        #             s = t
        #         idx = self.time_ext_states.index([s, t, 0])
        #         whittle_indices_lp[last_t] = self.whittle_indices[idx]
        #
        #     print(self.time_ext_states[idx])
        last_t = 0
        for t in range(self.time_period_len * 5):
            # for t in range(self.time_period_len):
            time_step = t % self.time_period_len
            if not self.is_in_action_win(time_step):
                if (t - last_t) >= self.num_mix_states - 1:
                    s = self.num_mix_states - 1
                else:

                    s = t - last_t
                idx = self.time_ext_states.index([s, time_step, 0])
            elif time_step in self.action_window[0] or time_step in self.action_window[1]:
                if time_step == self.action_window[0][1] or time_step == self.action_window[1][1]:
                    if (t - last_t) >= self.num_mix_states - 1:
                        s = self.num_mix_states - 1
                    else:

                        s = t - last_t
                    idx = self.time_ext_states.index([s, time_step, 0])
                    last_t = t
                else:
                    if (t - last_t) >= self.num_mix_states - 1:
                        s = self.num_mix_states - 1
                    else:

                        s = t - last_t
                    idx = self.time_ext_states.index([s, time_step, 0])

            whittle_indices_lp[t] = self.whittle_indices[idx]

            print(self.time_ext_states[idx])
        return whittle_indices_lp

    def debug(self):
        print("Arm mix states:", end="")
        print(self.mix_states)
        print("Rewards: ", end=" ")
        print(self.mix_rewards)
        print("Action Window", end=" ")
        print(self.action_window)
        print()
        print()

    def nan_count(self):
        tmp = np.array(self.whittle_indices)
        return np.sum(np.isnan(tmp))

    # def value_iter(self):
    #     vi = mdptoolbox.mdp.ValueIteration(np.array([self.P0, self.P1]),
    #                                        np.transpose(np.array([self.rewards, self.rewards])), self.discount)
    #     vi.run()
    #     self.V = vi.V
    #     q = mdptoolbox.mdp.QLearning(np.array([self.P0, self.P1]), np.transpose(np.array([self.rewards, self.rewards])),
    #                                  self.discount)
    #     q.run()
    #     self.Q = q.Q

    # def whittle_index(self, range, start_state_idx):
    #     left = range[0]
    #     right = range[1]
    #     last_m = 9999999
    #     m = (left + right) / 2
    #     while (1):
    #         if abs(last_m - m) < 1e-3:
    #             break
    #         Val_passive = m + self.rewards[start_state_idx] + self.Q[start_state_idx][0]
    #         Val_active = self.rewards[start_state_idx] + self.discount * self.Q[start_state_idx][1]
    #         if Val_passive >= Val_active:
    #             right = m
    #         else:
    #             left = m
    #         m = (left + right) / 2
    #         print("m is " + str(m))
    #         print("passive is " + str(Val_passive))
    #         print("active is " + str(Val_active))

    # print("The whittle index is :" + str(m))

class Arm:
    # def __init__(self, no_action_trans, action_trans, time_win, init_state=np.array([1, 0]), discount=1):
    def __init__(self, no_action_trans, action_trans, act_win, win_len, period_len, whittle_index, multi=[1],
                 init_state=np.array([1, 0]),
                 discount=1,
                 require_whittle_idx=True, need_trans = True):
        self.state = init_state
        self.action_trans = action_trans
        self.no_action_trans = no_action_trans
        self.discount = 0.99
        self.time_period_len = period_len
        # multi ins in one action window
        # self.multi = act_win[1]
        
        if type(act_win) is list:
            day1 = act_win[0]
            day2 = act_win[1]
        else:
            day1 = act_win
            day2 = None
        # day2 = act_win[0][1]
        # day3 = act_win[0][2]
        # self.action_window = [[i for i in range(day1, day1 + 5)]]
        # self.action_window = [act_win]

        self.action_window = [[i for i in range(day1, day1 + win_len)]]
        if day2:
            self.action_window.append([i for i in range(day2, day2 + win_len)])
        self.multi = [1] * len(self.action_window)
        self.mix_states = [init_state]
        self.current_state = 0
        self.days_in_good_state = 0
        # self.inspect_cnt = [0] * len(self.action_window)
        self.not_inspected = True
        self.cnt = 0
        self.win_len = win_len
        self.ins_record = []
        while (1):
            new_state = np.dot(self.mix_states[-1], self.no_action_trans)
            if (abs(new_state - self.mix_states[-1]) < 1e-2).all():
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
                if t != 0:
                    multi = self.multi[self.searchMulIdx(t)]
                    for mul in range(1, multi + 1):
                        self.time_ext_states.append([i, t, mul])

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
            if self.searchMulIdx(t) >= 0:

                mul = self.multi[self.searchMulIdx(t)]
            else:
                mul = -1
            if not (self.isLastEntry(t % self.time_period_len) or self.isLastDay(t % self.time_period_len)):
                next_state = [self.mix_traj_passive[s_prob], (t + 1) % self.time_period_len, is_inspected]
            else:
                next_state = [self.mix_traj_passive[s_prob], (t + 1) % self.time_period_len, 0]
            # self.passive_traj.append(self.time_ext_states.index(next_state))
            self.passive_traj.append(self.time_ext_states.index(next_state))
            # active transition
            if not self.isInWindow(t):
                next_state = next_state  # not in action, same as passive trans
            elif self.isLastEntry(t % self.time_period_len) or self.isLastDay(t % self.time_period_len):

                if is_inspected < mul:
                    next_state = [self.mix_traj_active[s_prob], (t + 1) % self.time_period_len, 0]
                else:
                    next_state = [self.mix_traj_passive[s_prob], (t + 1) % self.time_period_len, 0]
            else:
                if is_inspected < mul:
                    next_state = [self.mix_traj_active[s_prob], (t + 1) % self.time_period_len, is_inspected + 1]
                else:
                    next_state = [self.mix_traj_passive[s_prob], (t + 1) % self.time_period_len, is_inspected]
            self.active_traj.append(self.time_ext_states.index(next_state))
            # rewards for each states
            self.rewards.append(float(self.mix_states[s_prob][0]))
            # self.rewards.append(float(1 - s_prob / len(self.mix_states)))
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
        # self.P1 = [[0.0 for i in range(self.num_mix_states)] for j in range(self.num_mix_states)]
        # self.P0 = [[0.0 for i in range(self.num_mix_states)] for j in range(self.num_mix_states)]
        # for i in range(self.num_mix_states):
        #     for j in range(self.num_mix_states):
        #         if j == 0:
        #             self.P1[i][j] = 1.0
        #         if j - i == 1:
        #             self.P0[i][j] = 1.0
        # self.P0[-1][-1] = 1.0
        # self.rewards_P0P1 = []
        # self.P0 = np.array(self.P0)
        # self.P1 = np.array(self.P1)
        #
        # for state in self.mix_states:
        #     self.rewards_P0P1.append(state[0])
        # self.rewards_P0P1 = np.array(self.rewards_P0P1)
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
        #
        # self.passive_trans = np.zeros([self.num_states, self.num_states])
        # self.active_trans = np.zeros([self.num_states, self.num_states])
        # for i in range(self.num_states):
        #     self.passive_trans[i][self.passive_traj[i]] = 1.0
        #     self.active_trans[i][self.active_traj[i]] = 1.0
        # mb = bandit.restless_bandit_from_P0P1_R0R1(self.passive_trans, self.active_trans, self.rewards, self.rewards)
        # self.mbi = mb.whittle_indices(discount=0.99)
        #
        # if np.isnan(np.sum(self.mbi)):
        #     print(self.no_action_trans)
        #     print(self.mix_states)
        # print(mbi)

        if require_whittle_idx:
            if whittle_index is not None:
                if need_trans:
                    self.whittle_indices = whittle_index
                else:
                    self.mix_whittle_indices = whittle_index
            else:
                if need_trans:

                    self.whittle_indices = whittle.binary_search(self.num_states, self.passive_traj, self.active_traj,
                                                                 self.rewards, self.discount)
                else:
                    self.mix_whittle_indices = whittle.WhittleIndex_PKG(self.num_mix_states, self.mix_traj_passive,
                                                            self.mix_traj_active,
                                                            self.mix_rewards, 0.99999999)
        else:
            self.whittle_indices = None

        # if require_whittle_idx:
        #     start = time.time()
        #     self.whittle_indices = whittle.binary_search(self.num_states, self.passive_traj, self.active_traj, self.rewards, self.discount)
        #     print(self.num_states)
        #     print(len(self.whittle_indices))
        #     self.computation_time = time.time() - start
        #     self.Whittle_for_LP()
        # print(self.whittle_indices)
        # self.act_window_len = 3
        # self.noact_len = 9
    def isLastDay(self, t):
        return (t % self.time_period_len) == self.time_period_len - 1

    def reset_all(self):
        self.inspect_cnt = [0] * len(self.action_window)
        self.current_state = 0
    def reset(self):
        # self.current_state = 0
        # self.days_in_good_state = 0
        self.inspect_cnt = [0] * len(self.action_window)
        # self.not_inspected = True

    def isInWindow(self, t):
        for win in self.action_window:
            if t in win:
                return True
        return False

    def searchMulIdx(self, t):
        idx = -1
        for i in range(len(self.action_window)):
            if t in self.action_window[i]:
                idx = i
        if idx >= 0:
            return idx
        else:
            return -1

    def isLastEntry(self, t):
        for win in self.action_window:
            if t == win[-1]:
                return True
        return False

    def need_action(self, t):
        mulidx = self.searchMulIdx(t)
        #
        return self.inspect_cnt[mulidx] < self.multi[mulidx]
        # return self.not_inspected

    def get_risk(self):
        mix_idx = self.time_ext_states[self.current_state][0]
        return self.mix_states[mix_idx][1]

    def state_step(self, action, t):
        if not action:
            self.ins_record.append(0)
            self.current_state = self.passive_traj[self.current_state]
        else:
            self.ins_record.append(1)
            if self.isInWindow(t % self.time_period_len) and self.need_action(t % self.time_period_len):
                self.current_state = self.active_traj[self.current_state]
                mulidx = self.searchMulIdx(t % self.time_period_len)
                self.inspect_cnt[mulidx] += 1
            else:
                self.current_state = self.passive_traj[self.current_state]

        if self.sampling() == 0:
            self.days_in_good_state += 1

    def reset_ins_record(self):
        self.ins_record = []
        self.days_in_good_state = 0
        self.not_inspected = True

    def isCovered(self):
        total_ins = 0
        num_ins_inWin = 0
        num_ins_outWin = 0
        repeat = int(len(self.ins_record) / self.time_period_len)
        for i in range(repeat):
            num_ins_inWin = 0
            num_ins_outWin = 0
            for j in range(self.time_period_len):
                if self.ins_record[i * self.time_period_len + j]:
                    total_ins += 1
                    if self.isInWindow(j % self.time_period_len):
                        num_ins_inWin += 1
                    else:
                        num_ins_outWin += 1
            if sum(self.multi) > num_ins_inWin:
                return -1
        return 0
        # repeat = int(len(self.ins_record) / self.time_period_len)
        # total_ins_req = repeat * sum(self.multi)

        # if total_ins_req == total_ins:
        #     if num_ins_outWin == 0:
        #         return 0
        #     else:
        #         return 1
        # elif total_ins_req > total_ins:
        #     return 2
        # else:
        #     return 3
        # print(total_ins, num_ins_inWin, num_ins_outWin)
        #
        # if total_ins_req > num_ins_inWin:
        #     return -1
        # elif total_ins_req == num_ins_inWin:
        #     return 0
        # else:
        #     return 1

    def mix_state_step(self, action):
        if not action:
            self.current_state = self.mix_traj_passive[self.current_state]
        else:
            self.current_state = self.mix_traj_active[self.current_state]

    def get_state(self):
        mix_idx = self.time_ext_states[self.current_state][0]
        # return self.current_state, self.time_ext_states[self.current_state], self.mix_states[mix_idx]
        return mix_idx

    def is_in_action_win(self, t):
        return self.isInWindow(t)

    def sampling(self):
        mix_idx = self.time_ext_states[self.current_state][0]

        # print(x)
        rel = 0
        for i in range(20):
            x = np.random.rand()
            if x < self.mix_states[mix_idx][0]:
                rel += 1
            else:
                rel -= 1
        if rel >= 0:
            return 0
        else:
            return 1

    def get_index(self, t):
        assert self.current_state < len(self.whittle_indices)
        return self.whittle_indices[self.current_state]

    def get_mix_index(self, t):
        mix_idx = self.time_ext_states[self.current_state][0]
        return self.mix_whittle_indices[mix_idx]

    def reward_whittle(self):
        return self.rewards[self.current_state]

    def reward_mix_whittle(self):
        return self.mix_rewards[self.current_state]

    # def Whittle_for_LP(self):
    #     # plan for 12mon for now
    #     self.whittle_indices_lp = np.zeros((self.time_period_len, self.time_period_len, 2))
    #     for t in range(self.time_period_len):
    #         for last_t in range(self.time_period_len):
    #             s = last_t % self.num_mix_states
    #             idx = self.time_ext_states.index([s,t,0])
    #             self.whittle_indices_lp[t,last_t,0] = self.whittle_indices[idx]
    #             if t in self.action_window:
    #                 idx1 = self.time_ext_states.index([s,t,1])
    #                 self.whittle_indices_lp[t,last_t,1] = self.whittle_indices[idx1]
    #             else:
    #                 self.whittle_indices_lp[t,last_t,1] = self.whittle_indices[idx]

    def Whittle_for_LP(self):
        whittle_indices_lp = [-1] * self.time_period_len
        
        current_state = self.time_ext_states[self.current_state][0]
        if self.whittle_indices is not None:
            for t in range(self.time_period_len):

                time_step = t % self.time_period_len
                if not self.is_in_action_win(time_step):
                    continue
                else:

                    state = (current_state + t) % self.num_mix_states
                    idx = self.time_ext_states.index([state, t, 0])

                    whittle_indices_lp[t] = self.whittle_indices[idx]

            # print(self.time_ext_states[idx])


        return whittle_indices_lp

    def Whittle_for_LP_twice(self):
        # plan for 12mon for now and only
        debug_info = []
        whittle_indices_lp = [-1] * self.time_period_len * (self.win_len + 2)
        current_state = self.time_ext_states[self.current_state][0]
        for t in range(self.time_period_len):
            if self.searchMulIdx(t) == 0:
                next_s = (current_state + t) % self.num_mix_states
                idx = self.time_ext_states.index([next_s, t, 0])
                whittle_indices_lp[t] = self.whittle_indices[idx]
                debug_info.append([[next_s, t, 0]])
        last_t = 0
        for t in range(self.time_period_len, self.time_period_len * 2):
            if self.searchMulIdx(t % self.time_period_len) == 1:
                next_s = (t % self.time_period_len - last_t) % self.num_mix_states
                idx = self.time_ext_states.index([next_s, t % self.time_period_len, 0])
                whittle_indices_lp[t] = self.whittle_indices[idx]
                debug_info.append([[next_s, t % self.time_period_len, 0]])
        last_t = self.action_window[0][0]
        for t in range(self.time_period_len * 2, self.time_period_len * 3):
            if self.searchMulIdx(t % self.time_period_len) == 1:
                next_s = (t % self.time_period_len - last_t) % self.num_mix_states
                idx = self.time_ext_states.index([next_s, t % self.time_period_len, 0])
                whittle_indices_lp[t] = self.whittle_indices[idx]
                debug_info.append([[next_s, t % self.time_period_len, 0]])
        last_t = self.action_window[0][1]
        for t in range(self.time_period_len * 3, self.time_period_len * 4):
            if self.searchMulIdx(t % self.time_period_len) == 1:
                next_s = (t % self.time_period_len - last_t) % self.num_mix_states
                idx = self.time_ext_states.index([next_s, t % self.time_period_len, 0])
                whittle_indices_lp[t] = self.whittle_indices[idx]
                debug_info.append([[next_s, t % self.time_period_len, 0]])
        # print(debug_info)
        return whittle_indices_lp
        

    def debug(self):
        print("Arm mix states:", end="")
        print(self.mix_states)
        print("Rewards: ", end=" ")
        print(self.mix_rewards)
        print("Action Window", end=" ")
        print(self.action_window)
        print()
        print()

    def nan_count(self):
        tmp = np.array(self.whittle_indices)
        return np.sum(np.isnan(tmp))

    # def value_iter(self):
    #     vi = mdptoolbox.mdp.ValueIteration(np.array([self.P0, self.P1]),
    #                                        np.transpose(np.array([self.rewards, self.rewards])), self.discount)
    #     vi.run()
    #     self.V = vi.V
    #     q = mdptoolbox.mdp.QLearning(np.array([self.P0, self.P1]), np.transpose(np.array([self.rewards, self.rewards])),
    #                                  self.discount)
    #     q.run()
    #     self.Q = q.Q

    # def whittle_index(self, range, start_state_idx):
    #     left = range[0]
    #     right = range[1]
    #     last_m = 9999999
    #     m = (left + right) / 2
    #     while (1):
    #         if abs(last_m - m) < 1e-3:
    #             break
    #         Val_passive = m + self.rewards[start_state_idx] + self.Q[start_state_idx][0]
    #         Val_active = self.rewards[start_state_idx] + self.discount * self.Q[start_state_idx][1]
    #         if Val_passive >= Val_active:
    #             right = m
    #         else:
    #             left = m
    #         m = (left + right) / 2
    #         print("m is " + str(m))
    #         print("passive is " + str(Val_passive))
    #         print("active is " + str(Val_active))

    # print("The whittle index is :" + str(m))

class FullObservedArm:
    # def __init__(self, no_action_trans, action_trans, time_win, init_state=np.array([1, 0]), discount=1):
    def __init__(self, no_action_trans, action_trans, act_win, init_state=0, discount=0.99):
        self.whittle_indices = None
        self.state = init_state
        self.action_trans = action_trans
        self.no_action_trans = no_action_trans
        self.discount = discount
        self.time_period_len = 12
        self.action_window = act_win
        self.current_state = 0
        self.not_inspected = True
        self.days_in_good_state = 0
        self.inspect_cnt = 0
        self.whittle_index()
        self.cnt = 0

        # print(self.whittle_indices)

    def state_step(self, action, t):
        if not self.not_inspected:
            self.current_state = self.sampling(self.no_action_trans[self.current_state][0])
        else:
            if not action:
                self.current_state = self.sampling(self.no_action_trans[self.current_state][0])
            else:
                self.not_inspected = False
                self.cnt += 1
                self.current_state = self.sampling(self.action_trans[self.current_state][0])
        if self.current_state == 0:
            self.days_in_good_state += 1
        if action:
            self.inspect_cnt += 1

    def reset(self):
        self.current_state = 0
        self.not_inspected = True
        self.days_in_good_state = 0
        self.inspect_cnt = 0

    def get_state(self):
        mix_idx = self.time_ext_states[self.current_state][0]
        return self.current_state, self.time_ext_states[self.current_state], self.mix_states[mix_idx]

    def is_in_action_win(self, t):
        return t in self.action_window

    def sampling(self, p):
        x = np.random.rand()
        if x < p:
            return 0
        else:
            return 1

    def get_index(self, t):

        return self.whittle_indices[self.current_state]

    def reward_whittle(self):
        return 1 - self.current_state

    def debug(self):
        print(self.current_state)
        # print(self.P0)
        # print(self.P1)
        # print(self.rewards)
        # print(self.whittle_indices)
        # print(self.nan_count())

    def whittle_index(self):
        self.whittle_indices = whittle.binary_search1(2, self.no_action_trans, self.action_trans, [-1, 0],
                                                      self.discount)


class FullObservedArmTimed:
    # def __init__(self, no_action_trans, action_trans, time_win, init_state=np.array([1, 0]), discount=1):
    def __init__(self, no_action_trans, action_trans, act_win, init_state=0, discount=0.99):
        self.whittle_indices = None
        self.state = init_state
        self.action_trans = action_trans
        self.no_action_trans = no_action_trans
        self.discount = discount
        self.time_period_len = 12
        self.action_window = act_win
        self.current_state = 0
        self.not_inspected = True
        self.time_period_len = 12
        self.days_in_good_state = 0
        self.inspect_cnt = 0
        self.cnt = 0
        # self.whittle_index()
        # print(self.whittle_indices)
        self.time_ext_states = []
        self.rewards = []
        self.R = []
        self.mix_states = [0, 1]
        for i in range(len(self.mix_states)):
            for t in range(self.time_period_len):
                self.time_ext_states.append([i, t, 0])
                if t in self.action_window:
                    self.time_ext_states.append([i, t, 1])
        self.num_states = len(self.time_ext_states)
        for state in self.time_ext_states:
            self.rewards.append(1 - state[0])
            self.R.append(state[0])
        # generate new tran matrix
        self.active_trans = np.zeros([self.num_states, self.num_states])
        self.passive_trans = np.zeros([self.num_states, self.num_states])
        for i in range(self.num_states):
            # passive transition
            s = self.time_ext_states[i]
            s_prob = s[0]
            t = s[1]
            is_inspected = s[2]
            if t in self.action_window and t != self.action_window[-1]:
                next_state_good = [0, (t + 1) % self.time_period_len, is_inspected]
                next_state_bad = [1, (t + 1) % self.time_period_len, is_inspected]
            else:
                next_state_good = [0, (t + 1) % self.time_period_len, 0]
                next_state_bad = [1, (t + 1) % self.time_period_len, 0]
            good_idx = self.time_ext_states.index(next_state_good)
            bad_idx = self.time_ext_states.index(next_state_bad)
            self.passive_trans[i][good_idx] = self.no_action_trans[s_prob][0]
            self.passive_trans[i][bad_idx] = self.no_action_trans[s_prob][1]
            # active transition
            if not t in self.action_window:
                self.active_trans[i][good_idx] = self.no_action_trans[s_prob][0]
                self.active_trans[i][bad_idx] = self.no_action_trans[s_prob][1]
            elif t in self.action_window and t != self.action_window[-1]:
                if is_inspected:
                    next_state_good = [0, (t + 1) % self.time_period_len, 1]
                    next_state_bad = [1, (t + 1) % self.time_period_len, 1]
                    good_idx = self.time_ext_states.index(next_state_good)
                    bad_idx = self.time_ext_states.index(next_state_bad)
                    self.active_trans[i][good_idx] = self.no_action_trans[s_prob][0]
                    self.active_trans[i][bad_idx] = self.no_action_trans[s_prob][1]
                else:
                    next_state_good = [0, (t + 1) % self.time_period_len, 1]
                    next_state_bad = [1, (t + 1) % self.time_period_len, 1]
                    good_idx = self.time_ext_states.index(next_state_good)
                    bad_idx = self.time_ext_states.index(next_state_bad)
                    self.active_trans[i][good_idx] = self.action_trans[s_prob][0]
                    self.active_trans[i][bad_idx] = self.action_trans[s_prob][1]
            else:
                if is_inspected:
                    next_state_good = [0, (t + 1) % self.time_period_len, 0]
                    next_state_bad = [1, (t + 1) % self.time_period_len, 0]
                    good_idx = self.time_ext_states.index(next_state_good)
                    bad_idx = self.time_ext_states.index(next_state_bad)
                    self.active_trans[i][good_idx] = self.no_action_trans[s_prob][0]
                    self.active_trans[i][bad_idx] = self.no_action_trans[s_prob][1]
                else:
                    next_state_good = [0, (t + 1) % self.time_period_len, 0]
                    next_state_bad = [1, (t + 1) % self.time_period_len, 0]
                    good_idx = self.time_ext_states.index(next_state_good)
                    bad_idx = self.time_ext_states.index(next_state_bad)
                    self.active_trans[i][good_idx] = self.action_trans[s_prob][0]
                    self.active_trans[i][bad_idx] = self.action_trans[s_prob][1]
            # self.active_traj.append(tmp)
            # # rewards for each states
            # self.rewards.append(float(self.mix_states[s_prob][0]))

        self.whittle_index()
        # print(self.whittle_indices)

    def state_step(self, action, t):
        if not action:
            self.current_state = self.state_sampling(True)
        else:
            self.current_state = self.state_sampling(False)
        if self.time_ext_states[self.current_state][0] == 0:
            self.days_in_good_state += 1
        if action:
            self.inspect_cnt += 1

    def reset(self):
        self.not_inspected = True
        self.current_state = 0

    def get_state(self):
        mix_idx = self.time_ext_states[self.current_state][0]
        return self.current_state, self.time_ext_states[self.current_state], self.mix_states[mix_idx]

    def is_in_action_win(self, t):
        return t in self.action_window

    def state_sampling(self, passive=True):
        x = np.random.rand()
        if passive:
            states = []
            p = []
            for i in range(self.num_states):
                if self.passive_trans[self.current_state][i] > 0:
                    states.append(i)
                    p.append(self.passive_trans[self.current_state][i])
            assert len(states) <= 2
            if x < p[0]:
                return states[0]
            else:
                return states[1]
        else:
            states = []
            p = []
            for i in range(self.num_states):
                if self.active_trans[self.current_state][i] > 0:
                    states.append(i)
                    p.append(self.active_trans[self.current_state][i])
            assert len(states) <= 2
            if x < p[0]:
                return states[0]
            else:
                return states[1]

    def get_index(self, t):

        return self.whittle_indices[self.current_state]

    def reward_whittle(self):
        return 1 - self.time_ext_states[self.current_state][0]

    def debug(self):
        print(self.current_state)
        # print(self.P0)
        # print(self.P1)
        # print(self.rewards)
        # print(self.whittle_indices)
        # print(self.nan_count())

    def whittle_index(self):
        # mb = bandit.restless_bandit_from_P0P1_R0R1(self.passive_trans, self.active_trans, self.R, self.R)
        # mbi = mb.whittle_indices(discount=0.99)
        # self.whittle_indices = mbi
        self.whittle_indices = whittle.binary_search1(self.num_states, self.passive_trans, self.active_trans,
                                                      self.rewards, self.discount)


class Arm_No_Trans:
    # def __init__(self, no_action_trans, action_trans, time_win, init_state=np.array([1, 0]), discount=1):
    def __init__(self, no_action_trans, action_trans, act_win, win_len, period_len, whittle_index, multi=[1],
                 init_state=np.array([1, 0]),
                 discount=1,
                 require_whittle_idx=True):
        self.state = init_state
        self.action_trans = action_trans
        self.no_action_trans = no_action_trans
        self.discount = 0.99
        self.time_period_len = period_len
        # multi ins in one action window
        self.multi = [1]
        # self.multi = [2]
        day1 = act_win
        # day2 = act_win[0][1]
        # day3 = act_win[0][2]
        # self.action_window = [[i for i in range(day1, day1 + 5)]]
        # self.action_window = [act_win]
        self.action_window = [[i for i in range(day1, day1 + win_len)]]
        self.mix_states = [init_state]
        self.current_state = 0
        self.days_in_good_state = 0
        self.inspect_cnt = [0] * len(self.action_window)
        self.not_inspected = True
        self.cnt = 0
        self.ins_record = []
        while (1):
            new_state = np.dot(self.mix_states[-1], self.no_action_trans)
            if (abs(new_state - self.mix_states[-1]) < 1e-2).all():
                break
            self.mix_states.append(new_state)
        # print(self.mix_states)
        # for i in range(10):
        #     new_state = np.dot(self.mix_states[-1], self.no_action_trans)
        #     self.mix_states.append(new_state)
        #     # print(new_state)

        self.num_mix_states = len(self.mix_states)
        self.rewards = []
        self.num_states = self.num_mix_states
        self.mix_traj_active = [0] * self.num_mix_states
        self.mix_traj_passive = [i + 1 for i in range(self.num_mix_states)]
        self.mix_traj_passive[-1] = self.num_mix_states - 1
        self.mix_rewards = []
        for state in self.mix_states:
            self.rewards.append(float(state[0]))
        self.is_pulled = False
        if require_whittle_idx:
            self.whittle_indices = whittle.WhittleIndex_PKG(self.num_states, self.mix_traj_passive, self.mix_traj_active,
                                                        self.rewards, self.discount)
    def reset_all(self):
        self.is_pulled = False
        self.current_state = 0

    def reset(self):
        # self.current_state = 0
        # self.days_in_good_state = 0
        self.inspect_cnt = [0] * len(self.action_window)
        self.is_pulled = False

    def isInWindow(self, t):
        for win in self.action_window:
            if t in win:
                return True
        return False

    def searchMulIdx(self, t):
        idx = -1
        for i in range(len(self.action_window)):
            if t in self.action_window[i]:
                idx = i
        if idx >= 0:
            return idx
        else:
            return -1

    def isLastEntry(self, t):
        for win in self.action_window:
            if t == win[-1]:
                return True
        return False

    def need_action(self, t):
        return not self.is_pulled

    def get_risk(self):

        return self.mix_states[self.current_state][1]

    def state_step(self, action, t):
        if not action:
            self.ins_record.append(0)
            self.current_state = self.mix_traj_passive[self.current_state]
        else:
            self.ins_record.append(1)
            if self.isInWindow(t % self.time_period_len) and not self.is_pulled:
                self.current_state = self.mix_traj_active[self.current_state]
                self.is_pulled = True
            else:
                self.current_state = self.mix_traj_passive[self.current_state]

        if self.sampling() == 0:
            self.days_in_good_state += 1
        if action:
            idx = self.searchMulIdx(t)
            if idx >= 0:
                # print(self.inspect_cnt)
                self.inspect_cnt[idx] += 1

    def reset_ins_record(self):
        self.ins_record = []
        self.days_in_good_state = 0

    def isCovered(self):
        total_ins = 0
        num_ins_inWin = 0
        num_ins_outWin = 0
        repeat = int(len(self.ins_record) / self.time_period_len)
        for i in range(repeat):
            num_ins_inWin = 0
            num_ins_outWin = 0
            for j in range(self.time_period_len):
                if self.ins_record[i * self.time_period_len + j]:
                    total_ins += 1
                    if self.isInWindow(j % self.time_period_len):
                        num_ins_inWin += 1
                    else:
                        num_ins_outWin += 1
            if sum(self.multi) > num_ins_inWin:
                return -1
        return 0
        # repeat = int(len(self.ins_record) / self.time_period_len)
        # total_ins_req = repeat * sum(self.multi)

        # if total_ins_req == total_ins:
        #     if num_ins_outWin == 0:
        #         return 0
        #     else:
        #         return 1
        # elif total_ins_req > total_ins:
        #     return 2
        # else:
        #     return 3
        # print(total_ins, num_ins_inWin, num_ins_outWin)
        #
        # if total_ins_req > num_ins_inWin:
        #     return -1
        # elif total_ins_req == num_ins_inWin:
        #     return 0
        # else:
        #     return 1

    def mix_state_step(self, action):
        if not action:
            self.current_state = self.mix_traj_passive[self.current_state]
        else:
            self.current_state = self.mix_traj_active[self.current_state]

    def get_state(self):
        # mix_idx = self.time_ext_states[self.current_state][0]
        # # return self.current_state, self.time_ext_states[self.current_state], self.mix_states[mix_idx]
        return self.current_state

    def is_in_action_win(self, t):
        return self.isInWindow(t)

    def sampling(self):
        mix_idx = self.current_state

        # print(x)
        rel = 0
        for i in range(20):
            x = np.random.rand()
            if x < self.mix_states[mix_idx][0]:
                rel += 1
            else:
                rel -= 1
        if rel >= 0:
            return 0
        else:
            return 1

    def get_index(self, t):
        assert self.current_state < len(self.whittle_indices)
        return self.whittle_indices[self.current_state]

    def get_mix_index(self, t):
        return self.whittle_indices[self.current_state]

    def reward_whittle(self):
        return self.rewards[self.current_state]

    def reward_mix_whittle(self):
        return self.mix_rewards[self.current_state]

    # def Whittle_for_LP(self):
    #     # plan for 12mon for now
    #     self.whittle_indices_lp = np.zeros((self.time_period_len, self.time_period_len, 2))
    #     for t in range(self.time_period_len):
    #         for last_t in range(self.time_period_len):
    #             s = last_t % self.num_mix_states
    #             idx = self.time_ext_states.index([s,t,0])
    #             self.whittle_indices_lp[t,last_t,0] = self.whittle_indices[idx]
    #             if t in self.action_window:
    #                 idx1 = self.time_ext_states.index([s,t,1])
    #                 self.whittle_indices_lp[t,last_t,1] = self.whittle_indices[idx1]
    #             else:
    #                 self.whittle_indices_lp[t,last_t,1] = self.whittle_indices[idx]

    def Whittle_for_LP(self):
        # plan for 12mon for now and only
        whittle_indices_lp = [0] * self.time_period_len * 5

        last_t = 0
        for t in range(self.time_period_len * 5):
            # for t in range(self.time_period_len):
            time_step = t % self.time_period_len
            if not self.is_in_action_win(time_step):
                if (t - last_t) >= self.num_mix_states - 1:
                    s = self.num_mix_states - 1
                else:

                    s = t - last_t
                idx = self.time_ext_states.index([s, time_step, 0])
            elif self.isInWindow(time_step):
                if time_step == self.action_window[0][-1]:
                    if (t - last_t) >= self.num_mix_states - 1:
                        s = self.num_mix_states - 1
                    else:

                        s = t - last_t
                    idx = self.time_ext_states.index([s, time_step, 0])
                    last_t = t
                else:
                    if (t - last_t) >= self.num_mix_states - 1:
                        s = self.num_mix_states - 1
                    else:

                        s = t - last_t
                    idx = self.time_ext_states.index([s, time_step, 0])

            whittle_indices_lp[t] = self.whittle_indices[idx]

            # print(self.time_ext_states[idx])
        return whittle_indices_lp

    def Whittle_for_LP_multi(self):
        # plan for 12mon for now and only
        whittle_indices_lp = [0] * self.time_period_len * 5

        last_t = 0
        for t in range(self.time_period_len * 5):
            # for t in range(self.time_period_len):
            time_step = t % self.time_period_len
            if not self.is_in_action_win(time_step):
                if (t - last_t) >= self.num_mix_states - 1:
                    s = self.num_mix_states - 1
                else:

                    s = t - last_t
                idx = self.time_ext_states.index([s, time_step, 0])
            elif time_step in self.action_window[0] or time_step in self.action_window[1]:
                if time_step == self.action_window[0][1] or time_step == self.action_window[1][1]:
                    if (t - last_t) >= self.num_mix_states - 1:
                        s = self.num_mix_states - 1
                    else:

                        s = t - last_t
                    idx = self.time_ext_states.index([s, time_step, 0])
                    last_t = t
                else:
                    if (t - last_t) >= self.num_mix_states - 1:
                        s = self.num_mix_states - 1
                    else:

                        s = t - last_t
                    idx = self.time_ext_states.index([s, time_step, 0])

            whittle_indices_lp[t] = self.whittle_indices[idx]

            print(self.time_ext_states[idx])
        return whittle_indices_lp

    def debug(self):
        print("Arm mix states:", end="")
        print(self.mix_states)
        print("Rewards: ", end=" ")
        print(self.mix_rewards)
        print("Action Window", end=" ")
        print(self.action_window)
        print()
        print()

    def nan_count(self):
        tmp = np.array(self.whittle_indices)
        return np.sum(np.isnan(tmp))

    # def value_iter(self):
    #     vi = mdptoolbox.mdp.ValueIteration(np.array([self.P0, self.P1]),
    #                                        np.transpose(np.array([self.rewards, self.rewards])), self.discount)
    #     vi.run()
    #     self.V = vi.V
    #     q = mdptoolbox.mdp.QLearning(np.array([self.P0, self.P1]), np.transpose(np.array([self.rewards, self.rewards])),
    #                                  self.discount)
    #     q.run()
    #     self.Q = q.Q

    # def whittle_index(self, range, start_state_idx):
    #     left = range[0]
    #     right = range[1]
    #     last_m = 9999999
    #     m = (left + right) / 2
    #     while (1):
    #         if abs(last_m - m) < 1e-3:
    #             break
    #         Val_passive = m + self.rewards[start_state_idx] + self.Q[start_state_idx][0]
    #         Val_active = self.rewards[start_state_idx] + self.discount * self.Q[start_state_idx][1]
    #         if Val_passive >= Val_active:
    #             right = m
    #         else:
    #             left = m
    #         m = (left + right) / 2
    #         print("m is " + str(m))
    #         print("passive is " + str(Val_passive))
    #         print("active is " + str(Val_active))

    # print("The whittle index is :" + str(m))


class Arm_NO_Win:
    # def __init__(self, no_action_trans, action_trans, time_win, init_state=np.array([1, 0]), discount=1):
    def __init__(self, no_action_trans, action_trans, period_len, whittle_index=None, init_state=np.array([1, 0])):
        self.state = init_state
        self.action_trans = action_trans
        self.no_action_trans = no_action_trans
        self.discount = 0.99
        self.time_period_len = period_len
        self.inspect_step = -1
        # multi ins in one action window
        # self.multi = act_win[1]

        # day2 = act_win[0][1]
        # day3 = act_win[0][2]
        # self.action_window = [[i for i in range(day1, day1 + 5)]]
        # self.action_window = [act_win]
        self.action_window = [i for i in range(self.time_period_len)]
        self.mix_states = [init_state]
        self.current_state = 0
        self.days_in_good_state = 0
        # self.inspect_cnt = [0] * len(self.action_window)
        self.not_inspected = True
        self.cnt = 0
        self.ins_cnt = 0
        self.ins_record = []
        while (1):
            new_state = np.dot(self.mix_states[-1], self.no_action_trans)
            if (abs(new_state - self.mix_states[-1]) < 1e-2).all():
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
                if t != 0:

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
            mul = 1
            if (t % self.time_period_len != self.time_period_len - 1):
                next_state = [self.mix_traj_passive[s_prob], (t + 1) % self.time_period_len, is_inspected]
            else:
                next_state = [self.mix_traj_passive[s_prob], (t + 1) % self.time_period_len, 0]
            # self.passive_traj.append(self.time_ext_states.index(next_state))
            self.passive_traj.append(self.time_ext_states.index(next_state))
            # active transition
            if (t % self.time_period_len == self.time_period_len - 1):

                if is_inspected < mul:
                    next_state = [self.mix_traj_active[s_prob], (t + 1) % self.time_period_len, 0]
                else:
                    next_state = [self.mix_traj_passive[s_prob], (t + 1) % self.time_period_len, 0]
            else:
                if is_inspected < mul:
                    next_state = [self.mix_traj_active[s_prob], (t + 1) % self.time_period_len, is_inspected + 1]
                else:
                    next_state = [self.mix_traj_passive[s_prob], (t + 1) % self.time_period_len, is_inspected]
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
        # self.P1 = [[0.0 for i in range(self.num_mix_states)] for j in range(self.num_mix_states)]
        # self.P0 = [[0.0 for i in range(self.num_mix_states)] for j in range(self.num_mix_states)]
        # for i in range(self.num_mix_states):
        #     for j in range(self.num_mix_states):
        #         if j == 0:
        #             self.P1[i][j] = 1.0
        #         if j - i == 1:
        #             self.P0[i][j] = 1.0
        # self.P0[-1][-1] = 1.0
        # self.rewards_P0P1 = []
        # self.P0 = np.array(self.P0)
        # self.P1 = np.array(self.P1)
        #
        # for state in self.mix_states:
        #     self.rewards_P0P1.append(state[0])
        # self.rewards_P0P1 = np.array(self.rewards_P0P1)
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
        #
        # self.passive_trans = np.zeros([self.num_states, self.num_states])
        # self.active_trans = np.zeros([self.num_states, self.num_states])
        # for i in range(self.num_states):
        #     self.passive_trans[i][self.passive_traj[i]] = 1.0
        #     self.active_trans[i][self.active_traj[i]] = 1.0
        # mb = bandit.restless_bandit_from_P0P1_R0R1(self.passive_trans, self.active_trans, self.rewards, self.rewards)
        # self.mbi = mb.whittle_indices(discount=0.99)
        #
        # if np.isnan(np.sum(self.mbi)):
        #     print(self.no_action_trans)
        #     print(self.mix_states)
        # print(mbi)

        if whittle_index is not None:
            self.whittle_indices = whittle_index
        else:
            start = time.time()
            self.whittle_indices = whittle.binary_search(self.num_states, self.passive_traj, self.active_traj,
                                                         self.rewards, self.discount)
            # print("Compute Whittle Index for " + str(self.num_states) + " states")
            # self.whittle_indices = whittle.WhittleIndex_PKG(self.num_states, self.passive_traj, self.active_traj,
            #                                active_traj                 self.rewards, self.discount)
            total_time = time.time() - start
            assert len(self.whittle_indices) == self.num_states
            self.avg_compute = total_time / self.num_states
        # if require_whittle_idx:
        #     start = time.time()
        #     self.whittle_indices = whittle.binary_search(self.num_states, self.passive_traj, self.active_traj, self.rewards, self.discount)
        #     print(self.num_states)
        #     print(len(self.whittle_indices))
        #     self.computation_time = time.time() - start
        #     self.Whittle_for_LP()
        # print(self.whittle_indices)
        # self.act_window_len = 3
        # self.noact_len = 9

    def reset_all(self):
        self.not_inspected = True
        self.ins_cnt = 0
        self.current_state = 0

    def reset(self):
        # self.current_state = 0
        # self.days_in_good_state = 0
        # self.inspect_cnt = [0] * len(self.action_window)
        self.not_inspected = True
        self.ins_cnt = 0
        self.inspect_step = -1

    def isInWindow(self, t):
        return True

    def searchMulIdx(self, t):
        return 0

    def isLastEntry(self, t):
        return t % self.time_period_len == self.time_period_len - 1

    def need_action(self, t):
        # mulidx = self.searchMulIdx(t)
        #
        # return self.inspect_cnt[mulidx] < self.multi[mulidx]
        return self.not_inspected 
    def need_action_sec(self, t):
        # mulidx = self.searchMulIdx(t)
        #
        # return self.inspect_cnt[mulidx] < self.multi[mulidx]
        return self.ins_cnt < 2


    def get_risk(self):
        mix_idx = self.time_ext_states[self.current_state][0]
        return self.mix_states[mix_idx][1]

    def state_step(self, action, t):
        if not action:
            self.ins_record.append(0)
            self.current_state = self.passive_traj[self.current_state]
        else:

            self.current_state = self.active_traj[self.current_state]
            if self.inspect_step < 0:
                self.inspect_step = t % self.time_period_len

        if self.sampling() == 0:
            self.days_in_good_state += 1

    def reset_ins_record(self):
        self.ins_record = []
        self.days_in_good_state = 0

    def isCovered(self):
        total_ins = 0
        num_ins_inWin = 0
        num_ins_outWin = 0
        repeat = int(len(self.ins_record) / self.time_period_len)
        for i in range(repeat):
            num_ins_inWin = 0
            num_ins_outWin = 0
            for j in range(self.time_period_len):
                if self.ins_record[i * self.time_period_len + j]:
                    total_ins += 1
                    if self.isInWindow(j % self.time_period_len):
                        num_ins_inWin += 1
                    else:
                        num_ins_outWin += 1
            if sum(self.multi) > num_ins_inWin:
                return -1
        return 0
        # repeat = int(len(self.ins_record) / self.time_period_len)
        # total_ins_req = repeat * sum(self.multi)

        # if total_ins_req == total_ins:
        #     if num_ins_outWin == 0:
        #         return 0
        #     else:
        #         return 1
        # elif total_ins_req > total_ins:
        #     return 2
        # else:
        #     return 3
        # print(total_ins, num_ins_inWin, num_ins_outWin)
        #
        # if total_ins_req > num_ins_inWin:
        #     return -1
        # elif total_ins_req == num_ins_inWin:
        #     return 0
        # else:
        #     return 1

    def mix_state_step(self, action):
        if not action:
            self.current_state = self.mix_traj_passive[self.current_state]
        else:
            self.current_state = self.mix_traj_active[self.current_state]

    def get_state(self):
        mix_idx = self.time_ext_states[self.current_state][0]
        # return self.current_state, self.time_ext_states[self.current_state], self.mix_states[mix_idx]
        return mix_idx

    def is_in_action_win(self, t):
        return self.isInWindow(t)

    def sampling(self):
        mix_idx = self.time_ext_states[self.current_state][0]

        # print(x)
        rel = 0
        for i in range(20):
            x = np.random.rand()
            if x < self.mix_states[mix_idx][0]:
                rel += 1
            else:
                rel -= 1
        if rel >= 0:
            return 0
        else:
            return 1

    def get_index(self, t):
        assert self.current_state < len(self.whittle_indices)
        return self.whittle_indices[self.current_state]

    def get_mix_index(self, t):
        mix_idx = self.time_ext_states[self.current_state][0]
        return self.mix_whittle[mix_idx]

    def reward_whittle(self):
        return self.rewards[self.current_state]

    def reward_mix_whittle(self):
        return self.mix_rewards[self.current_state]

    # def Whittle_for_LP(self):
    #     # plan for 12mon for now
    #     self.whittle_indices_lp = np.zeros((self.time_period_len, self.time_period_len, 2))
    #     for t in range(self.time_period_len):
    #         for last_t in range(self.time_period_len):
    #             s = last_t % self.num_mix_states
    #             idx = self.time_ext_states.index([s,t,0])
    #             self.whittle_indices_lp[t,last_t,0] = self.whittle_indices[idx]
    #             if t in self.action_window:
    #                 idx1 = self.time_ext_states.index([s,t,1])
    #                 self.whittle_indices_lp[t,last_t,1] = self.whittle_indices[idx1]
    #             else:
    #                 self.whittle_indices_lp[t,last_t,1] = self.whittle_indices[idx]

    def Whittle_for_LP(self):
        # plan for 12mon for now and only
        whittle_indices_lp = [0] * self.time_period_len * 5
        last_t = 0
        for t in range(self.time_period_len * 5):
            # for t in range(self.time_period_len):
            time_step = t % self.time_period_len
            if not self.is_in_action_win(time_step):
                if (t - last_t) >= self.num_mix_states - 1:
                    s = self.num_mix_states - 1
                else:

                    s = t - last_t
                idx = self.time_ext_states.index([s, time_step, 0])
            elif self.isInWindow(time_step):
                if time_step == self.action_window[-1]:
                    if (t - last_t) >= self.num_mix_states - 1:
                        s = self.num_mix_states - 1
                    else:

                        s = t - last_t
                    idx = self.time_ext_states.index([s, time_step, 0])
                    last_t = t
                else:
                    if (t - last_t) >= self.num_mix_states - 1:
                        s = self.num_mix_states - 1
                    else:

                        s = t - last_t
                    idx = self.time_ext_states.index([s, time_step, 0])

            whittle_indices_lp[t] = self.whittle_indices[idx]

            # print(self.time_ext_states[idx])
        return whittle_indices_lp

        # return whittle_indices_lp

   

    def debug(self):
        print("Arm mix states:", end="")
        print(self.mix_states)
        print("Rewards: ", end=" ")
        print(self.mix_rewards)
        print("Action Window", end=" ")
        print(self.action_window)
        print()
        print()

    def nan_count(self):
        tmp = np.array(self.whittle_indices)
        return np.sum(np.isnan(tmp))

    # def value_iter(self):
    #     vi = mdptoolbox.mdp.ValueIteration(np.array([self.P0, self.P1]),
    #                                        np.transpose(np.array([self.rewards, self.rewards])), self.discount)
    #     vi.run()
    #     self.V = vi.V
    #     q = mdptoolbox.mdp.QLearning(np.array([self.P0, self.P1]), np.transpose(np.array([self.rewards, self.rewards])),
    #                                  self.discount)
    #     q.run()
    #     self.Q = q.Q

    # def whittle_index(self, range, start_state_idx):
    #     left = range[0]
    #     right = range[1]
    #     last_m = 9999999
    #     m = (left + right) / 2
    #     while (1):
    #         if abs(last_m - m) < 1e-3:
    #             break
    #         Val_passive = m + self.rewards[start_state_idx] + self.Q[start_state_idx][0]
    #         Val_active = self.rewards[start_state_idx] + self.discount * self.Q[start_state_idx][1]
    #         if Val_passive >= Val_active:
    #             right = m
    #         else:
    #             left = m
    #         m = (left + right) / 2
    #         print("m is " + str(m))
    #         print("passive is " + str(Val_passive))
    #         print("active is " + str(Val_active))

    # print("The whittle index is :" + str(m))

class Arm_NO_Win_Multi:
    # def __init__(self, no_action_trans, action_trans, time_win, init_state=np.array([1, 0]), discount=1):
    def __init__(self, no_action_trans, action_trans, period_len, whittle_index=None, init_state=np.array([1, 0])):
        self.state = init_state
        self.action_trans = action_trans
        self.no_action_trans = no_action_trans
        self.discount = 0.99
        self.time_period_len = period_len
        # multi ins in one action window
        # self.multi = act_win[1]

        # day2 = act_win[0][1]
        # day3 = act_win[0][2]
        # self.action_window = [[i for i in range(day1, day1 + 5)]]
        # self.action_window = [act_win]
        self.action_window = [i for i in range(self.time_period_len)]
        self.mix_states = [init_state]
        self.current_state = 0
        self.days_in_good_state = 0
        # self.inspect_cnt = [0] * len(self.action_window)
        self.not_inspected = True
        self.cnt = 0
        self.ins_record = []
        while (1):
            new_state = np.dot(self.mix_states[-1], self.no_action_trans)
            if (abs(new_state - self.mix_states[-1]) < 1e-2).all():
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
                self.time_ext_states.append([i, t])

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
            mul = 1
            if (t % self.time_period_len != self.time_period_len - 1):
                next_state = [self.mix_traj_passive[s_prob], (t + 1) % self.time_period_len, is_inspected]
            else:
                next_state = [self.mix_traj_passive[s_prob], (t + 1) % self.time_period_len, 0]
            # self.passive_traj.append(self.time_ext_states.index(next_state))
            self.passive_traj.append(self.time_ext_states.index(next_state))
            # active transition
            if (t % self.time_period_len == self.time_period_len - 1):

                if is_inspected < mul:
                    next_state = [self.mix_traj_active[s_prob], (t + 1) % self.time_period_len, 0]
                else:
                    next_state = [self.mix_traj_passive[s_prob], (t + 1) % self.time_period_len, 0]
            else:
                if is_inspected < mul:
                    next_state = [self.mix_traj_active[s_prob], (t + 1) % self.time_period_len, is_inspected + 1]
                else:
                    next_state = [self.mix_traj_passive[s_prob], (t + 1) % self.time_period_len, is_inspected]
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
        # self.P1 = [[0.0 for i in range(self.num_mix_states)] for j in range(self.num_mix_states)]
        # self.P0 = [[0.0 for i in range(self.num_mix_states)] for j in range(self.num_mix_states)]
        # for i in range(self.num_mix_states):
        #     for j in range(self.num_mix_states):
        #         if j == 0:
        #             self.P1[i][j] = 1.0
        #         if j - i == 1:
        #             self.P0[i][j] = 1.0
        # self.P0[-1][-1] = 1.0
        # self.rewards_P0P1 = []
        # self.P0 = np.array(self.P0)
        # self.P1 = np.array(self.P1)
        #
        # for state in self.mix_states:
        #     self.rewards_P0P1.append(state[0])
        # self.rewards_P0P1 = np.array(self.rewards_P0P1)
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
        #
        # self.passive_trans = np.zeros([self.num_states, self.num_states])
        # self.active_trans = np.zeros([self.num_states, self.num_states])
        # for i in range(self.num_states):
        #     self.passive_trans[i][self.passive_traj[i]] = 1.0
        #     self.active_trans[i][self.active_traj[i]] = 1.0
        # mb = bandit.restless_bandit_from_P0P1_R0R1(self.passive_trans, self.active_trans, self.rewards, self.rewards)
        # self.mbi = mb.whittle_indices(discount=0.99)
        #
        # if np.isnan(np.sum(self.mbi)):
        #     print(self.no_action_trans)
        #     print(self.mix_states)
        # print(mbi)

        if whittle_index is not None:
            self.whittle_indices = whittle_index
        else:
            start = time.time()
            self.whittle_indices = whittle.binary_search(self.num_states, self.passive_traj, self.active_traj,
                                                         self.rewards, self.discount)
            # print("Compute Whittle Index for " + str(self.num_states) + " states")
            # self.whittle_indices = whittle.WhittleIndex_PKG(self.num_states, self.passive_traj, self.active_traj,
            #                                active_traj                 self.rewards, self.discount)
            total_time = time.time() - start
            assert len(self.whittle_indices) == self.num_states
            self.avg_compute = total_time / self.num_states
        # if require_whittle_idx:
        #     start = time.time()
        #     self.whittle_indices = whittle.binary_search(self.num_states, self.passive_traj, self.active_traj, self.rewards, self.discount)
        #     print(self.num_states)
        #     print(len(self.whittle_indices))
        #     self.computation_time = time.time() - start
        #     self.Whittle_for_LP()
        # print(self.whittle_indices)
        # self.act_window_len = 3
        # self.noact_len = 9

    def reset_all(self):
        self.not_inspected = True
        self.current_state = 0

    def reset(self):
        # self.current_state = 0
        # self.days_in_good_state = 0
        # self.inspect_cnt = [0] * len(self.action_window)
        self.not_inspected = True

    def isInWindow(self, t):
        return True

    def searchMulIdx(self, t):
        return 0

    def isLastEntry(self, t):
        return t % self.time_period_len == self.time_period_len - 1

    def need_action(self, t):
        # mulidx = self.searchMulIdx(t)
        #
        # return self.inspect_cnt[mulidx] < self.multi[mulidx]
        return self.not_inspected

    def get_risk(self):
        mix_idx = self.time_ext_states[self.current_state][0]
        return self.mix_states[mix_idx][1]

    def state_step(self, action, t):
        if not action:
            self.ins_record.append(0)
            self.current_state = self.passive_traj[self.current_state]
        else:
            self.current_state = self.active_traj[self.current_state]

        if self.sampling() == 0:
            self.days_in_good_state += 1

    def reset_ins_record(self):
        self.ins_record = []
        self.days_in_good_state = 0

    def isCovered(self):
        total_ins = 0
        num_ins_inWin = 0
        num_ins_outWin = 0
        repeat = int(len(self.ins_record) / self.time_period_len)
        for i in range(repeat):
            num_ins_inWin = 0
            num_ins_outWin = 0
            for j in range(self.time_period_len):
                if self.ins_record[i * self.time_period_len + j]:
                    total_ins += 1
                    if self.isInWindow(j % self.time_period_len):
                        num_ins_inWin += 1
                    else:
                        num_ins_outWin += 1
            if sum(self.multi) > num_ins_inWin:
                return -1
        return 0
        # repeat = int(len(self.ins_record) / self.time_period_len)
        # total_ins_req = repeat * sum(self.multi)

        # if total_ins_req == total_ins:
        #     if num_ins_outWin == 0:
        #         return 0
        #     else:
        #         return 1
        # elif total_ins_req > total_ins:
        #     return 2
        # else:
        #     return 3
        # print(total_ins, num_ins_inWin, num_ins_outWin)
        #
        # if total_ins_req > num_ins_inWin:
        #     return -1
        # elif total_ins_req == num_ins_inWin:
        #     return 0
        # else:
        #     return 1

    def mix_state_step(self, action):
        if not action:
            self.current_state = self.mix_traj_passive[self.current_state]
        else:
            self.current_state = self.mix_traj_active[self.current_state]

    def get_state(self):
        mix_idx = self.time_ext_states[self.current_state][0]
        # return self.current_state, self.time_ext_states[self.current_state], self.mix_states[mix_idx]
        return mix_idx

    def is_in_action_win(self, t):
        return self.isInWindow(t)

    def sampling(self):
        mix_idx = self.time_ext_states[self.current_state][0]

        # print(x)
        rel = 0
        for i in range(20):
            x = np.random.rand()
            if x < self.mix_states[mix_idx][0]:
                rel += 1
            else:
                rel -= 1
        if rel >= 0:
            return 0
        else:
            return 1

    def get_index(self, t):
        assert self.current_state < len(self.whittle_indices)
        return self.whittle_indices[self.current_state]

    def get_mix_index(self, t):
        mix_idx = self.time_ext_states[self.current_state][0]
        return self.mix_whittle[mix_idx]

    def reward_whittle(self):
        return self.rewards[self.current_state]

    def reward_mix_whittle(self):
        return self.mix_rewards[self.current_state]

    # def Whittle_for_LP(self):
    #     # plan for 12mon for now
    #     self.whittle_indices_lp = np.zeros((self.time_period_len, self.time_period_len, 2))
    #     for t in range(self.time_period_len):
    #         for last_t in range(self.time_period_len):
    #             s = last_t % self.num_mix_states
    #             idx = self.time_ext_states.index([s,t,0])
    #             self.whittle_indices_lp[t,last_t,0] = self.whittle_indices[idx]
    #             if t in self.action_window:
    #                 idx1 = self.time_ext_states.index([s,t,1])
    #                 self.whittle_indices_lp[t,last_t,1] = self.whittle_indices[idx1]
    #             else:
    #                 self.whittle_indices_lp[t,last_t,1] = self.whittle_indices[idx]

    def Whittle_for_LP(self):
        # plan for 12mon for now and only
        whittle_indices_lp = [0] * self.time_period_len * 5
        last_t = 0
        for t in range(self.time_period_len * 5):
            # for t in range(self.time_period_len):
            time_step = t % self.time_period_len
            if not self.is_in_action_win(time_step):
                if (t - last_t) >= self.num_mix_states - 1:
                    s = self.num_mix_states - 1
                else:

                    s = t - last_t
                idx = self.time_ext_states.index([s, time_step, 0])
            elif self.isInWindow(time_step):
                if time_step == self.action_window[-1]:
                    if (t - last_t) >= self.num_mix_states - 1:
                        s = self.num_mix_states - 1
                    else:

                        s = t - last_t
                    idx = self.time_ext_states.index([s, time_step, 0])
                    last_t = t
                else:
                    if (t - last_t) >= self.num_mix_states - 1:
                        s = self.num_mix_states - 1
                    else:

                        s = t - last_t
                    idx = self.time_ext_states.index([s, time_step, 0])

            whittle_indices_lp[t] = self.whittle_indices[idx]

            # print(self.time_ext_states[idx])
        return whittle_indices_lp

        return whittle_indices_lp

    def Whittle_for_LP_multi(self):
        # plan for 12mon for now and only
        whittle_indices_lp = [0] * self.time_period_len * 5
        last_t = 0
        for t in range(self.time_period_len * 5):
            # for t in range(self.time_period_len):
            time_step = t % self.time_period_len
            if not self.is_in_action_win(time_step):
                if (t - last_t) >= self.num_mix_states - 1:
                    s = self.num_mix_states - 1
                else:

                    s = t - last_t
                idx = self.time_ext_states.index([s, time_step, 0])
            elif time_step in self.action_window[0] or time_step in self.action_window[1]:
                if time_step == self.action_window[0][1] or time_step == self.action_window[1][1]:
                    if (t - last_t) >= self.num_mix_states - 1:
                        s = self.num_mix_states - 1
                    else:

                        s = t - last_t
                    idx = self.time_ext_states.index([s, time_step, 0])
                    last_t = t
                else:
                    if (t - last_t) >= self.num_mix_states - 1:
                        s = self.num_mix_states - 1
                    else:

                        s = t - last_t
                    idx = self.time_ext_states.index([s, time_step, 0])

            whittle_indices_lp[t] = self.whittle_indices[idx]

            print(self.time_ext_states[idx])
        return whittle_indices_lp

    def debug(self):
        print("Arm mix states:", end="")
        print(self.mix_states)
        print("Rewards: ", end=" ")
        print(self.mix_rewards)
        print("Action Window", end=" ")
        print(self.action_window)
        print()
        print()

    def nan_count(self):
        tmp = np.array(self.whittle_indices)
        return np.sum(np.isnan(tmp))

    # def value_iter(self):
    #     vi = mdptoolbox.mdp.ValueIteration(np.array([self.P0, self.P1]),
    #                                        np.transpose(np.array([self.rewards, self.rewards])), self.discount)
    #     vi.run()
    #     self.V = vi.V
    #     q = mdptoolbox.mdp.QLearning(np.array([self.P0, self.P1]), np.transpose(np.array([self.rewards, self.rewards])),
    #                                  self.discount)
    #     q.run()
    #     self.Q = q.Q

    # def whittle_index(self, range, start_state_idx):
    #     left = range[0]
    #     right = range[1]
    #     last_m = 9999999
    #     m = (left + right) / 2
    #     while (1):
    #         if abs(last_m - m) < 1e-3:
    #             break
    #         Val_passive = m + self.rewards[start_state_idx] + self.Q[start_state_idx][0]
    #         Val_active = self.rewards[start_state_idx] + self.discount * self.Q[start_state_idx][1]
    #         if Val_passive >= Val_active:
    #             right = m
    #         else:
    #             left = m
    #         m = (left + right) / 2
    #         print("m is " + str(m))
    #         print("passive is " + str(Val_passive))
    #         print("active is " + str(Val_active))

    # print("The whittle index is :" + str(m))

class Arm_2OR1:
    # def __init__(self, no_action_trans, action_trans, time_win, init_state=np.array([1, 0]), discount=1):
    def __init__(self, no_action_trans, action_trans, period_len, whittle_index=None, init_state=np.array([1, 0])):
        self.state = init_state
        self.action_trans = action_trans
        self.no_action_trans = no_action_trans
        self.discount = 0.99
        self.time_period_len = period_len
        self.inspect_step = -1
        # multi ins in one action window
        # self.multi = act_win[1]

        # day2 = act_win[0][1]
        # day3 = act_win[0][2]
        # self.action_window = [[i for i in range(day1, day1 + 5)]]
        # self.action_window = [act_win]
        self.action_window = [i for i in range(self.time_period_len)]
        self.mix_states = [init_state]
        self.current_state = 0
        self.days_in_good_state = 0
        # self.inspect_cnt = [0] * len(self.action_window)
        self.not_inspected = True
        self.cnt = 0
        self.ins_cnt = 0
        self.ins_record = []
        while (1):
            new_state = np.dot(self.mix_states[-1], self.no_action_trans)
            if (abs(new_state - self.mix_states[-1]) < 1e-2).all():
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
                if t != 0:

                    self.time_ext_states.append([i, t, 1])
                    self.time_ext_states.append([i, t, 2])

        self.num_states = len(self.time_ext_states)
        self.mix_traj_active = [0] * self.num_mix_states
        self.mix_traj_passive = [i + 1 for i in range(self.num_mix_states)]
        self.mix_traj_passive[-1] = self.num_mix_states - 1
        self.mix_rewards = []
        self.reward_records = []
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
            mul = 2
            if (t % self.time_period_len != self.time_period_len - 1):
                next_state = [self.mix_traj_passive[s_prob], (t + 1) % self.time_period_len, is_inspected]
            else:
                next_state = [self.mix_traj_passive[s_prob], (t + 1) % self.time_period_len, 0]
            # self.passive_traj.append(self.time_ext_states.index(next_state))
            self.passive_traj.append(self.time_ext_states.index(next_state))
            # active transition
            if (t % self.time_period_len == self.time_period_len - 1):

                if is_inspected < mul:
                    next_state = [self.mix_traj_active[s_prob], (t + 1) % self.time_period_len, 0]
                else:
                    next_state = [self.mix_traj_passive[s_prob], (t + 1) % self.time_period_len, 0]
            else:
                if is_inspected < mul:
                    next_state = [self.mix_traj_active[s_prob], (t + 1) % self.time_period_len, is_inspected + 1]
                else:
                    next_state = [self.mix_traj_passive[s_prob], (t + 1) % self.time_period_len, is_inspected]
            self.active_traj.append(self.time_ext_states.index(next_state))
            # rewards for each states
            self.rewards.append(float(self.mix_states[s_prob][0]))
       

        if whittle_index is not None:
            self.whittle_indices = whittle_index
        else:
            start = time.time()
            self.whittle_indices = whittle.binary_search(self.num_states, self.passive_traj, self.active_traj,
                                                         self.rewards, self.discount)
            # print("Compute Whittle Index for " + str(self.num_states) + " states")
            # self.whittle_indices = whittle.WhittleIndex_PKG(self.num_states, self.passive_traj, self.active_traj,
            #                                active_traj                 self.rewards, self.discount)
            total_time = time.time() - start
            assert len(self.whittle_indices) == self.num_states
            self.avg_compute = total_time / self.num_states
       
    def reset_all(self):
        self.not_inspected = True
        self.ins_cnt = 0
        self.current_state = 0

    def reset(self):
        # self.current_state = 0
        # self.days_in_good_state = 0
        # self.inspect_cnt = [0] * len(self.action_window)
        self.not_inspected = True
        self.ins_cnt = 0
        self.inspect_step = -1

    def isInWindow(self, t):
        return True

    def searchMulIdx(self, t):
        return 0

    def isLastEntry(self, t):
        return t % self.time_period_len == self.time_period_len - 1

    def need_action_first(self, t):
        # mulidx = self.searchMulIdx(t)
        #
        # return self.inspect_cnt[mulidx] < self.multi[mulidx]
        return self.not_inspected 
    def need_action_sec(self, t):
        # mulidx = self.searchMulIdx(t)
        #
        # return self.inspect_cnt[mulidx] < self.multi[mulidx]
        return self.ins_cnt < 2


    def get_risk(self):
        mix_idx = self.time_ext_states[self.current_state][0]
        return self.mix_states[mix_idx][1]

    def state_step(self, action, t):
        if not action:
            self.ins_record.append(0)
            self.current_state = self.passive_traj[self.current_state]
        else:

            self.current_state = self.active_traj[self.current_state]
            if self.inspect_step < 0:
                self.inspect_step = t % self.time_period_len

        if self.sampling() == 0:
            self.days_in_good_state += 1

    def reset_ins_record(self):
        self.ins_record = []
        self.days_in_good_state = 0

    def isCovered(self):
        total_ins = 0
        num_ins_inWin = 0
        num_ins_outWin = 0
        repeat = int(len(self.ins_record) / self.time_period_len)
        for i in range(repeat):
            num_ins_inWin = 0
            num_ins_outWin = 0
            for j in range(self.time_period_len):
                if self.ins_record[i * self.time_period_len + j]:
                    total_ins += 1
                    if self.isInWindow(j % self.time_period_len):
                        num_ins_inWin += 1
                    else:
                        num_ins_outWin += 1
            if sum(self.multi) > num_ins_inWin:
                return -1
        return 0
        # repeat = int(len(self.ins_record) / self.time_period_len)
        # total_ins_req = repeat * sum(self.multi)

        # if total_ins_req == total_ins:
        #     if num_ins_outWin == 0:
        #         return 0
        #     else:
        #         return 1
        # elif total_ins_req > total_ins:
        #     return 2
        # else:
        #     return 3
        # print(total_ins, num_ins_inWin, num_ins_outWin)
        #
        # if total_ins_req > num_ins_inWin:
        #     return -1
        # elif total_ins_req == num_ins_inWin:
        #     return 0
        # else:
        #     return 1

    def mix_state_step(self, action):
        if not action:
            self.current_state = self.mix_traj_passive[self.current_state]
        else:
            self.current_state = self.mix_traj_active[self.current_state]

    def get_state(self):
        mix_idx = self.time_ext_states[self.current_state][0]
        # return self.current_state, self.time_ext_states[self.current_state], self.mix_states[mix_idx]
        return mix_idx

    def is_in_action_win(self, t):
        return self.isInWindow(t)

    def sampling(self):
        mix_idx = self.time_ext_states[self.current_state][0]

        # print(x)
        rel = 0
        for i in range(20):
            x = np.random.rand()
            if x < self.mix_states[mix_idx][0]:
                rel += 1
            else:
                rel -= 1
        if rel >= 0:
            return 0
        else:
            return 1

    def get_index(self, t):
        assert self.current_state < len(self.whittle_indices)
        return self.whittle_indices[self.current_state]

    def get_mix_index(self):
        mix_idx = self.time_ext_states[self.current_state][0]
        return mix_idx

    def reward_whittle(self):
        self.reward_records.append(self.rewards[self.current_state])
        return self.rewards[self.current_state]

    def reward_mix_whittle(self):
        return self.mix_rewards[self.current_state]

    

    def Whittle_for_LP(self):
        # plan for 12mon for now and only
        whittle_indices_lp = [[-10000000 for i in range(self.time_period_len + 1)] for j in range(self.time_period_len + 1)]
        s = self.time_ext_states[self.current_state][0]
        debug = [[-10000000 for i in range(self.time_period_len + 1)] for j in range(self.time_period_len + 1)]
        for i in range(self.time_period_len + 1):
            if i == 0:
                for j in range(i, self.time_period_len):
                  
                    
                    
                    next_s = [min(s + j - i, self.num_mix_states - 1),j, 0]
                    idx = self.time_ext_states.index(next_s)
                    whittle_indices_lp[i][j] = self.whittle_indices[idx]
                    debug[i][j] = next_s
            else:
                for j in range(i, self.time_period_len):
                    if i == 0:
                        next_s = [min(s + j, self.num_mix_states - 1), j, 0]
                    
                    else:
                        next_s = [min(s + j - i, self.num_mix_states - 1),j, 1]
                    idx = self.time_ext_states.index(next_s)
                    whittle_indices_lp[i][j] = self.whittle_indices[idx]
                    debug[i][j] = next_s


        

        return whittle_indices_lp

 
    # def Whittle_for_LP(self):
    # # plan for 12mon for now and only
    #     whittle_indices_lp = [[-10000 for i in range(self.time_period_len + 1)] for j in range(self.time_period_len + 1)]
    #     s = self.time_ext_states[self.current_state][0]
    #     # debug = [[-10000000 for i in range(self.time_period_len + 1)] for j in range(self.time_period_len + 1)]
    #     for i in range(self.time_period_len + 1):
    #         if i == 0:
    #             for j in range(i, self.time_period_len):
                    
                    
                    
    #                 next_s = min(s + j - i, self.num_mix_states - 1)
    #                 whittle_indices_lp[i][j] =  self.mix_states[next_s][0]
                
    #         else:
    #             for j in range(i, self.time_period_len):
    #                 if i == 0:
    #                     next_s = min(s + j, self.num_mix_states - 1)
                    
    #                 else:
    #                     next_s = min(s + j - i, self.num_mix_states - 1)
    #                 # idx = self.time_ext_states.index(next_s)
    #                 whittle_indices_lp[i][j] =  self.mix_states[next_s][0]
    #                 # debug[i][j] = next_s


        

    #     return whittle_indices_lp


    def debug(self):
        print("Arm mix states:", end="")
        print(self.mix_states)
        print("Rewards: ", end=" ")
        print(self.mix_rewards)
        print("Action Window", end=" ")
        print(self.action_window)
        print()
        print()

    def nan_count(self):
        tmp = np.array(self.whittle_indices)
        return np.sum(np.isnan(tmp))

    # def value_iter(self):
    #     vi = mdptoolbox.mdp.ValueIteration(np.array([self.P0, self.P1]),
    #                                        np.transpose(np.array([self.rewards, self.rewards])), self.discount)
    #     vi.run()
    #     self.V = vi.V
    #     q = mdptoolbox.mdp.QLearning(np.array([self.P0, self.P1]), np.transpose(np.array([self.rewards, self.rewards])),
    #                                  self.discount)
    #     q.run()
    #     self.Q = q.Q

    # def whittle_index(self, range, start_state_idx):
    #     left = range[0]
    #     right = range[1]
    #     last_m = 9999999
    #     m = (left + right) / 2
    #     while (1):
    #         if abs(last_m - m) < 1e-3:
    #             break
    #         Val_passive = m + self.rewards[start_state_idx] + self.Q[start_state_idx][0]
    #         Val_active = self.rewards[start_state_idx] + self.discount * self.Q[start_state_idx][1]
    #         if Val_passive >= Val_active:
    #             right = m
    #         else:
    #             left = m
    #         m = (left + right) / 2
    #         print("m is " + str(m))
    #         print("passive is " + str(Val_passive))
    #         print("active is " + str(Val_active))

    # print("The whittle index is :" + str(m))



    # def __init__(self, no_action_trans, action_trans, time_win, init_state=np.array([1, 0]), discount=1):
    def __init__(self, no_action_trans, action_trans, period_len, noise_mean, noise_std, init_state=np.array([1, 0])):
        self.state = init_state
        self.action_trans = action_trans
        self.no_action_trans = no_action_trans
        self.discount = 0.99
        self.time_period_len = period_len
        self.inspect_step = -1
      
        self.action_window = [i for i in range(self.time_period_len)]
        self.mix_states = [init_state]
        self.current_state = 0
        self.days_in_good_state = 0
        # self.inspect_cnt = [0] * len(self.action_window)
        self.not_inspected = True
        self.cnt = 0
        self.ins_cnt = 0
        self.ins_record = []
        while (1):
            new_state = np.dot(self.mix_states[-1], self.no_action_trans)
            if (abs(new_state - self.mix_states[-1]) < 1e-2).all():
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
                if t != 0:

                    self.time_ext_states.append([i, t, 1])

        self.num_states = len(self.time_ext_states)
        self.mix_traj_active = [0] * self.num_mix_states
        self.mix_traj_passive = [i + 1 for i in range(self.num_mix_states)]
        self.mix_traj_passive[-1] = self.num_mix_states - 1
        self.mix_rewards = []
        for state in self.mix_states:
            self.mix_rewards.append(float(state[0]))

       
        self.passive_traj = []
        self.active_traj = []
        self.rewards = []
        for i in range(self.num_states):
            # passive transition
            s = self.time_ext_states[i]
            s_prob = s[0]
            t = s[1]
            is_inspected = s[2]
            mul = 1
            if (t % self.time_period_len != self.time_period_len - 1):
                next_state = [self.mix_traj_passive[s_prob], (t + 1) % self.time_period_len, is_inspected]
            else:
                next_state = [self.mix_traj_passive[s_prob], (t + 1) % self.time_period_len, 0]
            # self.passive_traj.append(self.time_ext_states.index(next_state))
            self.passive_traj.append(self.time_ext_states.index(next_state))
            # active transition
            if (t % self.time_period_len == self.time_period_len - 1):

                if is_inspected < mul:
                    next_state = [self.mix_traj_active[s_prob], (t + 1) % self.time_period_len, 0]
                else:
                    next_state = [self.mix_traj_passive[s_prob], (t + 1) % self.time_period_len, 0]
            else:
                if is_inspected < mul:
                    next_state = [self.mix_traj_active[s_prob], (t + 1) % self.time_period_len, is_inspected + 1]
                else:
                    next_state = [self.mix_traj_passive[s_prob], (t + 1) % self.time_period_len, is_inspected]
            self.active_traj.append(self.time_ext_states.index(next_state))
            # rewards for each states
            self.rewards.append(float(self.mix_states[s_prob][0]))
        
       
        start = time.time()
        self.whittle_indices = whittle.binary_search(self.num_states, self.passive_traj, self.active_traj,
                                                        self.rewards, self.discount)
        # print("Compute Whittle Index for " + str(self.num_states) + " states")
        # self.whittle_indices = whittle.WhittleIndex_PKG(self.num_states, self.passive_traj, self.active_traj,
        #                                active_traj                 self.rewards, self.discount)
        total_time = time.time() - start
        assert len(self.whittle_indices) == self.num_states
        self.avg_compute = total_time / self.num_states
        # if require_whittle_idx:
        #     start = time.time()
        #     self.whittle_indices = whittle.binary_search(self.num_states, self.passive_traj, self.active_traj, self.rewards, self.discount)
        #     print(self.num_states)
        #     print(len(self.whittle_indices))
        #     self.computation_time = time.time() - start
        #     self.Whittle_for_LP()
        # print(self.whittle_indices)
        # self.act_window_len = 3
        # self.noact_len = 9

    def reset_all(self):
        self.not_inspected = True
        self.ins_cnt = 0
        self.current_state = 0

    def reset(self):
        # self.current_state = 0
        # self.days_in_good_state = 0
        # self.inspect_cnt = [0] * len(self.action_window)
        self.not_inspected = True
        self.ins_cnt = 0
        self.inspect_step = -1

    def isInWindow(self, t):
        return True

    def searchMulIdx(self, t):
        return 0

    def isLastEntry(self, t):
        return t % self.time_period_len == self.time_period_len - 1

    def need_action(self, t):
        # mulidx = self.searchMulIdx(t)
        #
        # return self.inspect_cnt[mulidx] < self.multi[mulidx]
        return self.not_inspected 
    def need_action_sec(self, t):
        # mulidx = self.searchMulIdx(t)
        #
        # return self.inspect_cnt[mulidx] < self.multi[mulidx]
        return self.ins_cnt < 2


    def get_risk(self):
        mix_idx = self.time_ext_states[self.current_state][0]
        return self.mix_states[mix_idx][1]

    def state_step(self, action, t):
        if not action:
            self.ins_record.append(0)
            self.current_state = self.passive_traj[self.current_state]
        else:

            self.current_state = self.active_traj[self.current_state]
            if self.inspect_step < 0:
                self.inspect_step = t % self.time_period_len

        if self.sampling() == 0:
            self.days_in_good_state += 1

    def reset_ins_record(self):
        self.ins_record = []
        self.days_in_good_state = 0

    def isCovered(self):
        total_ins = 0
        num_ins_inWin = 0
        num_ins_outWin = 0
        repeat = int(len(self.ins_record) / self.time_period_len)
        for i in range(repeat):
            num_ins_inWin = 0
            num_ins_outWin = 0
            for j in range(self.time_period_len):
                if self.ins_record[i * self.time_period_len + j]:
                    total_ins += 1
                    if self.isInWindow(j % self.time_period_len):
                        num_ins_inWin += 1
                    else:
                        num_ins_outWin += 1
            if sum(self.multi) > num_ins_inWin:
                return -1
        return 0
        # repeat = int(len(self.ins_record) / self.time_period_len)
        # total_ins_req = repeat * sum(self.multi)

        # if total_ins_req == total_ins:
        #     if num_ins_outWin == 0:
        #         return 0
        #     else:
        #         return 1
        # elif total_ins_req > total_ins:
        #     return 2
        # else:
        #     return 3
        # print(total_ins, num_ins_inWin, num_ins_outWin)
        #
        # if total_ins_req > num_ins_inWin:
        #     return -1
        # elif total_ins_req == num_ins_inWin:
        #     return 0
        # else:
        #     return 1

    def mix_state_step(self, action):
        if not action:
            self.current_state = self.mix_traj_passive[self.current_state]
        else:
            self.current_state = self.mix_traj_active[self.current_state]

    def get_state(self):
        mix_idx = self.time_ext_states[self.current_state][0]
        # return self.current_state, self.time_ext_states[self.current_state], self.mix_states[mix_idx]
        return mix_idx

    def is_in_action_win(self, t):
        return self.isInWindow(t)

    def sampling(self):
        mix_idx = self.time_ext_states[self.current_state][0]

        # print(x)
        rel = 0
        for i in range(20):
            x = np.random.rand()
            if x < self.mix_states[mix_idx][0]:
                rel += 1
            else:
                rel -= 1
        if rel >= 0:
            return 0
        else:
            return 1

    def get_index(self, t):
        assert self.current_state < len(self.whittle_indices)
        return self.whittle_indices[self.current_state]

    def get_mix_index(self, t):
        mix_idx = self.time_ext_states[self.current_state][0]
        return self.mix_whittle[mix_idx]

    def reward_whittle(self):
        return self.rewards[self.current_state]

    def reward_mix_whittle(self):
        return self.mix_rewards[self.current_state]

    # def Whittle_for_LP(self):
    #     # plan for 12mon for now
    #     self.whittle_indices_lp = np.zeros((self.time_period_len, self.time_period_len, 2))
    #     for t in range(self.time_period_len):
    #         for last_t in range(self.time_period_len):
    #             s = last_t % self.num_mix_states
    #             idx = self.time_ext_states.index([s,t,0])
    #             self.whittle_indices_lp[t,last_t,0] = self.whittle_indices[idx]
    #             if t in self.action_window:
    #                 idx1 = self.time_ext_states.index([s,t,1])
    #                 self.whittle_indices_lp[t,last_t,1] = self.whittle_indices[idx1]
    #             else:
    #                 self.whittle_indices_lp[t,last_t,1] = self.whittle_indices[idx]

    def Whittle_for_LP(self):
        # plan for 12mon for now and only
        whittle_indices_lp = [0] * self.time_period_len * 5
        last_t = 0
        for t in range(self.time_period_len * 5):
            # for t in range(self.time_period_len):
            time_step = t % self.time_period_len
            if not self.is_in_action_win(time_step):
                if (t - last_t) >= self.num_mix_states - 1:
                    s = self.num_mix_states - 1
                else:

                    s = t - last_t
                idx = self.time_ext_states.index([s, time_step, 0])
            elif self.isInWindow(time_step):
                if time_step == self.action_window[-1]:
                    if (t - last_t) >= self.num_mix_states - 1:
                        s = self.num_mix_states - 1
                    else:

                        s = t - last_t
                    idx = self.time_ext_states.index([s, time_step, 0])
                    last_t = t
                else:
                    if (t - last_t) >= self.num_mix_states - 1:
                        s = self.num_mix_states - 1
                    else:

                        s = t - last_t
                    idx = self.time_ext_states.index([s, time_step, 0])

            whittle_indices_lp[t] = self.whittle_indices[idx]

            # print(self.time_ext_states[idx])
        return whittle_indices_lp

        # return whittle_indices_lp

   

    def debug(self):
        print("Arm mix states:", end="")
        print(self.mix_states)
        print("Rewards: ", end=" ")
        print(self.mix_rewards)
        print("Action Window", end=" ")
        print(self.action_window)
        print()
        print()

    def nan_count(self):
        tmp = np.array(self.whittle_indices)
        return np.sum(np.isnan(tmp))

    # def value_iter(self):
    #     vi = mdptoolbox.mdp.ValueIteration(np.array([self.P0, self.P1]),
    #                                        np.transpose(np.array([self.rewards, self.rewards])), self.discount)
    #     vi.run()
    #     self.V = vi.V
    #     q = mdptoolbox.mdp.QLearning(np.array([self.P0, self.P1]), np.transpose(np.array([self.rewards, self.rewards])),
    #                                  self.discount)
    #     q.run()
    #     self.Q = q.Q

    # def whittle_index(self, range, start_state_idx):
    #     left = range[0]
    #     right = range[1]
    #     last_m = 9999999
    #     m = (left + right) / 2
    #     while (1):
    #         if abs(last_m - m) < 1e-3:
    #             break
    #         Val_passive = m + self.rewards[start_state_idx] + self.Q[start_state_idx][0]
    #         Val_active = self.rewards[start_state_idx] + self.discount * self.Q[start_state_idx][1]
    #         if Val_passive >= Val_active:
    #             right = m
    #         else:
    #             left = m
    #         m = (left + right) / 2
    #         print("m is " + str(m))
    #         print("passive is " + str(Val_passive))
    #         print("active is " + str(Val_active))

    # print("The whittle index is :" + str(m))

if __name__ == '__main__':
    matrix_no_act = np.array([[0.8, 0.2], [0.1, 0.9]])
    # matrix_no_act = fractional_matrix_power(matrix_no_act, 1 / 4.5)
    print(matrix_no_act)
    # matrix_no_act = np.array([[0.21092801, 0.78907199], [0.93852318, 0.06147682]])
    matrix_act = np.array([[1, 0], [1, 0]])

    arm1 = Arm_2OR1(matrix_no_act, matrix_act, 12,None)
    print(arm1.Whittle_for_LP())
   
    # arm2 = Arm_No_Trans(matrix_no_act, matrix_act, 2, 2, 12, None)
    # print(arm1.Whittle_for_LP())

