import time

import numpy
import numpy as np
import pickle
from arm import Arm, FullObservedArm, FullObservedArmTimed, WIP_Arm, Arm_No_Trans, Arm_NO_Win, WIP, Arm_MultiPull, Arm_2OR1
from reward import reward_naive, reward_whittle, reward_mix_whittle
from policy import policy_naive1, policy_naive2, policy_whittle, policy_random, policy_NO_WIN, policy_WIP, policy_multipull
from tools import avg_and_std
from WhittleLP import LP_Optimize, LP_Optimize_IP_WIN, LP_Optimize_twice, LP_ML
import json
from numba import jit
import os, sys
from win_assign import LP_Assign, onehot_to_int, actions_to_assign
from scipy.linalg import fractional_matrix_power
from tqdm import tqdm
from collections import defaultdict
time_period_len = 12
win_len = 2


def initarms(num_arms, all_trans, all_times, type, whittle_idx, noise_mean, noise_std):
    rel = []

    whittle_rel = []
    for i in tqdm (range (num_arms), desc="Initing Arms"):
        no_action_trans = fractional_matrix_power(all_trans[i][0], 12 / time_period_len)
        if type == 0 or type == 8:
            tmp = Arm_NO_Win(no_action_trans, all_trans[i][1], time_period_len, whittle_idx[i])
        elif type == 12:
            good_to_good = no_action_trans[0][0] + np.random.normal(noise_mean, noise_std)
            bad_to_good = no_action_trans[1][0] + np.random.normal(noise_mean, noise_std)
            if good_to_good < 0:
                good_to_good = 0
            elif good_to_good > 1:
                good_to_good  = 1
            if bad_to_good < 0:
                bad_to_good = 0
            elif bad_to_good > 1:
                bad_to_good  = 1
            matrix = np.array([[good_to_good, 1 - good_to_good],[bad_to_good, 1- bad_to_good]])  
            tmp = Arm_NO_Win(matrix, all_trans[i][1], time_period_len, None)
        elif type == 1:
            tmp = Arm(no_action_trans, all_trans[i][1], all_times[i], win_len, time_period_len, whittle_idx[i],
                      require_whittle_idx=False, need_trans=(type != 0))
        elif type == 13:
            good_to_good = no_action_trans[0][0] + np.random.normal(noise_mean, noise_std)
            bad_to_good = no_action_trans[1][0] + np.random.normal(noise_mean, noise_std)
            if good_to_good < 0:
                good_to_good = 0
            elif good_to_good > 1:
                good_to_good  = 1
            if bad_to_good < 0:
                bad_to_good = 0
            elif bad_to_good > 1:
                bad_to_good  = 1
            matrix = np.array([[good_to_good, 1 - good_to_good],[bad_to_good, 1- bad_to_good]]) 
            tmp = Arm(matrix, all_trans[i][1], all_times[i], win_len, time_period_len, whittle_idx[i],
                      require_whittle_idx=False, need_trans=(type != 0))

        elif type == 2:
            tmp = Arm(no_action_trans, all_trans[i][1], all_times[i], win_len, time_period_len, whittle_idx[i],
                      require_whittle_idx=True, need_trans=(type != 0))

        elif type == 4:
            tmp = Arm(no_action_trans, all_trans[i][1], all_times[i], win_len, time_period_len, whittle_idx[i],
                      require_whittle_idx=True, need_trans=(type != 0))
        elif type == 5:
            tmp = WIP_Arm(no_action_trans, all_trans[i][1])
        elif type == 6:
            tmp = Arm_MultiPull(no_action_trans, all_trans[i][1], all_times[i], win_len, time_period_len, whittle_idx[i],
                      require_whittle_idx=True, need_trans=(type != 0))
        elif type == 9 or type == 11:
            tmp = Arm_2OR1(no_action_trans, all_trans[i][1], time_period_len, whittle_idx[i])
        else:
            tmp = Arm(no_action_trans, all_trans[i][1], all_times[i], win_len, time_period_len, whittle_idx[i],
                      require_whittle_idx=False, need_trans=(type != 0))
        rel.append(tmp)
        if type in [0,2,4,5,6,8,9,11]:
            whittle_rel.append(tmp.whittle_indices)

        else:
            whittle_rel.append([])


    return rel, whittle_rel


class Bandit:

    def __init__(self, config_file, times_file, time_arr=None, budget=1, type=2, noise_mean=None, noise_std=None):
        with open(config_file, 'rb') as f:
            all_trans = np.load(f)
        if time_arr is None:
            with open(times_file, 'rb') as t:
                all_times = np.load(t)
        else:
            all_times = time_arr
        assert 0 <= all_trans.all() <= 1.0
        self.num_arms = all_trans.shape[0]
        print("We have " + str(self.num_arms) + "arms")
        self.arms = []
        self.nanCount = 0
        self.whittle_indice_json = {}
        self.first_window = 0
        self.second_window = 0
        self.outof_window = 0
        self.valid_pulled_arms = []
        self.type = type
        if type == 2:
            whittle_file = config_file + "P4.json"
        elif type == 11:
            whittle_file = config_file + "P9.json"
        else:
            whittle_file = config_file + "P" + str(type) + ".json"
        if os.path.isfile(whittle_file):
            with open(whittle_file, "rb") as fp:
                whittle_idx = pickle.load(fp)
        else:
            whittle_idx = [None] * self.num_arms
        # whittle_idx = [None] * self.num_arms
        self.arms, whittle_rel = initarms(self.num_arms, all_trans, all_times, type, whittle_idx, noise_mean, noise_std)

        if not os.path.isfile(whittle_file):
            with open(whittle_file, "wb") as fp:
                pickle.dump(whittle_rel, fp)

        self.budget = budget
        self.actions_record = []
        # print("MAB initialized: " + str(self.num_arms)  + " arms.")
        # print(self.nanCount)
        # json_object = json.dumps(self.whittle_indice_json, indent=4)
        # with open(config_file[6:-4] + "_whittle.json", "w") as outfile:
        #     outfile.write(json_object)

    def onestep(self, t):
        actions = self.policy(t)
        # print(actions)

        # input actions into IP to get win assignments
        self.actions_record.append(actions)
        for i in range(self.num_arms):
            tmp = False
            if actions[i] == 1:
                self.arms[i].state_step(True, t)
                # if t == self.arms[i].action_window[0][0]:
                #     self.first_window += 1
                #     tmp = True
                # elif t == self.arms[i].action_window[0][1]:
                #     self.second_window += 1
                #     tmp = True
                # else:
                #     self.outof_window += 12
            else:
                self.arms[i].state_step(False, t)
            # if tmp and not i in self.valid_pulled_arms:
            #     self.valid_pulled_arms.append(i)

    def policy(self, t):

        # print(self.type)
        if self.type in [0, 8, 12]:
            # print("Policy whittle WIN")
            return policy_NO_WIN(self.arms, self.budget, t)
        elif self.type == 3:
           return [0] * self.num_arms
        elif self.type == 4:
            return policy_WIP(self.arms, self.budget, t)
        elif self.type == 5:
            # print("Policy WIP_NO_WIN")
            return policy_WIP(self.arms, self.budget, t)
        elif self.type == 6:
            # print("Policy no win")
            return  policy_multipull(self.arms, self.budget, t)
        elif self.type == 7:
            return [0] * self.num_arms
        else:
            # print("Policy rp")
            return policy_random(self.arms, self.budget, True)

    def onestep_reward(self):
        return reward_whittle(self.arms)

    def show_all_states(self):
        for i in range(self.num_arms):
            print("Arm " + str(i) + ": ", end="")
            print(self.arms[i].get_state())
            # print(self.arms[i].sampling())

    def current_whittle(self, t):
        tmp = []
        for arm in self.arms:
            tmp.append(arm.get_index(t))
        return tmp

    def current_mix_state(self):
        tmp = []
        for arm in self.arms:
            tmp.append(arm.get_state())
        return tmp

    def run(self, time=60):
        reward_total = 0
        # print("Rewards:")
        for i in range(time):
            # print("Round " + str(i))
            if i % time_period_len == 0:
                self.reset_ins()
            # if self.type in [0, 1, 2, 5]:
            #     print(self.current_whittle(i))
            # print(self.current_mix_state())

            self.onestep(i)
            reward = self.onestep_reward()
            tmp = []
            for arm in self.arms:
                tmp.append(arm.reward_whittle())
            # print(tmp)
            # print(reward)
            # print("\n\n")
            reward_total += reward
            # print(reward_total)
            # self.show_all_states()
            # print()
            # print()
        return reward_total
        # print("Total Reward is: " + str(reward_total))

    def simulate_onestep(self, t, actions):
        # print(actions)
        for i in range(self.num_arms):
            if actions[i] == 1:
                self.arms[i].state_step(True, t)
                # if t == self.arms[i].action_window[0]:
                #     self.first_window + 1
                # elif t == self.arms[i].action_window[1]:
                #     self.second_window += 1
                # else:
                #     self.outof_window += 1
            else:
                self.arms[i].state_step(False, t)

    def pulls_count(self, actions):
        cnt = defaultdict(int)
        print(f'We have {self.num_arms} arms')
        for i in range(self.num_arms):
            for t in range(time_period_len):
                if actions[i,t]:
                    cnt[i] += 1
        cnt1 = 0
        cnt2 = 0
        for arm in cnt.keys():
            if cnt[arm] == 1:
                cnt1 += 1
            if cnt[arm] == 2:
                cnt2 += 1
        print(f"{cnt1} one pull, {cnt2} twice")
        return cnt
    def simulate_run(self, total_steps, is_tcb, unlimted_pulls):
        assert total_steps % time_period_len == 0
        reward_total = 0
        for i in range(total_steps // time_period_len):
            actions = LP_Optimize(self.arms, self.LP_whittle(), self.budget, time_period_len, is_tcb, unlimted_pulls)
            self.reset_ins()
           
            for i in range(time_period_len):
                
                self.simulate_onestep(i, actions[:, i])
                reward = self.onestep_reward()
                reward_total+=reward

        return reward_total,actions
    
    def simulate_run_atleast(self, total_steps, is_baseline):
        assert total_steps % time_period_len == 0
        reward_total = 0
        
        for i in range(total_steps // time_period_len):
            actions, w = LP_Optimize_twice(self.arms, self.LP_whittle_twice(), self.budget, time_period_len, win_len, is_baseline)
            self.reset_ins()
            
            for i in range(time_period_len):
                
                self.simulate_onestep(i, actions[:, i])
                # print(actions[:, i])
                reward = self.onestep_reward()
                reward_total+=reward

        return reward_total, actions, w
    def simulate_run_ml(self, total_steps):
        assert total_steps % time_period_len == 0
        reward_total = 0
        
        for i in range(total_steps // time_period_len):
            actions = LP_ML(self.arms, self.budget, time_period_len, win_len)
            self.reset_ins()
            
            for i in range(time_period_len):
                
                self.simulate_onestep(i, actions[:, i])
                print(actions[:, i])
                reward = self.onestep_reward()
                reward_total+=reward

        return reward_total

    def simulate_run_ip_win(self, total_steps):
        assert total_steps % time_period_len == 0
        reward_total = 0
        for i in range(total_steps // time_period_len):
            self.run(time_period_len)
            actions = LP_Optimize_IP_WIN(self.arms, self.budget, time_period_len)
            self.reset_ins()
            for i in range(time_period_len):
                self.simulate_onestep(i, actions[:, i])
                reward = self.onestep_reward()
                reward_total += reward

        return reward_total


    def avg_compute(self):
        time = []
        for arm in self.arms:
            time.append(arm.avg_compute)
        return sum(time) / len(time)

    def reset(self):
        for arm in self.arms:
            arm.reset()

    def reset_ins(self):
        for arm in self.arms:
            arm.reset()

    def reset_bandit(self):
        for arm in self.arms:
            arm.reset_all()

    def computation_time(self):
        rel = 0
        for arm in self.arms:
            rel += arm.computation_time
        return rel

    def LP_whittle(self):
        rel = []
        for arm in self.arms:
            rel.append(arm.Whittle_for_LP())
        return rel

    def LP_whittle_twice(self):
        rel = []
        for arm in self.arms:
            rel.append(arm.Whittle_for_LP())
        return rel


def mean_and_std(test_list):
    mean = sum(test_list) / len(test_list)
    variance = sum([((x - mean) ** 2) for x in test_list]) / len(test_list)
    res = variance ** 0.5
    return round(mean, 3), round(res, 3)


def inspection_avg(arms):
    nums = []
    for arm in arms:
        nums.append(arm.inspect_cnt)
    return mean_and_std(nums)


def gridtest(type, budget, trans_file_name, times_file_name):
    if budget <= 0:
        return
    repeat = 1
    start_time = time.time()
    bandit = Bandit(trans_file_name, times_file_name, 0, type=type)
    prepare_time = time.time() - start_time
    # print(str(num_arms) + " arms " + str(budget) + ": ")
    bandit.budget = budget
    time_rel = []
    rewards_rel = []
    days_rel = []
    for i in range(repeat):
        bandit.reset_bandit()
        # print("Rel: " + str(prepare_time), end="/")
        time_rel.append(prepare_time)
        # print(str(bandit.run(60)),end="/")
        rewards_rel.append(bandit.run(12))
        days_in_good = 0
        for arm in bandit.arms:
            days_in_good += arm.days_in_good_state
        # print(str(days_in_good / num_arms))
        days_rel.append(days_in_good / 6570)
    time_mean, time_err = mean_and_std(time_rel)
    days_mean, days_err = mean_and_std(days_rel)
    rel_mean, rel_err = mean_and_std(rewards_rel)

    # count each arm's pulls
    pulls = {}
    for arm in bandit.arms:
        if arm.inspect_cnt in pulls.keys():
            pulls[arm.inspect_cnt] += 1
        else:
            pulls[arm.inspect_cnt] = 1
    return bandit.first_window, bandit.second_window
    # return [rel_mean, rel_err, time_mean, time_err, days_mean, days_err, pulls, bandit.first_window, bandit.second_window, len(bandit.valid_pulled_arms)]
    # print("Rel: " + str(time_mean)+"±"+str(time_err), end="/")
    # time_mean, time_err = mean_and_std(rewards_rel)
    # print(str(time_mean) + "±" + str(time_err), end="/")
    # time_mean, time_err = mean_and_std(days_rel)
    # print(str(time_mean) + "±" + str(time_err), end="/")
    # inspect_cnt_avg, err = inspection_avg(bandit.arms)
    # print("Inspection Distribution:" ,end = " ")
    # print(str(inspect_cnt_avg) + "±" + str(err))
    # print()
    # print()


# bandit = Bandit("trans_config_10.npy", "times_config_10.npy", 1, 0)
# print(bandit.run(60))
def prev_exp():

    V = 20
    policy = ["WIP_WIN", "RP", "TCB", "RFP", "TCB_NO_WIN", "IP"]
    # policy = ["WIP", "RP", "TCB_NO_WIN", "RFP"]
    arms = 1000
    budgets = [ 0.03, 0.04, 0.06, .08, 0.1]
    # budgets = [0.08]
    # budgets = [ 0.04, 0.06, .08]
    rel_table = numpy.zeros((len(policy), len(budgets), V))
    rounds = 5
    # for p in range(len(policy)):
    #     for budget_idx in range(len(budgets)):
    #         for v in range(V):
    rel_no_win = 0
    rel_assign = 0
    total_rounds = time_period_len * rounds
    for v in range(V):
        for p in range(len(policy)):
            config_file = "./testfiles/trans_multi_config_" + str(arms) + "_V" + str(v) + ".npy"
            times_file = "./testfiles/times_multi_config_" + str(arms) + "_V" + str(v) + ".npy"
            bt = Bandit(config_file, times_file, None, 1, p)
            for budget_idx in range(len(budgets)):
                print("V: %2d P: %s Budget: %d" % (v, policy[p], budget_idx))
                budget = budgets[budget_idx]
                bt.budget = int(budget * arms)
                bt.reset_bandit()
                if p != 5:
                    rel_table[p][budget_idx][v] = bt.run(total_rounds)
                else:
                    actions = LP_Optimize(bt, False, p)
                    rel_table[p][budget_idx][v] = bt.simulate_run(actions)
                    # print(actions.shape)

                    # print(bt.run(total_rounds))

    print(rel_table)
    result_file = "./result.npy"
    with open(result_file, "wb") as fp:
        np.save(fp, rel_table)

def assignments_exp():
    V = int(sys.argv[1])
    # policy = ["TCB", "IP"]
    policy = ["WIP", "RP", "TCB", "TCB_NO_WIN", "RFP", "IP", "IP2"]
    arms = 1000
    budgets = [0.03, 0.04, 0.06, .08, 0.1]
    # budgets = [0.08]
    # budgets = [ 0.04, 0.06, .08]
    # rel_table = [[None for i in range(len(budget))] for j in range(len(policy))]
    rel_table = numpy.zeros((len(policy), len(budgets)))
    win_assign_distribution = []
    rounds = 5
    # for p in range(len(policy)):
    #     for budget_idx in range(len(budgets)):
    #         for v in range(V):
    rel_no_win = 0
    rel_assign = 0
    total_rounds = time_period_len * rounds
    for v in range(V, V+1):
        for p in [3]:
            config_file = "./testfiles/trans_multi_config_" + str(arms) + "_V" + str(v) + ".npy"
            times_file = "./testfiles/times_multi_config_" + str(arms) + "_V" + str(v) + ".npy"
            bt = Bandit(config_file, times_file, None, 1, p)
            for budget_idx in range(len(budgets)):
                print("V: %2d P: %s Budget: %d" % (v, policy[p], budget_idx))
                budget = budgets[budget_idx]
                bt.budget = int(budget * arms)
                bt.reset_bandit()
                bt.actions_record = []
                rel_table[p][budget_idx] = bt.run(total_rounds)
                for i in range(rounds):
                    windows, assigns = LP_Assign(bt.num_arms, win_len, time_period_len, bt.actions_record[i * time_period_len: (i+1)*time_period_len])
                    windows = onehot_to_int(windows)
                    dist = [0] * win_len
                    for i in range(bt.num_arms):
                        dist[assigns[i] - windows[i]] += 1
                    win_assign_distribution.append(dist)

                         
                
                    # print(actions.shape)

                    # print(bt.run(total_rounds))
    print(win_assign_distribution)
    win_assign_distribution = np.array(win_assign_distribution)
    print(rel_table)
    result_file1 = "./result_nowin_atmostonce_rewards_V" +str(V)+".npy"
    result_file2 = "./result_assginprob_V" +str(V)+".npy"
    with open(result_file1, "wb") as fp:
        np.save(fp, rel_table)
    with open(result_file2, "wb") as fp:
        np.save(fp, win_assign_distribution)

def reward_exp():
    V = int(sys.argv[1])
    P = int(sys.argv[2])
    policy = ["TCB_NO_WIN", "IP", "TCB_IP", "Nothing"]
    # policy = ["WIP", "RP", "TCB", "TCB_NO_WIN", "RFP", "IP", "IP2", "Nothing"]
    arms = 1000
    budgets = [ 0.1, .12, 0.15]
    # budgets = [0.08]
    # budgets = [ 0.04, 0.06, .08]
    rel_table = numpy.zeros((len(policy), len(budgets)))
    rounds = 5
    # for p in range(len(policy)):
    #     for budget_idx in range(len(budgets)):
    #         for v in range(V):
    rel_no_win = 0
    rel_assign = 0
    total_rounds = time_period_len * rounds
    v = V
    begin = time.time()
    for p in [P]:
        config_file = "./testfiles/trans_multi_config_" + str(arms) + "_V" + str(v) + ".npy"
        times_file = "./testfiles/times_multi_config_" + str(arms) + "_V" + str(v) + ".npy"
        # config_file = "./testfiles/trans_real.npy"
        # times_file = "./testfiles/times_real.npy"
        bt = Bandit(config_file, times_file, None, 1, p)
        for budget_idx in [0]:
            print("V: %2d P: %s Budget: %d" % (v, policy[p], budget_idx))
            budget = budgets[budget_idx]
            bt.budget = int(budget * arms)
            bt.reset_bandit()
            if p in [0, 3]:
                rel_table[p][budget_idx] = bt.run(total_rounds)
            elif p == 2:

                rel_table[p][budget_idx] = bt.simulate_run(total_rounds, True)
            else:
                rel_table[p][budget_idx] = bt.simulate_run(total_rounds, False)
                # print(actions.shape)

                # print(bt.run(total_rounds))
    for p in range(len(policy) - 1):
        for budget_idx in range(len(budgets)):
            rel_table[p][budget_idx] -= rel_table[-1][budget_idx]
    print(rel_table)
    print(time.time() - begin)
    import psutil
    process = psutil.Process()
    print(process.memory_info().rss / (1024 * 1024))  # in bytes 

    # result_file = "./result_rewards_TCBIP_V" + str(V) + ".npy"
    result_file = "./result_rewards_real_TCBIP.npy"
    with open(result_file, "wb") as fp:
        np.save(fp, rel_table)

def pulls_ana(actions1, actions2):
    dict1 = defaultdict(list)
    dict2 = defaultdict(list)
    for i in range(len(actions1)):
        for t in range(time_period_len):
            if actions1[i,t]:
                dict1[i].append(t)
            if actions2[i,t]:
                dict2[i].append(t)
    cnt1 = 0
    cnt2 = 0
    for i in range(len(actions1)):
        print(f"Arm {i} : One: {dict1[i]} Two:{dict2[i]}")
        if len(dict2[i]) == 1:
            cnt1 += 1
        else:
            cnt2 += 2
    print(f"One:{cnt1} Two:{cnt2}")
def cost_exp(V, noise_mean, noise_std):
    
    policy = ["TCB_NO_WIN", "IP", "TCB_IP", "Nothing", "TCB", "WIP", "WIN_MULTI", "TCB_NO_WIN_MULTI", "IP_OP_WIN", "TCB_At_least", "ML", "atleast_bsl","Noise_Study", "Noise_base_ip"]
    polices_names = ["(Opt, Opt, =1)", "(Opt, IP, =1)","(Rdm, Opt, =1)", "No inspections"]
    # 0 "TCB_NO_WIN"
    # 1 "IP"
    # 2 "TCB_IP"
    # 3 "Nothing"
    # 4 "TCB"
    # 5 "WIP"
    # 6 "WIN_MULTI" 
    # 7 "TCB_NO_WIN_MULTI"
    # 8 "IP_OP_WIN"
    # 9 "TCB_At_least" 
    # 10 "ML"
    # 11 "atleast_bsl"
    # 12 "Noise_Study"
    # 13 "Noise_base_ip"

    arms = 1000
    budgets = [ 0.1, .12, 0.15]
    # budgets = [0.08]
    # budgets = [ 0.04, 0.06, .08]
    # rel_table = numpy.zeros((len(policy), len(budgets)))
    rel_table = [[None for i in range(len(budgets))] for j in range(len(policy) + 1)]
    rounds = 5
    # for p in range(len(policy)):
    #     for budget_idx in range(len(budgets)):
    #         for v in range(V):
    rel_no_win = 0
    rel_assign = 0
    total_rounds = time_period_len * rounds
    v = V
    begin = time.time()
    # for p in range(len(policy)):
    for p in [0,1,2,3]:
        config_file = "./testfiles/trans_config_" + str(arms) + "_V" + str(v) + ".npy"
        times_file = "./testfiles/times_config_" + str(arms) + "_V" + str(v) + ".npy"
        # config_file = "./testfiles/trans_real.npy"
        # times_file = "./testfiles/times_real.npy"
        bt = Bandit(config_file, times_file, None, 1, p, noise_mean, noise_std)
        for budget_idx in [0]:
            print("V: %2d P: %s Budget: %d" % (v, policy[p], budget_idx))
            budget = budgets[budget_idx]
            bt.budget = int(budget * len(bt.arms))
            bt.reset_bandit()
            if p in [0, 3, 4, 5, 6, 7]:
                rel_table[p][budget_idx] = bt.run(total_rounds)
            elif p == 12:
                rel_table[p][budget_idx] = bt.run(total_rounds)
            elif p == 1:
                rel_table[p][budget_idx],_ = bt.simulate_run(total_rounds, False, False)
            elif p == 13:
                rel_table[p][budget_idx],_ = bt.simulate_run(total_rounds, False, False)
            elif p == 2:
                rel_table[p][budget_idx], tmp1 = bt.simulate_run(total_rounds, True, False)
            elif p == 8:
                rel_table[p][budget_idx] = bt.simulate_run_ip_win(total_rounds)
            elif p == 9:
                rel_table[p][budget_idx], tmp2, w = bt.simulate_run_atleast(total_rounds, False)
            elif p == 11:
                rel_table[p][budget_idx], tmp2, w = bt.simulate_run_atleast(total_rounds, True)
            else:
                rel_table[p][budget_idx] = bt.simulate_run_ml(total_rounds)

              
    # pulls_ana(tmp1, tmp2)
                # print(bt.run(total_rounds))
    # for p in range(len(policy)):
    #     if p == 3:
    #         continue
    #     for budget_idx in range(len(budgets)):
    #         rel_table[p][budget_idx] -= rel_table[3][budget_idx]
    # rel_table = np.array(rel_table)

    #output result table
    for i in [0, 1, 2, 3]:
        print(polices_names[i], end=': ')
        print(rel_table[i])
 

    # print(time.time() - begin)
    # import psutil
    # process = psutil.Process()
    # print(process.memory_info().rss / (1024 * 1024))  # in bytes 
    # print(noise_std)
    # result_file = "./result_rewards_cost_atleast_V" + str(V) + ".npy"
    # # result_file = "./result_rewards_cost_new_real.npy"
    # with open(result_file, "wb") as fp:
    #     np.save(fp, rel_table)
    # return rel_table[0][0], rel_table[12][0]




def mixing_study():
    import math
    v = 0
    arms = 1000
    config_file = "./testfiles/trans_config_" + str(arms) + "_V" + str(v) + ".npy"
    times_file = "./testfiles/times_config_" + str(arms) + "_V" + str(v) + ".npy"
    # config_file = "./testfiles/trans_real.npy"
    # times_file = "./testfiles/times_real.npy"
    bt = Bandit(config_file, times_file, None, 1, 1)
    mixing_steps = []
    mixing_states = []
    for i in range(bt.num_arms):
        mixing_states.append(bt.arms[i].mix_states[-1][0])
        mixing_steps.append(bt.arms[i].num_mix_states)  
    print(np.mean(mixing_states))
    print(np.std(mixing_states) / math.sqrt(len(mixing_states)))
    print(np.mean(mixing_steps))
    print(np.std(mixing_steps) / math.sqrt(len(mixing_steps)))


if __name__ == '__main__':
    V = int(sys.argv[1])
    cost_exp(V, 0, 0)
