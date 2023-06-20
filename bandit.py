import numpy as np
from arm import Arm
from reward import reward_naive, reward_whittle, reward_mix_whittle
from policy import policy_naive1,policy_naive2,policy_whittle, policy_random, policy_routine, policy_whittle_win
class Bandit:
    def __init__(self, config_file, times_file, budget = 1):
        with open(config_file, 'rb') as f:
            all_trans = np.load(f)
        with open(times_file, 'rb') as t:
            all_times = np.load(t)
        self.num_arms = all_trans.shape[0]
        self.arms = []
        self.nanCount = 0
        for i in range(self.num_arms):
            tmp = Arm(all_trans[i][0], all_trans[i][1], all_times[i])
            self.arms.append(tmp)
            tmp.debug()
            # print(tmp.P1)
            # print(tmp.rewards)
            # tmp.value_iter()
            # print(tmp.action_trans)
            # print(tmp.no_action_trans)
            # print("Whittle Index: ", end=" ")
            # print(tmp.whittle_indices)
            # print()
            # print()
            # self.nanCount += tmp.nan_count()

        self.budget = budget
        print("MAB initialized: " + str(self.num_arms)  + " arms.")
        print(self.nanCount)


    def onestep(self, t):
        actions = self.policy(t)
        print(actions)
        for i in range(self.num_arms):
            if actions[i] == 1:
                self.arms[i].state_step(True)
            else:
                self.arms[i].state_step(False)


    def policy(self, t):
        return policy_whittle_win(self.arms, self.budget, t)
        # return policy_naive2(self.arms, self.budget, t, True)
        # return policy_random(self.arms, self.budget, True)

    def onestep_reward(self):
        return reward_whittle(self.arms)

    def show_all_states(self):
        for i in range(self.num_arms):
            print("Arm " + str(i) + ": ", end="")
            print(self.arms[i].get_state())
            # print(self.arms[i].sampling())

    def run(self, time=60):
        reward_total = 0
        print("Rewards:")
        for i in range(time):
            self.onestep(i)
            reward = self.onestep_reward()
            reward_total += reward
            print(reward_total)
            self.show_all_states()
            print()
            print()
        # print("Total Reward is: " + str(reward_total))





if __name__ == "__main__":
    time_period = 10
    bandit = Bandit("trans_config.npy","times_config.npy",2)
    bandit.run(60)
    # print(bandit.arms[1].no_action_trans)
    # print(bandit.arms[1].whittle_indices)
    # print(bandit.arms[1].mix_states)
