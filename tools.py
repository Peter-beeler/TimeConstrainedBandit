import json as js
import random
import numpy as np

# config parameters
num_restaurant = 10 # how many arms we have
normal_ratio = 1
file_path = "trans_config.npy"
file_path2 = "times_config.npy"
no_action_range1 = [0.8, 0.9]
no_action_range2 = [0.6, 0.7]
action_range1 = [0.98, 1]
action_range2 = [0.95, 1]

# no_action_range1 = [0, 1]
# no_action_range2 = [0, 1]
# action_range1 = [0, 1]
# action_range2 = [0, 1]

inspect_times = 5
config = []
num_normal_trans = int(num_restaurant * normal_ratio)
num_random_trans = num_restaurant - num_normal_trans
times = []
for i in range(num_normal_trans):
    tmp = []
    # x1 = random.uniform(no_action_range1[0], no_action_range1[1])
    # x2 = random.uniform(no_action_range2[0], no_action_range2[1])
    # x3 = random.uniform(action_range1[0], action_range1[1])
    # x4 = random.uniform(action_range1[0], action_range1[1])
    x1 = random.random()
    x2 = random.random()
    x3 = 1
    x4 = 1
    randomlist = random.sample(range(0, 60), inspect_times * 2)

    tmp.append([[x1, 1 - x1], [x2, 1 - x2]])
    tmp.append([[x3, 1 - x3], [x4, 1 - x4]])
    randomlist.sort()
    times.append(randomlist)
    config.append(tmp)

config = np.array(config)
print(config.shape)

with open(file_path, 'wb') as f:
    np.save(f, config)

times = np.array(times)
with open(file_path2, 'wb') as f:
    np.save(f, times)
print(times.shape)
