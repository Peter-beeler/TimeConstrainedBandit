import functools
import numpy as np
import random

time_period_len = 52

def policy_naive1(arms, budget):
    num = len(arms)
    actions = num * [0]
    probalities = []
    for arm in arms:
        probalities.append(arm.no_action_trans[0][1])
    probalities = np.array(probalities)
    indexes = np.argpartition(probalities, -budget)[-budget:]  # M most big
    for index in indexes:
        actions[index] = 1
    return actions


def policy_naive2(arms, budget, t, is_time_constrain=False):
    t = t % time_period_len 
    num = len(arms)
    actions = num * [0]
    probalities = []
    for arm in arms:
        # if is_time_constrain and not (arm.is_in_action_win(t) and arm.need_action(t)):
        if not (arm.is_in_action_win(t) and arm.need_action(t)):
            probalities.append(-1)
        else:

            probalities.append(arm.get_risk())
            # probalities.append(arm.mix_states[arm.current_state][1])
    probalities = np.array(probalities)
    # print(probalities)
    indexes = np.argpartition(probalities, -budget)[-budget:]
    for index in indexes:
        actions[index] = 1
    return actions

def policy_multipull(arms, budget, timestamp):
    timestamp = timestamp % time_period_len 
    indices = []
    for arm in arms:
        # indices.append(arm.get_index(timestamp))
        
        indices.append(arm.get_index(timestamp))
        
    index = np.argpartition(indices, - budget)[-budget:]


    rel = [0] * len(arms)
    for x in index:
        rel[x] = 1
    return rel

def policy_whittle(arms, budget, timestamp):
    timestamp = timestamp % time_period_len 
    indices = []
    for arm in arms:
        # indices.append(arm.get_index(timestamp))
        if arm.is_in_action_win(timestamp):
            indices.append(arm.get_index(timestamp))
        else:
            indices.append(-1)

    index = np.argpartition(indices, - budget)[-budget:]


    rel = [0] * len(arms)
    for x in index:
        rel[x] = 1
    return rel

def policy_WIP(arms, budget, timestamp):
    timestamp = timestamp % time_period_len
    indices = []
    for arm in arms:
        
        indices.append(arm.get_index(timestamp))
 


    index = np.argpartition(indices, - budget)[-budget:]


    rel = [0] * len(arms)
    for x in index:
        rel[x] = 1
    return rel

def policy_NO_WIN(arms, budget, timestamp):
    timestamp = timestamp % time_period_len
    indices = []

    for arm in arms:
        if arm.need_action(timestamp):
            indices.append(arm.get_index(timestamp))
        else:
            indices.append(-1)



    index = np.argpartition(indices, - budget)[-budget:]

    # tmp = indices[index[0]]
    # indices[index[0]] = -1
    # if abs(max(indices) - tmp) < 1e-9:
    #     idx = indices.index(max(indices))
    #     if arms[idx].get_state() > arms[index[0]].get_state():
    #         index = [idx]

    rel = [0] * len(arms)
    # print(indices)
    for x in index:
        rel[x] = 1
        # arms[x].is_pulled = True
    return rel

def policy_whittle_win(arms, budget, timestamp):
    timestamp = timestamp % time_period_len 
    indices = []
    for arm in arms:
        
        indices.append(arm.get_index(timestamp))
       
    index = np.argpartition(indices, - budget)[-budget:]
    rel = [0] * len(arms)
    for x in index:
        rel[x] = 1
    return rel


def policy_random(arms, budget, t, is_time_constrain=False):
    t = t % time_period_len 
    if is_time_constrain:
        allowed_arms = []
        for i in range(len(arms)):
            if arms[i].is_in_action_win(t) and arms[i].need_action(t):
                allowed_arms.append(i)
        rel = [0] * len(arms)
        if budget > len(allowed_arms):
            seq = allowed_arms
        else:
            seq = random.sample(allowed_arms, budget)
        for i in seq:
            rel[i] = 1
        return rel
    else:
        rel = [0] * len(arms)
        for i in range(budget):
            index = random.randint(0, len(arms) - 1)
            rel[index] = 1
        return rel


def policy_routine(arms, budget, t, is_time_constrain=False):
    rel = [0] * len(arms)
    count = 0
    index = (t * budget) % len(arms)
    for i in range(budget):
        rel[(count + index) % len(arms)] = 1
        count += 1
    return rel
