
def reward_naive(arms):
    num = len(arms)
    reward = 0
    for arm in arms:
        if arm.sampling() == 0:
            reward += 1
    reward = reward / num
    return reward

def reward_whittle(arms):
    reward = 0
    for arm in arms:
        reward += arm.reward_whittle()
    return reward
