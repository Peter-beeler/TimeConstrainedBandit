# from bandit import Bandit
import numpy as np
import sys
from tools import avg_and_std
import gurobipy as gp
from gurobipy import GRB


def LP_Optimize(arms, W, budget, time_period_len = 52, is_tcb = True, unlimted_pulls = False):
    W = np.array(W)
    time_period = W.shape[1]
    assert time_period == time_period_len
    num_arms = W.shape[0]
    

    try:

        # Create a new model
        model = gp.Model("mip1")

        # Create variables
        a = model.addMVar(shape = (num_arms, time_period), vtype=GRB.BINARY, name="action")

        # Set objective
        obj_tcb = gp.quicksum( a[i, t] * W[i, t]
                        for i in range(num_arms)
                        for t in range(time_period))
        obj_IP = gp.quicksum(a[i, t]
                        for i in range(num_arms)
                        for t in range(time_period))
        if is_tcb:
            model.setObjective(obj_tcb, GRB.MAXIMIZE)
        else:
            model.setObjective(obj_IP, GRB.MAXIMIZE)

        for j in range(time_period):
            model.addConstr((gp.quicksum(a[i,j] for i in range(num_arms) )) <= budget)
        # for t in range(int(time_period / time_period_len)):
        #     for i in range(num_arms):
        #         model.addConstr((gp.quicksum(a[i, j] for j in range(t * time_period_len, t * time_period_len + time_period_len))) <= 1)
        if not unlimted_pulls:
            for i in range(num_arms):
                model.addConstr((gp.quicksum(a[i, j] for j in range(time_period))) == 1)

        for i in range(num_arms):
            for j in range(time_period):
                if not arms[i].is_in_action_win(j % time_period_len):
                    model.addConstr(a[i,j]==0)




        # Optimize model
        model.optimize()
        # print("Model Done!")
        return np.array(a.X)
        # print('Obj: %g' % model.ObjVal)

    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ': ' + str(e))
        return None

    except AttributeError:
        print('Encountered an attribute error')
        return None
    return rel


def LP_Optimize_IP_WIN(arms, budget, time_period_len = 52):
    num_arms = len(arms)
    

    try:

        # Create a new model
        model = gp.Model("mip1")

        # Create variables
        a = model.addMVar(shape = (num_arms, time_period_len), vtype=GRB.BINARY, name="action")

        # Set objective
        obj_IP = gp.quicksum(a[i, t]
                        for i in range(num_arms)
                        for t in range(time_period_len))
       
        model.setObjective(obj_IP, GRB.MAXIMIZE)

        for j in range(time_period_len):
            model.addConstr((gp.quicksum(a[i,j] for i in range(num_arms) )) <= budget)
       
        for i in range(num_arms):
            model.addConstr((gp.quicksum(a[i, j] for j in range(time_period_len))) == 1)

        for i in range(num_arms):
            if arms[i].inspect_step < 0:
                    continue
            for j in range(time_period_len):
                if not (0 <= arms[i].inspect_step - j <time_period_len ):
                    model.addConstr(a[i,j]==0)




        # Optimize model
        model.optimize()
        # print("Model Done!")
        return np.array(a.X)
        # print('Obj: %g' % model.ObjVal)

    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ': ' + str(e))
        return None

    except AttributeError:
        print('Encountered an attribute error')
        return None
    return rel

def random_LP(arms, W, budget, time_period_len = 52, act_win_len=2):
    a = np.zeros((len(arms), time_period_len))
    start = 0
    for t in range(time_period_len):
        for i in range(start, start + budget):
            idx = i % len(arms)
            a[idx,t] = 1
        start += budget
    return a, None
def LP_Optimize_twice(arms, W, budget, time_period_len = 52, act_win_len=2, is_baseline = False):
    W = np.array(W)
    time_period = time_period_len
    # assert time_period == time_period_len
    num_arms = W.shape[0]
    w = np.zeros((num_arms, time_period ))
    for i in range(num_arms):
        for j in range(time_period):
            w[i,j] = W[i][0][j]

    try:

        # Create a new model
        model = gp.Model("WHITTLE")

        # Create variables
        a = model.addMVar(shape = (num_arms, time_period ), vtype=GRB.BINARY, name="action")
        # w = model.addMVar(shape = (num_arms, time_period ), vtype=GRB.CONTINUOUS, name="whittle_value")
        # Set objective
        if not is_baseline:
            obj_tcb = gp.quicksum((a[i,t]*  w[i, t])
                            for i in range(num_arms)
                            for t in range(time_period))
        else:
            # return random_LP(arms, W, budget, time_period_len, act_win_len)
            obj_tcb = 1
        model.setObjective(obj_tcb, GRB.MAXIMIZE)
      
        #contraints for budgets at one time step

        for t in range(time_period):
            
            model.addConstr(gp.quicksum(a[i,t] for i in range(num_arms) ) <= budget)
            model.addConstr(gp.quicksum(a[i,t] for i in range(num_arms) ) >= 0.9 * budget)
       
        #constraints: at least once most twice
        for i in range(num_arms):
            model.addConstr(  gp.quicksum(a[i,t] for t in range(time_period_len)) <= 2)
            model.addConstr(  gp.quicksum(a[i,t] for t in range(time_period_len)) >= 1)

        # #constraints for w cannot bigger than that no pulls before t:
        # for i in range(num_arms):
        #     for t in range(time_period_len):
        #         model.addConstr(w[i,t] <= W[i][0][t])
               
    
        # # constraints: conditioned pull for w
        # M = 1000000000
        # if not is_baseline:
        #     for i in range(num_arms):
        #         for t1 in range(time_period):
        #             for t2 in range(time_period):
        #                 if t1 < t2:
        #                     model.addConstr(w[i,t2] <= (1 - a[i,t1]) * M + W[i][t1+1][t2])

        # Optimize model
        # model.Params.Presolve = 0
        # model.Params.MIPGap = 0.05
        # model.Params.TimeLimit = 100
        model.setParam(GRB.Param.PoolSearchMode, 2)  # Continue searching after optimal
        model.setParam(GRB.Param.PoolSolutions, 10)  # Store up to 10 solutions
        model.setParam(GRB.Param.TimeLimit, 60)  # Allow up to 60 seconds
        model.setParam(GRB.Param.MIPGap, 0.0)  # Ensure only optimal solutions are considered
        model.optimize()
        
        num_solutions = model.SolCount

       
        # Randomly select one solution index
        random_index = np.random.randint(0, num_solutions)
        
        # Set the solution number
        model.setParam(GRB.Param.SolutionNumber, random_index)
        
       
        ret = np.array(a.Xn)
        # ret = np.rint(np.array(a.X))
        # ret_w = np.array(w.X)
        
        
        return ret, None
        # print('Obj: %g' % model.ObjVal)

    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ': ' + str(e))
        return None

    except AttributeError:
        print('Encountered an attribute error')
        return None
    return rel

def LP_ML(arms,budget, time_period_len = 52, act_win_len=2):
    time_period = time_period_len
    # assert time_period == time_period_len
    num_arms = len(arms)
    with open("./xg_scores_1801.npy", 'rb') as f:
        ml_scores = np.load(f)
    
    assert num_arms == len(ml_scores)
  

    try:

        # Create a new model
        model = gp.Model("WHITTLE")

        # Create variables
        a = model.addMVar(shape = (num_arms, time_period ), lb=0, ub=1, vtype=GRB.CONTINUOUS, name="action")
        # Set objective
       
        obj_tcb = gp.quicksum((a[i,t]*  ml_scores[i])
                        for i in range(num_arms)
                        for t in range(time_period))


        model.setObjective(obj_tcb, GRB.MAXIMIZE)
      
        #contraints for budgets at one time step

        for t in range(time_period):
            
            model.addConstr(gp.quicksum(a[i,t] for i in range(num_arms) ) <= budget)
       
        #constraints: at least once most twice
        for i in range(num_arms):
            model.addConstr(  gp.quicksum(a[i,t] for t in range(time_period_len)) == 1)

       
               
    
      
        # Optimize model
        # model.Params.Presolve = 0
        # model.Params.MIPGap = 0.05
        # model.Params.TimeLimit = 100
        model.optimize()
        
        #     flag = False
        #     for i in range(num_arms):
        #         for t in range(time_period):
        #             if not(abs(a[i,t].X) < 1e-6 or abs(a[i,t].X - 1) < 1e-6):
        #                 a[i,t].VType = GRB.BINARY
        #                 flag = True
        #     model.optimize()
        # print("Model Done!")
        
        ret = np.rint(np.array(a.X))
        
        
        
        return ret
        # print('Obj: %g' % model.ObjVal)

    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ': ' + str(e))
        return None

    except AttributeError:
        print('Encountered an attribute error')
        return None
    return rel

# def LP_Optimize_twice(arms, W, budget, time_period_len = 52, act_win_len=2):
#     time_period = time_period_len
#     # assert time_period == time_period_len
#     num_arms = len(W)
#     num_mix_states = []
#     s0 = []
#     for arm in arms:
#         num_mix_states.append(arm.num_mix_states)
#         s0.append(arm.get_mix_index())

#     try:

#         # Create a new model
#         model = gp.Model("WHITTLE")

#         # Create variables
#         a = model.addMVar(shape = (num_arms, time_period, time_period), vtype=GRB.BINARY, name="action")
#         # Set objective
#         obj_tcb = gp.quicksum( a[i,t1, t2]  * (W[i][ (t1 + s0[i]) % num_mix_states[i],t1,0] + W[i][ (t2-t1)%num_mix_states[i], t2, 1])
#                         for t1 in range(time_period)
#                         for t2 in range(t1, time_period)
#                         for i in range(num_arms))

#         model.setObjective(obj_tcb, GRB.MAXIMIZE)
      
#         #contraints for budgets at one time step
#         for t in range(time_period):
#             model.addConstr(gp.quicksum(a[i,t,t1] + a[i,t2,t] for i in range(num_arms) for t1 in range(t, time_period) for t2 in range(t)) <= budget)
       
#         #constraints: at least once most twice
#         for i in range(num_arms):
#             model.addConstr(gp.quicksum(a[i,t1, t2] for t1 in range(time_period) for t2 in range(time_period)) == 1)
#             for t1 in range(time_period):
#                 for t2 in range(time_period):
#                     if t1 > t2:
#                         model.addConstr(a[i,t1,t2] == 0)
               

        

#         # Optimize model
#         # model.Params.Presolve = 0
#         model.optimize()
        
#         #     flag = False
#         #     for i in range(num_arms):
#         #         for t in range(time_period):
#         #             if not(abs(a[i,t].X) < 1e-6 or abs(a[i,t].X - 1) < 1e-6):
#         #                 a[i,t].VType = GRB.BINARY
#         #                 flag = True
#         #     model.optimize()
#         # print("Model Done!")
        
#         solution = a.X
       
        
#         ret = np.zeros((num_arms, time_period))
#         for i in range(num_arms):
#             for t1 in range(time_period):
#                 for t2 in range(time_period):
#                     if solution[i,t1, t2] == 1:
#                         ret[i][t1] = 1
#                         ret[i][t2] = 1
#         # print(k)
#         return ret
#         # print('Obj: %g' % model.ObjVal)

#     except gp.GurobiError as e:
#         print('Error code ' + str(e.errno) + ': ' + str(e))
#         return None

#     except AttributeError:
#         print('Encountered an attribute error')
#         return None
#     return rel

# def LP_Optimize_multi(bt, force_equal=False, ins=1):
#     W = bt.LP_whittle()
#     W = np.array(W)
#     time_period = W.shape[1]
#     num_arms = W.shape[0]
#     # print("LP model with %d arms and time period is %d" % (num_arms, time_period))
#     # print(W[5])
#     # print(bt.arms[5].action_window)
#     # for i in bt.arms:
#     #     print(i.action_window)
#     budget = bt.budget

#     try:

#         # Create a new model
#         model = gp.Model("mip1")

#         # Create variables
#         a = model.addMVar(shape=(num_arms, time_period), vtype=GRB.BINARY, name="action")

#         # Set objective
#         obj = gp.quicksum(a[i, t] * W[i, t]
#                           for i in range(num_arms)
#                           for t in range(time_period))
#         obj_possible_max = gp.quicksum(a[i, t]
#                                        for i in range(num_arms)
#                                        for t in range(time_period))
#         # obj_balance = gp.quicksum( a[i, t] * W[i, t, t]
#         #                 for i in range(num_arms)
#         #                 for t in range(time_period)) + gp.quicksum(a[i, t]
#         #                 for i in range(num_arms)
#         #                 for t in range(time_period))
#         model.setObjective( obj_possible_max, GRB.MAXIMIZE)

#         # Add constraint: x + 2 y + 3 z <= 4
#         # if force_equal:
#         #     for i in range(num_arms):
#         #         model.addConstr((gp.quicksum(a[i,j] for j in range(time_period) )) == ins * 5)
#         # else:
#         #     for i in range(num_arms):
#         #         model.addConstr((gp.quicksum(a[i,j] for j in range(time_period) )) <= ins * 5)

#         for j in range(time_period):
#             model.addConstr((gp.quicksum(a[i, j] for i in range(num_arms))) <= budget)

#         for i in range(len(bt.arms)):
#             for j in range(time_period):
#                 if not bt.arms[i].is_in_action_win(j % 12):
#                     model.addConstr(a[i, j] == 0)

#         if force_equal:
#             for i in range(len(bt.arms)):
#                 action_wins = bt.arms[i].action_window
#                 for win in action_wins:
#                     for repeat in range(5):
#                         rel_win = [win[0] + repeat * 12, win[1] + repeat * 12]
#                         model.addConstr((gp.quicksum(a[i, j] for j in rel_win)) == 1)
#         else:
#             for i in range(len(bt.arms)):
#                 action_wins = bt.arms[i].action_window
#                 for win in action_wins:
#                     for repeat in range(5):
#                         rel_win = [win[0] + repeat * 12, win[1] + repeat * 12]
#                         model.addConstr((gp.quicksum(a[i, j] for j in rel_win)) <= 1)

#         # Optimize model
#         model.optimize()
#         # print("Model Done!")
#         return np.array(a.X)
#         # print('Obj: %g' % model.ObjVal)

#     except gp.GurobiError as e:
#         print('Error code ' + str(e.errno) + ': ' + str(e))
#         return None

#     except AttributeError:
#         print('Encountered an attribute error')
#         return None
#     return rel
# def is_valid(actions, bt):
#     n1 , n2 = actions.shape
#     cnt1 = 0
#     cnt2 = 0
#     cnt_inspeted = 0
#     for i in range(n1):
#         inspected = False
#         for j in range(n2):
#             if actions[i,j] == 1 and not j in bt.arms[i].action_window :
#                 print("Arm %d is not valid for act in %d" % (i, j))
#                 print(bt.arms[i].action_window)
#             if actions[i,j] == 1 and  j == bt.arms[i].action_window[0] :
#                 cnt1 += 1
#                 inspected = True
#             if actions[i,j] == 1 and  j == bt.arms[i].action_window[1] :
#                 cnt2 += 1
#                 inspected = True
#         if inspected:
#             cnt_inspeted += 1


#     # print(cnt1)
#     # print(cnt2)
#     print("%f arms choose first window, %f choose second, total %d arms inspected validly" % (1.0 * cnt1 / n1, 1.0 * cnt2 / n1, cnt_inspeted))
#     return cnt_inspeted

# if __name__ == '__main__':
#     rel = []
#     V = sys.argv[1]
#     config_file = "./trans_config_5000_V" + str(V) + ".npy"
#     times_file = "./times_config_5000_V" + str(V) + ".npy"
#     bt = Bandit(config_file, times_file, budget=500)
#     str_rel = ""
#     while(1):
#         actions = LP_Optimize(bt, 500)

    # for bg in [416,430,450,500]:
    # # for bg in [1,10]:
    #     bt.reset()
    #     value = 0
    #     # for i in range(5):
    #     bt.reset()
    #     actions = LP_Optimize(bt, bg)
    #     print(GRB.OPTIMAL)
    #     # if GRB.OPTIMAL  == 3:
    #     #     str_rel += "V" +str(V) + "no solution with budegt" + str(bg) + "\n"
    #     #     continue
    #     try:
    #     # total = is_valid(actions, bt)
    #         value = bt.simulate_run(actions)
    #     except:
    #         str_rel += "V" +str(V) + "no solution with budegt" + str(bg) + "\n"
    #         continue
    #     # print("bg %d LP reward: %f" % (bg,value))
    #     # bt.reset()
    #     # print("Whittle reward: %f" % (bt.run(60)))
    #     # print()
    #     # print()
    #     # rel.append([ value])
    #     str_rel += "V" +str(V) + " value : " + str(value) + " with budegt" + str(bg) + "\n"
    # print(str_rel)
# import time
# # 
        



# if __name__ == '__main__':


#     rel_whittle_time = []
#     rel_lp = []
#     # num_arm = sys.argv[1]
#     arms = int(sys.argv[1])
#     dic = {}
#     str_rel = ""
#     for num in [arms]:
#         for budget in [0.01, .05, .1, .2]:
#             if int(num * budget) < 1:
#                 continue
#             rel = []
#             for v in range(10):

#                 config_file = "./trans_config_"+str(num)+"_V" + str(v) + ".npy"
#                 times_file = "./times_config_"+str(num)+"_V" + str(v) + ".npy"
#                 start = time.time()
#                 if config_file in dic.keys():
#                     bt = dic[config_file]
#                 else:
#                     bt = Bandit(config_file, times_file, budget=1, type=0)
#                     dic[config_file] = bt
#                 # bt = Bandit(config_file, times_file, budget=500)
#                 end = time.time()
#                 rel_whittle_time.append(end - start)
#                 # for bg in [380,400,416,430,450]:
#                 bt.reset()
#                 bt.budget = int(num * budget)
#                 start = time.time()
#                 value = 0
#                 for i in range(5):
#                     # bt.reset()
#                     actions = LP_Optimize(bt, int(num * budget))
#                     value += bt.simulate_run(actions)
#                 end = time.time()
#                 # total = is_valid(actions, bt)
#                 # value = bt.simulate_run(actions)
#                 # print("LP reward: %f" % (value))
#                 # rel.append([total, value])
#                 rel_lp.append(end-start)
#                 rel.append(value)
            
#             # print(rel_whittle_time)
#             # print(rel_lp)
#             avg,std = avg_and_std(rel)
#             str_rel += "IP_" + str(num)+"_"+str(budget)+":"
#             str_rel += "{:.2f} ± {:.2f}".format(avg,std)
#             str_rel += "\n"
#     print(str_rel)





# import time
# if __name__ == '__main__':

#     config_file = "./trans_config_6570.npy"
#     times_file = "./times_config_6570.npy"
#     # config_file = "./trans_config_100_V2.npy"
#     # times_file = "./times_config_100_V2.npy"
#     start = time.time()

#     bt = Bandit(config_file, times_file, budget=657, type=0)
#     end = time.time()
#     bt.reset()
#     bt.budget =657
#     start = time.time()
#     # actions = LP_Optimize(bt, 10)
#     end = time.time()
#     # total = is_valid(actions, bt)
#     value = 0
#     for i in range(5):
#         actions = LP_Optimize(bt, 657)
#         value += bt.simulate_run(actions)
#     # print("LP reward: %f" % (value))
#     # rel.append([total, value])
#     print(value)
#     rel = []
#     for arm in bt.arms:
#         rel.append(arm.days_in_good_state)
#     print("months in good for real data  domain ip policy: %.2f  %.2f" % (avg_and_std(rel)))
  

