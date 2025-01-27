import gurobipy as gp
from gurobipy import GRB
import numpy as np


def actions_to_assign(actions):  # turn actions in time series to each arm's inspection timestep.
    actions = np.array(actions)
    total_steps = len(actions)
    num_arm = len(actions[0])
    # for i in range(num_arm):
    #     print(actions[:, i])
    assigns = [-1] * num_arm
    for n in range(num_arm):
        for i in range(total_steps):
            if actions[i][n] == 1 and assigns[n] == -1:
                assigns[n] = i
    final_actions = np.zeros((total_steps, num_arm))
    # print(assigns)
    for i in range(num_arm):
        final_actions[assigns[i]][i] = 1
    # print(final_actions)

    assert len(assigns) == num_arm
    return assigns, final_actions


def onehot_to_int(assigns):
    num_arms = len(assigns)
    rel = []
    for assign in assigns:
        rel.append(int(np.where(assign == 1)[0]))
    assert len(rel) == num_arms
    return rel


def LP_Assign(num_arms, win_len, time_period, actions):
    ins_per_year = 1

    assigns, final_actions = actions_to_assign(actions)
    try:

        # Create a new model
        model = gp.Model("mip1")

        # Create variables
        x = model.addMVar(shape=(num_arms, time_period - win_len + 1), vtype=GRB.BINARY, name="assign_win")
        # set intermediate var(counter)
        # w = model.addMVar(shape=(num_arms, time_period, win_len, win_len), vtype=GRB.INTEGER, name="assign_win")
        # Set objective
        obj = 0
        for t in range(0, time_period - win_len + 1):
          
            temp_obj = []
            for j in range(win_len):
                temp_obj.append(gp.quicksum(x[n, t] * final_actions[j + t][n] for n in range(num_arms)))
            # tmp_obj2 = gp.quicksum(x[n, t] * final_actions[1 + t][n] for n in range(num_arms))
            for i in range(len(temp_obj)):
                for j in range(i+1, len(temp_obj)):
                    diff = model.addVar(vtype=GRB.INTEGER)
                    model.addConstr(diff >= temp_obj[i] - temp_obj[j])
                    model.addConstr(diff >= temp_obj[j] - temp_obj[i])
                    obj += diff

            # obj += abs_var

        for i in range(num_arms):
            for j in range(time_period - win_len + 1):
                if not (0 <= assigns[i] - j <= win_len-1):
                    model.addConstr(x[i, j] == 0)

        for i in range(num_arms):
            model.addConstr((gp.quicksum(x[i, j] for j in range(time_period - win_len + 1))) == ins_per_year)

        model.setObjective(obj)
        # Optimize model

        model.optimize()
        # print("Model Done!")
        return np.array(x.X), assigns
        # print('Obj: %g' % model.ObjVal)

    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ': ' + str(e))
        return None

    except AttributeError:
        print('Encountered an attribute error')
        return None
    return rel
