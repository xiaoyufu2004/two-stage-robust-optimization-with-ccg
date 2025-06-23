from gurobipy import *
import numpy as np

""" The input parameter """
facility_num = 3
customer_num = 3
fixed_cost = [400, 414, 326]
unit_capacity_cost = [18, 25, 20]
trans_cost = [[22, 33, 24],
              [33, 23, 30],
              [20, 25, 27]]
max_capacity = 800

demand_nominal = [206, 274, 220]
demand_var = [40, 40, 40]

""" build initial master problem """
""" Create variables """
master = Model('master problem')
x_master = {}
z = {}
y = {}
eta = master.addVar(lb=0, vtype=GRB.CONTINUOUS, name='eta')

for i in range(facility_num):
    y[i] = master.addVar(vtype=GRB.BINARY, name='y_%s' % (i))
    z[i] = master.addVar(lb=0, vtype=GRB.CONTINUOUS, name='z_%s' % (i))

""" Set objective """
obj = LinExpr()
for i in range(facility_num):
    obj.addTerms(fixed_cost[i], y[i])
    obj.addTerms(unit_capacity_cost[i], z[i])
obj.addTerms(1, eta)

master.setObjective(obj, GRB.MINIMIZE)

""" Add Constraints  """
# cons 1
for i in range(facility_num):
    master.addConstr(z[i] <= max_capacity * y[i])

""" Add initial value Constraints  """
# create new variables x
iter_cnt = 0
for i in range(facility_num):
    for j in range(customer_num):
        x_master[iter_cnt, i, j] = master.addVar(lb=0
                                                 , ub=GRB.INFINITY
                                                 , vtype=GRB.CONTINUOUS
                                                 , name='x_%s_%s_%s' % (iter_cnt, i, j))
# create new constraints
expr = LinExpr()
for i in range(facility_num):
    for j in range(customer_num):
        expr.addTerms(trans_cost[i][j], x_master[iter_cnt, i, j])
master.addConstr(eta >= expr)

expr = LinExpr()
for i in range(facility_num):
    expr.addTerms(1, z[i])
master.addConstr(expr >= 772)  # 206 + 274 + 220 + 40 * 1.8

""" solve the model and output """
master.optimize()
print('Obj = {}'.format(master.ObjVal))
print('-----  location ----- ')
for key in z.keys():
    print('facility : {}, location: {}, capacity: {}'.format(key, y[key].x, z[key].x))


def print_sub_sol(model, d, g, x):
    d_sol = {}
    if model.status != GRB.OPTIMAL:
        print('The problem is infeasible or unbounded!')
        print('Status: {}'.format(model.status))
        d_sol[0] = 0
        d_sol[1] = 0
        d_sol[2] = 0
    else:
        print('Obj(sub) : {:.2f}'.format(model.ObjVal), end='\t | ')
        for key in d.keys():
            d_sol[key] = d[key].x
    return d_sol


""" Column-and-constraint generation """

LB = -np.inf
UB = np.inf
iter_cnt = 0
max_iter = 30
eps = 0.001
Gap = np.inf

z_sol = {}
for key in z.keys():
    z_sol[key] = z[key].x

""" solve the master problem and update bound """
master.optimize()

""" 
 Update the Lower bound 
"""
LB = master.ObjVal
print('LB: {:.2f}'.format(LB))

''' create the subproblem '''
subProblem = Model('sub problem')
x = {}  # transportation decision variables in subproblem
d = {}  # true demand
g = {}  # uncertainty part: var part
pi = {}  # dual variable
theta = {}  # dual variable
v = {}  # aux var for capacity constraint
w = {}  # aux var for demand constraint
h = {}  # aux var for transportation constraint
big_M = 10000  # Big M value for linearization

# Create variables
for i in range(facility_num):
    pi[i] = subProblem.addVar(lb=-GRB.INFINITY, ub=0, vtype=GRB.CONTINUOUS, name='pi_%s' % i)
    v[i] = subProblem.addVar(vtype=GRB.BINARY, name='v_%s' % i)
    
for j in range(customer_num):
    w[j] = subProblem.addVar(vtype=GRB.BINARY, name='w_%s' % j)
    g[j] = subProblem.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name='g_%s' % j)
    theta[j] = subProblem.addVar(lb=-GRB.INFINITY, ub=0, vtype=GRB.CONTINUOUS, name='theta_%s' % j)
    d[j] = subProblem.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='d_%s' % j)
    
for i in range(facility_num):
    for j in range(customer_num):
        h[i, j] = subProblem.addVar(vtype=GRB.BINARY, name='h_%s_%s' % (i, j))
        x[i, j] = subProblem.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='x_%s_%s' % (i, j))

""" set objective """
sub_obj = LinExpr()
for i in range(facility_num):
    for j in range(customer_num):
        sub_obj.addTerms(trans_cost[i][j], x[i, j])
subProblem.setObjective(sub_obj, GRB.MAXIMIZE)

""" add constraints to subproblem """
# cons 1: Capacity constraints
for i in range(facility_num):
    expr = LinExpr()
    for j in range(customer_num):
        expr.addTerms(1, x[i, j])
    subProblem.addConstr(expr <= z_sol[i], name='sub_capacity_%s' % i)

# cons 2: Demand satisfaction
for j in range(customer_num):
    expr = LinExpr()
    for i in range(facility_num):
        expr.addTerms(1, x[i, j])
    subProblem.addConstr(expr >= d[j], name='sub_demand_%s' % j)

# cons 3: Dual constraint
for i in range(facility_num):
    for j in range(customer_num):
        subProblem.addConstr(pi[i] - theta[j] <= trans_cost[i][j], 
                            name='dual_constraint_%s_%s' % (i, j))

""" demand constraints """
for j in range(customer_num):
    subProblem.addConstr(d[j] == demand_nominal[j] + g[j] * demand_var[j], 
                        name='demand_expr_%s' % j)

subProblem.addConstr(g[0] + g[1] + g[2] <= 1.8, name='g_sum1')
subProblem.addConstr(g[0] + g[1] <= 1.2, name='g_sum2')

""" logic constraints for KKT conditions """
# Logic 1: Capacity constraints complementary slackness
for i in range(facility_num):
    # π_i >= -M * v_i
    subProblem.addConstr(pi[i] >= -big_M * v[i], name='logic1a_%s' % i)
    # z_i - sum_j x_ij <= M * (1 - v_i)
    subProblem.addConstr(z_sol[i] - sum(x[i, j] for j in range(customer_num)) <= big_M * (1 - v[i]), 
                        name='logic1b_%s' % i)

# Logic 2: Demand constraints complementary slackness
for j in range(customer_num):
    # θ_j >= -M * w_j
    subProblem.addConstr(theta[j] >= -big_M * w[j], name='logic2a_%s' % j)
    # sum_i x_ij - d_j <= M * (1 - w_j)
    subProblem.addConstr(sum(x[i, j] for i in range(facility_num)) - d[j] <= big_M * (1 - w[j]), 
                        name='logic2b_%s' % j)

# Logic 3: Transportation constraints complementary slackness
for i in range(facility_num):
    for j in range(customer_num):
        # x_ij <= M * h_{ij}
        subProblem.addConstr(x[i, j] <= big_M * h[i, j], name='logic3a_%s_%s' % (i, j))
        # c_ij - π_i + θ_j <= M * (1 - h_{ij})
        subProblem.addConstr(trans_cost[i][j] - pi[i] + theta[j] <= big_M * (1 - h[i, j]), 
                            name='logic3b_%s_%s' % (i, j))

subProblem.update()
subProblem.write('SP.lp')
subProblem.optimize()
d_sol = {}

print('\n\n\n *******            C&CG starts          *******  ')
print('\n **                Initial Solution             ** ')

d_sol = print_sub_sol(subProblem, d, g, x)

""" 
 Update the initial Upper bound 
"""
if subProblem.status == GRB.OPTIMAL:
    UB = min(UB, subProblem.ObjVal + master.ObjVal - eta.x)
print('UB (iter {}): {:.2f}'.format(iter_cnt, UB))

# close the outputflag
master.setParam('Outputflag', 0)
subProblem.setParam('Outputflag', 0)

"""
 Main loop of CCG algorithm 
"""
while (UB - LB > eps and iter_cnt <= max_iter):
    iter_cnt += 1
    print('\n iter : {} '.format(iter_cnt), end='\t | ')

    # Create new variables for this iteration
    for i in range(facility_num):
        for j in range(customer_num):
            x_master[iter_cnt, i, j] = master.addVar(
                lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, 
                name='x_%s_%s_%s' % (iter_cnt, i, j))

    # Get the worst-case demand from subproblem
    d_worst = [d[j].x for j in range(customer_num)]
    
    # If subproblem is feasible and bounded
    if subProblem.status == GRB.OPTIMAL:
        # Create new optimality cut: eta >= sum c_ij x_ij^l
        expr = LinExpr()
        for i in range(facility_num):
            for j in range(customer_num):
                expr.addTerms(trans_cost[i][j], x_master[iter_cnt, i, j])
        master.addConstr(eta >= expr, name='opt_cut_%s' % iter_cnt)

        # Add constraints for this scenario
        # Capacity constraints for this scenario
        for i in range(facility_num):
            expr = LinExpr()
            for j in range(customer_num):
                expr.addTerms(1, x_master[iter_cnt, i, j])
            master.addConstr(expr <= z[i], name='cap_scen_%s_%s' % (iter_cnt, i))
        
        # Demand constraints for this scenario
        for j in range(customer_num):
            expr = LinExpr()
            for i in range(facility_num):
                expr.addTerms(1, x_master[iter_cnt, i, j])
            master.addConstr(expr >= d_worst[j], name='dem_scen_%s_%s' % (iter_cnt, j))

        # Solve the master problem
        master.optimize()
        print('Obj(master): {:.2f}'.format(master.ObjVal), end='\t | ')
        
        """ Update the LB """
        LB = master.ObjVal
        print('LB: {:.2f}'.format(LB), end='\t | ')
        
        """ Update the subproblem """
        # Update z_sol from master solution
        for key in z.keys():
            z_sol[key] = z[key].x
        
        # Update capacity constraints in subproblem
        for i in range(facility_num):
            # Remove old constraint
            constr_name = 'sub_capacity_%s' % i
            constr = subProblem.getConstrByName(constr_name)
            if constr:
                subProblem.remove(constr)
            
            # Add new constraint with updated z_sol
            expr = LinExpr()
            for j in range(customer_num):
                expr.addTerms(1, x[i, j])
            subProblem.addConstr(expr <= z_sol[i], name=constr_name)
        
        # Re-optimize subproblem
        subProblem.optimize()
        d_sol = print_sub_sol(subProblem, d, g, x)
        
        """ Update the Upper bound """
        if subProblem.status == GRB.OPTIMAL:
            total_cost = sum(fixed_cost[i] * y[i].x for i in range(facility_num))
            total_cost += sum(unit_capacity_cost[i] * z[i].x for i in range(facility_num))
            total_cost += subProblem.ObjVal
            
            UB = min(UB, total_cost)
        
        print('UB: {:.2f}'.format(UB), end='\t | ')
        Gap = 100 * (UB - LB) / UB if UB != 0 else 100
        print('Gap: {:.2f}%'.format(Gap), end='\t')
        
    # If the subproblem is infeasible (shouldn't happen with relative complete recourse)
    else:
        print('Subproblem infeasible! Adding feasibility cut.')
        # Create feasibility cut (not needed in this case study)
        # For completeness, we add the scenario constraints
        for j in range(customer_num):
            expr = LinExpr()
            for i in range(facility_num):
                expr.addTerms(1, x_master[iter_cnt, i, j])
            master.addConstr(expr >= d_worst[j], name='feas_cut_%s_%s' % (iter_cnt, j))
        
        master.optimize()
        print('Obj(master): {:.2f}'.format(master.ObjVal))
        LB = master.ObjVal
        print('LB: {:.2f}'.format(LB))
        
        # Update subproblem with new z
        for key in z.keys():
            z_sol[key] = z[key].x
        
        for i in range(facility_num):
            constr_name = 'sub_capacity_%s' % i
            constr = subProblem.getConstrByName(constr_name)
            if constr:
                subProblem.remove(constr)
            
            expr = LinExpr()
            for j in range(customer_num):
                expr.addTerms(1, x[i, j])
            subProblem.addConstr(expr <= z_sol[i], name=constr_name)
        
        subProblem.optimize()
        d_sol = print_sub_sol(subProblem, d, g, x)
        
        if subProblem.status == GRB.OPTIMAL:
            total_cost = sum(fixed_cost[i] * y[i].x for i in range(facility_num))
            total_cost += sum(unit_capacity_cost[i] * z[i].x for i in range(facility_num))
            total_cost += subProblem.ObjVal
            UB = min(UB, total_cost)
        
        print('UB: {:.2f}'.format(UB))
        Gap = 100 * (UB - LB) / UB if UB != 0 else 100
        print('Gap: {:.2f}%'.format(Gap))

# Final output
master.write('finalMP.lp')
print('\n\nOptimal solution found!')
print('Optimal Obj: {:.2f}'.format(master.ObjVal))
print(' ** Final Gap: {:.2f}% ** '.format(Gap))
print('\n ** Facility Solution ** ')
for i in range(facility_num):
    print('Facility {}: Open={}, Capacity={:.2f}'.format(i, int(y[i].x), z[i].x))

print('\n ** Worst-case Demand ** ')
for j in range(customer_num):
    print('Customer {}: Nominal={}, Worst-case={:.2f}, Perturbation={:.2f}'.format(
        j, demand_nominal[j], d[j].x, g[j].x))

print('\n ** Transportation Solution ** ')
total_trans_cost = 0
for i in range(facility_num):
    for j in range(customer_num):
        if x[i, j].x > 1e-4:  # Only print non-zero flows
            cost = trans_cost[i][j] * x[i, j].x
            total_trans_cost += cost
            print('From facility {} to customer {}: {:.2f} units, Cost={:.2f}'.format(
                i, j, x[i, j].x, cost))

print('\nTotal transportation cost: {:.2f}'.format(total_trans_cost))
print('Total fixed cost: {:.2f}'.format(
    sum(fixed_cost[i] * y[i].x for i in range(facility_num))))
print('Total capacity cost: {:.2f}'.format(
    sum(unit_capacity_cost[i] * z[i].x for i in range(facility_num))))
print('Total cost: {:.2f}'.format(total_trans_cost+sum(fixed_cost[i] * y[i].x for i in range(facility_num))+sum(unit_capacity_cost[i] * z[i].x for i in range(facility_num))))