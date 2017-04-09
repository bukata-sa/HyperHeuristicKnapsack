import numpy as np
from cvxpy import *

from knapsack.hyper.multi import problem


def ksp_solve_lp_relaxed(costs, weights, sizes):
    x = Variable(len(sizes), len(costs))
    weights_param = Parameter(rows=len(sizes), cols=len(costs))
    weights_param.value = np.asarray(weights)
    costs_param = Parameter(len(costs))
    costs_param.value = costs

    constr = [diag(x * weights_param.T) < sizes, sum_entries(x, axis=0).T <= [1] * len(costs), 0 <= x, x <= 1]
    objective = Maximize(sum_entries(x * costs_param))

    solution = Problem(objective, constr).solve()
    print(x.value)
    print(problem.validate(x.value, costs, weights, sizes))
    print(problem.solve(x.value, costs, weights, sizes))
    return solution


if __name__ == '__main__':
    optimal_selection = [[1, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0, 1, 0, 0, 0]]
    costs = [92, 57, 49, 68, 60, 43, 67, 84, 87, 72]
    weights = [[23, 31, 29, 44, 53, 38, 63, 85, 89, 82],
               [23, 31, 29, 44, 53, 38, 63, 85, 89, 82]]
    sizes = [70, 127]
    optimal_cost = problem.solve(optimal_selection, costs, weights, sizes)
    print(ksp_solve_lp_relaxed(costs, weights, sizes))
