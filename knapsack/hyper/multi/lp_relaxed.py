import numpy as np
from cvxpy import *

from knapsack.hyper.multi import problem


def ksp_solve_lp_relaxed(costs, weights, sizes):
    x = Variable(len(sizes), len(costs))
    weights_param = Parameter(rows=len(sizes), cols=len(costs))
    weights_param.value = np.asarray(weights)

    constr1 = [diag(x * weights_param.T) < sizes]
    constr2 = [sum_entries(x, axis=0).T <= [1] * len(costs)]
    constr3 = [0 <= x, x <= 1]
    objective = Maximize(sum_entries(np.dot(optimal_selection, costs)))

    return Problem(objective, constr1 + constr2 + constr3).solve()


if __name__ == '__main__':
    optimal_selection = [[1, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0, 1, 0, 0, 0]]
    costs = [92, 57, 49, 68, 60, 43, 67, 84, 87, 72]
    weights = [[23, 31, 29, 44, 53, 38, 63, 85, 89, 82],
               [23, 31, 29, 44, 53, 38, 63, 85, 89, 82]]
    sizes = [70, 127]
    optimal_cost = problem.solve(optimal_selection, costs, weights, sizes)
    print(ksp_solve_lp_relaxed(costs, weights, sizes))
