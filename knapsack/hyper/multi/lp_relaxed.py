import operator

import numpy as np
from cvxpy import *

from knapsack.hyper.multi import problem


def ksp_solve_lp_relaxed_convex(costs, weights, sizes):
    x = Variable(len(sizes), len(costs))
    weights_param = Parameter(rows=len(sizes), cols=len(costs))
    weights_param.value = np.asarray(weights)
    costs_param = Parameter(len(costs))
    costs_param.value = costs

    constr = [diag(x * weights_param.T) < sizes, sum_entries(x, axis=0).T <= [1] * len(costs), 0 < x, x < 1]
    objective = Maximize(sum_entries(x * costs_param))

    solution = Problem(objective, constr).solve()
    print(x.value)
    print(problem.validate(x.value, costs, weights, sizes))
    print(problem.solve(x.value, costs, weights, sizes))
    return solution


def ksp_solve_lp_relaxed_greedy(costs, weights, sizes):
    costs = np.asarray(costs)
    weights = np.asarray(weights)
    sizes = np.asarray(sizes)

    included = []
    for ksp_index, weight in enumerate(weights):
        current_characteristics = list(sorted(enumerate(costs / weight), key=operator.itemgetter(1), reverse=True))
        median_index = 0
        while np.sum(weight[:median_index + 1]) < sizes[ksp_index]:
            median_index += 1
        median = current_characteristics[median_index]
        current_included = [0] * len(costs)
        rest = (sizes[ksp_index] - np.sum(weight[costs / weight > median[1]])) / weight[median_index]
        for item_index, characteristic in current_characteristics:
            if characteristic > median[1]:
                current_included[item_index] = 1
            elif characteristic == median[1]:
                current_included[item_index] = rest
            else:
                current_included[item_index] = 0
        included.append(current_included)

    included = np.asarray(included)
    for item_index in range(len(costs)):
        if np.sum(included, axis=0)[item_index] <= 1:
            continue
        vertical_characteristics = list(sorted(enumerate(costs[item_index] / weights[:, item_index]),
                                               key=operator.itemgetter(1)))
        new_included = [0]*len(sizes)
        while True:
            ksp_index, _ = vertical_characteristics.pop()
            if np.sum(new_included) + included[ksp_index][item_index] <= 1:
                new_included[ksp_index] =+ included[ksp_index][item_index]
            else:
                break
        included[:, item_index] = new_included
    return problem.solve(included, costs, weights, sizes)

if __name__ == '__main__':
    optimal_selection = [[1, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0, 1, 0, 0, 0]]
    costs = [92, 57, 49, 68, 60, 43, 67, 84, 87, 72]
    weights = [[23, 31, 29, 44, 53, 38, 63, 85, 89, 82],
               [23, 31, 29, 44, 53, 38, 63, 85, 89, 82]]
    sizes = [70, 127]
    optimal_cost = problem.solve(optimal_selection, costs, weights, sizes)
    print(ksp_solve_lp_relaxed_greedy(costs, weights, sizes))
