import numpy as np

from knapsack.hyper.multi import genetic
from knapsack.hyper.multi import lp_relaxed as lp

optimal_selection = [[1, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0, 1, 0, 0, 0]]
costs = [92, 57, 49, 68, 60, 43, 67, 84, 87, 72]
weights = [[23, 31, 29, 44, 53, 38, 63, 85, 89, 82],
           [23, 31, 29, 44, 53, 38, 63, 85, 89, 82]]
sizes = [70, 127]
# optimal_cost = problem.solve(optimal_selection, costs, weights, sizes)
optimal_cost = lp.ksp_solve_lp_relaxed_convex(costs, weights, sizes)
# TODO generate initial state using LP-relaxed solution
start = np.zeros((len(sizes), len(costs)))

if __name__ == '__main__':
    result = 0
    cumulative_gap = 0
    for i in range(1, 50):
        optimal_funcs = genetic.minimize(50, weights=weights, costs=costs, sizes=sizes, included=start)
        current = genetic.fitness_hyper_ksp(optimal_funcs, weights=weights, costs=costs, sizes=sizes, included=start)
        print("Current:\t" + str(optimal_cost - current))
        result += optimal_cost - current
        current_gap = 100 * (optimal_cost - current) / optimal_cost
        print("Normalized:\t" + str(current_gap))
        cumulative_gap += current_gap
        print("Normed cum:\t" + str(cumulative_gap / i))
