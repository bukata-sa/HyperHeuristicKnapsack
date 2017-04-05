import numpy as np

from knapsack.hyper.multi import genetic

optimal_cost = 3800
costs = [100, 600, 1200, 2400, 500, 2000]
weights = [[8, 12, 13, 64, 22, 41],
           [8, 12, 13, 75, 22, 41],
           [3, 6, 4, 18, 6, 4],
           [5, 10, 8, 32, 6, 12],
           [5, 13, 8, 42, 6, 20],
           [5, 13, 8, 48, 6, 20],
           [0, 0, 0, 0, 8, 0],
           [3, 0, 4, 0, 8, 0],
           [3, 2, 4, 0, 8, 4],
           [3, 2, 4, 8, 8, 4]]
sizes = [80, 96, 20, 36, 44, 48, 10, 18, 22, 24]
start = np.zeros((len(sizes), len(costs)))

if __name__ == '__main__':
    result = 0
    cumulative_gap = 0
    for i in range(1, 50):
        optimal_funcs = genetic.minimize(50, weights=weights, costs=costs, sizes=sizes, included=start)
        current = genetic.fitness_hyper_ksp(optimal_funcs, weights=weights, costs=costs, sizes=sizes, included=start)
        print("Current:\t" + str(3800 - current))
        result += 13549094 - current
        current_gap = 100 * (optimal_cost - current) / optimal_cost
        print("Normalized:\t" + str(current_gap))
        cumulative_gap += current_gap
        print("Normed cum:\t" + str(cumulative_gap / i))
