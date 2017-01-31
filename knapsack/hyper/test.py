import knapsack.hyper.genetic as hyper_gene

size = 165
weights = [23, 31, 29, 44, 53, 38, 63, 85, 89, 82]
costs = [92, 57, 49, 68, 60, 43, 67, 84, 87, 72]
start = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
optimal = [1, 1, 1, 1, 0, 1, 0, 0, 0, 0]

optimal_funcs = hyper_gene.minimize(10, weights=weights, costs=costs, size=size, included=start)
print(optimal_funcs)
result = hyper_gene.fitness_hyper_ksp(optimal_funcs, weights=weights, costs=costs, size=size, included=start)
print(result)
