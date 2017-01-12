import numpy as np

import knapsack.annealing as ann
import knapsack.genetic as gen

success = 0

size = 165
weights = [23, 31, 29, 44, 53, 38, 63, 85, 89, 82]
profits = [92, 57, 49, 68, 60, 43, 67, 84, 87, 72]
optimal = [1, 1, 1, 1, 0, 1, 0, 0, 0, 0]

ann_minimize = ann.minimize([False, False, False, False, False, False, False, False, False, False], 1, 10000,
                            costs=profits, weights=weights, size=size)
print(ann_minimize)
print(np.sum(np.array(ann_minimize) * np.array(profits)))
gen_minimize = gen.minimize(10, costs=profits, weights=weights, size=size)
print(np.sum(np.array(gen_minimize) * np.array(profits)))
print(gen_minimize)
