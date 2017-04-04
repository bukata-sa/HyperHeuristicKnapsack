import operator

import numpy as np

from knapsack.hyper.single import heurs1knpsck as single


def get_single_heurs_for_knapsack_with_least_cost():
    single_heuristics = single.get_all_single_heuristics()
    result = []
    for single_heuristic in single_heuristics:
        def knapsack_with_least_cost_heuristic(current, tabooed_indexes, my_single_heuristic=single_heuristic,
                                               **kwargs):
            weights = np.asarray(kwargs["weights"])
            indexed_weights = np.sum(np.asarray(current) * weights, axis=1)
            indexed_weights = enumerate(indexed_weights)
            indexed_weights = list(sorted(indexed_weights, key=operator.itemgetter(1)))
            modified_index = -1
            while modified_index == -1 and len(indexed_weights) > 0:
                ksp_index = indexed_weights.pop()[0]
                single_ksp_kwargs = {"costs": kwargs["costs"], "weights": kwargs["weights"][ksp_index],
                                     "size": kwargs["sizes"][ksp_index]}
                multi_include_constraint = build_multi_include_constraint(current, ksp_index)
                tabooed_indexes = list(set(tabooed_indexes).union(set(multi_include_constraint)))
                new_included, modified_index = my_single_heuristic(current[ksp_index], tabooed_indexes,
                                                                   **single_ksp_kwargs)
                current[ksp_index] = new_included
            return current, modified_index

        result.append(knapsack_with_least_cost_heuristic)
    return result


def build_multi_include_constraint(current, ksp_index):
    tabu = []
    column_sums = np.sum(current, axis=0)
    for i in range(len(current[ksp_index])):
        if current[ksp_index][i] == 1:
            continue
        if column_sums[i] == 1:
            tabu.append(i)
    return tabu


if __name__ == '__main__':
    heurs = get_single_heurs_for_knapsack_with_least_cost()

    costs = [8, 12, 13, 64, 22, 41]
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
    start = np.zeros((len(sizes), len(costs))).tolist()
    for heur in heurs:
        print(heur(start, [], costs=costs, weights=weights, sizes=sizes))
