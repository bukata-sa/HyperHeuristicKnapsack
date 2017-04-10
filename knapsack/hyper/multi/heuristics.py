import operator

import numpy as np

from knapsack.hyper.single import heurs1knpsck as single


def get_single_heurs_for_knapsack_with_least_cost():
    single_heuristics = single.get_all_single_heuristics()
    result = []
    ksp_choice_functions = [least_weight_overall, most_weight_overall, least_cost_overall, most_cost_overall]
    for single_heuristic in single_heuristics:
        for ksp_choice_function in ksp_choice_functions:
            def single_knapsack_with_exteme_property(current, tabooed_indexes,
                                                     ksp_choice_function=ksp_choice_function,
                                                     my_single_heuristic=single_heuristic, **kwargs):
                current = list(current)
                indexed_properties = ksp_choice_function(current, **kwargs)
                modified_index = -1
                while modified_index == -1 and len(indexed_properties) > 0:
                    ksp_index = indexed_properties.pop()[0]
                    single_ksp_kwargs = {"costs": kwargs["costs"], "weights": kwargs["weights"][ksp_index],
                                         "size": kwargs["sizes"][ksp_index]}
                    # TODO should multi include constraint be here?
                    multi_include_constraint = build_multi_include_constraint(current, ksp_index)
                    tabooed_indexes = list(set(tabooed_indexes).union(set(multi_include_constraint)))
                    new_included, modified_index = my_single_heuristic(current[ksp_index], tabooed_indexes,
                                                                       **single_ksp_kwargs)
                    current[ksp_index] = new_included
                return current, modified_index

            result.append(single_knapsack_with_exteme_property)
    return result


def least_weight_overall(included, **kwargs):
    weights = np.asarray(kwargs["weights"])
    indexed_properties = np.sum(np.asarray(included) * weights, axis=1)
    indexed_properties = enumerate(indexed_properties)
    indexed_properties = list(sorted(indexed_properties, key=operator.itemgetter(1), reverse=True))
    return indexed_properties


def most_weight_overall(included, **kwargs):
    weights = np.asarray(kwargs["weights"])
    indexed_properties = np.sum(np.asarray(included) * weights, axis=1)
    indexed_properties = enumerate(indexed_properties)
    indexed_properties = list(sorted(indexed_properties, key=operator.itemgetter(1)))
    return indexed_properties


def least_cost_overall(included, **kwargs):
    costs = np.asarray(kwargs["costs"])
    indexed_properties = np.sum(np.asarray(included) * costs, axis=1)
    indexed_properties = enumerate(indexed_properties)
    indexed_properties = list(sorted(indexed_properties, key=operator.itemgetter(1), reverse=True))
    return indexed_properties


def most_cost_overall(included, **kwargs):
    costs = np.asarray(kwargs["costs"])
    indexed_properties = np.sum(np.asarray(included) * costs, axis=1)
    indexed_properties = enumerate(indexed_properties)
    indexed_properties = list(sorted(indexed_properties, key=operator.itemgetter(1)))
    return indexed_properties


def most_efficiency_overall(included, **kwargs):
    pass


def least_efficiency_overall(included, **kwargs):
    pass


def potentially_most_cost(included, **kwargs):
    pass


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
    start = np.zeros((len(sizes), len(costs))).tolist()
    for heur in heurs:
        print(heur(start, [], costs=costs, weights=weights, sizes=sizes))
