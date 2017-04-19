import operator

from knapsack.hyper.multidim.problem import *

cached_heuristics = None


def get_heuristics():
    global cached_heuristics
    if cached_heuristics is not None:
        return cached_heuristics
    # probability to select add heuristic is 0.9
    add_heurs = [add_lightest, add_heaviest, add_least_cost, add_most_cost, add_best]
    add_heurs = list(zip(add_heurs, [0.9 / len(add_heurs)] * len(add_heurs)))
    # probability to select remove heuristic is 0.1
    remove_heurs = [remove_lightest, remove_heaviest, remove_least_cost, remove_most_cost, remove_worst]
    remove_heurs = list(zip(remove_heurs, [0.1 / len(remove_heurs)] * len(remove_heurs)))
    cached_heuristics = add_heurs + remove_heurs
    return cached_heuristics


def update_ksp_extreme_property(current, is_add, is_max, properties, tabooed_indexes=None, **kwargs):
    index_property_list = [element for element in enumerate(properties)]
    indexes_property_sorted = [index[0] for index in
                               sorted(index_property_list, key=operator.itemgetter(1), reverse=not is_max)]
    while len(indexes_property_sorted) > 0:
        result = np.asarray(list(current))
        candidate_index = indexes_property_sorted.pop()

        if result[candidate_index] == is_add:
            continue

        result[candidate_index] = 1
        if candidate_index not in tabooed_indexes and (not is_add or validate(result, **kwargs)):
            return result, candidate_index
    return current, -1


def add_lightest(current, tabooed_indexes=None, **kwargs):
    return update_ksp_extreme_property(current, True, False, kwargs["weights"][0], tabooed_indexes=tabooed_indexes,
                                       **kwargs)


def add_heaviest(current, tabooed_indexes=None, **kwargs):
    return update_ksp_extreme_property(current, True, True, kwargs["weights"][0], tabooed_indexes=tabooed_indexes,
                                       **kwargs)


def remove_lightest(current, tabooed_indexes=None, **kwargs):
    return update_ksp_extreme_property(current, False, False, kwargs["weights"][0], tabooed_indexes=tabooed_indexes,
                                       **kwargs)


def remove_heaviest(current, tabooed_indexes=None, **kwargs):
    return update_ksp_extreme_property(current, False, True, kwargs["weights"][0], tabooed_indexes=tabooed_indexes,
                                       **kwargs)


def add_least_cost(current, tabooed_indexes=None, **kwargs):
    return update_ksp_extreme_property(current, True, False, kwargs["costs"], tabooed_indexes=tabooed_indexes, **kwargs)


def add_most_cost(current, tabooed_indexes=None, **kwargs):
    return update_ksp_extreme_property(current, True, True, kwargs["costs"], tabooed_indexes=tabooed_indexes, **kwargs)


def remove_least_cost(current, tabooed_indexes=None, **kwargs):
    return update_ksp_extreme_property(current, False, False, kwargs["costs"], tabooed_indexes=tabooed_indexes,
                                       **kwargs)


def remove_most_cost(current, tabooed_indexes=None, **kwargs):
    return update_ksp_extreme_property(current, False, True, kwargs["costs"], tabooed_indexes=tabooed_indexes, **kwargs)


def weight_cost_priority_list(costs, weights):
    return list(map(lambda x: x[0] / float(x[1]) if x[1] != 0 else np.inf, zip(costs, weights)))


def add_best(current, tabooed_indexes=None, **kwargs):
    priorities = weight_cost_priority_list(kwargs["costs"], kwargs["weights"][0])
    return update_ksp_extreme_property(current, True, True, priorities, tabooed_indexes=tabooed_indexes, **kwargs)


def remove_worst(current, tabooed_indexes=None, **kwargs):
    priorities = weight_cost_priority_list(kwargs["costs"], kwargs["weights"][0])
    return update_ksp_extreme_property(current, False, False, priorities, tabooed_indexes=tabooed_indexes, **kwargs)
