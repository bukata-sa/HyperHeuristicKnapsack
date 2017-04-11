import operator

from knapsack.hyper.single.problem import *


def get_all_single_heuristics():
    return [add_lightest, add_heaviest, add_least_cost, add_most_cost, add_best]


def update_ksp_extreme_property(current, is_add, is_max, properties, tabooed_indexes=None, **kwargs):
    index_property_list = [element for element in enumerate(properties)]
    indexes_property_sorted = [index[0] for index in
                               sorted(index_property_list, key=operator.itemgetter(1), reverse=not is_max)]
    while len(indexes_property_sorted) > 0:
        result = list(current)
        candidate_index = indexes_property_sorted.pop()

        if result[candidate_index] == is_add:
            continue

        result[candidate_index] = int(is_add)
        if candidate_index not in tabooed_indexes and (not is_add or validate(result, **kwargs)):
            return result, candidate_index
    return current, -1


def add_lightest(current, tabooed_indexes=None, **kwargs):
    return update_ksp_extreme_property(current, True, False, kwargs["weights"], tabooed_indexes=tabooed_indexes,
                                       **kwargs)


def add_heaviest(current, tabooed_indexes=None, **kwargs):
    return update_ksp_extreme_property(current, True, True, kwargs["weights"], tabooed_indexes=tabooed_indexes,
                                       **kwargs)


def remove_lightest(current, tabooed_indexes=None, **kwargs):
    return update_ksp_extreme_property(current, False, False, kwargs["weights"], tabooed_indexes=tabooed_indexes,
                                       **kwargs)


def remove_heaviest(current, tabooed_indexes=None, **kwargs):
    return update_ksp_extreme_property(current, False, True, kwargs["weights"], tabooed_indexes=tabooed_indexes,
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
    priorities = weight_cost_priority_list(kwargs["costs"], kwargs["weights"])
    return update_ksp_extreme_property(current, True, True, priorities, tabooed_indexes=tabooed_indexes, **kwargs)


def remove_worst(current, tabooed_indexes=None, **kwargs):
    priorities = weight_cost_priority_list(kwargs["costs"], kwargs["weights"])
    return update_ksp_extreme_property(current, False, False, priorities, tabooed_indexes=tabooed_indexes, **kwargs)
