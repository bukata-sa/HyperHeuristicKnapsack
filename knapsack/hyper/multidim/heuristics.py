import operator

from knapsack.hyper.multidim.problem import *

cached_heuristics = None


def get_heuristics():
    global cached_heuristics
    if cached_heuristics is not None:
        return cached_heuristics
    # probability to select add heuristic is 0.9
    add_heurs = [add_lightest, add_heaviest, add_least_cost, add_most_cost, add_best, add_heaviest_item_heaviest_condition, add_lightest_item_heaviest_condition, add_heaviest_item_lightest_condition,
                 add_lightest_item_lightest_condition]
    add_heurs = list(zip(add_heurs, [0.8 / len(add_heurs)] * len(add_heurs)))
    # probability to select remove heuristic is 0.1
    remove_heurs = [remove_lightest, remove_heaviest, remove_least_cost, remove_most_cost, remove_worst, remove_heaviest_item_heaviest_condition, remove_lightest_item_heaviest_condition,
                    remove_heaviest_item_lightest_condition, remove_lightest_item_lightest_condition]
    remove_heurs = list(zip(remove_heurs, [0.3 / len(remove_heurs)] * len(remove_heurs)))
    cached_heuristics = add_heurs + remove_heurs
    return cached_heuristics


def update_ksp_extreme_property(current, is_add, is_max, properties, tabooed_indexes=None, **kwargs):
    index_property_list = (element for element in enumerate(properties))
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
    properties = calculate_relative_weights(kwargs["weights"], kwargs["sizes"])
    return update_ksp_extreme_property(current, True, False, properties, tabooed_indexes=tabooed_indexes,
                                       **kwargs)


def add_heaviest(current, tabooed_indexes=None, **kwargs):
    properties = calculate_relative_weights(kwargs["weights"], kwargs["sizes"])
    return update_ksp_extreme_property(current, True, True, properties, tabooed_indexes=tabooed_indexes,
                                       **kwargs)


def remove_lightest(current, tabooed_indexes=None, **kwargs):
    properties = calculate_relative_weights(kwargs["weights"], kwargs["sizes"])
    return update_ksp_extreme_property(current, False, False, properties, tabooed_indexes=tabooed_indexes,
                                       **kwargs)


def remove_heaviest(current, tabooed_indexes=None, **kwargs):
    properties = calculate_relative_weights(kwargs["weights"], kwargs["sizes"])
    return update_ksp_extreme_property(current, False, True, properties, tabooed_indexes=tabooed_indexes,
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
    return map(lambda x: x[0] / float(x[1]) if x[1] != 0 else np.inf, zip(costs, weights))


def add_best(current, tabooed_indexes=None, **kwargs):
    properties = calculate_relative_weights(kwargs["weights"], kwargs["sizes"])
    priorities = weight_cost_priority_list(kwargs["costs"], properties)
    return update_ksp_extreme_property(current, True, True, priorities, tabooed_indexes=tabooed_indexes, **kwargs)


def remove_worst(current, tabooed_indexes=None, **kwargs):
    properties = calculate_relative_weights(kwargs["weights"], kwargs["sizes"])
    priorities = weight_cost_priority_list(kwargs["costs"], properties)
    return update_ksp_extreme_property(current, False, False, priorities, tabooed_indexes=tabooed_indexes, **kwargs)


def add_heaviest_item_heaviest_condition(current, tabooed_indexes=None, **kwargs):
    weights = np.asarray(kwargs["weights"])
    sizes = np.asarray(kwargs["sizes"])
    properties = np.sum(weights, axis=1) / sizes
    property_index = np.argmax(properties)
    return update_ksp_extreme_property(current, True, True, weights[property_index], tabooed_indexes=tabooed_indexes,
                                       **kwargs)


def remove_heaviest_item_heaviest_condition(current, tabooed_indexes=None, **kwargs):
    weights = np.asarray(kwargs["weights"])
    sizes = np.asarray(kwargs["sizes"])
    properties = np.sum(weights, axis=1) / sizes
    property_index = np.argmax(properties)
    return update_ksp_extreme_property(current, False, True, weights[property_index], tabooed_indexes=tabooed_indexes,
                                       **kwargs)


def add_lightest_item_heaviest_condition(current, tabooed_indexes=None, **kwargs):
    weights = np.asarray(kwargs["weights"])
    sizes = np.asarray(kwargs["sizes"])
    properties = np.sum(weights, axis=1) / sizes
    property_index = np.argmax(properties)
    return update_ksp_extreme_property(current, True, False, weights[property_index], tabooed_indexes=tabooed_indexes,
                                       **kwargs)


def remove_lightest_item_heaviest_condition(current, tabooed_indexes=None, **kwargs):
    weights = np.asarray(kwargs["weights"])
    sizes = np.asarray(kwargs["sizes"])
    properties = np.sum(weights, axis=1) / sizes
    property_index = np.argmax(properties)
    return update_ksp_extreme_property(current, False, False, weights[property_index], tabooed_indexes=tabooed_indexes,
                                       **kwargs)


def add_lightest_item_lightest_condition(current, tabooed_indexes=None, **kwargs):
    weights = np.asarray(kwargs["weights"])
    sizes = np.asarray(kwargs["sizes"])
    properties = np.sum(weights, axis=1) / sizes
    property_index = np.argmin(properties)
    return update_ksp_extreme_property(current, True, False, weights[property_index], tabooed_indexes=tabooed_indexes,
                                       **kwargs)


def remove_lightest_item_lightest_condition(current, tabooed_indexes=None, **kwargs):
    weights = np.asarray(kwargs["weights"])
    sizes = np.asarray(kwargs["sizes"])
    properties = np.sum(weights, axis=1) / sizes
    property_index = np.argmin(properties)
    return update_ksp_extreme_property(current, False, False, weights[property_index], tabooed_indexes=tabooed_indexes,
                                       **kwargs)


def add_heaviest_item_lightest_condition(current, tabooed_indexes=None, **kwargs):
    weights = np.asarray(kwargs["weights"])
    sizes = np.asarray(kwargs["sizes"])
    properties = np.sum(weights, axis=1) / sizes
    property_index = np.argmin(properties)
    return update_ksp_extreme_property(current, True, True, weights[property_index], tabooed_indexes=tabooed_indexes,
                                       **kwargs)


def remove_heaviest_item_lightest_condition(current, tabooed_indexes=None, **kwargs):
    weights = np.asarray(kwargs["weights"])
    sizes = np.asarray(kwargs["sizes"])
    properties = np.sum(weights, axis=1) / sizes
    property_index = np.argmin(properties)
    return update_ksp_extreme_property(current, False, True, weights[property_index], tabooed_indexes=tabooed_indexes,
                                       **kwargs)


def calculate_relative_weights(weights, sizes):
    weights = np.asarray(weights)
    sizes = np.asarray(sizes)
    properties = zip(sizes, weights)
    properties = list(map(lambda x: x[0] / x[1], properties))
    properties = np.sum(properties, axis=0)
    return properties
