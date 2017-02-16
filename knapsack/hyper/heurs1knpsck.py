import operator

from knapsack.problem import *


def diffed_index(state1, state2):
    subtract = np.subtract(state1, state2)
    diffed = np.where(subtract != 0)
    result = int(diffed[0]) if len(diffed[0]) > 0 else -1
    return result


# def tabu_operation_chain(current, **kwargs):
#     success = False
#     operation_list = [add_lightest, add_heaviest, add_least_cost, add_most_cost, add_best, remove_lightest,
#                       remove_heaviest, remove_least_cost, remove_most_cost, remove_worst]
#     add_operation = add_lightest
#     result = add_operation(list(current), validation=False, **kwargs)
#     taboed_item = diffed_index(result, current)
#     taboed_weights = list(kwargs["weights"])
#     taboed_weights[taboed_item] = 0
#     while not success:
#         next_operation = operation_list[randint(0, len(operation_list) - 1)]
#         state_candidate = next_operation(result, costs=kwargs["costs"], weights=taboed_weights, size=kwargs["size"])
#         if taboed_item != diffed_index(result, state_candidate):
#             result = state_candidate
#             success = True
#     result[taboed_item] = 0
#     return result


def update_ksp_extreme_property(current, is_add, is_max, properties, validation=True, **kwargs):
    index_property_list = [element for element in enumerate(properties)]
    indexes_property_sorted = [index[0] for index in
                               sorted(index_property_list, key=operator.itemgetter(1), reverse=not is_max)]
    while len(indexes_property_sorted) > 0:
        result = list(current)
        candidate_index = indexes_property_sorted.pop()

        if result[candidate_index] == is_add:
            continue

        result[candidate_index] = int(is_add)
        if not validation or not is_add or validate(result, **kwargs):
            return result
    return current


def add_lightest(current, validation=True, **kwargs):
    return update_ksp_extreme_property(current, True, False, kwargs["weights"], validation=validation, **kwargs)


def add_heaviest(current, validation=True, **kwargs):
    return update_ksp_extreme_property(current, True, True, kwargs["weights"], validation=validation, **kwargs)


def remove_lightest(current, validation=True, **kwargs):
    return update_ksp_extreme_property(current, False, False, kwargs["weights"], validation=validation, **kwargs)


def remove_heaviest(current, validation=True, **kwargs):
    return update_ksp_extreme_property(current, False, True, kwargs["weights"], validation=validation, **kwargs)


def add_least_cost(current, validation=True, **kwargs):
    return update_ksp_extreme_property(current, True, False, kwargs["costs"], validation=validation, **kwargs)


def add_most_cost(current, validation=True, **kwargs):
    return update_ksp_extreme_property(current, True, True, kwargs["costs"], validation=validation, **kwargs)


def remove_least_cost(current, validation=True, **kwargs):
    return update_ksp_extreme_property(current, False, False, kwargs["costs"], validation=validation, **kwargs)


def remove_most_cost(current, validation=True, **kwargs):
    return update_ksp_extreme_property(current, False, True, kwargs["costs"], validation=validation, **kwargs)


def weight_cost_priority_list(costs, weights):
    return list(map(lambda x: x[0] / float(x[1]) if x[1] != 0 else np.inf, zip(costs, weights)))


def add_best(current, validation=True, **kwargs):
    priorities = weight_cost_priority_list(kwargs["costs"], kwargs["weights"])
    return update_ksp_extreme_property(current, True, True, priorities, validation=validation, **kwargs)


def remove_worst(current, validation=True, **kwargs):
    priorities = weight_cost_priority_list(kwargs["costs"], kwargs["weights"])
    return update_ksp_extreme_property(current, False, False, priorities, validation=validation, **kwargs)
