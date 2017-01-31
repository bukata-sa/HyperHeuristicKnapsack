from random import shuffle, randint

from knapsack.problem import *


def add_random(current, validation=True, **kwargs):
    indexes = list(range(len(kwargs["weights"])))
    shuffle(indexes)
    while len(indexes) > 0:
        result = list(current)
        candidate_index = indexes.pop()

        if result[candidate_index] != 0:
            continue

        result[candidate_index] = 1
        if not validation or validate(result, kwargs["costs"], kwargs["weights"], kwargs["size"]):
            return result
    return current


def add_best(current, validation=True, **kwargs):
    priorities = list(map(lambda x: x[0] / float(x[1]), zip(kwargs["costs"], kwargs["weights"])))
    index_priorities = list(zip(range(len(kwargs["weights"])), priorities))
    index_priorities.sort(key=lambda x: x[1])

    while len(index_priorities) > 0:
        result = list(current)
        candidate_index, _ = index_priorities.pop()

        if result[candidate_index] != 0:
            continue

        result[candidate_index] = 1
        if not validation or validate(result, kwargs["costs"], kwargs["weights"], kwargs["size"]):
            return result
    return current


def remove_random(current, **kwargs):
    indexes = [i for i in range(len(current)) if current[i] == 1]

    if len(indexes) == 0:
        return current

    delete_index = indexes[randint(0, len(indexes) - 1)]
    current[delete_index] = 0
    return current


def remove_worst(current, **kwargs):
    indexes = [i for i in range(len(current)) if current[i] == 1]
    if len(indexes) == 0:
        return current
    included_weights = [weight for i, weight in enumerate(kwargs["weights"]) if current[i] == 1]
    included_costs = [cost for i, cost in enumerate(kwargs["costs"]) if current[i] == 1]
    priorities = list(map(lambda x: x[0] / float(x[1]), zip(included_costs, included_weights)))
    index_priorities = list(zip(indexes, priorities))
    min_priority_index = min(index_priorities, key=lambda x: x[1])[0]
    current[min_priority_index] = 0
    return current

# size = 165
# weights = [23, 31, 29, 44, 53, 38, 63, 85, 89, 82]
# costs = [92, 57, 49, 68, 60, 43, 67, 84, 87, 72]
# current = [1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
# print(add_random(list(current), size=size, weights=weights, costs=costs))
# print(add_best(list(current), size=size, weights=weights, costs=costs))
# print(remove_random(list(current), size=size, weights=weights, costs=costs))
# print(remove_worst(list(current), size=size, weights=weights, costs=costs))
