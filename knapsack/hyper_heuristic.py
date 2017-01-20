import random as rnd

import knapsack.problem as ksp


def add_random(current, costs, weights, size):
    unincluded_indexes = [i for i, included in enumerate(current) if not included]

    while len(unincluded_indexes) > 0:
        candidate_index = rnd.randint(0, len(unincluded_indexes))
        candidate = unincluded_indexes.pop(candidate_index)
        clone = list(current)

        if clone[candidate]:
            continue

        clone[candidate] = 1
        if ksp.validate(clone, costs=costs, weights=weights, size=size):
            current[candidate] = 1
            break

    return current


def remove_random(current, costs, weights, size):
    included_indexes = [i for i, included in enumerate(current) if included]

    while len(included_indexes) > 0:
        candidate_index = rnd.randint(0, len(included_indexes))
        candidate = included_indexes.pop(candidate_index)

        if not current[candidate]:
            continue

        clone = list(current)
        clone[candidate] = 0
        if ksp.validate(clone, costs=costs, weights=weights, size=size):
            current[candidate] = 0
            break

    return current


def add_best(current, costs, weights, size):
    pass


def remove_worst(current, costs, weights, size):
    pass
