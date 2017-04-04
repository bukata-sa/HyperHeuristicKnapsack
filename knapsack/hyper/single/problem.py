import numpy as np


def validate(included, costs, weights, size):
    weights = np.asarray(weights)
    costs = np.asarray(costs)
    included = np.asarray(included)
    # Weights, costs, included must be in same dimensionality
    if not (len(weights) == len(costs) == len(included)):
        return False

    # Sum of all included weights must be less or equal to knapsack size
    if not (np.sum(weights * included) <= size):
        return False

    return True


def solve(included, costs, weights, size):
    weights = np.asarray(weights)
    costs = np.asarray(costs)
    included = np.asarray(included)
    return np.sum(costs * included) if validate(included, costs, weights, size) else float("-inf")
