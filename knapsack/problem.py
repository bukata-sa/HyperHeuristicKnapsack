import numpy as np


def validate(included, costs, weights, knapsack_size):
    weights = np.asarray(weights)
    costs = np.asarray(costs)
    included = np.asarray(included)
    # Weights, costs, included must be in same dimensionality
    if not (len(weights) == len(costs) == len(included)):
        return False

    # Sum of all included weights must be less or equal to knapsack size
    if not (np.sum(weights * included) <= knapsack_size):
        return False

    return True


def solve(included, costs, weights, knapsack_size):
    weights = np.asarray(weights)
    costs = np.asarray(costs)
    included = np.asarray(included)
    return np.sum(costs * included) if validate(included, costs, weights, knapsack_size) else float("-inf")
