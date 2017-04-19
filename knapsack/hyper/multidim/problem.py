import numpy as np


def solve(included, costs, weights, sizes):
    costs = np.asarray(costs)
    included = np.asarray(included)

    if not validate(included, costs, weights, sizes):
        return float("-inf")

    return np.sum(costs * included)


def validate(included, costs, weights, sizes):
    weights = np.asarray(weights)
    costs = np.asarray(costs)
    included = np.asarray(included)
    sizes = np.asarray(sizes)

    if not (sizes.size == weights.shape[0]):
        return False

    if not (weights.shape[1] == costs.size == included.size):
        return False

    if not all(np.sum(weights * included, axis=1) <= sizes):
        return False

    return True


if __name__ == '__main__':
    included = [1, 0, 0, 0, 0, 0]
    costs = [1, 2, 3, 4, 5, 6]
    weights = [[1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2], [3, 3, 3, 3, 3, 3], [4, 4, 4, 4, 4, 4], [5, 5, 5, 5, 5, 5]]
    sizes = [5, 5, 5, 5, 5]
    print(solve(included, costs, weights, sizes))
