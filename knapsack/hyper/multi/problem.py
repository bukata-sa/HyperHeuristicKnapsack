import numpy as np


def solve(included, costs, weights, sizes):
    costs = np.asarray(costs)
    included = np.asarray(included)

    if not validate(included, costs, weights, sizes):
        return float("-inf")

    return np.sum(costs * np.sum(included, axis=0))


def validate(included, costs, weights, sizes):
    weights = np.asarray(weights)
    costs = np.asarray(costs)
    included = np.asarray(included)
    sizes = np.asarray(sizes)

    if not (sizes.size == weights.shape[0] == included.shape[0]):
        return False

    if not (weights.shape[1] == costs.size == included.shape[1]):
        return False

    vertical_sum_included = np.sum(included, axis=0)
    if np.where(vertical_sum_included > 1)[0].size > 0:
        return False

    sum_included_weights = np.sum(weights * included, axis=1)
    if np.where(sum_included_weights > sizes)[0].size > 0:
        return False

    return True


if __name__ == '__main__':
    included = [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0]]
    costs = [1, 2, 3, 4, 5, 6]
    weights = [[1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2], [3, 3, 3, 3, 3, 3], [4, 4, 4, 4, 4, 4], [5, 5, 5, 5, 5, 5]]
    sizes = [5, 5, 5, 5, 5]
    print(solve(included, costs, weights, sizes))
