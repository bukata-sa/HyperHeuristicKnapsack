from knapsack.annealing import minimize
from knapsack.problem import solve

success = 0

for i in range(1, 100):
    result = minimize([False, False, False, False, False], 1, 10000,
                      costs=[1, 2, 3, 4, 5], weights=[1, 2, 3, 4, 5], size=10)
    knapsack_cost = solve(result, [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], 10)
    print(knapsack_cost)
    if knapsack_cost == 10:
        success += 1

    print(str(success) + "%")
