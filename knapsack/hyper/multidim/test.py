import pickle

import numpy as np

from knapsack.hyper.multidim import genetic
from knapsack.hyper.multidim import read_write_file as io

mknap1_path, mknap2_path = "./resources/mknap1.txt", "./resources/mknap2.txt"
mknapcbs_pathes = ["./resources/mknapcb1.txt", "./resources/mknapcb2.txt", "./resources/mknapcb3.txt",
                   "./resources/mknapcb4.txt",
                   "./resources/mknapcb5.txt", "./resources/mknapcb6.txt", "./resources/mknapcb7.txt",
                   "./resources/mknapcb8.txt",
                   "./resources/mknapcb9.txt"]
mknapres_path = "./resources/mkcbres.txt"


def generate_initial_knapsack(optimal, weights, costs, sizes):
    initial_knapsack = np.zeros(len(costs))

    # for item_index in list(range(len(costs))):
    #     pseudo_optimal_value = pseudo_optimal[item_index]
    #     initial_knapsack[item_index] = random.randint(0, 1) if pseudo_optimal_value == 1 else 0
    #
    # initial_fitness = problem.solve(initial_knapsack, costs, weights, sizes)
    return initial_knapsack


def solve(knapsack, attempts=50):
    optimal_fitness = knapsack["optimal"]
    result = 0
    cumulative_gap = 0
    print("Optimal_fitness:\t" + str(optimal_fitness))
    solved = []
    # TODO generate initial state using LP-relaxed solution
    for i in range(1, attempts + 1):
        start = generate_initial_knapsack(**knapsack)
        optimal_funcs = genetic.minimize(weights=knapsack["weights"], costs=knapsack["costs"],
                                         sizes=knapsack["sizes"], included=start)
        current = genetic.fitness_hyper_ksp(optimal_funcs, weights=knapsack["weights"], costs=knapsack["costs"],
                                            sizes=knapsack["sizes"], included=start)
        fitness_current_diff = optimal_fitness - current
        print("Current:\t" + str(fitness_current_diff))
        solved.append(fitness_current_diff)
        result += fitness_current_diff
        current_gap = 100 * (fitness_current_diff) / optimal_fitness
        print("Normalized:\t" + str(current_gap))
        cumulative_gap += current_gap
        print("Normed cum:\t" + str(cumulative_gap / (attempts + 1)))
    return solved, cumulative_gap / attempts


if __name__ == '__main__':
    # knapsacks = io.parse_mknapcb(mknapcbs_pathes, mknapres_path)
    knapsacks = io.parse_mknap1(mknap1_path)
    # lp_optimals = [lp.ksp_solve_lp_relaxed_greedy(**knapsack) for knapsack in knapsacks]
    optimals = []
    results = []
    ksp_number = 0
    for knapsack in list(knapsacks):
        print("KNAPSACK:")
        print("Number of constraints: " + str(len(knapsack["sizes"])))
        print("Number of items: " + str(len(knapsack["costs"])))
        solved, normalized = solve(knapsack, attempts=3)
        optimals.append(normalized)
        with open("resources/output/mknap1_out_" + str(ksp_number) + ".pckl", 'wb') as out:
            pickle.dump({
                "const": len(knapsack["sizes"]),
                "items": len(knapsack["costs"]),
                "solved": solved,
                "normalized:": normalized
            }, out)
        ksp_number += 1
    print("CUMULATIVE GAP OVER ALL TEST DATA: " + str(optimals))