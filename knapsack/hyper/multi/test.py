import numpy as np

from knapsack.hyper.multi import genetic
from knapsack.hyper.multi import lp_relaxed as lp
from knapsack.hyper.multi import read_write_file as io

mknap1_path, mknap2_path = "./resources/mknap1.txt", "./resources/mknap2.txt"
mknapcbs_pathes = ["./resources/mknapcb1.txt", "./resources/mknapcb2.txt", "./resources/mknapcb3.txt", "./resources/mknapcb4.txt",
                   "./resources/mknapcb5.txt", "./resources/mknapcb6.txt", "./resources/mknapcb7.txt", "./resources/mknapcb8.txt",
                   "./resources/mknapcb9.txt"]


def solve(knapsack, optimal, attempts=50):
    result = 0
    cumulative_gap = 0
    print("Pseudo-optimal:\t" + str(optimal))
    # TODO generate initial state using LP-relaxed solution
    start = np.zeros((len(knapsack["sizes"]), len(knapsack["costs"])))
    for i in range(1, attempts + 1):
        optimal_funcs = genetic.minimize(50, weights=knapsack["weights"], costs=knapsack["costs"], sizes=knapsack["sizes"], included=start)
        current = genetic.fitness_hyper_ksp(optimal_funcs, weights=knapsack["weights"], costs=knapsack["costs"], sizes=knapsack["sizes"], included=start)
        print("Current:\t" + str(optimal - current))
        result += optimal - current
        current_gap = 100 * (optimal - current) / optimal
        print("Normalized:\t" + str(current_gap))
        cumulative_gap += current_gap
        print("Normed cum:\t" + str(cumulative_gap / i))
    return cumulative_gap / attempts


if __name__ == '__main__':
    knapsacks = io.parse_mknap2(mknap2_path)
    lp_optimal = [lp.ksp_solve_lp_relaxed_greedy(**knapsack) for knapsack in knapsacks]
    optimals = []
    for knapsack, lp_optimal in zip(knapsacks, lp_optimal):
        print("KNAPSACK:")
        print("Number of KSPs: " + str(len(knapsack["sizes"])))
        print("Number of Items: " + str(len(knapsack["costs"])))
        optimals.append(solve(knapsack, lp_optimal))
    print("CUMULATIVE GAP OVER ALL TEST DATA: " + str(optimals))
