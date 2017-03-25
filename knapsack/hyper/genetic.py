import random as rnd

import algorithms.genetic as gene
import knapsack.hyper.heurs1knpsck as heurs
import knapsack.problem as problem


def simple_state_generator_hyper_ksp(dimension):
    state = []
    candidates = [heurs.add_lightest, heurs.add_heaviest, heurs.add_least_cost, heurs.add_most_cost, heurs.add_best,
                  heurs.remove_lightest, heurs.remove_heaviest, heurs.remove_least_cost, heurs.remove_most_cost,
                  heurs.remove_worst]
    while len(state) < dimension:
        index = rnd.randint(0, len(candidates) - 1)
        probability = rnd.random()
        tabu_generation = rnd.randint(1, 6) if probability < 0.2 else 0
        state.append((candidates[index], tabu_generation))
    return state


def initial_population_generator_hyper_ksp(amount, dimension, state_generator=simple_state_generator_hyper_ksp,
                                           **kwargs):
    population = []
    while len(population) < amount:
        population.append(state_generator(dimension))
    return population


def crossover_reproduction_hyper_ksp(first_parent, second_parent, **kwargs):
    # TODO: try other genetic operators (different types of crossover, for example)
    # nor first allel nor last allel
    crossover_index = rnd.randint(1, len(first_parent) - 1)
    child_candidate = first_parent[:crossover_index + 1] + second_parent[crossover_index + 1:]
    return child_candidate


def fitness_hyper_ksp(state, **kwargs):
    included = list(kwargs["included"])
    tabooed_items_generations = [0] * len(included)
    for operation, tabu_generation in state:
        tabooed_items = [index for index, generation in enumerate(tabooed_items_generations) if generation > 0]
        included, modified_index = operation(included, tabooed_indexes=tabooed_items, costs=kwargs["costs"],
                                             weights=kwargs["weights"], size=kwargs["size"])
        tabooed_items_generations = list(map(lambda x: x - 1 if x > 0 else 0, tabooed_items_generations))

        # TODO if modified index is -1, should we exclude the operation from the state?
        if tabu_generation > 0 and modified_index >= 0:
            tabooed_items_generations[modified_index] = tabu_generation
    return problem.solve(included, costs=kwargs["costs"], weights=kwargs["weights"], size=kwargs["size"])


def minimize(dimension, **kwargs):
    return gene.minimize(dimension, initial_population_generator_hyper_ksp, crossover_reproduction_hyper_ksp,
                         fitness_hyper_ksp, **kwargs)


if __name__ == '__main__':
    size = 165
    weights = [23, 31, 29, 44, 53, 38, 63, 85, 89, 82]
    costs = [92, 57, 49, 68, 60, 43, 67, 84, 87, 72]
    start = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    optimal = [1, 1, 1, 1, 0, 1, 0, 0, 0, 0]
    heuristics = [(heurs.add_least_cost, 2), (heurs.add_least_cost, 1), (heurs.remove_least_cost, 0)]
    print(fitness_hyper_ksp(heuristics, weights=weights, costs=costs,
                            size=size, included=start))
