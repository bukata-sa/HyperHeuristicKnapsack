import random as rnd

import algorithms.genetic as gene
import knapsack.hyper.heurs1knpsck as heurs
import knapsack.problem as problem


def simple_state_generator_hyper_ksp(dimension):
    state = []
    candidates = [heurs.add_lightest, heurs.add_heaviest, heurs.add_least_cost, heurs.add_most_cost, heurs.add_best,
                  heurs.remove_lightest, heurs.remove_heaviest, heurs.remove_least_cost, heurs.remove_most_cost,
                  heurs.remove_worst, """heurs.tabu_operation_chain"""]
    while len(state) < dimension:
        index = rnd.randint(0, len(candidates) - 1)
        state.append(candidates[index])
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
    for action in state:
        included = action(included, costs=kwargs["costs"], weights=kwargs["weights"], size=kwargs["size"])
    return problem.solve(included, costs=kwargs["costs"], weights=kwargs["weights"], size=kwargs["size"])


def minimize(dimension, **kwargs):
    return gene.minimize(dimension, initial_population_generator_hyper_ksp, crossover_reproduction_hyper_ksp,
                         fitness_hyper_ksp, **kwargs)
