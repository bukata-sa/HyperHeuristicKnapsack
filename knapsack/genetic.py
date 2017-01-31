import random as rnd

import algorithms as algs
import knapsack.problem as ksp


def simple_state_generator_ksp(dimension):
    state = []
    for i in range(0, dimension):
        random_boolean = False if rnd.randint(0, 1) == 0 else True
        state.append(random_boolean)
    return state


def initial_population_generator_ksp(amount, dimension, validator=ksp.validate,
                                     state_generator=simple_state_generator_ksp,
                                     **kwargs):
    population = []
    while len(population) < amount:
        population_candidate = state_generator(dimension)
        if validator(population_candidate, **kwargs):
            population.append(population_candidate)
    return population


def compare_state_ksp(state1, state2):
    for i in range(0, len(state1)):
        if state1 != state2:
            return True
    return False


def crossover_func_ksp(first_parent, second_parent, **kwargs):
    iteration = 0
    while iteration < len(first_parent) ** 2:
        # TODO: try other genetic operators (different types of crossover, for example)
        # nor first allel nor last allel
        crossover_index = rnd.randint(1, len(first_parent) - 1)
        child_candidate = first_parent[:crossover_index + 1] + second_parent[crossover_index + 1:]
        if ksp.validate(child_candidate, **kwargs):
            return child_candidate
    return None


def minimize(dimension, **kwargs):
    return algs.genetic.minimize(dimension, initial_population_generator_ksp, crossover_func_ksp, ksp.solve, **kwargs)
