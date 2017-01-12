import random as rnd

import numpy as np

from knapsack.problem import solve, validate


def compare_state_ksp(state1, state2):
    for i in range(0, len(state1)):
        if state1 != state2:
            return True
    return False


def crossover_selection_ksp(population, fitness_func,
                            validate_func=validate,
                            compare_state=compare_state_ksp,
                            **kwargs):
    iteration = 0
    while iteration < 100:
        iteration += 1
        first_parent_index = rnd.randint(0, len(population) - 1)
        second_parent_index = rnd.randint(0, len(population) - 1)
        first_parent = population[first_parent_index]
        second_parent = population[second_parent_index]
        if (first_parent_index != second_parent_index and
                not compare_state(first_parent, second_parent) or iteration > len(population) ** 2):
            break
    valid_child = False

    iteration = 0
    while not (valid_child or iteration > len(first_parent) ** 2):
        # TODO: try other genetic operators (different types of crossover, for example)
        # nor first allel nor last allel
        crossover_index = rnd.randint(1, len(first_parent) - 2)
        child_candidate = first_parent[:crossover_index + 1] + second_parent[crossover_index + 1:]
        valid_child = validate_func(child_candidate, **kwargs)

    # TODO: mutate child

    if not valid_child:
        return

    for i, state in enumerate(population):
        if fitness_func(child_candidate, **kwargs) > fitness_func(state, **kwargs):
            # TODO: avoid dominant genome
            population[i] = child_candidate
            break


def simple_state_generator_ksp(dim):
    state = []
    for i in range(0, dim):
        random_boolean = False if rnd.randint(0, 1) == 0 else True
        state.append(random_boolean)
    return state


def initial_population_generator_ksp(amount, dim, validator=validate,
                                     state_generator=simple_state_generator_ksp,
                                     **kwargs):
    population = []
    while len(population) < amount:
        population_candidate = state_generator(dim)
        if validator(population_candidate, **kwargs):
            population.append(population_candidate)
    return population


def minimize(dimension,
             initial_population_generator=initial_population_generator_ksp,
             fitness_func=solve,
             selection_func=crossover_selection_ksp,
             **kwargs):
    population = initial_population_generator(30, dimension, **kwargs)
    max_fitness = 0
    iteration = 0
    while iteration < 100:
        selection_func(population, fitness_func, **kwargs)
        max_fitness_state_index = np.argmax([fitness_func(state, **kwargs) for state in population])
        current_max_fitness = fitness_func(population[max_fitness_state_index], **kwargs)
        if max_fitness >= current_max_fitness:
            iteration += 1
        else:
            max_fitness = current_max_fitness
            iteration = 0
    return population[max_fitness_state_index]
