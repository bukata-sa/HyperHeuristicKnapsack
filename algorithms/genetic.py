import random as rnd

import numpy as np


def crossover_selection(population, fitness_func, crossover_reproduction_func, **kwargs):
    iteration = 0
    first_parent = None
    second_parent = None
    while iteration < 100:
        iteration += 1
        first_parent_index = rnd.randint(0, len(population) - 1)
        second_parent_index = rnd.randint(0, len(population) - 1)
        first_parent = population[first_parent_index]
        second_parent = population[second_parent_index]
        if (first_parent_index != second_parent_index and not first_parent == second_parent) \
                or iteration > len(population) ** 2:
            first_parent = None
            second_parent = None
        break

    if first_parent is None or second_parent is None:
        return

    child = crossover_reproduction_func(first_parent, second_parent, **kwargs)

    if child is None:
        return

    # TODO: mutate child

    for i, state in enumerate(population):
        if fitness_func(child, **kwargs) > fitness_func(state, **kwargs):
            # TODO: avoid dominant genome
            population[i] = child
            break


def minimize(dimension, initial_population_generator, crossover_reproduction, fitness_func, **kwargs):
    population = initial_population_generator(30, dimension, **kwargs)
    max_fitness = 0
    iteration = 0
    while iteration < 100:
        crossover_selection(population, fitness_func, crossover_reproduction, **kwargs)
        max_fitness_state_index = np.argmax([fitness_func(state, **kwargs) for state in population])
        current_max_fitness = fitness_func(population[max_fitness_state_index], **kwargs)
        if max_fitness >= current_max_fitness:
            iteration += 1
        else:
            max_fitness = current_max_fitness
            iteration = 0
    return population[max_fitness_state_index]
