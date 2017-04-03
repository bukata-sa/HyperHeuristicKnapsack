import operator
import random as rnd

import numpy as np


def crossover_selection(population, fitness_func, crossover_reproduction_func, **kwargs):
    iteration = 0
    first_parent = None
    second_parent = None

    mean = np.mean(list(map(operator.itemgetter(1), population)))
    top_half_individuals = list(filter(lambda x: x[1] > mean, population))

    while iteration < 100:
        iteration += 1
        first_parent_index = rnd.randint(0, len(top_half_individuals) - 1)
        second_parent_index = rnd.randint(0, len(top_half_individuals) - 1)

        if first_parent_index != second_parent_index and not \
                        top_half_individuals[first_parent_index] == top_half_individuals[second_parent_index]:
            first_parent = population[first_parent_index][0]
            second_parent = population[second_parent_index][0]
            break

    if first_parent is None or second_parent is None:
        return

    child = crossover_reproduction_func(first_parent, second_parent, **kwargs)

    if child is None:
        return

    # TODO: mutate child
    population.append((child, fitness_func(child, **kwargs)))


def minimize(dimension, initial_population_generator, crossover_reproduction, fitness_func, **kwargs):
    # TODO how ti initialize population amount?
    population = initial_population_generator(100, dimension, **kwargs)
    max_fitness = 0
    iteration = 0
    # TODO finish condition?
    while iteration < 50:
        crossover_selection(population, fitness_func, crossover_reproduction, **kwargs)
        max_fitness_state_index = np.argmax([fitness for state, fitness in population])
        current_max_fitness = population[max_fitness_state_index][1]
        if max_fitness >= current_max_fitness:
            iteration += 1
        else:
            max_fitness = current_max_fitness
            iteration = 0
    return population[max_fitness_state_index][0]
