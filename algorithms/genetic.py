import operator
import random as rnd

import numpy as np


def crossover_selection(population, part_of_best_to_stay_alive, crossover_reproduction_func, mutation_func,
                        fitness_func, **kwargs):
    iteration = 0
    first_parent = None
    second_parent = None

    population = list(sorted(population, key=operator.itemgetter("fitness"), reverse=True))[:part_of_best_to_stay_alive]

    while iteration < 50:
        iteration += 1
        first_parent_index = rnd.randint(0, len(population) - 1)
        second_parent_index = rnd.randint(0, len(population) - 1)

        if first_parent_index != second_parent_index and not \
                        population[first_parent_index] == population[second_parent_index]:
            first_parent = population[first_parent_index]["heuristics"]
            second_parent = population[second_parent_index]["heuristics"]
            break

    if first_parent is None or second_parent is None:
        return population

    child = crossover_reproduction_func(first_parent, second_parent, **kwargs)

    if child is None:
        return population

    mutation_func(child, **kwargs)

    new_individual = {"heuristics": child, "fitness": fitness_func(child, **kwargs)}
    population.append(new_individual)
    return population


def minimize(dimension, initial_population_generator, crossover_reproduction, mutation_func, fitness_func, **kwargs):
    # TODO how ti initialize population amount?
    population = initial_population_generator(10, dimension, **kwargs)
    part_of_best_to_stay_alive = int(len(population) * 0.25)
    max_fitness = 0
    iteration = 0
    # TODO finish condition?
    while iteration < 15:
        max_fitness_state_index = np.argmax([individual["fitness"] for individual in population])
        current_max_fitness = population[max_fitness_state_index]["fitness"]
        if max_fitness >= current_max_fitness:
            iteration += 1
        else:
            if max_fitness > 0:
                print("WE FOUND A BETTER SOLUTION: " + str(current_max_fitness - max_fitness))
            max_fitness = current_max_fitness
            iteration = 0
        population = crossover_selection(population, part_of_best_to_stay_alive, crossover_reproduction, mutation_func,
                                         fitness_func, **kwargs)
    return population[max_fitness_state_index]["heuristics"]
