import operator
import time

import numpy as np


def crossover_selection(population, part_of_best_to_stay_alive, crossover_reproduction_func, mutation_func,
                        fitness_func, **kwargs):
    population = list(sorted(population, key=operator.itemgetter("fitness"), reverse=True))[:len(population) // 2]

    childs = crossover_reproduction_func(population, **kwargs)

    if childs is None or len(childs) == 0:
        return population

    for child in childs:
        mutation_func(child, **kwargs)

        new_individual = {"heuristics": child, "fitness": fitness_func(child, **kwargs)}
        population.append(new_individual)
    return population


def minimize(dimension, initial_population_generator, crossover_reproduction, mutation_func, fitness_func, **kwargs):
    first = time.perf_counter()
    population = initial_population_generator(300, dimension, **kwargs)
    print("Population generation time: " + str(time.perf_counter() - first))
    part_of_best_to_stay_alive = int(len(population) * 0.5)
    max_fitness = 0
    max_fitness_heuristics = 0
    iteration = 0
    # TODO finish condition?
    while iteration < 10:
        current_max_fitness_state_index = np.argmax([individual["fitness"] for individual in population])
        current_max_fitness = population[current_max_fitness_state_index]["fitness"]
        if max_fitness >= current_max_fitness:
            iteration += 1
        else:
            if max_fitness > 0:
                print("WE FOUND A BETTER SOLUTION: " + str(current_max_fitness - max_fitness))
            max_fitness = current_max_fitness
            max_fitness_heuristics = population[current_max_fitness_state_index]["heuristics"]
            iteration = 0
        population = crossover_selection(population, part_of_best_to_stay_alive, crossover_reproduction, mutation_func,
                                         fitness_func, **kwargs)
    return max_fitness_heuristics
