import time

import numpy as np
from pathos.multiprocessing import ProcessPool as Pool


def crossover(population, crossover_reproduction_func, mutation_func,
              fitness_func, **kwargs):
    childs = crossover_reproduction_func(population, **kwargs)

    if childs is None or len(childs) == 0:
        return population

    def worker(child):
        mutation_func(child, **kwargs)
        return {"heuristics": child, "fitness": fitness_func(child, **kwargs)}

    pool = Pool()
    new_individuals = pool.map(worker, childs)
    population += new_individuals
    return population


def minimize(initial_population_generator, selection_func, crossover_reproduction, mutation_func, fitness_func,
             **kwargs):
    first = time.perf_counter()
    population = initial_population_generator(300, **kwargs)
    print("Population generation time: " + str(time.perf_counter() - first))
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
        population = selection_func(population)
        population = crossover(population, crossover_reproduction, mutation_func, fitness_func, **kwargs)
    return max_fitness_heuristics
