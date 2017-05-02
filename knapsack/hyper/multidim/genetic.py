import itertools
import math
import operator
import random as rnd

import numpy as np
from pathos.multiprocessing import ProcessPool as Pool

from algorithms import genetic as gene
from knapsack.hyper.multidim import heuristics
from knapsack.hyper.multidim import problem
from knapsack.hyper.single.genetic import mutation_hyper_ksp


def fitness_hyper_ksp(state, **kwargs):
    # TODO: avoid copypaste
    included = list(kwargs["included"])
    tabooed_items_generations = [0] * len(kwargs["costs"])
    for operation, tabu_generation in state:
        tabooed_items = [index for index, generation in enumerate(tabooed_items_generations) if generation > 0]
        included, modified_index = operation(included, tabooed_indexes=tabooed_items, costs=kwargs["costs"],
                                             weights=kwargs["weights"], sizes=kwargs["sizes"])
        tabooed_items_generations = list(map(lambda x: x - 1 if x > 0 else 0, tabooed_items_generations))

        # TODO if modified index is -1, should we exclude the operation from the state?
        if tabu_generation > 0 and modified_index >= 0:
            tabooed_items_generations[modified_index] = tabu_generation
    return problem.solve(included, costs=kwargs["costs"], weights=kwargs["weights"], sizes=kwargs["sizes"])


def crossover_reproduction_hyper_ksp(population, **kwargs):
    chunk_size = int(math.sqrt(len(population)))
    chunks = [population[i:i + chunk_size] for i in range(0, len(population), chunk_size)]
    champions = []
    for chunk in chunks:
        champions.append(max(chunk, key=operator.itemgetter("fitness")))

    childs = []
    for champion1, champion2 in itertools.permutations(champions, 2):
        crossover_point = min(len(champion1["heuristics"]), len(champion2["heuristics"])) // 2 - 1
        child = champion1["heuristics"][:crossover_point] + champion2["heuristics"][crossover_point + 1:]
        child = fit_generated_child(child, **kwargs)
        childs.append(child)
        return childs


def fit_generated_child(child, **kwargs):
    included = list(kwargs["included"])
    tabooed_items_generations = [0] * len(kwargs["costs"])

    deleted_offset = 0
    for index, operation_and_tabu_generation in enumerate(child[:]):
        operation, tabu_generation = operation_and_tabu_generation
        tabooed_items = [index for index, generation in enumerate(tabooed_items_generations) if generation > 0]
        included, modified_index = operation(included, tabooed_indexes=tabooed_items, costs=kwargs["costs"],
                                             weights=kwargs["weights"], sizes=kwargs["sizes"])
        tabooed_items_generations = list(map(lambda x: x - 1 if x > 0 else 0, tabooed_items_generations))

        if modified_index == -1:
            del child[index - deleted_offset]
            deleted_offset += 1
            continue

        if tabu_generation > 0:
            tabooed_items_generations[modified_index] = tabu_generation

    fitted_child = simple_state_generator_hyper_ksp(child, heuristics.get_heuristics(), included=included, costs=kwargs["costs"], weights=kwargs["weights"], sizes=kwargs["sizes"])
    return fitted_child


# deprecated
def initial_population_generator_hyper_ksp(amount, dimension, **kwargs):
    population = []
    heuristics_candidates = heuristics.get_heuristics()
    for i in range(amount):
        state = simple_state_generator_hyper_ksp(dimension, heuristics_candidates, **kwargs)
        population.append({"heuristics": state, "fitness": fitness_hyper_ksp(state, **kwargs)})
    return population


def initial_population_generator_hyper_ksp_multiproc(amount, **kwargs):
    heuristics_candidates = heuristics.get_heuristics()

    def worker(_):
        state = simple_state_generator_hyper_ksp([], heuristics_candidates, **kwargs)
        result = {"heuristics": state, "fitness": fitness_hyper_ksp(state, **kwargs)}
        return result

    pool = Pool()
    population = pool.map(worker, range(amount))
    return population


def local_max_not_reached(weights, included, sizes):
    items_not_included = np.where(included < 1)[0]
    reached = True
    for item in items_not_included:
        if all(np.sum(weights * included, axis=1) + weights[:, item] < sizes):
            reached = False
            break
    return not reached


def simple_state_generator_hyper_ksp(state, heuristics_candidates, **kwargs):
    weights = np.asarray(kwargs["weights"])
    included = np.asarray(kwargs["included"])
    sizes = np.asarray(kwargs["sizes"])
    tabooed_items_generations = [0] * len(kwargs["costs"])

    # select heuristics for state while not reached local maximum
    while any(included < 1) and local_max_not_reached(weights, included, sizes):
        probability = rnd.random()
        tabu_generation = rnd.randint(1, 5) if probability < 0.3 else 0

        cumulative_probability = 0
        for index, heuristics_candidate in enumerate(heuristics_candidates):
            cumulative_probability += heuristics_candidate[1]
            if probability <= cumulative_probability:
                break
        operation = heuristics_candidates[index][0]

        tabooed_items = [index for index, generation in enumerate(tabooed_items_generations) if generation > 0]
        included, modified_index = operation(included, tabooed_indexes=tabooed_items, costs=kwargs["costs"],
                                             weights=kwargs["weights"], sizes=kwargs["sizes"])

        if modified_index == -1:
            continue

        state.append((operation, tabu_generation))
        tabooed_items_generations = list(map(lambda x: x - 1 if x > 0 else 0, tabooed_items_generations))

        if tabu_generation > 0:
            tabooed_items_generations[modified_index] = tabu_generation
    return state


def mutation_hyper_multi_ksp(state, **kwargs):
    return mutation_hyper_ksp(state, heuristics)


def minimize(**kwargs):
    return gene.minimize(initial_population_generator_hyper_ksp_multiproc, crossover_reproduction_hyper_ksp,
                         mutation_hyper_multi_ksp, fitness_hyper_ksp, **kwargs)
