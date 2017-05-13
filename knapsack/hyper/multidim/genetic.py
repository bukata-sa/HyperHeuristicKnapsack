import itertools
import operator
import random as rnd
from functools import partial

import numpy as np
from pathos.multiprocessing import ProcessPool as Pool

from algorithms import genetic as gene
from knapsack.hyper.multidim import heuristics
from knapsack.hyper.multidim import problem
from knapsack.hyper.single.genetic import mutation_hyper_ksp


def fitness_hyper_ksp(state, **kwargs):
    included = list(kwargs["included"])
    for operation in state:
        included, modified_index = operation(included, tabooed_indexes=[], costs=kwargs["costs"],
                                             weights=kwargs["weights"], sizes=kwargs["sizes"])
    return problem.solve(included, costs=kwargs["costs"], weights=kwargs["weights"], sizes=kwargs["sizes"])


def fit_generated_child(child, **kwargs):
    included = list(kwargs["included"])

    deleted_offset = 0
    for index, operation in enumerate(child[:]):
        included, modified_index = operation(included, tabooed_indexes=[], costs=kwargs["costs"],
                                             weights=kwargs["weights"], sizes=kwargs["sizes"])

        if modified_index == -1:
            del child[index - deleted_offset]
            deleted_offset += 1
            continue

    fitted_child = simple_state_generator_hyper_ksp(child, heuristics.get_heuristics(), included=included,
                                                    costs=kwargs["costs"], weights=kwargs["weights"],
                                                    sizes=kwargs["sizes"])
    return fitted_child


# test purposes only
def initial_population_generator_hyper_ksp(amount, **kwargs):
    population = []
    heuristics_candidates = heuristics.get_heuristics()
    for i in range(amount):
        state = simple_state_generator_hyper_ksp([], heuristics_candidates, **kwargs)
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


def local_max_reached(weights, included, sizes, tabooed_items):
    items_not_included = np.where(included < 1)[0]
    tabooed_items = np.asarray(tabooed_items)
    items_tabooed = np.where(tabooed_items > 0)[0]
    available_items = set(items_not_included).difference(set(items_tabooed))
    reached = True
    for item in available_items:
        if all(np.sum(weights * included, axis=1) + weights[:, item] < sizes):
            reached = False
            break
    return reached


def simple_state_generator_hyper_ksp(state, heuristics_candidates, **kwargs):
    weights = np.asarray(kwargs["weights"])
    included = np.asarray(kwargs["included"])
    sizes = np.asarray(kwargs["sizes"])
    shortened_kwargs = dict(kwargs)
    del shortened_kwargs["included"]
    tabooed_items_generations = [0] * len(kwargs["costs"])

    # select heuristics for state while not reached local maximum
    max_reached_times = 0
    max_state = None
    max_state_fitness = 0

    while any(included < 1):
        probability = rnd.random()
        cumulative_probability = 0
        for heuristic_operation, heuristic_probability in heuristics_candidates:
            cumulative_probability += heuristic_probability
            if probability <= cumulative_probability:
                break
        operation = heuristic_operation

        tabooed_items = (index for index, generation in enumerate(tabooed_items_generations) if generation > 0)
        included, modified_index = operation(included, tabooed_indexes=tabooed_items, costs=kwargs["costs"],
                                             weights=kwargs["weights"], sizes=kwargs["sizes"])

        state.append(operation)
        tabooed_items_generations = list(map(lambda x: x - 1 if x > 0 else 0, tabooed_items_generations))

        if local_max_reached(weights, included, sizes, tabooed_items_generations):
            current_max_state_fitness = problem.solve(included, **shortened_kwargs)
            if current_max_state_fitness > max_state_fitness:
                max_state_fitness = current_max_state_fitness
                max_state = state[:]

            max_reached_times += 1
            if max_reached_times == 20:
                break
            state.pop()

            if modified_index == -1:
                continue
            tabooed_items_generations[modified_index] = rnd.randint(3, 7)
    return max_state


def selection_hyper_ksp(population):
    chunk_size = 2
    chunks = (population[i:i + chunk_size] for i in range(0, len(population), chunk_size))

    champions = list(map(lambda chunk: max(chunk, key=operator.itemgetter("fitness")), chunks))
    return champions


def crossover_reproduction_hyper_ksp(population, **kwargs):
    fitness_sum = sum(map(operator.itemgetter("fitness"), population))
    probabilities = list(map(lambda person: person["fitness"] / fitness_sum, population))
    candidates = np.random.choice(population, 14, p=probabilities)
    # candidates = np.random.choice(population, 14)
    cross_child_partial = partial(cross_child, **kwargs)
    pool = Pool()
    childs = pool.map(cross_child_partial, *zip(*itertools.permutations(candidates, 2)))
    return childs


def cross_child(child1, child2, **kwargs):
    crossover_point = min(len(child1["heuristics"]), len(child2["heuristics"])) // 2 - 1
    child = child1["heuristics"][:crossover_point] + child2["heuristics"][crossover_point + 1:]
    child = fit_generated_child(child, **kwargs)
    return child


def mutation_hyper_multi_ksp(state, **kwargs):
    return mutation_hyper_ksp(state, heuristics)


def minimize(**kwargs):
    return gene.minimize(initial_population_generator_hyper_ksp_multiproc, selection_hyper_ksp,
                         crossover_reproduction_hyper_ksp, mutation_hyper_multi_ksp, fitness_hyper_ksp, **kwargs)
