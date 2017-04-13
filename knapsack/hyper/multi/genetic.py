import random as rnd

from pathos.multiprocessing import ProcessPool as Pool

from algorithms import genetic as gene
from knapsack.hyper.multi import heuristics
from knapsack.hyper.multi import problem
from knapsack.hyper.single.genetic import simple_state_generator_hyper_ksp, mutation_hyper_ksp


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


def crossover_reproduction_hyper_ksp(first_parent, second_parent, **kwargs):
    # TODO: try other genetic operators (different types of crossover, for example)
    # nor first allel nor last allel
    crossover_index = rnd.randint(1, len(first_parent) - 1)
    child_candidate = first_parent[:crossover_index + 1] + second_parent[crossover_index + 1:]
    return child_candidate


def initial_population_generator_hyper_ksp(amount, dimension, **kwargs):
    population = []
    heuristics_candidates = heuristics.get_heuristics()
    for i in range(amount):
        state = simple_state_generator_hyper_ksp(dimension, heuristics_candidates)
        population.append({"heuristics": state, "fitness": fitness_hyper_ksp(state, **kwargs)})
    return population


def initial_population_generator_hyper_ksp_multiproc(amount, dimension, **kwargs):
    heuristics_candidates = heuristics.get_heuristics()

    def worker(_):
        state = simple_state_generator_hyper_ksp(dimension, heuristics_candidates)
        result = {"heuristics": state, "fitness": fitness_hyper_ksp(state, **kwargs)}
        return result

    pool = Pool()
    population = pool.map(worker, range(amount))
    return population


def mutation_hyper_multi_ksp(state, **kwargs):
    return mutation_hyper_ksp(state, heuristics)


def minimize(dimension, **kwargs):
    return gene.minimize(dimension, initial_population_generator_hyper_ksp_multiproc, crossover_reproduction_hyper_ksp,
                         mutation_hyper_multi_ksp, fitness_hyper_ksp, **kwargs)
