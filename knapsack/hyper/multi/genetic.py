import random as rnd

from algorithms import genetic as gene
from knapsack.hyper.multi import heuristics
from knapsack.hyper.multi import problem
from knapsack.hyper.single.genetic import simple_state_generator_hyper_ksp


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
    heuristics_candidates = heuristics.get_single_heurs_for_multi_knapsack()
    for i in range(amount):
        state = simple_state_generator_hyper_ksp(dimension, heuristics_candidates)
        population.append({"heuristics": state, "fitness": fitness_hyper_ksp(state, **kwargs)})
    return population


def mutation_hyper_ksp(state, **kwargs):
    # mutate heuristics order with tabu (reorder tuples)
    if rnd.random() < 0.12:
        shuffle_start_index = rnd.randint(0, len(state) - 5)
        to_shuffle = state[shuffle_start_index:shuffle_start_index + 5]
        rnd.shuffle(to_shuffle)
        state[shuffle_start_index:shuffle_start_index + 5] = to_shuffle

    # mutate heuristics (replace with random)
    if rnd.random() < 0.12:
        heurs_indexes_to_update = rnd.sample(range(len(state)), 5)
        while len(heurs_indexes_to_update) > 0:
            index = heurs_indexes_to_update.pop()
            heurs, tabu = state[index]
            candidate_heurs = rnd.choice(heuristics.get_single_heurs_for_multi_knapsack())
            while candidate_heurs == heurs:
                candidate_heurs = rnd.choice(heuristics.get_single_heurs_for_multi_knapsack())
            state[index] = candidate_heurs, tabu

    # mutate tabu indexes
    if rnd.random() < 0.12:
        tabu_indexes_to_update = rnd.sample(range(len(state)), 5)
        while len(tabu_indexes_to_update) > 0:
            index = tabu_indexes_to_update.pop()
            heurs, tabu = state[index]
            candidate_tabu = rnd.randint(0, 3)
            while candidate_tabu == tabu:
                candidate_tabu = rnd.randint(0, 3)
            state[index] = heurs, candidate_tabu

    # mutate heuristics order without tabu
    if rnd.random() < 0.12:
        shuffle_start_index = rnd.randint(0, len(state) - 5)
        to_shuffle = state[shuffle_start_index:shuffle_start_index + 5]
        rnd.shuffle(to_shuffle)
        to_shuffle = zip(to_shuffle, state[shuffle_start_index:shuffle_start_index + 5])
        to_shuffle = list(map(lambda x: (x[0][0], x[1][1]), to_shuffle))
        state[shuffle_start_index:shuffle_start_index + 5] = to_shuffle

    return state


def minimize(dimension, **kwargs):
    return gene.minimize(dimension, initial_population_generator_hyper_ksp, crossover_reproduction_hyper_ksp,
                         mutation_hyper_ksp, fitness_hyper_ksp, **kwargs)
