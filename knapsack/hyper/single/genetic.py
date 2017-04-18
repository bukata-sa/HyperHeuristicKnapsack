import random as rnd

import algorithms.genetic as gene
import knapsack.hyper.single.heurs1knpsck as heurs
import knapsack.hyper.single.problem as problem


def simple_state_generator_hyper_ksp(dimension, heuristics_candidates):
    state = []
    while len(state) < dimension:
        probability = rnd.random()
        tabu_generation = rnd.randint(3, 6) if probability < 0.1 else 0
        probability = rnd.random()
        cumulative_probability = 0
        for index, heuristics_candidate in enumerate(heuristics_candidates):
            cumulative_probability += heuristics_candidate[1]
            if probability <= cumulative_probability:
                break
        state.append((heuristics_candidates[index][0], tabu_generation))
    return state


def initial_population_generator_hyper_ksp(amount, dimension, state_generator=simple_state_generator_hyper_ksp,
                                           **kwargs):
    population = []
    heuristics_candidates = heurs.get_all_single_heuristics()
    while len(population) < amount:
        state = state_generator(dimension, heuristics_candidates)
        population.append({"heuristics": state, "fitness": fitness_hyper_ksp(state, **kwargs)})
    return population


def crossover_reproduction_hyper_ksp(first_parent, second_parent, **kwargs):
    # TODO: try other genetic operators (different types of crossover, for example)
    # nor first allel nor last allel
    crossover_index = rnd.randint(1, len(first_parent) - 1)
    child_candidate = first_parent[:crossover_index + 1] + second_parent[crossover_index + 1:]
    return child_candidate


def mutation_hyper_single_ksp(state, **kwargs):
    return mutation_hyper_ksp(state, heurs, **kwargs)


# probability 0.6 that state won't be changed
# probability 0.4 that at least one mutation will be applied
def mutation_hyper_ksp(state, heuristic_source, **kwargs):
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
            candidate_heurs = rnd.choice(heuristic_source.get_heuristics())[0]
            while candidate_heurs == heurs:
                candidate_heurs = rnd.choice(heuristic_source.get_heuristics())[0]
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


def fitness_hyper_ksp(state, **kwargs):
    included = list(kwargs["included"])
    tabooed_items_generations = [0] * len(included)
    for operation, tabu_generation in state:
        tabooed_items = [index for index, generation in enumerate(tabooed_items_generations) if generation > 0]
        included, modified_index = operation(included, tabooed_indexes=tabooed_items, costs=kwargs["costs"],
                                             weights=kwargs["weights"], size=kwargs["size"])
        tabooed_items_generations = list(map(lambda x: x - 1 if x > 0 else 0, tabooed_items_generations))

        # TODO if modified index is -1, should we exclude the operation from the state?
        if tabu_generation > 0 and modified_index >= 0:
            tabooed_items_generations[modified_index] = tabu_generation
    return problem.solve(included, costs=kwargs["costs"], weights=kwargs["weights"], size=kwargs["size"])


def minimize(dimension, **kwargs):
    return gene.minimize(dimension, initial_population_generator_hyper_ksp, crossover_reproduction_hyper_ksp,
                         mutation_hyper_single_ksp, fitness_hyper_ksp, **kwargs)


if __name__ == '__main__':
    size = 165
    weights = [23, 31, 29, 44, 53, 38, 63, 85, 89, 82]
    costs = [92, 57, 49, 68, 60, 43, 67, 84, 87, 72]
    start = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    optimal = [1, 1, 1, 1, 0, 1, 0, 0, 0, 0]
    heuristics = [(heurs.add_least_cost, 2), (heurs.add_least_cost, 1), (heurs.remove_least_cost, 0)]
    print(fitness_hyper_ksp(heuristics, weights=weights, costs=costs,
                            size=size, included=start))
