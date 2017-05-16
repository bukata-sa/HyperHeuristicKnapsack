import itertools
import operator
import random as rnd
import time
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


def repair_func(child, **kwargs):
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


def pick_heuristic_random(heuristics_candidates):
    heuristics_candidates_items, heuristics_candidates_probabilities = zip(*heuristics_candidates)
    operation = np.random.choice(list(heuristics_candidates_items), p=list(heuristics_candidates_probabilities))
    return operation


class ChoiceFunction:
    __choice_function1 = {}
    __choice_function2 = {}
    __choice_function3 = {}

    def __init__(self, heuristic_candidates):
        self.__heuristic_candidates = heuristic_candidates
        self.alpha = 0.4
        self.beta = 0.4
        self.gamma = 0.2

    def calculate_heuristic_rang(self, heuristic):
        func1_value = self.__choice_function1.get(heuristic, 0)
        func2_value = self.__choice_function2.get(heuristic, 0)
        func3_value = self.__choice_function3.get(heuristic, 10000)
        return heuristic[0], self.alpha * func1_value + self.beta * func2_value + self.gamma * func3_value

    def pick_heuristic(self):
        heuristics_priorities = map(self.calculate_heuristic_rang, self.__heuristic_candidates)
        heuristics_priorities = list(sorted(heuristics_priorities, key=operator.itemgetter(1)))
        best_heuristic_priority = heuristics_priorities[-1][1]
        top_priority_heuristics = [heuristic[0] for heuristic in heuristics_priorities if
                                   heuristic[1] == best_heuristic_priority]
        return np.random.choice(top_priority_heuristics)

    def recalculate_choice_functions(self, heuristic, previous_heuristic, time_taken, heuristic_impact):
        self.__choice_function1[heuristic] = self.__choice_function1.get(heuristic, 0) * self.alpha + heuristic_impact

        if previous_heuristic is not None:
            self.__choice_function2[(previous_heuristic, heuristic)] = \
                self.__choice_function2.get((previous_heuristic, heuristic), 0) * self.beta + heuristic_impact

        self.__choice_function3.update(
            {heuristic: iterations + 1 for heuristic, iterations in self.__choice_function3.items()})
        self.__choice_function3[heuristic] = 0


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
    previous_knapsack_fitness = 0

    heuristic_picker = ChoiceFunction(heuristics_candidates)
    while any(included < 1):
        operation = heuristic_picker.pick_heuristic()
        tabooed_items = (index for index, generation in enumerate(tabooed_items_generations) if generation > 0)
        current_time = time.perf_counter()
        included, modified_index = operation(included, tabooed_indexes=tabooed_items, costs=kwargs["costs"],
                                             weights=kwargs["weights"], sizes=kwargs["sizes"])
        heuristic_completion_time = time.perf_counter() - current_time

        current_knapsack_fitness = problem.solve(included, **shortened_kwargs)
        heuristic_picker.recalculate_choice_functions(operation, state[-1] if len(state) > 0 else None,
                                                      heuristic_completion_time,
                                                      current_knapsack_fitness - previous_knapsack_fitness)
        previous_knapsack_fitness = current_knapsack_fitness

        state.append(operation)
        tabooed_items_generations = list(map(lambda x: x - 1 if x > 0 else 0, tabooed_items_generations))

        if local_max_reached(weights, included, sizes, tabooed_items_generations):
            current_max_state_fitness = problem.solve(included, **shortened_kwargs)
            if current_max_state_fitness > max_state_fitness:
                max_state_fitness = current_max_state_fitness
                max_state = state[:]

            max_reached_times += 1
            if max_reached_times == 50:
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
    childs = list(map(cross_child_partial, *zip(*itertools.permutations(candidates, 2))))
    return childs


def cross_child(child1, child2, **kwargs):
    crossover_point = min(len(child1["heuristics"]), len(child2["heuristics"])) // 2 - 1
    child = child1["heuristics"][:crossover_point] + child2["heuristics"][crossover_point + 1:]
    return child


def mutation_hyper_multi_ksp(state, **kwargs):
    return mutation_hyper_ksp(state, heuristics)


def minimize(**kwargs):
    return gene.minimize(initial_population_generator_hyper_ksp_multiproc, selection_hyper_ksp,
                         crossover_reproduction_hyper_ksp, mutation_hyper_multi_ksp, repair_func,
                         fitness_hyper_ksp, **kwargs)
