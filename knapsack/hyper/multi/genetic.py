from algorithms import genetic as gene


def fitness_hyper_ksp(state, **kwargs):
    pass


def simple_state_generator_hyper_ksp(dimension):
    pass


def crossover_reproduction_hyper_ksp(first_parent, second_parent, **kwargs):
    pass


def initial_population_generator_hyper_ksp(amount, dimension, **kwargs):
    pass


def mutation_hyper_ksp(state, **kwargs):
    pass


def minimize(dimension, **kwargs):
    return gene.minimize(dimension, initial_population_generator_hyper_ksp, crossover_reproduction_hyper_ksp,
                         mutation_hyper_ksp, fitness_hyper_ksp, **kwargs)
