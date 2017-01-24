import knapsack.problem as ksp


def initial_population_generator_hyper_ksp(amount, dim, validator=validate,
                                           state_generator=simple_state_generator_ksp,
                                           **kwargs):
    pass


def crossover_selection_hyper_ksp(population, fitness_func,
                                  validate_func=ksp.validate,
                                  compare_state=compare_state_ksp,
                                  **kwargs):
    pass
