import random as rand
from math import exp, sqrt

from knapsack.hyper.single.problem import solve, validate


def temperature_ksp(t, iteration):
    return sqrt(t)


def change_state_candidate_ksp(validator, seq, **kwargs):
    while True:
        copy = list(seq)
        n = len(copy)
        position_to_invert = rand.randint(0, n - 1)
        copy[position_to_invert] = not copy[position_to_invert]
        if validator(copy, **kwargs):
            return copy


def simple_probability_change(current_state, candidate_state, delta_energy, temperature):
    value = rand.random()

    if value <= exp(-delta_energy / temperature):
        return current_state
    else:
        return candidate_state


def energy_calculator_ksp(seq, **kwargs):
    profit = solve(seq, **kwargs)
    return 1 / profit if profit > 0 else float("inf")


def minimize(initial_state, tmin, tmax,
             energy_calculator=energy_calculator_ksp,
             change_state_candidate=change_state_candidate_ksp,
             validator=validate,
             temperature_change=temperature_ksp,
             probability_change=simple_probability_change,
             **kwargs):
    t = tmax
    current_state = initial_state
    i = 1
    while t > tmin:
        candidate_state = change_state_candidate(validator, current_state, **kwargs)
        delta_energy = energy_calculator(candidate_state, **kwargs) - energy_calculator(current_state, **kwargs)
        if delta_energy <= 0:
            current_state = candidate_state
        else:
            current_state = probability_change(current_state, candidate_state, delta_energy, t)
        t = temperature_change(t, i)
        i += 1
    return current_state
