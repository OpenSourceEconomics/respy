"""This module includes test to specifically test that randomness is held constant."""
import numpy as np
import pytest

import respy as rp
from respy.config import EXAMPLE_MODELS
from respy.tests.random_model import generate_random_model


@pytest.mark.parametrize("model_or_seed", EXAMPLE_MODELS + list(range(5)))
def test_invariance_of_model_solution_in_solve_and_simulation(model_or_seed):
    if isinstance(model_or_seed, str):
        params_spec, options_spec = rp.get_example_model(model_or_seed)
    else:
        np.random.seed(model_or_seed)
        params_spec, options_spec = generate_random_model()

    state_space = rp.solve(params_spec, options_spec)
    state_space_, _ = rp.simulate(params_spec, options_spec)

    np.array_equal(state_space.states, state_space_.states)
    np.array_equal(state_space.covariates, state_space_.covariates)
    np.array_equal(state_space.rewards, state_space_.rewards)
    np.array_equal(state_space.emaxs, state_space_.emaxs)
    np.array_equal(state_space.base_draws_sol, state_space_.base_draws_sol)


@pytest.mark.parametrize("model_or_seed", EXAMPLE_MODELS + list(range(5)))
def test_invariance_of_model_solution_in_solve_and_crit_func(model_or_seed):
    if isinstance(model_or_seed, str):
        params_spec, options_spec = rp.get_example_model(model_or_seed)
    else:
        np.random.seed(model_or_seed)
        params_spec, options_spec = generate_random_model()

    state_space = rp.solve(params_spec, options_spec)

    _, df = rp.simulate(params_spec, options_spec)

    crit_func = rp.get_crit_func(params_spec, options_spec, df)
    crit_func(params_spec)

    state_space_ = crit_func.keywords["state_space"]

    np.array_equal(state_space.states, state_space_.states)
    np.array_equal(state_space.covariates, state_space_.covariates)
    np.array_equal(state_space.rewards, state_space_.rewards)
    np.array_equal(state_space.emaxs, state_space_.emaxs)
    np.array_equal(state_space.base_draws_sol, state_space_.base_draws_sol)
