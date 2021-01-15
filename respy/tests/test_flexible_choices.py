"""Contains code for testing for flexible choices."""
import pandas as pd
import pytest

import respy as rp
from respy.pre_processing.model_processing import process_params_and_options
from respy.simulate import get_simulate_func
from respy.state_space import _create_dense_period_choice
from respy.state_space import create_state_space_class
from respy.tests.utils import process_model_or_seed


@pytest.mark.integration
def test_choice_restrictions():
    """Basic first test."""
    # Load model.
    params, options = process_model_or_seed("robinson_crusoe_extended")

    # Extend with observable characteristic.
    params.loc[("observable_health_well", "probability"), "value"] = 0.9
    params.loc[("observable_health_sick", "probability"), "value"] = 0.1

    # Sick people can never work.
    options["negative_choice_set"] = {
        "fishing": ["health == 'sick' & period < 2", "health == 'sick' & period >= 2"],
        "friday": ["period < 2", "exp_fishing == 0"],
    }
    # Create internal specification objects.
    optim_paras, options = process_params_and_options(params, options)

    state_space = create_state_space_class(optim_paras, options)

    for x in state_space.dense_key_to_complex.values():
        if (x[0] < 2) & (x[2] == (0,)):
            assert x[1] == (False, False, True)
        elif x[2] == (0,):
            assert x[1] in [(False, False, True), (False, True, True)]
        elif (x[0] < 2) & (x[2] == (1,)):
            assert x[1] == (True, False, True)
        elif x[2] == (1,):
            assert x[1] in [(True, False, True), (True, True, True)]


@pytest.mark.end_to_end
def test_simulation_with_flexible_choice_sets():
    params, options = process_model_or_seed("robinson_crusoe_basic")

    # Extend with observable characteristic.
    params.loc[("observable_health_well", "probability"), "value"] = 0.9
    params.loc[("observable_health_sick", "probability"), "value"] = 0.1

    # Sick people can never work.
    options["negative_choice_set"] = {
        "fishing": ["health == 'sick'"],
        "friday": ["period < 2", "exp_fishing == 0"],
    }
    # Create internal specification objects.
    optim_paras, options = process_params_and_options(params, options)
    simulate = get_simulate_func(params, options)
    df = simulate(params)

    assert isinstance(df, pd.DataFrame)


def test_dense_period_choice():
    params, options = rp.get_example_model("kw_94_one", with_data=False)
    options["negative_choice_set"] = {}
    options["negative_choice_set"]["b"] = ["period < 5"]

    optim_paras, options = process_params_and_options(params, options)
    state_space = create_state_space_class(optim_paras, options)

    check = _create_dense_period_choice(
        state_space.core,
        state_space.dense,
        state_space.core_key_to_core_indices,
        state_space.core_key_to_complex,
        optim_paras,
        options,
    )

    for key in check:
        if key[0] < 5:
            assert ~key[1][1]
