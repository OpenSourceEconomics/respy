"""
This module contains tests for felx choices!
"""
import pandas as pd
import pytest

import respy as rp
from respy.pre_processing.model_processing import process_params_and_options
from respy.simulate import get_simulate_func
from respy.state_space import create_state_space_class


@pytest.mark.integration
def test_choice_restrictions():
    """
    Basic first test!
    """
    # Load model.
    params, options = rp.get_example_model("robinson_crusoe_extended", with_data=False)

    # Extend with observable characteristic.
    params.loc[("observable_health_well", "probability"), "value"] = 0.9
    params.loc[("observable_health_sick", "probability"), "value"] = 0.1

    # Sick people can never work.
    options["inadmissible_states"] = {
        "fishing": ["health == 'sick' & period < 2", "health == 'sick' & period >= 2"],
        "friday": ["period < 2", "exp_fishing == 0"],
    }
    # Create internal specification objects.
    optim_paras, options = process_params_and_options(params, options)

    sp = create_state_space_class(optim_paras, options)
    check = sp.dense_index_to_complex

    for x in check.values():
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
    params, options = rp.get_example_model("robinson_crusoe_extended", with_data=False)

    # Extend with observable characteristic.
    params.loc[("observable_health_well", "probability"), "value"] = 0.9
    params.loc[("observable_health_sick", "probability"), "value"] = 0.1

    # Sick people can never work.
    options["inadmissible_states"] = {
        "fishing": ["health == 'sick'"],
        "friday": ["period < 2", "exp_fishing == 0"],
    }
    # Create internal specification objects.
    optim_paras, options = process_params_and_options(params, options)
    simulate = get_simulate_func(params, options)
    df = simulate(params)

    assert isinstance(df, pd.DataFrame)
