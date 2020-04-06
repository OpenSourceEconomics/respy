"""
This module contains tests for felx choices!
"""
import pandas as pd
import respy as rp

from respy.pre_processing.model_processing import process_params_and_options
from respy.state_space import _create_core_and_indexer, create_state_space_class
from respy.solve import _create_choice_rewards


def test_period_choice_dense_cores():
    """
    Basic first test!
    """
    # Load model.
    params, options = rp.get_example_model("robinson_crusoe_extended", with_data=False)

    # Extend with observable characteristic.
    params.loc[("observable_health_well", "probability"), "value"] = 0.9
    params.loc[("observable_health_sick", "probability"), "value"] = 0.1

    # Sick people can never work.
    options["inadmissible_choices"] = {
        "fishing": ["health == 1 & period < 2", "health == 1 & period >= 2"],
        "friday": ["period < 2", "exp_fishing == 0"],
    }
    # Create internal specification objects.
    optim_paras, options = process_params_and_options(params, options)

    sp = create_state_space_class(options, optim_paras)
    check = sp.period_choice_cores

    for x in check[(1,)].keys():
        if x[0] < 2:
            assert x[1] == (False, False, True)
        else:
            assert x[1] in [(False, False, True), (False, True, True)]

    for x in check[(0,)].keys():
        if x[0] < 2:
            assert x[1] == (True, False, True)
        else:
            assert x[1] in [(True, False, True), (True, True, True)]


def test_robustness_solution():
    params, options = rp.get_example_model("robinson_crusoe_extended", with_data=False)

    # Extend with observable characteristic.
    params.loc[("observable_health_well", "probability"), "value"] = 0.9
    params.loc[("observable_health_sick", "probability"), "value"] = 0.1

    # Sick people can never work.
    options["inadmissible_choices"] = {
        "fishing": ["health == 1"],
        "friday": ["period < 2", "exp_fishing == 0"],
    }
    # Create internal specification objects.
    optim_paras, options = process_params_and_options(params, options)

    state_space = create_state_space_class(options, optim_paras)

    states = state_space.states
    period_choice_cores = state_space.period_choice_cores

    wages, nonpecs = _create_choice_rewards(states, period_choice_cores, optim_paras)
    state_space.set_attribute("wages", wages)


def test_robustness_mixed_constrainst():
    """
    Test whether mixed constraints work for now!
    Will write more general! Test in the future:
    Write mixed cosntriants that are equivalent to core constriants and
    check for equality!

    """
    params, options = rp.get_example_model("robinson_crusoe_extended", with_data=False)

    # Extend with observable characteristic.
    params.loc[("observable_health_well", "probability"), "value"] = 0.9
    params.loc[("observable_health_sick", "probability"), "value"] = 0.1

    # Sick people can never work.
    options["inadmissible_choices"] = {
        "fishing": ["health == 1 & period < 2", "health == 1 & period >= 2"],
        "friday": ["period < 2", "exp_fishing == 0"],
    }
    simulate = rp.get_simulate_func(params, options)
    df_mixed = simulate(params)

    # Equivalent model
    params, options = rp.get_example_model("robinson_crusoe_extended", with_data=False)

    # Extend with observable characteristic.
    params.loc[("observable_health_well", "probability"), "value"] = 0.9
    params.loc[("observable_health_sick", "probability"), "value"] = 0.1

    options["inadmissible_choices"] = {
        "fishing": ["health == 1"],
        "friday": ["period < 2", "exp_fishing == 0"],
    }
    simulate = rp.get_simulate_func(params, options)
    df_core = simulate(params)

    pd.testing.assert_series_equal(df_core["Choice"], df_mixed["Choice"])


def test_equality_solution():
    """
    Rough take at adaptation of reg tests! 
    """
    pass
