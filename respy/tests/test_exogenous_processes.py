import numpy as np
import pandas as pd
import pytest

from respy.interface import get_example_model
from respy.simulate import get_simulate_func
from respy.solve import get_solve_func


@pytest.fixture
def model_with_one_exog_proc():
    params, options = get_example_model("robinson_crusoe_basic", with_data=False)
    params.loc[("nonpec_fishing", "sick"), "value"] = -2
    params.loc[("observable_illness_sick", "probability"), "value"] = 0.1
    params.loc[("observable_illness_healthy", "probability"), "value"] = 0.9
    params.loc[("exogenous_process_illness_sick", "probability"), "value"] = 0.1
    params.loc[("exogenous_process_illness_healthy", "probability"), "value"] = 0.9
    options["covariates"]["sick"] = "illness == 'sick'"

    return params, options


@pytest.fixture
def model_with_two_exog_proc(model_with_one_exog_proc):
    params, options = model_with_one_exog_proc
    params.loc[("nonpec_fishing", "stormy"), "value"] = -1
    params.loc[("observable_weather_stormy", "probability"), "value"] = 0.2
    params.loc[("observable_weather_normal", "probability"), "value"] = 0.8
    params.loc[("exogenous_process_weather_stormy", "probability"), "value"] = 0.2
    params.loc[("exogenous_process_weather_normal", "probability"), "value"] = 0.8
    options["covariates"]["stormy"] = "weather == 'stormy'"

    return params, options


def test_transition_probabilities_for_one_exogenous_process(model_with_one_exog_proc):
    params, options = model_with_one_exog_proc

    simulate = get_simulate_func(params, options)
    df = simulate(params)

    df["Prev_Illness"] = df.groupby("Identifier")["Illness"].shift()
    probs = pd.crosstab(df["Illness"], df["Prev_Illness"], normalize=True)

    assert np.allclose(probs, [[0.81, 0.09], [0.09, 0.01]], atol=0.01)


def test_transition_probabilities_for_two_exogenous_processes(model_with_two_exog_proc):
    params, options = model_with_two_exog_proc

    simulate = get_simulate_func(params, options)
    df = simulate(params)

    df["Prev_Illness"] = df.groupby("Identifier")["Illness"].shift()
    probs = pd.crosstab(df["Illness"], df["Prev_Illness"], normalize=True)
    assert np.allclose(probs, [[0.81, 0.09], [0.09, 0.01]], atol=0.01)

    df["Prev_Weather"] = df.groupby("Identifier")["Weather"].shift()
    probs = pd.crosstab(df["Weather"], df["Prev_Weather"], normalize=True)
    assert np.allclose(probs, [[0.64, 0.16], [0.16, 0.04]], atol=0.01)


def test_weight_continuation_values_for_one_exogenous_process(model_with_one_exog_proc):
    params, options = model_with_one_exog_proc

    solve = get_solve_func(params, options)
    state_space = solve(params)

    state_space.get_attribute("expected_value_functions")[(0,)][:] = 100
    state_space.get_attribute("expected_value_functions")[(1,)][:] = 20

    for period in range(options["n_periods"] - 1):
        continuation_values = state_space.get_continuation_values(period=period)
        assert np.allclose(continuation_values[(0,)], 92)
        assert np.allclose(continuation_values[(1,)], 92)
