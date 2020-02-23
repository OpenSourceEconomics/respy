"""Test model generation."""
import io
import textwrap

import numpy as np
import pandas as pd
import pytest

from respy.config import EXAMPLE_MODELS
from respy.likelihood import get_crit_func
from respy.pre_processing.model_checking import validate_options
from respy.pre_processing.model_processing import _convert_labels_in_formulas_to_codes
from respy.pre_processing.model_processing import _parse_initial_and_max_experience
from respy.pre_processing.model_processing import _parse_measurement_errors
from respy.pre_processing.model_processing import _parse_observables
from respy.pre_processing.model_processing import _parse_shocks
from respy.pre_processing.model_processing import process_params_and_options
from respy.tests.random_model import generate_random_model
from respy.tests.random_model import simulate_truncated_data
from respy.tests.utils import process_model_or_seed


def test_generate_random_model():
    """Test if random model specifications can be simulated and processed."""
    params, options = generate_random_model()

    df = simulate_truncated_data(params, options)

    crit_func = get_crit_func(params, options, df)

    crit_val = crit_func(params)

    assert isinstance(crit_val, float)


@pytest.mark.parametrize("model_or_seed", EXAMPLE_MODELS)
def test_model_options(model_or_seed):
    _, options = process_model_or_seed(model_or_seed)

    _, options = process_params_and_options(_, options)

    validate_options(options)


@pytest.mark.parametrize("model_or_seed", EXAMPLE_MODELS)
def test_sorting_of_type_probability_parameters(model_or_seed):
    # Set configuration for random models.
    n_types = np.random.randint(2, 5)
    n_type_covariates = np.random.randint(2, 4)

    params, options = process_model_or_seed(
        model_or_seed, n_types=n_types, n_type_covariates=n_type_covariates
    )

    optim_paras, options = process_params_and_options(params, options)

    # Update variables if not a random model, but an example model is tested.
    if isinstance(model_or_seed, str):
        n_types = optim_paras["n_types"]
        n_type_covariates = (
            None if n_types == 1 else len(optim_paras["type_covariates"])
        )

    if optim_paras["n_types"] > 1:
        # Resort type probability parameters.
        types = [f"type_{i}" for i in range(1, optim_paras["n_types"])]
        params.loc[types] = params.sort_index(ascending=False).loc[types]

        optim_paras_, _ = process_params_and_options(params, options)

        for (level, coeffs), (level_, coeffs_) in zip(
            optim_paras["type_prob"].items(), optim_paras_["type_prob"].items()
        ):
            assert level == level_
            assert np.all(coeffs == coeffs_)


def test_parse_initial_and_max_experience():
    """Test ensures that probabilities are transformed with logs and rest passes."""
    choices = ["a", "b"]

    options = {"n_periods": 10}
    optim_paras = {"choices_w_exp": choices, "choices": {"a": {}, "b": {}}}
    params = pd.DataFrame(
        {
            "category": [
                "initial_exp_a_0",
                "initial_exp_a_5",
                "initial_exp_b_0",
                "initial_exp_b_5",
                "maximum_exp",
            ],
            "name": ["probability"] * 2 + ["constant"] * 2 + ["b"],
            "value": [2, 2, np.log(2), np.log(2), 5],
        }
    ).set_index(["category", "name"])["value"]

    optim_paras = _parse_initial_and_max_experience(optim_paras, params, options)

    assert (
        optim_paras["choices"]["a"]["start"][0]
        == optim_paras["choices"]["a"]["start"][5]
    ).all()
    assert (
        optim_paras["choices"]["b"]["start"][0]
        == optim_paras["choices"]["b"]["start"][5]
    ).all()
    assert optim_paras["choices"]["a"]["max"] == options["n_periods"] - 1 + max(
        optim_paras["choices"]["a"]["start"]
    )
    assert optim_paras["choices"]["b"]["max"] == 5


def test_normalize_probabilities():
    constraints = {"observables": [3]}
    params, options = generate_random_model(point_constr=constraints)
    optim_paras_1, _ = process_params_and_options(params, options)

    for group in ["initial_exp_edu", "observable_"]:
        mask = params.index.get_level_values(0).str.contains(group)
        params.loc[mask, "value"] = params.loc[mask, "value"].to_numpy() / 2

    optim_paras_2, _ = process_params_and_options(params, options)

    for key in optim_paras_1["choices"]["edu"]["start"]:
        np.testing.assert_array_almost_equal(
            optim_paras_1["choices"]["edu"]["start"][key],
            optim_paras_2["choices"]["edu"]["start"][key],
        )
    for level in optim_paras_1["observables"]["observable_0"]:
        np.testing.assert_array_almost_equal(
            optim_paras_1["observables"]["observable_0"][level],
            optim_paras_2["observables"]["observable_0"][level],
        )


def test_convert_labels_in_covariates_to_codes():
    optim_paras = {
        "choices": ["fishing", "hammock"],
        "observables": {"fishing_grounds": ["poor", "rich"]},
        "choices_w_exp": ["fishing"],
        "exogenous_processes": {},
    }

    options = {
        "covariates": {
            "rich_fishing_grounds": "fishing_grounds == 'rich'",
            "do_fishing": "choice == 'fishing'",
            "do_hammock": 'choice == "hammock"',
        },
        "core_state_space_filters": [],
        "inadmissible_states": {},
    }

    options = _convert_labels_in_formulas_to_codes(options, optim_paras)

    expected = {
        "rich_fishing_grounds": "fishing_grounds == 1",
        "do_fishing": "choice == 0",
        "do_hammock": "choice == 1",
    }

    assert options["covariates"] == expected


def test_parse_observables():
    params = pd.read_csv(
        io.StringIO(
            textwrap.dedent(
                """
                category,name,value
                observable_fishing_grounds_rich_grounds,probability,0.5
                observable_fishing_grounds_poor_grounds,probability,0.5
                observable_ability_low_middle,probability,0.5
                observable_ability_high,probability,0.5
                """
            )
        ),
        index_col=["category", "name"],
    )["value"]
    optim_paras = _parse_observables({}, params)

    expected = {
        "fishing_grounds": {
            "rich_grounds": pd.Series(data=np.log(0.5), index=["constant"]),
            "poor_grounds": pd.Series(data=np.log(0.5), index=["constant"]),
        },
        "ability": {
            "low_middle": pd.Series(data=np.log(0.5), index=["constant"]),
            "high": pd.Series(data=np.log(0.5), index=["constant"]),
        },
    }

    for observable, level_dict in optim_paras["observables"].items():
        for level in level_dict:
            assert optim_paras["observables"][observable][level].equals(
                expected[observable][level]
            )


def test_raise_exception_for_missing_meas_error():
    params, options = generate_random_model()

    params = params.drop(index=("meas_error", "sd_b"))

    with pytest.raises(KeyError):
        _parse_measurement_errors(params, options)


def test_raise_exception_for_missing_shock_matrix():
    params, _ = generate_random_model()

    params = params.drop(index="shocks_sdcorr", level="category")

    with pytest.raises(KeyError):
        _parse_shocks({}, params)


@pytest.mark.parametrize("observables", [[2], [2, 2]])
def test_raise_exception_for_observable_with_one_value(observables):
    point_constr = {"observables": observables}
    params, _ = generate_random_model(point_constr=point_constr)

    params = params.drop(index="observable_observable_0_0", level="category")["value"]

    with pytest.raises(ValueError, match=r"Observables and exogenous processes"):
        _parse_observables({}, params)
