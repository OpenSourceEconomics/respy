"""
Test the msm interface of respy!
"""
import respy as rp
import numpy as np
import pytest

from respy.pre_processing.model_processing import process_params_and_options
from respy.tests.random_model import get_mock_moment_func
from respy.method_of_simulated_moments import get_msm_func


@pytest.fixture()
def msm_input():
    params, options, _ = rp.get_example_model("kw_94_one")
    simulate = rp.get_simulate_func(params, options)
    df = simulate(params)
    optim_paras, _ = process_params_and_options(params, options)
    get_moments = get_mock_moment_func(df, optim_paras)
    moments_base = get_moments(df)
    return params, options, moments_base, get_moments


@pytest.mark.parametrize(range(5))
def test_msm_base(msm_input):
    msm = get_msm_func(msm_input[0], msm_input[1],msm_input[2],msm_input[3])
    rslt = msm(msm_input[0])
    assert rslt == 0


@pytest.mark.parametrize(range(5))
def test_msm_random(msm_input):
    msm_input[1]["simulation_seed"] = msm_input[1]["simulation_seed"] + 5235
    msm = get_msm_func(msm_input[0], msm_input[1],msm_input[2],msm_input[3])
    rslt = msm(msm_input[0])
    assert rslt != 0


@pytest.mark.parametrize(range(5))
def test_msm_vec(msm_input):
    msm = get_msm_func(msm_input[0], msm_input[1],msm_input[2],msm_input[3], all_dims=True)
    rslt = msm(msm_input[0])
    np.testing.assert_array_equal(rslt, np.zeros(len(msm_input[2])))


@pytest.mark.parametrize(range(5))
def test_msm_missing(msm_input):
    msm_input[1]["n_periods"] = msm_input[1]["n_periods"] - 1
    msm = get_msm_func(msm_input[0], msm_input[1],msm_input[2],msm_input[3], all_dims=True)
    rslt = msm(msm_input[0])
    np.testing.assert_array_equal(rslt, np.zeros(len(msm_input[2])-4))

