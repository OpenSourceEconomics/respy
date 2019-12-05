"""
Test the msm interface of respy!
"""
import respy as rp
import numpy as np

from respy.pre_processing.model_processing import process_params_and_options
from respy.tests.random_model import get_mock_moment_func
from respy.method_of_simulated_moments import get_msm_func

def test_msm_base():
    params, options,_ = rp.get_example_model("kw_94_one")
    simulate = rp.get_simulate_func(params, options)
    df = simulate(params)
    optim_paras, _ = process_params_and_options(params, options)
    get_moments = get_mock_moment_func(df, optim_paras)
    moments_base = get_moments(df)
    msm = get_msm_func(params, options, moments_base, get_moments)
    rslt = msm(params)
    assert rslt == 0

def test_msm_vec():
    params, options,_ = rp.get_example_model("kw_94_one")
    simulate = rp.get_simulate_func(params, options)
    df = simulate(params)
    optim_paras, _ = process_params_and_options(params, options)
    get_moments = get_mock_moment_func(df, optim_paras)
    moments_base = get_moments(df)
    msm = get_msm_func(params, options, moments_base, get_moments, all_dims=True)
    rslt = msm(params)
    np.testing.assert_array_equal(rslt, np.zeros(len(moments_base)))
