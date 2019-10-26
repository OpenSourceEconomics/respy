import functools

import numpy as np

from respy.pre_processing.model_processing import process_params_and_options
from respy.simulate import get_simulate_func


def get_smm_func(params, options, moments, calc_moments, weighting_matrix=None):
    optim_paras, options = process_params_and_options(params, options)

    simulate = get_simulate_func(params, options)

    if weighting_matrix is None:
        weighting_matrix = np.eye(len(moments))

    smm_function = functools.partial(
        smm,
        simulate=simulate,
        moments=moments,
        weighting_matrix=weighting_matrix,
        calc_moments=calc_moments,
        options=options,
    )

    return smm_function


def smm(params, simulate, moments, weighting_matrix, calc_moments, options):
    df = simulate(params)

    estimated_moments = calc_moments(df)

    # Do we want more methods? E.g. percentage errors which work great if non of the
    # real moments is zero.
    moments_error = estimated_moments - moments

    return moments_error @ weighting_matrix @ moments_error
