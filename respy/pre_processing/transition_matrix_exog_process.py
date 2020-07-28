"""This module allows to transform a transition matrix of a exogenous process.

The output are a subset of the params and options object used in respy. The
probabilities are transformed to logit coefficents.
"""
import numpy as np


def create_covariates(states, process_name, process_states):
    """Create a covariate template for the user.

    :param states:
    :param process_name:
    :param process_states:
    :return:
    """
    covariates = {}
    for state in states:
        i = 0
        for process_state in process_states:
            if process_state in state:
                destination_state = process_state
                i += 1
        if i == 0:
            covariates[str(state)] = "?"
        elif i == 1:
            covariates[str(state)] = f"{process_name} == {destination_state} & ?"
        else:
            raise ValueError(f"{state} contains more than one process state.")
    return covariates


def check_numerics(matrix, n_states):
    """Check numeric conditions on a transition matrix.

    :param matrix:
    :param n_states:
    :return:
    """
    if not np.allclose(matrix.sum(axis=1), np.full(n_states, 1)):
        raise ValueError("The rows of the transition matrix don't sum to 1.")
    if not (((matrix >= 0) & (matrix <= 1)).all()).all():
        raise ValueError("The values of the matrix are not between 0 and 1.")
