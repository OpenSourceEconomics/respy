"""This module allows to transform a transition matrix of a exogenous process.

The output are a subset of the params and options object used in respy. The
probabilities are transformed to logit coefficents.
"""
import itertools

import numpy as np
import pandas as pd


def parse_transition_matrix_for_exog_processes(matrix, process_name):
    """Parse the transition matrix for an exogenous process to respy layout.

    Parse a transition matrix to logit coefficients and create params file and
    covariates template for the respy initial files.
    :param matrix:
    :param process_name:
    :return:
    """
    # Check transition matrix conditions
    check_numerics(matrix)

    states = matrix.index
    process_states = matrix.columns
    covariates = create_covariates(states, process_name, process_states)
    transformed_matrix = transform_matrix(matrix)
    params = create_params(transformed_matrix, states, process_name, process_states)
    return params, covariates


def create_params(transformed_matrix, states, process_name, process_states):
    """Create the params entries for the respy initial files.

    Create entries with logit coefficents instead of probabilities.
    :param transformed_matrix:
    :param states:
    :param process_name:
    :param process_states:
    :return:
    """
    categories = [
        f"exogenous_process_{process_name}_{state}" for state in process_states
    ]
    index = pd.MultiIndex.from_tuples(
        itertools.product(categories, states), names=["category", "name"]
    )
    params = pd.DataFrame(index=index, columns=["value"])
    for process_state in process_states:
        params.loc[
            (f"exogenous_process_{process_name}_{process_state}", states), "value"
        ] = transformed_matrix[process_state].values
    return params


def transform_matrix(matrix):
    """Transform probabilities to logit coefficients.

    :param matrix:
    :return:
    """
    transformed_matrix = matrix.copy()
    transformed_matrix[transformed_matrix == 0] = -1e300
    transformed_matrix[transformed_matrix > 0] = np.log(
        transformed_matrix[transformed_matrix > 0]
    )
    transformed_matrix.index = transformed_matrix.index.map(str)
    return transformed_matrix


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


def check_numerics(matrix):
    """Check numeric conditions on a transition matrix.

    :param matrix:
    :return:
    """
    n_states = matrix.shape[0]
    if not np.allclose(matrix.sum(axis=1), np.full(n_states, 1)):
        raise ValueError("The rows of the transition matrix don't sum to 1.")
    if not (((matrix >= 0) & (matrix <= 1)).all()).all():
        raise ValueError("The values of the matrix are not between 0 and 1.")
