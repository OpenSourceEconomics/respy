"""This module allows to transform a transition matrix of a exogenous process.

The output are a subset of the params and options object used in respy. The
probabilities are transformed to logit coefficents.
"""
import itertools

import numpy as np
import pandas as pd


def parse_transition_matrix_for_exogenous_processes(matrix, process_name):
    """Parse the transition matrix for an exogenous process to respy layout.

    Parse a transition matrix to logit coefficients and create params file and
    covariates template for the respy initial files.
    Parameters
    ----------
    matrix : pandas.DataFrame
        The transition matrix for an exogenous process.
    process_name : string
        The name of the exogenous process.
    Returns
    -------
    params : pandas.DataFrame
        A subset of the params initial file for respy. Contains all information on
        the exogenous process.
    covariates : dictionary
        A template for the specification of the covariates required for the exogenous
        process.

    """
    # Check transition matrix conditions
    check_numerics(matrix)

    states = matrix.index
    process_states = matrix.columns
    # Create the covariates template
    covariates = create_covariates(states, process_name, process_states)
    # Create logit values
    transformed_matrix = transform_matrix(matrix)
    # Finally transfer to the params layout
    params = create_params(transformed_matrix, states, process_name, process_states)
    return params, covariates


def create_params(transformed_matrix, states, process_name, process_states):
    """Create the params entries for the respy initial files.

    Create entries with logit coefficents instead of probabilities.
    Parameters
    ----------
    transformed_matrix : pandas.DataFrame
        The original DataFrame converted to logit values.
    states : list
        A list of the states on which probabilities for the exogenous process are
        definded.
    process_name : string
        The name of the exogenous process.
    process_states : list
        A list of the possible outcomes of the exogenous process.

    Returns
    -------
    params : pandas.DataFrame
        A subset of the params initial file for respy. Contains all information on
        the exogenous process.
    """
    # Create categories for the different states of the process
    categories = [
        f"exogenous_process_{process_name}_{state}" for state in process_states
    ]
    index = pd.MultiIndex.from_tuples(
        itertools.product(categories, states), names=["category", "name"]
    )
    params = pd.DataFrame(index=index, columns=["value"])
    # Transfer values
    for process_state in process_states:
        params.loc[
            (f"exogenous_process_{process_name}_{process_state}", states), "value"
        ] = transformed_matrix[process_state].values
    return params


def transform_matrix(matrix):
    """Transform probabilities to logit coefficients.

    Parameters
    ----------
    matrix : pandas.DataFrame
        The transition matrix for an exogenous process.
    Returns
    -------
    transformed_matrix : pandas.DataFrame
        The original DataFrame converted to logit values.

    """
    transformed_matrix = matrix.copy()
    # Asign large negative numbers for zero probabilities
    transformed_matrix[transformed_matrix == 0] = -1e300
    transformed_matrix[transformed_matrix > 0] = np.log(
        transformed_matrix[transformed_matrix > 0]
    )
    transformed_matrix.index = transformed_matrix.index.map(str)
    return transformed_matrix


def create_covariates(states, process_name, process_states):
    """Create a covariate template for the user.

    Parameters
    ----------
    states : list
        A list of the states on which probabilities for the exogenous process are
        definded.
    process_name : string
        The name of the exogenous process.
    process_states : list
        A list of the possible outcomes of the exogenous process.

    Returns
    -------
    covariates : dictionary
        A template for the specification of the covariates required for the exogenous
        process.

    """
    covariates = {}
    for state in states:
        i = 0
        for process_state in process_states:
            if process_state in state:
                destination_state = process_state
                i += 1
        # If state is independent of exogenous process, return without any specification
        if i == 0:
            covariates[str(state)] = "?"
        elif i == 1:
            # If only dependent on exogenous process, return full covariate statement
            if state in process_states:
                covariates[str(state)] = f"{process_name} == {destination_state}"
            # If dependent on exogenous process and covariate, return template
            else:
                covariates[str(state)] = f"{process_name} == {destination_state} & ?"
        else:
            raise ValueError(
                f"{state} contains more than one states of the exogenous " f"process."
            )
    return covariates


def check_numerics(matrix):
    """Check numeric conditions on a transition matrix.

    Parameters
    ----------
    matrix : pandas.DataFrame
        The transition matrix for an exogenous process.
    Returns
    -------
    Nothing. Only raises errors if conditions are not fullfilled.
    """
    n_states = matrix.shape[0]
    # Rows should sum to 1
    if not np.allclose(matrix.sum(axis=1), np.full(n_states, 1)):
        raise ValueError("The rows of the transition matrix don't sum to 1.")
    # Probabilities are between 0 and 1
    if not (((matrix >= 0) & (matrix <= 1)).all()).all():
        raise ValueError("The values of the matrix are not between 0 and 1.")
