"""Utils for the inclusion of exogenous processes."""
import functools

import numpy as np
import pandas as pd
from scipy import special

from respy.parallelization import parallelize_across_dense_dimensions
from respy.shared import get_exogenous_from_dense_covariates
from respy.shared import load_objects
from respy.shared import pandas_dot


def compute_transition_probabilities(
    states, transit_keys, optim_paras, dense_key_to_dense_covariates
):
    """Insert Docstring."""
    exogenous_processes = optim_paras["exogenous_processes"]

    dense_key_to_exogenous = {
        key: get_exogenous_from_dense_covariates(
            dense_key_to_dense_covariates[key], optim_paras
        )
        for key in transit_keys
    }

    # Compute the probabilities for every exogenous process.
    probabilities = []
    for exog_proc in exogenous_processes:

        # Create the dot product of covariates and parameters.
        x_betas = []
        for params in exogenous_processes[exog_proc].values():
            x_beta = pandas_dot(states[params.index], params)
            x_betas.append(x_beta)

        probs = special.softmax(np.column_stack(x_betas), axis=1)
        probabilities.append(probs)

    # Prepare full Dataframe. If issues arrise we might want to switch typed dicts
    df = pd.DataFrame(index=states.index)

    for dense in dense_key_to_exogenous:
        array = functools.reduce(
            np.multiply,
            [
                probabilities[proc][:, val]
                for proc, val in enumerate(dense_key_to_exogenous[dense])
            ],
        )
        df[str(dense)] = array

    return df


@parallelize_across_dense_dimensions
def weight_continuation_values(
    complex_, options, continuation_values, transit_key_to_choice_set
):
    """Insert Docstring."""
    transition_df = load_objects("transition", complex_, options)

    choice_set = complex_[1]
    # Reconsider this setup
    choice_positons = {
        key: _get_representation_cols(value, choice_set)
        for key, value in transit_key_to_choice_set.items()
    }
    continuation_values_adjusted = {
        ftr_key: continuation_values[int(ftr_key)][
            :, list(choice_positons[int(ftr_key)])
        ]
        for ftr_key in transition_df.columns
    }

    weighted_columns = [
        transition_df[ftr_key].values.reshape((transition_df.shape[0], 1))
        * continuation_values_adjusted[ftr_key]
        for ftr_key in transition_df.columns
    ]

    continuation_values = functools.reduce(np.add, weighted_columns)
    return continuation_values


def create_transit_choice_set(
    dense_key_to_transit_representation, dense_key_to_choice_set
):
    """Return max representation choice set of each dense choice core."""
    continuation_choice_sets = {}
    for dense_key in dense_key_to_transit_representation:
        continuation_choice_sets[dense_key] = []
        for transit_representation in dense_key_to_transit_representation:
            if dense_key in dense_key_to_transit_representation[transit_representation]:
                continuation_choice_sets[dense_key].append(
                    dense_key_to_choice_set[transit_representation]
                )
    out = {}
    for dense_key in continuation_choice_sets:
        out[dense_key] = _get_maximal_choice_set(continuation_choice_sets[dense_key])
    return out


def _get_maximal_choice_set(list_of_choice_sets):
    array = np.concatenate(
        [np.expand_dims(x, axis=0) for x in list_of_choice_sets], axis=0
    )
    out = array.any(axis=0)
    return out


def _get_representation_cols(rep_choice_set, choice_set):
    """Return index of array cols."""
    # Subset choice sets.
    return np.array(choice_set)[np.array(rep_choice_set)]


@parallelize_across_dense_dimensions
def create_transition_objects(
    dense_covariates,
    core_key,
    exogenous_grid,
    n_exog,
    dense_covariates_to_dense_index,
    core_key_and_dense_index_to_dense_key,
):
    """Insert Doctring."""
    static_dense = dense_covariates[:-n_exog]

    reachable_dense_indices = [
        dense_covariates_to_dense_index[(*static_dense, *exog)]
        for exog in exogenous_grid
    ]
    reachable_dense_keys = [
        core_key_and_dense_index_to_dense_key[(core_key, dense_index)]
        for dense_index in reachable_dense_indices
    ]

    return reachable_dense_keys
