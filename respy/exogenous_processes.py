import functools
import itertools

import pandas as pd
import numpy as np
from scipy import special

from respy.parallelization import parallelize_across_dense_dimensions
from respy.shared import load_objects
from respy.shared import create_dense_state_space_columns
from respy.shared import pandas_dot

def compute_transition_probabilities(
        states,
        transit_keys,
        exogenous_grid,
        optim_paras,
        dense_key_to_exogenous
    ):
    exogenous_processes = optim_paras["exogenous_processes"]

    dense_key_to_exogenous = dense_key_to_exogenous[transit_keys]

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
        array = functools.reduce(np.multiply,
                                 [probabilities[proc][:, val] \
                                  for proc, val in
                                  enumerate(dense_key_to_exogenous[dense])])
        df[str(dense)] = array
    
    return df


@parallelize_across_dense_dimensions
def weight_continuation_values(
        complex,
        options,
        continuation_values,
        transit_key_to_choice_set
        ):
    transition_df = load_objects("transition", complex, options)
    choice_set = complex[1]
    continuation_values_adjusted = [

    ]

    weighted_columns = \
        [transition_df[ftr_key].values.reshape((transition_df.shape[0], 1)) \
         * continuation_values[int(ftr_key)] for ftr_key in transition_df.columns]

    continuation_values = functools.reduce(np.add,weighted_columns)
    return continuation_values


def create_continuation_choice_set(
    dense_key_to_transit_representation,
    dense_key_to_complex

):
    """
    Return a dict containing the max representation choice
    set of each dense choice core. 
    """
    continuation_choice_sets = dict()
    for dense_key in dense_key_to_transit_representation:
        continuation_choice_sets[dense_key] = list()
        for transit_representation in dense_key_to_transit_representation:
            if dense_key in dense_key_to_transit_representation[transit_representation]:
                continuation_choice_sets[dense_key].append(
                    dense_key_to_choice_set[transit_representation]
            )
    out = dict()
    for dense_key in continuation_choice_sets:
        out[sense_key] = _get_maximal_choice_set(
            continuation_choice_sets[dense_key]
        )
    return out

def _get_maximal_choice_set(list_of_choice_sets):
    array = np.concatenate([np.expand_dims(x,axis=0) for x in a],axis=0)
    out = array.any(axis=0)
    return out

def _get_representation_cols(rep_choice_set, choice_set):
    """Return index of array cols"""
    # Subset choice sets.
    return np.array(choice_set)[np.array(rep_choice_set)]


@parallelize_across_dense_dimensions
def create_transition_objects(
    dense_covariates,
    core_key,
    exog_grid,
    n_exog,
    optim_paras,
    dense_covariates_to_dense_index,
    core_key_and_dense_index_to_dense_key
):
"""Insert Docstring!"""
    # Needs to be created in here since that is dense-period-choice-core specific.
    static_dense = dense_covariates[:n_exog]
   
    reachable_dense_indices = [
        dense_covariates_to_dense_index[(*static_dense, *exog)]]
    
    reachable_dense_keys = {
        core_key_and_dense_index_to_dense_key[(core_key, key)]: value for key, value in
        dense_index_to_exogenous.items()}
    
    return reachable_dense_keys
