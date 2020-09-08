import functools
import itertools

import pandas as pd
import numpy as np
from scipy import special

from respy.parallelization import parallelize_across_dense_dimensions
from respy.shared import load_objects
from respy.shared import create_dense_state_space_columns
from respy.shared import pandas_dot

@parallelize_across_dense_dimensions
def compute_transition_probabilities(core_key,
                                     complex_,
                                     dense_covariates_to_dense_index,
                                     core_key_and_dense_index_to_dense_key,
                                     optim_paras,
                                     options
                                     ):
    """Insert Docstring Here!"""
    states = load_objects("states", complex_, options)
    exogenous_processes = optim_paras["exogenous_processes"]

    # How does the accounting work here again? Would that actually work?
    dense_columns = create_dense_state_space_columns(optim_paras)
    static_dense_columns = [x for x in dense_columns if x not in exogenous_processes]

    if static_dense_columns != []:
        static_dense = list(states.groupby(static_dense_columns).groups)[0]
        static_dense = (static_dense,) if type(static_dense) is int else static_dense
    else:
        static_dense = []

    levels_of_processes = [range(len(i)) for i in exogenous_processes.values()]
    comb_exog_procs = itertools.product(*levels_of_processes)

    # Needs to be created in here since that is dense-period-choice-core specific.
    dense_index_to_exogenous = {
        dense_covariates_to_dense_index[(*static_dense, *exog)]: exog for exog in
        comb_exog_procs}
    dense_key_to_exogenous = {
        core_key_and_dense_index_to_dense_key[(core_key, key)]: value for key, value in
        dense_index_to_exogenous.items()}

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
    print(dense_key_to_exogenous)

    for dense in dense_key_to_exogenous:
        array = functools.reduce(np.multiply,
                                 [probabilities[proc][:, val] \
                                  for proc, val in
                                  enumerate(dense_key_to_exogenous[dense])])
        df[dense] = array
    return df

@parallelize_across_dense_dimensions
def weight_continuation_values(
        complex,
        options,
        continuation_values):
    transition_df = load_objects("transition", complex, options)

    weighted_columns = \
        [transition_df[ftr_key].values.reshape((transition_df.shape[0], 1)) \
         * continuation_values[ftr_key] for ftr_key in transition_df.columns]

    continuation_values = functools.reduce(np.add,weighted_columns)
    return continuation_values
