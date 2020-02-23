"""This module contains everything related to exogenous processes.

In models without exogenous processes the law of motion, the transition of states over
periods, is deterministic. Given a state and a choice the state in the next period is
certain.

Exogenous processes introduce a stochastic element to the law of motion. Given a state
and a choice there exist multiple states in the next period which are valid successors.
The transition to one of the successors is determined by a probability which depends on
the characteristics of the state.

See Also
--------
respy.pre_processing.model_processing._parse_exogenous_processes
respy.state_space._MultiDimStateSpace._weight_continuation_values
respy.simulate._apply_law_of_motion

"""
import itertools

import numpy as np
from scipy import special

from respy.parallelization import parallelize_across_dense_dimensions
from respy.shared import create_dense_state_space_columns


@parallelize_across_dense_dimensions
def compute_transition_probabilities(states, optim_paras):
    """Compute the transition probabilities between states due to exogenous processes.

    This function returns a dictionary for each `respy.state_space._SingleDimStateSpace`
    whose keys are the dense state space indices to where the states can transition to.
    To each key belongs a probability vector which is the product of probabilities
    defined by the exogenous processes.

    As an example, assume a model without observables and types but one exogenous
    processes named "exog_proc" with two possible values, zero and one. The return
    taking into account the decorator will look like this:

    .. code-block:: python

        {  # Outer dictionary created by deco.
            (0,): {(0,): prob_vector, (1,): prob_vector},  # Inner dictionary create by
            (1,): {(0,): prob_vector, (1,): prob_vector},  # the function.
        }

    """
    exogenous_processes = optim_paras["exogenous_processes"]
    dense_columns = create_dense_state_space_columns(optim_paras)

    # Get the dense index of the current states instead of having an additional funcarg.
    dense_idx = list(states.groupby(dense_columns).groups)[0]
    dense_idx = (dense_idx,) if np.isscalar(dense_idx) else dense_idx

    probabilities = []
    for exog_proc in exogenous_processes:

        # Create the dot product of covariates and parameters.
        x_betas = []
        for params in exogenous_processes[exog_proc].values():
            x_beta = np.dot(states[params.index].to_numpy(), params.to_numpy())
            x_betas.append(x_beta)

        probs = special.softmax(np.column_stack(x_betas), axis=1)
        probabilities.append(probs)

    transition_probabilities = {}
    comb_exog_procs = itertools.product(
        *[range(len(levels)) for levels in exogenous_processes.values()]
    )
    n_exog_procs = len(exogenous_processes)
    pos = dense_columns.index(list(exogenous_processes)[0])
    for exog_proc_idx in comb_exog_procs:
        probs = np.multiply.reduce(
            [probabilities[i][:, idx] for i, idx in enumerate(exog_proc_idx)]
        )
        new_dense_idx = (
            *dense_idx[:pos],
            *exog_proc_idx,
            *dense_idx[pos + n_exog_procs :],
        )
        transition_probabilities[new_dense_idx] = probs

    return transition_probabilities
