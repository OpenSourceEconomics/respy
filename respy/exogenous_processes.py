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
respy.pre_processing.model_processing._parse_observables
respy.state_space._MultiDimStateSpace._weight_continuation_values
respy.simulate._apply_law_of_motion

"""
import numpy as np
from scipy import special

from respy.config import COVARIATES_DOT_PRODUCT_DTYPE
from respy.parallelization import parallelize_across_dense_dimensions
from respy.shared import create_dense_state_space_columns


@parallelize_across_dense_dimensions
def compute_process_specific_transition_probabilities(states, optim_paras):
    """Compute process-specific transition probabilities for all exogenous processes.

    This function computes for each exogenous process the probability of transitioning
    to another realization of the exogenous process given the current value. For each
    exogenous process, the result is a matrix with as many rows as states and a column
    for each realization of the process. The values are the probabilities.

    The matrices of all exogenous processes are stored in a list where the position is
    determined by the order of exogenous processes in ``optim_paras``.

    Why do we store the probabilities like this? Another alternative would be to combine
    the process-specific probabilities to calculate the transition probabilities to
    other dense dimensions of the state space.

    The reason we do not do this is to save memory. Assume you have one exogenous
    process with two realizations and another one with three. The process-specific
    transition probabilities are two matrices with two and three columns. If we would
    compute the transition probabilities to each combination of the realizations of the
    exogenous processes, we would end up with six (2 * 3) vectors of transition
    probabilities.

    Parameters
    ----------
    states : pandas.DataFrame
        DataFrame containing the states for one dense vector .
    optim_paras : dict
        Dictionary containing parameters.

    Returns
    -------
    probabilities : list
        List containing a matrix for each exogenous process. Each matrix has as many
        rows as states and as many columns as number of realizations of the process.

    """
    exogenous_processes = optim_paras["exogenous_processes"]
    dense_columns = create_dense_state_space_columns(optim_paras)

    # Get the dense index of the current states instead of having an additional funcarg.
    dense_idx = list(states.groupby(dense_columns).groups)[0]
    dense_idx = (dense_idx,) if np.isscalar(dense_idx) else dense_idx

    # Compute the probabilities for every exogenous process.
    probabilities = []
    for exog_proc in exogenous_processes:

        # Create the dot product of covariates and parameters.
        x_betas = []
        for params in exogenous_processes[exog_proc].values():
            x_beta = np.dot(
                states[params.index].to_numpy(dtype=COVARIATES_DOT_PRODUCT_DTYPE),
                params.to_numpy(),
            )
            x_betas.append(x_beta)

        probs = special.softmax(np.column_stack(x_betas), axis=1)
        probabilities.append(probs)

    return probabilities
