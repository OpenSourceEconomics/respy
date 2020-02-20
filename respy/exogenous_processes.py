import numpy as np
from scipy import special

from respy.parallelization import parallelize_across_dense_dimensions


@parallelize_across_dense_dimensions
def compute_transition_probabilities(states, optim_paras):
    """Compute the transition probs between states due to exogenous processes.

    TODO: Think about some reductions:
    - If there is only one unique value per column, reduce matrix to one-dimensional
      array.
    - If the arrays has only one unique value, replace with scalar.

    """
    exogenous_processes = optim_paras["exogenous_processes"]

    transition_probabilities = {}
    for exog_proc in exogenous_processes:
        n_levels = len(exogenous_processes[exog_proc])
        x_betas = np.empty((states.shape[0], n_levels))

        for i, params in enumerate(exogenous_processes[exog_proc].values()):
            covariates = params.index
            x_betas[:, i] = np.dot(states[covariates].to_numpy(), params.to_numpy())

        transition_probabilities[exog_proc] = special.softmax(x_betas, axis=1)

    return transition_probabilities


def weight_continuation_values_by_exogenous_processes(
    continuation_values, transition_probabilities
):
    pass
