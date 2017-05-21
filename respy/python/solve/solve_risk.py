import numpy as np

from respy.python.shared.shared_auxiliary import get_total_values


def construct_emax_risk(num_periods, num_draws_emax, period, k, draws_emax_risk, rewards_systematic,
        periods_emax, states_all, mapping_state_idx, edu_spec, optim_paras):
    """ Simulate expected future value for a given distribution of the unobservables.
    """
    # Antibugging
    assert np.all(draws_emax_risk[:, :2] >= 0)

    # Calculate maximum value
    emax = 0.0
    for i in range(num_draws_emax):

        # Select draws for this draw
        draws = draws_emax_risk[i, :]

        # Get total value of admissible states
        total_values = get_total_values(period, num_periods, optim_paras, rewards_systematic,
            draws, edu_spec, mapping_state_idx, periods_emax, k, states_all)

        # Determine optimal choice
        maximum = max(total_values)

        # Recording expected future value
        emax += maximum

    # Scaling
    emax = emax / num_draws_emax

    # Finishing
    return emax
