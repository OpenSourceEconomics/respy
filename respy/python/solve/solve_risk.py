from respy.python.shared.shared_auxiliary import get_total_values


def construct_emax_risk(num_periods, num_draws_emax, period, k,
        draws_emax_transformed, rewards_systematic, edu_max, edu_start,
        periods_emax, states_all, mapping_state_idx, delta):
    """ Simulate expected future value for a given distribution of the
    unobservables.
    """
    # Calculate maximum value
    emax = 0.0
    for i in range(num_draws_emax):

        # Select draws for this draw
        draws = draws_emax_transformed[i, :]

        # Get total value of admissible states
        total_values = get_total_values(period, num_periods, delta,
            rewards_systematic, draws, edu_max, edu_start, mapping_state_idx,
            periods_emax, k, states_all)

        # Determine optimal choice
        maximum = max(total_values)

        # Recording expected future value
        emax += maximum

    # Scaling
    emax = emax / num_draws_emax

    # Finishing
    return emax
