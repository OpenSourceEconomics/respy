""" This module contains the PYTHON implementations fo several functions where FORTRAN alternatives are available.
"""
import numpy as np

def get_future_payoffs(edu_max, edu_start, mapping_state_idx, period, emax, k,
                            states_all):
        """ Get future payoffs for additional choices.
        """

        # Distribute state space
        exp_A, exp_B, edu, edu_lagged = states_all[period, k, :]

        # Future utilities
        future_payoffs = np.tile(np.nan, 4)

        # Working in occupation A
        future_idx = mapping_state_idx[period + 1, exp_A + 1, exp_B,
                                       edu, 0]
        future_payoffs[0] = emax[period + 1, future_idx]

        # Working in occupation B
        future_idx = mapping_state_idx[period + 1, exp_A, exp_B + 1,
                                       edu, 0]

        future_payoffs[1] = emax[period + 1, future_idx]

        # Increasing schooling. Note that adding an additional year
        # of schooling is only possible for those that have strictly
        # less than the maximum level of additional education allowed.
        if edu < edu_max - edu_start:

            future_idx = mapping_state_idx[period + 1, exp_A, exp_B,
                                           edu + 1, 1]

            future_payoffs[2] = emax[period + 1, future_idx]

        else:

            future_payoffs[2] = -np.inf

        # Staying at home
        future_idx = mapping_state_idx[period + 1, exp_A, exp_B,
                                       edu, 0]

        future_payoffs[3] = emax[period + 1, future_idx]

        # Ensuring that schooling does not increase beyond the
        # maximum allowed level. This is necessary as in the
        # special case where delta is equal to zero,
        # (-np.inf * 0) evaluates to NAN.
        if edu >= edu_max - edu_start:
            future_payoffs[2] = -np.inf

        # Finishing
        return future_payoffs

def calculate_payoffs_ex_ante(num_periods, states_number_period, states_all,
                                edu_start, coeffs_A, coeffs_B, coeffs_edu,
                                coeffs_home, max_states_period):
        """ Calculate ex ante payoffs.
        """

        # Initialize
        period_payoffs_ex_ante = np.tile(np.nan, (num_periods, max_states_period,
                                                  4))

        # Calculate systematic instantaneous payoffs
        for period in range(num_periods - 1, -1, -1):

            # Loop over all possible states
            for k in range(states_number_period[period]):

                # Distribute state space
                exp_A, exp_B, edu, edu_lagged = states_all[period, k, :]

                # Auxiliary objects
                covars = [1.0, edu + edu_start, exp_A, exp_A ** 2, exp_B,
                          exp_B ** 2]

                # Calculate systematic part of wages in occupation A
                period_payoffs_ex_ante[period, k, 0] = np.exp(
                    np.dot(coeffs_A, covars))

                # Calculate systematic part pf wages in occupation B
                period_payoffs_ex_ante[period, k, 1] = np.exp(
                    np.dot(coeffs_B, covars))

                # Calculate systematic part of schooling utility
                payoff = coeffs_edu[0]

                # Tuition cost for higher education if agents move
                # beyond high school.
                if edu + edu_start > 12:
                    payoff += coeffs_edu[1]
                # Psychic cost of going back to school
                if edu_lagged == 0:
                    payoff += coeffs_edu[2]

                period_payoffs_ex_ante[period, k, 2] = payoff

                # Calculate systematic part of HOME
                period_payoffs_ex_ante[period, k, 3] = coeffs_home[0]

        # Finishing
        return period_payoffs_ex_ante

