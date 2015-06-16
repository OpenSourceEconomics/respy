""" This module contains all the capabilities to solve the dynamic
programming problem.
"""

# standard library
import numpy as np

''' Public function
'''


def solve(init_dict):
    # Initialization of BASICS
    DELTA = init_dict['BASICS']['delta']
    NUM_PERIODS = init_dict['BASICS']['periods']
    NUM_AGENTS = init_dict['BASICS']['agents']

    # Initialization of COMPUTATION
    NUM_DRAWS = init_dict['COMPUTATION']['draws']

    # Initialization of EDUCATION
    EDU_START = init_dict['EDUCATION']['initial']
    EDU_MAX = init_dict['EDUCATION']['maximum']


    # Create grid of possible/admissible state space values
    k_state, k_period, f_state = _create_state_space(init_dict)

    k_max = max(k_period)

    # Draw random variable
    np.random.seed(init_dict['COMPUTATION']['seed'])

    eps = np.random.multivariate_normal(np.zeros(4), init_dict['SHOCKS'],
                                        (NUM_PERIODS, NUM_DRAWS))

    ''' Solve finite horizon dynamic programming problem by backward induction.
    Speed consideration are of the outmost importance here.
    '''
    emax = np.tile(np.nan, (NUM_PERIODS, k_max))

    # Systematic components of payoff
    payoffs_ex_ante = np.tile(np.nan, (NUM_PERIODS, k_max, 4))

    # Calculate systematic instantaneous payoffs
    for period in range(NUM_PERIODS - 1, -1, -1):

        # Loop over all possible states
        for k in range(k_period[period]):

            # Distribute state space
            exp_A = k_state[period, k, 0]
            exp_B = k_state[period, k, 1]

            edu = k_state[period, k, 2]
            edu_lagged = k_state[period, k, 3]

            # Auxiliary objects
            coeffs = dict()
            coeffs['A'] = [init_dict['A']['int']] + init_dict['A']['coeff']
            coeffs['B'] = [init_dict['B']['int']] + init_dict['B']['coeff']

            covars = [1.0, edu, exp_A, exp_A ** 2, exp_B, exp_B ** 2]

            # Calculate systematic part of wages in occupation A
            payoffs_ex_ante[period, k, 0] = np.exp(np.dot(coeffs['A'], covars))

            # Calculate systematic part pf wages in occupation B
            payoffs_ex_ante[period, k, 1] = np.exp(np.dot(coeffs['B'], covars))

            # Calculate systematic part of schooling utility
            payoff = init_dict['EDUCATION']['int']

            if edu > 12:  # Tuition cost for higher education
                payoff -= init_dict['EDUCATION']['coeff'][0]

            if edu_lagged == 0:  # Psychic cost of going back to school
                payoff -= init_dict['EDUCATION']['coeff'][1]

            payoffs_ex_ante[period, k, 2] = payoff

            # Calculate systematic part of HOME
            payoffs_ex_ante[period, k, 3] = init_dict['HOME']['int']

    for period in range(NUM_PERIODS - 1, -1, -1):

        # Loop over all possible states
        for k in range(k_period[period]):

            # Initialize container
            emax[period, k] = 0

            payoffs_ex_post = np.tile(np.nan, 4)

            # Distribute state space
            exp_A = k_state[period, k, 0]
            exp_B = k_state[period, k, 1]

            edu = k_state[period, k, 2]
            edu_lagged = k_state[period, k, 3]

            # Calculate systematic part of wages
            wA_systm = payoffs_ex_ante[period, k, 0]

            wB_systm = payoffs_ex_ante[period, k, 1]

            # Calculate systematic part of schooling utility
            edu_systm = payoffs_ex_ante[period, k, 2]

            # Calculate systematic part of HOME
            home_systm = payoffs_ex_ante[period, k, 3]

            vmax = 0

            for i in range(NUM_DRAWS):

                # Calculate ex post payoffs
                for j in [0, 1]:
                    payoffs_ex_post[j] = payoffs_ex_ante[period, k, j] * \
                                         np.exp(eps[period, i, j])

                for j in [2, 3]:
                    payoffs_ex_post[j] = payoffs_ex_ante[period, k, j] + \
                                         eps[period, i, j]

                # Future utilities
                wA_total, wB_total, edu_total, home_total = payoffs_ex_post

                # Future utilities
                if period != (NUM_PERIODS - 1):

                    # Working in occupation A
                    future_idx = f_state[period + 1, exp_A + 1, exp_B,
                                         (edu - EDU_START), 0]

                    wA_total += DELTA * emax[period + 1, future_idx]

                    # Working in occupation B
                    future_idx = f_state[period + 1, exp_A, exp_B + 1,
                                         (edu - EDU_START), 0]

                    wB_total += DELTA * emax[period + 1, future_idx]

                    # Increasing schooling. Note that adding an additional year
                    # of schooling is only possible for those that have strictly
                    # less than the maximum level of education allowed.
                    #
                    #
                    # Is this a valid way to impose this restriction?
                    if edu < EDU_MAX:
                        future_idx = f_state[period + 1, exp_A, exp_B,
                                             (edu - EDU_START + 1), 1]

                        edu_total += DELTA * emax[period + 1, future_idx]
                    else:

                        edu_total = - 40000

                    # Staying at home
                    future_idx = f_state[period + 1, exp_A, exp_B,
                                         (edu - EDU_START), 0]

                    home_total += DELTA * emax[period + 1, future_idx]

                # Hypothetical choice
                vmax = max(wA_total, wB_total, edu_total, home_total)

                # Recording expected future value
                emax[period, k] += vmax

            # Scaling
            emax[period, k] = emax[period, k] / NUM_DRAWS

    # Checking validity of expected future values. All valid values need to be
    # finite.
    for period in range(NUM_PERIODS):
        assert (np.all(np.isfinite(emax[period, :k_period[period]])))

    # Finishing
    return emax, k_period, k_state, eps, payoffs_ex_ante, f_state


''' Private functions
'''


def _create_state_space(init_dict):
    """ Create grid for state space.
    """

    # Initialization of BASICS
    num_periods = init_dict['BASICS']['periods']

    # Initialization of EDUCATION
    edu_start = init_dict['EDUCATION']['initial']
    edu_max = init_dict['EDUCATION']['maximum']

    # Array for possible realization of state space by period
    k_state = np.tile(np.nan, (num_periods, 100000, 4))

    # Array for maximum number of realizations of state space by period
    k_period = np.tile(np.nan, num_periods)

    # Array for future states by period
    f_state = np.tile(np.nan, (num_periods, num_periods, num_periods,
                               min(num_periods, edu_max), 2))

    for period in range(num_periods):

        # Count admissible realizations of state space by period
        k = 0

        # Loop over all admissible education levels
        for edu in range(edu_start, edu_max + 1):

            # Loop over all admissible work experiences for occupation A
            for exp_A in range(num_periods):

                # Loop over all admissible work experience for occupation B
                for exp_B in range(num_periods):

                    # Loop over all admissible values for leisure. Note that
                    # the leisure variable takes only zero/value. The time path
                    # does not matter.
                    for edu_lagged in [0, 1]:

                        # Check if lagged education admissible. (1) In the first
                        # period all agents have lagged schooling equal to one.
                        if (edu_lagged == 0) and (period == 0): continue
                        # (2) Whenever an agent has ten years of education and
                        # we are not in the first period, then this cannot be
                        # the case.
                        if (edu_lagged == 1) and (edu == 10) and (period > 0):
                            continue

                        # Check if admissible for time constraints
                        total = (edu - edu_start) + exp_A + exp_B

                        # Note that the total number of activities does not have
                        # is less or equal to the total possible number of
                        # activites as the rest is implicitly filled with leisure.
                        if total > period: continue

                        # Collect opportunities
                        k_state[period, k, :] = [exp_A, exp_B, edu, edu_lagged]

                        # Collect future state. Note that the -1 shift is
                        # required as Python indexing starts at 0.
                        f_state[period, exp_A, exp_B, (edu - edu_start),
                                edu_lagged] = k

                        # Update count
                        k += 1

        # Record maximum number of state space realizations by time period
        k_period[period] = k

    # Type transformation
    k_period = np.array(k_period, dtype='int')
    k_state = np.array(k_state, dtype='int')

    # Auxiliary objects
    k_max = k_period[-1]

    # Cutting to size
    k_state = k_state[:, :k_max, :]

    # Checking validity of state space values. All valid values need to be
    # finite.
    for period in range(num_periods):
        assert (np.all(np.isfinite(k_state[period, :k_period[period]])))

    # Finishing
    return k_state, k_period, f_state
