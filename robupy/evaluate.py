""" This module contains all the capabilities to evaluate the likelihood of
an observed dataset.
"""


from scipy.stats import norm
import numpy as np

from robupy.python.py.auxiliary import opt_get_model_parameters
from robupy.python.py.auxiliary import get_total_value
from robupy.auxiliary import create_disturbances

from robupy.python.solve_python import solve_python


from robupy.constants import TINY_FLOAT


def evaluate(robupy_obj, data_array):
    """ Evaluate likelihood function.
    """

    edu_max = robupy_obj.get_attr('edu_max')
    delta = robupy_obj.get_attr('delta')
    edu_start = robupy_obj.get_attr('edu_start')
    is_debug = robupy_obj.get_attr('is_debug')
    is_interpolated = robupy_obj.get_attr('is_interpolated')
    is_python = robupy_obj.get_attr('is_python')
    level = robupy_obj.get_attr('level')
    measure = robupy_obj.get_attr('measure')
    min_idx = robupy_obj.get_attr('min_idx')
    num_draws = robupy_obj.get_attr('num_draws')
    num_periods = robupy_obj.get_attr('num_periods')
    num_points = robupy_obj.get_attr('num_points')
    num_sims = robupy_obj.get_attr('num_sims')

    is_ambiguous = robupy_obj.get_attr('is_ambiguous')
    num_agents = robupy_obj.get_attr('num_agents')

    seed_solution = robupy_obj.get_attr('seed_solution')
    seed_estimation = robupy_obj.get_attr('seed_estimation')
    model_paras = robupy_obj.get_attr('model_paras')

    coeffs_a = model_paras['coeffs_a']
    coeffs_b = model_paras['coeffs_b']

    coeffs_edu = model_paras['coeffs_edu']
    coeffs_home = model_paras['coeffs_home']

    shocks = model_paras['shocks']
    eps_cholesky = model_paras['eps_cholesky']


    # Auxiliary objects
    standard_deviates = create_disturbances(num_sims, seed_estimation,
                                               eps_cholesky,
                                               is_ambiguous,
                                               num_periods,
                                               is_debug, 'estimation')

    # Update parameters
    #coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks, eps_cholesky = \
    #    opt_get_model_parameters(x, is_debug)

    # Solve the model for updated parametrization
    args = solve_python(coeffs_a, coeffs_b, coeffs_edu, coeffs_home,
                shocks, eps_cholesky, edu_max, delta, edu_start, is_debug,
                is_interpolated, is_python, level, measure, min_idx,
                num_draws, num_periods, num_points, is_ambiguous, seed_solution)

    # Distribute return arguments
    mapping_state_idx, periods_emax, periods_future_payoffs = args[:3]
    periods_payoffs_ex_post, periods_payoffs_systematic, states_all = args[3:6]

    # Initialize auxiliary objects
    likl, j = [], 0

    # Calculate the probability over agents and time.
    for i in range(num_agents):
        for period in range(num_periods):
            # Extract observable components of state space as well as agent
            # decision.
            exp_A, exp_B, edu, edu_lagged = data_array[j, 4:].astype(int)
            choice_indicator = data_array[j, 2].astype(int)

            # Transform total years of education to additional years of
            # education and create an index from the choice.
            edu, idx = edu - edu_start, choice_indicator - 1

            # Get state indicator to obtain the systematic component of the
            # agents payoffs. These feed into the simulation of choice
            # probabilities.
            k = mapping_state_idx[period, exp_A, exp_B, edu, edu_lagged]
            payoffs_systematic = periods_payoffs_systematic[period, k, :]

            # Extract relevant deviates from standard normal distribution.
            deviates = standard_deviates[period, :, :].copy()

            # Prepare to calculate product of likelihood contributions.
            likl_contrib = 1.0

            # If an agent is observed working, then the the labor market shocks
            # are observed and the conditional distribution is used to determine
            # the choice probabilities.
            if choice_indicator in [1, 2]:
                # Calculate the disturbance, which follows from a normal
                # distribution.
                eps = np.log(data_array[j, 3].astype(float)) - \
                        np.log(payoffs_systematic[idx])
                # Construct independent normal draws implied by the observed
                # wages.
                if choice_indicator == 1:
                    deviates[:, idx] = eps / np.sqrt(shocks[idx, idx])
                else:
                    deviates[:, idx] = (eps - eps_cholesky[idx, 0] *
                        deviates[:, 0]) / eps_cholesky[idx, idx]
                # Record contribution of wage observation.
                likl_contrib *= norm.pdf(eps, 0.0, np.sqrt(shocks[idx, idx]))

            # Determine conditional deviates. These correspond to the
            # unconditional draws if the agent did not work in the labor market.
            conditional_deviates = np.dot(eps_cholesky, deviates.T).T

            # Simulate the conditional distribution of alternative-specific
            # value functions and determine the choice probabilities.
            counts = np.tile(0.0, 4)

            for s in range(num_sims):
                # Extract deviates from (un-)conditional normal distributions.
                disturbances = conditional_deviates[s, :]

                disturbances[0] = np.exp(disturbances[0])
                disturbances[1] = np.exp(disturbances[1])

                # Calculate total payoff.
                total_payoffs, _, _ = get_total_value(period, num_periods,
                    delta, payoffs_systematic, disturbances, edu_max,
                    edu_start, mapping_state_idx, periods_emax, k, states_all)
                # Record optimal choices
                counts[np.argmax(total_payoffs)] += 1.0

            # Determine relative shares
            choice_probabilities = counts / num_sims

            # Adjust  and record likelihood contribution
            likl_contrib *= choice_probabilities[idx]
            likl += [likl_contrib]

            j += 1

    # Scaling
    likl = -np.mean(np.log(np.clip(likl, TINY_FLOAT, np.inf)))

    # Checks TODO: Refactor
    assert (isinstance(likl, float))
    assert (np.isfinite(likl))

    # Finishing
    return likl