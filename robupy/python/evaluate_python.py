""" This module provides the interface to the functionality needed to
evaluate the likelihood function.
"""

# standard library
from scipy.stats import norm

import numpy as np

# project library
from robupy.python.solve_python import solve_python_bare
from robupy.python.py.auxiliary import get_total_value

from robupy.auxiliary import distribute_model_paras
from robupy.auxiliary import create_draws

from robupy.constants import SMALL_FLOAT
from robupy.constants import TINY_FLOAT
from robupy.constants import HUGE_FLOAT

''' Main function
'''


def evaluate_python(robupy_obj, data_frame):
    """ Evaluate the criterion function of the model using PYTHON/F2PY code.
    This purpose of this wrapper is to extract all relevant information from
    the project class to pass it on to the actual evaluation functions. This is
    required to align the functions across the PYTHON and F2PY implementations.
    """
    # Distribute class attribute
    is_interpolated = robupy_obj.get_attr('is_interpolated')

    seed_prob = robupy_obj.get_attr('seed_prob')

    seed_emax = robupy_obj.get_attr('seed_emax')

    is_ambiguous = robupy_obj.get_attr('is_ambiguous')

    is_deterministic = robupy_obj.get_attr('is_deterministic')

    model_paras = robupy_obj.get_attr('model_paras')

    num_periods = robupy_obj.get_attr('num_periods')

    num_points = robupy_obj.get_attr('num_points')

    num_agents = robupy_obj.get_attr('num_agents')

    is_python = robupy_obj.get_attr('is_python')

    edu_start = robupy_obj.get_attr('edu_start')

    num_draws_emax = robupy_obj.get_attr('num_draws_emax')

    is_myopic = robupy_obj.get_attr('is_myopic')

    is_debug = robupy_obj.get_attr('is_debug')

    num_draws_prob = robupy_obj.get_attr('num_draws_prob')

    edu_max = robupy_obj.get_attr('edu_max')

    measure = robupy_obj.get_attr('measure')

    min_idx = robupy_obj.get_attr('min_idx')

    level = robupy_obj.get_attr('level')

    delta = robupy_obj.get_attr('delta')

    # Transform dataset to array for easy access
    data_array = data_frame.as_matrix()

    # Distribute model parameters
    coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cov, shocks_cholesky = \
        distribute_model_paras(model_paras, is_debug)

    # Draw standard normal deviates for choice probability integration
    periods_draws_prob = create_draws(num_periods, num_draws_prob,
        seed_prob, is_debug, 'prob', shocks_cholesky)

    # Draw standard normal deviates for EMAX integration
    periods_draws_emax = create_draws(num_periods, num_draws_emax,
        seed_emax, is_debug, 'emax', shocks_cholesky)

    # Solve model for given parametrization
    args = solve_python_bare(coeffs_a, coeffs_b, coeffs_edu, coeffs_home,
        shocks_cov, edu_max, delta, edu_start, is_debug, is_interpolated, level,
        measure, min_idx, num_draws_emax, num_periods, num_points, is_ambiguous,
        periods_draws_emax, is_deterministic, is_myopic, shocks_cholesky,
        is_python)

    # Distribute return arguments from solution run
    mapping_state_idx, periods_emax, periods_payoffs_future = args[:3]
    periods_payoffs_ex_post, periods_payoffs_systematic, states_all = args[3:6]

    # Evaluate the criterion function
    likl = _evaluate_python_bare(mapping_state_idx, periods_emax,
                periods_payoffs_systematic, states_all, shocks_cov, edu_max,
                delta, edu_start, num_periods, shocks_cholesky, num_agents,
                num_draws_prob, data_array, periods_draws_prob, is_deterministic,
                is_python)

    # Finishing
    return robupy_obj, likl


''' Auxiliary functions
'''


def _evaluate_python_bare(mapping_state_idx, periods_emax,
                          periods_payoffs_systematic, states_all, shocks_cov, edu_max, delta,
                          edu_start, num_periods, shocks_cholesky, num_agents, num_draws_prob,
                          data_array, periods_draws_prob, is_deterministic, is_python):
    """ This function is required to ensure a full analogy to F2PY and
    FORTRAN implementations. The first part of the interface is identical to
    the solution request functions.
    """

    if is_python:
        likl = evaluate_criterion_function(mapping_state_idx, periods_emax,
                                           periods_payoffs_systematic, states_all, shocks_cov, edu_max, delta,
                                           edu_start, num_periods, shocks_cholesky, num_agents, num_draws_prob,
                                           data_array, periods_draws_prob, is_deterministic)

    else:
        import robupy.python.f2py.f2py_library as f2py_library
        likl = f2py_library.wrapper_evaluate_criterion_function(
            mapping_state_idx, periods_emax, periods_payoffs_systematic,
            states_all, shocks_cov, edu_max, delta, edu_start, num_periods,
            shocks_cholesky, num_agents, num_draws_prob, data_array,
            periods_draws_prob, is_deterministic)

    # Finishing
    return likl


# Solve the model for given parametrization
def evaluate_criterion_function(mapping_state_idx, periods_emax,
                                periods_payoffs_systematic, states_all, shocks_cov, edu_max, delta,
                                edu_start, num_periods, shocks_cholesky, num_agents, num_draws_prob,
                                data_array, periods_draws_prob, is_deterministic):
    """ Evaluate criterion function. This code allows for a deterministic
    model, where there is no random variation in the rewards. If that is the
    case and all agents have corresponding experiences, then one is returned.
    If a single agent violates the implications, then the zero is returned.
    """
    # Initialize auxiliary objects
    likl, j = [], 0

    # Calculate the probability over agents and time.
    for i in range(num_agents):
        for period in range(num_periods):
            # Extract observable components of state space as well as agent
            # decision.
            exp_a, exp_b, edu, edu_lagged = data_array[j, 4:].astype(int)
            choice = data_array[j, 2].astype(int)
            is_working = choice in [1, 2]

            # Transform total years of education to additional years of
            # education and create an index from the choice.
            edu, idx = edu - edu_start, choice - 1

            # Get state indicator to obtain the systematic component of the
            # agents payoffs. These feed into the simulation of choice
            # probabilities.
            k = mapping_state_idx[period, exp_a, exp_b, edu, edu_lagged]
            payoffs_systematic = periods_payoffs_systematic[period, k, :]

            # Extract relevant deviates from standard normal distribution.
            draws_prob = periods_draws_prob[period, :, :].copy()

            # Prepare to calculate product of likelihood contributions.
            likl_contrib = 1.0

            # If an agent is observed working, then the the labor market shocks
            # are observed and the conditional distribution is used to determine
            # the choice probabilities.
            if is_working:
                # Calculate the disturbance, which follows a normal
                # distribution.
                dist = np.log(data_array[j, 3].astype(float)) - \
                        np.log(payoffs_systematic[idx])

                # If there is no random variation in payoffs, then the
                # observed wages need to be identical their systematic
                # components. The discrepancy between the observed wages and
                # their systematic components might be small due to the
                # reading in of the dataset (FORTRAN only).
                if is_deterministic and (dist > SMALL_FLOAT):
                    return 0.0

                # Construct independent normal draws implied by the observed
                # wages.
                if choice == 1:
                    draws_prob[:, idx] = dist / np.sqrt(shocks_cov[idx, idx])
                else:
                    draws_prob[:, idx] = (dist - shocks_cholesky[idx, 0] *
                        draws_prob[:, 0]) / shocks_cholesky[idx, idx]

                # Record contribution of wage observation.
                likl_contrib *= norm.pdf(dist, 0.0, np.sqrt(shocks_cov[idx, idx]))
            # Determine conditional deviates. These correspond to the
            # unconditional draws if the agent did not work in the labor market.
            conditional_draws = np.dot(shocks_cholesky, draws_prob.T).T

            # Simulate the conditional distribution of alternative-specific
            # value functions and determine the choice probabilities.
            counts = np.tile(0.0, 4)

            for s in range(num_draws_prob):
                # Extract deviates from (un-)conditional normal distributions.
                draws = conditional_draws[s, :]

                draws[0] = np.exp(draws[0])
                draws[1] = np.exp(draws[1])

                # Calculate total payoff.
                total_payoffs, _, _ = get_total_value(period, num_periods,
                    delta, payoffs_systematic, draws, edu_max,
                    edu_start, mapping_state_idx, periods_emax, k, states_all)

                # Record optimal choices
                counts[np.argmax(total_payoffs)] += 1.0

            # Determine relative shares
            choice_probabilities = counts / num_draws_prob

            # If there is no random variation in payoffs, then this implies a
            # unique optimal choice.
            if is_deterministic and (not (max(counts) == num_draws_prob)):
                return 0.0

            # Adjust  and record likelihood contribution
            likl_contrib *= choice_probabilities[idx]
            likl += [likl_contrib]

            j += 1

    # Scaling
    likl = -np.mean(np.log(np.clip(likl, TINY_FLOAT, HUGE_FLOAT)))

    # If there is no random variation in payoffs and no agent violated the
    # implications of observed wages and choices, then the evaluation return
    # a value of one.
    if is_deterministic:
        likl = 1.0

    # Finishing
    return likl

