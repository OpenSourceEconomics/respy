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
from robupy.auxiliary import create_disturbances

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

    model_paras = robupy_obj.get_attr('model_paras')

    num_periods = robupy_obj.get_attr('num_periods')

    num_points = robupy_obj.get_attr('num_points')

    num_agents = robupy_obj.get_attr('num_agents')

    is_python = robupy_obj.get_attr('is_python')

    edu_start = robupy_obj.get_attr('edu_start')

    num_draws_emax = robupy_obj.get_attr('num_draws_emax')

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
    coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks, shocks_cholesky = \
        distribute_model_paras(model_paras, is_debug)

    # Draw standard normal deviates for S-ML approach
    disturbances_prob = create_disturbances(num_periods, num_draws_prob,
        seed_prob, is_debug, 'prob', shocks_cholesky, is_ambiguous)

    # Get the relevant set of disturbances. These are standard normal draws
    # in the case of an ambiguous world. This function is located outside the
    # actual bare solution algorithm to ease testing across implementations.
    disturbances_emax = create_disturbances(num_periods, num_draws_emax,
        seed_emax, is_debug, 'emax', shocks_cholesky, is_ambiguous)

    # Solve model for given parametrization
    args = solve_python_bare(coeffs_a, coeffs_b, coeffs_edu, coeffs_home,
        shocks, edu_max, delta, edu_start, is_debug, is_interpolated, level,
        measure, min_idx, num_draws_emax, num_periods, num_points, is_ambiguous,
        disturbances_emax, is_python)

    # Distribute return arguments from solution run
    mapping_state_idx, periods_emax, periods_payoffs_future = args[:3]
    periods_payoffs_ex_post, periods_payoffs_systematic, states_all = args[3:6]

    # Evaluate the criterion function
    likl = _evaluate_python_bare(mapping_state_idx, periods_emax,
                periods_payoffs_systematic, states_all, shocks, edu_max,
                delta, edu_start, num_periods, shocks_cholesky, num_agents,
                num_draws_prob, data_array, disturbances_prob, is_python)

    # Finishing
    return robupy_obj, likl


''' Auxiliary functions
'''


def _evaluate_python_bare(mapping_state_idx, periods_emax,
        periods_payoffs_systematic, states_all, shocks, edu_max, delta,
        edu_start, num_periods,  shocks_cholesky, num_agents, num_draws_prob,
        data_array, disturbances_prob, is_python):
    """ This function is required to ensure a full analogy to F2PY and
    FORTRAN implementations. The first part of the interface is identical to
    the solution request functions.
    """

    if is_python:
        likl = evaluate_criterion_function(mapping_state_idx, periods_emax,
            periods_payoffs_systematic, states_all, shocks, edu_max, delta,
            edu_start, num_periods, shocks_cholesky, num_agents, num_draws_prob,
            data_array, disturbances_prob)

    else:
        import robupy.python.f2py.f2py_library as f2py_library
        likl = f2py_library.wrapper_evaluate_criterion_function(
            mapping_state_idx, periods_emax, periods_payoffs_systematic,
            states_all, shocks, edu_max, delta, edu_start, num_periods,
            shocks_cholesky, num_agents, num_draws_prob, data_array,
            disturbances_prob)

    # Finishing
    return likl


# Solve the model for given parametrization
def evaluate_criterion_function(mapping_state_idx, periods_emax,
        periods_payoffs_systematic, states_all, shocks, edu_max, delta,
        edu_start, num_periods, shocks_cholesky, num_agents, num_draws_prob,
        data_array, disturbances_prob):
    """ Evaluate criterion function.
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
            deviates = disturbances_prob[period, :, :].copy()

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
                # Construct independent normal draws implied by the observed
                # wages.
                if choice == 1:
                    deviates[:, idx] = dist / np.sqrt(shocks[idx, idx])
                else:
                    deviates[:, idx] = (dist - shocks_cholesky[idx, 0] *
                        deviates[:, 0]) / shocks_cholesky[idx, idx]

                # Record contribution of wage observation.
                likl_contrib *= norm.pdf(dist, 0.0, np.sqrt(shocks[idx, idx]))
            # Determine conditional deviates. These correspond to the
            # unconditional draws if the agent did not work in the labor market.
            conditional_deviates = np.dot(shocks_cholesky, deviates.T).T

            # Simulate the conditional distribution of alternative-specific
            # value functions and determine the choice probabilities.
            counts = np.tile(0.0, 4)

            for s in range(num_draws_prob):
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
            choice_probabilities = counts / num_draws_prob

            # Adjust  and record likelihood contribution
            likl_contrib *= choice_probabilities[idx]
            likl += [likl_contrib]

            j += 1

    # Scaling
    likl = -np.mean(np.log(np.clip(likl, TINY_FLOAT, HUGE_FLOAT)))

    # Finishing
    return likl

