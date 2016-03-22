""" This module contains the interface to solve the model.
"""

# standard library
import logging
import shlex
import os

# project library
from robupy.auxiliary import distribute_model_paras
from robupy.fortran.solve_fortran import solve_fortran
from robupy.python.solve_python import solve_python
from robupy.auxiliary import create_draws

''' Main function
'''


def solve(robupy_obj):
    """ Solve dynamic programming problem by backward induction.
    """
    # Checks, cleanup, start logger
    assert _check_solve(robupy_obj)

    _cleanup()

    _start_logging()

    # Distribute class attributes
    is_ambiguous = robupy_obj.get_attr('is_ambiguous')

    is_myopic = robupy_obj.get_attr('is_myopic')

    is_debug = robupy_obj.get_attr('is_debug')

    version = robupy_obj.get_attr('version')

    store = robupy_obj.get_attr('store')

    is_deterministic = robupy_obj.get_attr('is_deterministic')

    is_interpolated = robupy_obj.get_attr('is_interpolated')

    num_draws_emax = robupy_obj.get_attr('num_draws_emax')

    is_ambiguous = robupy_obj.get_attr('is_ambiguous')

    num_periods = robupy_obj.get_attr('num_periods')

    model_paras = robupy_obj.get_attr('model_paras')

    num_points = robupy_obj.get_attr('num_points')

    seed_emax = robupy_obj.get_attr('seed_emax')

    edu_start = robupy_obj.get_attr('edu_start')

    is_python = robupy_obj.get_attr('is_python')

    is_myopic = robupy_obj.get_attr('is_myopic')

    is_debug = robupy_obj.get_attr('is_debug')

    measure = robupy_obj.get_attr('measure')

    edu_max = robupy_obj.get_attr('edu_max')

    min_idx = robupy_obj.get_attr('min_idx')

    store = robupy_obj.get_attr('store')

    delta = robupy_obj.get_attr('delta')

    level = robupy_obj.get_attr('level')

    # Select appropriate interface
    if version == 'FORTRAN':
        robupy_obj = solve_fortran(robupy_obj)
    else:

        # Construct auxiliary objects
        _start_ambiguity_logging(is_ambiguous, is_debug)

        # Distribute model parameters
        coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cov, shocks_cholesky = \
            distribute_model_paras(model_paras, is_debug)

        # Get the relevant set of disturbances. These are standard normal draws
        # in the case of an ambiguous world. This function is located outside
        # the actual bare solution algorithm to ease testing across
        # implementations.
        periods_draws_emax = create_draws(num_periods, num_draws_emax,
                seed_emax, is_debug, 'emax', shocks_cholesky)

        # Collect arguments
        args = (coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cov,
            shocks_cholesky, is_deterministic, is_interpolated, num_draws_emax,
            periods_draws_emax, is_ambiguous, num_periods, num_points, edu_start,
            is_myopic, is_debug, measure, edu_max, min_idx, delta, level,
            is_python)

        args = solve_python(*args)

        periods_payoffs_systematic, periods_payoffs_ex_post = args[:2]
        periods_payoffs_future, states_number_period = args[2:4]
        mapping_state_idx, periods_emax, states_all = args[4:7]

        robupy_obj.unlock()

        robupy_obj.set_attr('periods_payoffs_systematic', periods_payoffs_systematic)

        robupy_obj.set_attr('periods_payoffs_ex_post', periods_payoffs_ex_post)

        robupy_obj.set_attr('periods_payoffs_future', periods_payoffs_future)

        robupy_obj.set_attr('states_number_period', states_number_period)

        robupy_obj.set_attr('mapping_state_idx', mapping_state_idx)

        robupy_obj.set_attr('periods_emax', periods_emax)

        robupy_obj.set_attr('states_all', states_all)

        robupy_obj.set_attr('is_solved', True)

        robupy_obj.lock()

        # Store object to file
        if store:
            robupy_obj.store('solution.robupy.pkl')

    # Summarize optimizations in case of ambiguity
    if is_debug and is_ambiguous and (not is_myopic):
        _summarize_ambiguity(robupy_obj)

    # Set flag that object includes the solution objects
    robupy_obj.unlock()

    robupy_obj.set_attr('is_solved', True)

    robupy_obj.lock()

    # Store results if requested
    if store:
        robupy_obj.store('solution.robupy.pkl')

    # Orderly shutdown of logging capability.
    _stop_logging()

    # Finishing
    return robupy_obj

''' Auxiliary functions
'''


def _check_solve(robupy_obj):
    """ Check likelihood calculation.
    """
    # Check integrity of the request
    assert (robupy_obj.get_attr('is_solved') is False)
    assert (robupy_obj.get_status())

    # Finishing
    return True


def _stop_logging():
    """ Ensure orderly shutdown of logging capabilities.
    """
    # Collect all loggers for shutdown.
    loggers = []
    loggers += [logging.getLogger('ROBUPY_SOLVE')]
    loggers += [logging.getLogger('ROBUPY_SIMULATE')]

    # Close file handlers
    for logger in loggers:
        handlers = logger.handlers[:]
        for handler in handlers:
            handler.close()
            logger.removeHandler(handler)


def _start_logging():
    """ Initialize logging setup for the solution of the model.
    """

    formatter = logging.Formatter('  %(message)s \n')

    logger = logging.getLogger('ROBUPY_SOLVE')

    handler = logging.FileHandler('logging.robupy.sol.log', mode='w',
                                  delay=False)

    handler.setFormatter(formatter)

    logger.setLevel(logging.INFO)

    logger.addHandler(handler)

    logger = logging.getLogger('ROBUPY_SIMULATE')

    handler = logging.FileHandler('logging.robupy.sim.log', mode='w',
                                  delay=False)

    handler.setFormatter(formatter)

    logger.setLevel(logging.INFO)

    logger.addHandler(handler)


def _summarize_ambiguity(robupy_obj):
    """ Summarize optimizations in case of ambiguity.
    """

    def _process_cases(list_internal):
        """ Process cases and determine whether keyword or empty line.
        """
        # Antibugging
        assert (isinstance(list_internal, list))

        # Get information
        is_empty_internal = (len(list_internal) == 0)

        if not is_empty_internal:
            is_block_internal = list_internal[0].isupper()
        else:
            is_block_internal = False

        # Antibugging
        assert (is_block_internal in [True, False])
        assert (is_empty_internal in [True, False])

        # Finishing
        return is_empty_internal, is_block_internal

    # Distribute class attributes
    num_periods = robupy_obj.get_attr('num_periods')

    dict_ = dict()

    for line in open('ambiguity.robupy.log').readlines():

        # Split line
        list_ = shlex.split(line)

        # Determine special cases
        is_empty, is_block = _process_cases(list_)

        # Applicability
        if is_empty:
            continue

        # Prepare dictionary
        if is_block:

            period = int(list_[1])

            if period in dict_.keys():
                continue
    
            dict_[period] = {}
            dict_[period]['success'] = 0
            dict_[period]['failure'] = 0
            dict_[period]['total'] = 0

        # Collect success indicator
        if list_[0] == 'Success':
            dict_[period]['total'] += 1

            is_success = (list_[1] == 'True')
            if is_success:
                dict_[period]['success'] += 1
            else:
                dict_[period]['failure'] += 1

    with open('ambiguity.robupy.log', 'a') as file_:

        file_.write('\nSUMMARY\n\n')

        string = '''{0[0]:>10} {0[1]:>10} {0[2]:>10} {0[3]:>10}\n'''
        file_.write(string.format(['Period', 'Total', 'Success', 'Failure']))

        file_.write('\n')

        for period in range(num_periods):
            total = dict_[period]['total']

            success = dict_[period]['success']/total
            failure = dict_[period]['failure']/total

            string = '''{0[0]:>10} {0[1]:>10} {0[2]:10.2f} {0[3]:10.2f}\n'''
            file_.write(string.format([period, total, success, failure]))


def _cleanup():
    """ Cleanup all selected files. Note that not simply all *.robupy.*
    files can be deleted as the blank logging files are already created.
    """
    if os.path.exists('ambiguity.robupy.log'):
        os.unlink('ambiguity.robupy.log')


def _start_ambiguity_logging(is_ambiguous, is_debug):
    """ Start logging for ambiguity.
    """
    # Start logging if required
    if os.path.exists('ambiguity.robupy.log'):
        os.remove('ambiguity.robupy.log')

    if is_debug and is_ambiguous:
        open('ambiguity.robupy.log', 'w').close()