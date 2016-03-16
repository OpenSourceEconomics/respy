""" This modules contains some additional tests that are only used in long-run
development tests.
"""

# standard library
from pandas.util.testing import assert_frame_equal

import pandas as pd
import numpy as np

import pytest

# testing library
from codes.auxiliary import write_interpolation_grid
from codes.auxiliary import write_disturbances

# ROBUPY import
from robupy import simulate
from robupy import evaluate
from robupy import process
from robupy import solve
from robupy import read

from robupy.tests.codes.random_init import generate_random_dict
from robupy.tests.codes.random_init import print_random_dict
from robupy.tests.codes.random_init import generate_init

from robupy.python.py.python_library import create_state_space


''' Main
'''


@pytest.mark.usefixtures('fresh_directory', 'set_seed', 'supply_resources')
class TestClass:

    def test_1(self):
        """ Testing whether random model specifications can be solved, simulated
        and processed.
        """
        # Generate random initialization file
        generate_init()

        robupy_obj = read('test.robupy.ini')

        solve(robupy_obj)

        simulate(robupy_obj)

        process(robupy_obj)

    def test_2(self):
        """ Testing the equality of an evaluation of the criterion function for
        a random request.
        """
        # Run evaluation for multiple random requests.
        is_deterministic = np.random.choice([True, False], p=[0.10, 0.9])
        is_interpolated = np.random.choice([True, False], p=[0.10, 0.9])
        is_myopic = np.random.choice([True, False], p=[0.10, 0.9])
        max_draws = np.random.random_integers(10, 100)

        # Generate random initialization file
        constraints = dict()
        constraints['is_deterministic'] = is_deterministic
        constraints['is_myopic'] = is_myopic
        constraints['max_draws'] = max_draws

        init_dict = generate_random_dict(constraints)

        # The use of the interpolation routines is a another special case.
        # Constructing a request that actually involves the use of the
        # interpolation routine is a little involved as the number of
        # interpolation points needs to be lower than the actual number of
        # states. And to know the number of states each period, I need to
        # construct the whole state space.
        if is_interpolated:
            # Extract from future initialization file the information
            # required to construct the state space. The number of periods
            # needs to be at least three in order to provide enough state
            # points.
            num_periods = np.random.random_integers(3, 6)
            edu_start = init_dict['EDUCATION']['start']
            edu_max = init_dict['EDUCATION']['max']
            min_idx = min(num_periods, (edu_max - edu_start + 1))

            max_states_period = create_state_space(num_periods, edu_start,
                                            edu_max, min_idx)[3]

            # Updates to initialization dictionary that trigger a use of the
            # interpolation code.
            init_dict['BASICS']['periods'] = num_periods
            init_dict['INTERPOLATION']['apply'] = True
            init_dict['INTERPOLATION']['points'] = \
                np.random.random_integers(10, max_states_period)

        # Print out the relevant initialization file.
        print_random_dict(init_dict)

        # Write out random components and interpolation grid to align the
        # three implementations.
        num_periods = init_dict['BASICS']['periods']
        write_disturbances(num_periods, max_draws)
        write_interpolation_grid('test.robupy.ini')

        # Clean evaluations based on interpolation grid,
        base_eval, base_data = None, None

        for version in ['PYTHON', 'F2PY', 'FORTRAN']:

            robupy_obj = read('test.robupy.ini')

            # Modify the version of the program for the different requests.
            robupy_obj.unlock()
            robupy_obj.set_attr('version',  version)
            robupy_obj.lock()

            # Solve the model
            robupy_obj = solve(robupy_obj)

            # This parts checks the equality of simulated dataset for the
            # different versions of the code.
            data_frame = pd.read_csv('data.robupy.dat', delim_whitespace=True)

            if base_data is None:
                base_data = data_frame.copy()

            assert_frame_equal(base_data, data_frame)

            # This part checks the equality of an evaluation of the
            # criterion function.
            data_frame = simulate(robupy_obj)

            robupy_obj, eval_ = evaluate(robupy_obj, data_frame)

            if base_eval is None:
                base_eval = eval_

            np.testing.assert_allclose(base_eval, eval_, rtol=1e-05,
                                       atol=1e-06)

            # We know even more for the deterministic case.
            if constraints['is_deterministic']:
                assert (eval_ in [0.0, 1.0])

    def test_3(self):
        """ Testing whether the systematic and ex post payoffs are identical if
        there is no random variation in the payoffs (all disturbances set to
        zero).
        """
        # Generate constraint periods
        constraints = dict()
        constraints['is_deterministic'] = True
        constraints['level'] = 0.00

        # Generate random initialization file
        generate_init(constraints)

        # Perform toolbox actions
        robupy_obj = read('test.robupy.ini')

        robupy_obj = solve(robupy_obj)

        # Distribute class attributes
        systematic = robupy_obj.get_attr('periods_payoffs_systematic')
        ex_post = robupy_obj.get_attr('periods_payoffs_ex_post')

        # Check
        assert (np.ma.all(np.ma.masked_invalid(systematic) ==
                    np.ma.masked_invalid(ex_post)))

    def test_4(slf):
        """ If there is no random variation in payoffs then the number of
        draws to simulate the expected future value should have no effect.
        """
        # Generate constraints
        constraints = dict()
        constraints['is_deterministic'] = True
        constraints['level'] = 0.0

        # Generate random initialization file
        generate_init(constraints)

        # Initialize auxiliary objects
        base = None

        for _ in range(2):

            # Draw a random number of draws for
            # expected future value calculations.
            num_draws_emax = np.random.randint(1, 100)

            # Perform toolbox actions
            robupy_obj = read('test.robupy.ini')

            robupy_obj.unlock()

            robupy_obj.set_attr('num_draws_emax', num_draws_emax)

            robupy_obj.lock()

            robupy_obj = solve(robupy_obj)

            # Distribute class attributes
            periods_emax = robupy_obj.get_attr('periods_emax')

            if base is None:
                base = periods_emax.copy()

            # Statistic
            diff = np.max(abs(np.ma.masked_invalid(base) - np.ma.masked_invalid(
                periods_emax)))

            # Checks
            assert (np.isfinite(diff))
            assert (diff < 10e-10)

    def test_5(self):
        """ Testing whether the risk code is identical to the ambiguity code for
        very, very small levels of ambiguity.
        """
        # Generate random initialization dictionary
        constraints = dict()

        init_dict = generate_random_dict(constraints)

        # Initialize containers
        base = None

        # Loop over different uncertain environments.
        for level in [0.00, 0.000000000000001]:

            # Set varying constraints
            init_dict['AMBIGUITY']['level'] = level

            # Print to dictionary
            print_random_dict(init_dict)

            # Perform toolbox actions
            robupy_obj = read('test.robupy.ini')

            robupy_obj = solve(robupy_obj)

            # Distribute class attributes
            periods_emax = robupy_obj.get_attr('periods_emax')

            if base is None:
                base = periods_emax.copy()

            # Checks
            np.testing.assert_allclose(base, periods_emax, rtol=1e-06)

    def test_6(self):
        """ Testing whether the systematic payoff calculation is unaffected by
        the level of ambiguity.
        """
        # Select version

        # Generate constraints
        constraints = dict()

        # Generate random initialization dictionary
        init_dict = generate_random_dict(constraints)

        # Initialize containers
        base = None

        # Loop over different uncertain environments.
        for _ in range(2):

            # Set varying constraints
            init_dict['AMBIGUITY']['level'] = np.random.choice(
                [0.00, np.random.uniform()])

            # Print to dictionary
            print_random_dict(init_dict)

            # Perform toolbox actions
            robupy_obj = read('test.robupy.ini')

            robupy_obj = solve(robupy_obj)

            # Distribute class attributes
            systematic = robupy_obj.get_attr('periods_payoffs_systematic')

            if base is None:
                base = systematic.copy()

            # Checks
            np.testing.assert_allclose(base, systematic)