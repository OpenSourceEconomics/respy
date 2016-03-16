""" This modules contains some additional tests that are only used in long-run
development tests.
"""

# standard library
from pandas.util.testing import assert_frame_equal

import pandas as pd
import numpy as np

import pytest
import sys
import os

# testing library
from material.auxiliary import write_interpolation_grid
from material.auxiliary import write_disturbances

# ROBUPY import
sys.path.insert(0, os.environ['ROBUPY'])
from robupy import simulate
from robupy import evaluate
from robupy import process
from robupy import solve
from robupy import read

from robupy.tests.random_init import generate_random_dict
from robupy.tests.random_init import print_random_dict
from robupy.tests.random_init import generate_init

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

