
""" This modules contains some additional tests that are only used in
long-run development tests.
"""

# standard library
from pandas.util.testing import assert_frame_equal
import pandas as pd
import sys
import os

# project library
from modules.auxiliary import compile_package


# ROBUPY import
sys.path.insert(0, os.environ['ROBUPY'])
from robupy.tests.random_init import generate_random_dict, print_random_dict

''' Main
'''
def test_99():
    """ Testing whether the results from a fast and slow execution of the
    code result in identical simulate datasets.
    """
    # Set up constraints
    compile_package('fast')

    # Constraint to risk model
    constraints = dict()
    constraints['level'] = 0.0

    # Generate random initialization
    init_dict = generate_random_dict(constraints)

    # Initialize containers
    base = None

    for fast in ['True', 'False']:

        # Prepare initialization file
        init_dict['SOLUTION']['fast'] = fast

        print_random_dict(init_dict)

        # Simulate the ROBUPY package
        os.system('robupy-solve --simulate --model test.robupy.ini')

        # Load simulated data frame
        data_frame = pd.read_csv('data.robupy.dat')

        # Compare
        if base is None:
            base = data_frame.copy()

        assert_frame_equal(base, data_frame)