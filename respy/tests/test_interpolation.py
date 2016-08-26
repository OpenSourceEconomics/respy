import numpy as np
import pytest

from respy.python.shared.shared_auxiliary import dist_class_attributes
from respy.python.shared.shared_auxiliary import print_init_dict
from codes.random_init import generate_random_dict
from respy import RespyCls
from respy import simulate


@pytest.mark.usefixtures('fresh_directory', 'set_seed')
class TestClass(object):
    """ This class groups together some tests.
    """
    def test_1(self):
        """ This is the special case where the EMAX better be equal to the MAXE.
        """
        # Set initial constraints
        constraints = dict()
        constraints['flag_interpolation'] = False
        constraints['periods'] = np.random.randint(3, 6)
        constraints['is_deterministic'] = True

        # Initialize request
        init_dict = generate_random_dict(constraints)
        baseline = None

        # Solve with and without interpolation code
        for _ in range(2):

            # Write out request
            print_init_dict(init_dict)

            # Process and solve
            respy_obj = RespyCls('test.respy.ini')
            respy_obj = simulate(respy_obj)

            # Extract class attributes
            states_number_period, periods_emax = \
                dist_class_attributes(respy_obj,
                    'states_number_period', 'periods_emax')

            # Store and check results
            if baseline is None:
                baseline = periods_emax
            else:
                np.testing.assert_array_almost_equal(baseline, periods_emax)

            # Updates for second iteration. This ensures that there is at least
            # one interpolation taking place.
            init_dict['INTERPOLATION']['points'] = max(states_number_period) - 1
            init_dict['INTERPOLATION']['flag'] = True

    def test_2(self):
        """ This test compares the results from a solution using the
        interpolation code for the special case where the number of interpolation
        points is exactly the number of states in the final period. In this case
        the interpolation code is run and then all predicted values replaced
        with their actual values.
        """
        # Set initial constraints
        constraints = dict()
        constraints['flag_interpolation'] = False
        constraints['periods'] = np.random.randint(3, 6)

        # Initialize request
        init_dict = generate_random_dict(constraints)
        baseline = None

        # Solve with and without interpolation code
        for _ in range(2):

            # Write out request
            print_init_dict(init_dict)

            # Process and solve
            respy_obj = RespyCls('test.respy.ini')
            respy_obj = simulate(respy_obj)

            # Extract class attributes
            states_number_period, periods_emax = \
                dist_class_attributes(respy_obj,
                    'states_number_period', 'periods_emax')

            # Store and check results
            if baseline is None:
                baseline = periods_emax
            else:
                np.testing.assert_array_almost_equal(baseline, periods_emax)

            # Updates for second iteration
            init_dict['INTERPOLATION']['points'] = max(states_number_period)
            init_dict['INTERPOLATION']['flag'] = True

