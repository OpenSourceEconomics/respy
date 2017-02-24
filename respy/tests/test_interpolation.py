import numpy as np
import pytest

from respy.python.shared.shared_auxiliary import dist_class_attributes
from respy.python.shared.shared_auxiliary import print_init_dict
from codes.random_init import generate_random_dict
from codes.auxiliary import simulate_observed
from respy import RespyCls


@pytest.mark.usefixtures('fresh_directory', 'set_seed')
class TestClass(object):
    """ This class groups together some tests.
    """
    def test_1(self):
        """ This is the special case where the EMAX better be equal to the MAXE.
        """
        # Set initial constraints
        constr = dict()
        constr['flag_interpolation'] = False
        constr['periods'] = np.random.randint(3, 6)
        constr['flag_deterministic'] = True
        constr['level'] = 0.00

        # Initialize request
        init_dict = generate_random_dict(constr)
        baseline = None

        # Solve with and without interpolation code
        for _ in range(2):

            # Write out request
            print_init_dict(init_dict)

            # Process and solve
            respy_obj = RespyCls('test.respy.ini')
            respy_obj = simulate_observed(respy_obj)

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
        constr = dict()
        constr['flag_interpolation'] = False
        constr['periods'] = np.random.randint(3, 6)

        # Initialize request
        init_dict = generate_random_dict(constr)
        baseline = None

        # Solve with and without interpolation code
        for _ in range(2):

            # Write out request
            print_init_dict(init_dict)

            # Process and solve
            respy_obj = RespyCls('test.respy.ini')
            respy_obj = simulate_observed(respy_obj)

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

