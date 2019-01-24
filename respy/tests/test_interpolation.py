import numpy as np

from respy.python.shared.shared_auxiliary import dist_class_attributes
from respy.tests.codes.random_model import generate_random_model
from respy.tests.codes.auxiliary import simulate_observed
from respy import RespyCls


class TestClass(object):
    """ This class groups together some tests.
    """

    def test_1(self):
        """ This is the special case where the EMAX better be equal to the MAXE.
        """
        # Set initial constraints
        constr = {
            "interpolation": {"flag": False},
            "num_periods": np.random.randint(3, 6)
        }

        params_spec, options_spec = generate_random_model(point_constr=constr, deterministic=True)

        baseline = None

        # Solve with and without interpolation code
        for _ in range(2):
            respy_obj = RespyCls(params_spec, options_spec)
            respy_obj = simulate_observed(respy_obj)

            # Extract class attributes
            states_number_period, periods_emax = dist_class_attributes(
                respy_obj, "states_number_period", "periods_emax"
            )

            # Store and check results
            if baseline is None:
                baseline = periods_emax
            else:
                np.testing.assert_array_almost_equal(baseline, periods_emax)

            # Updates for second iteration. This ensures that there is at least one interpolation
            #  taking place.
            options_spec["interpolation"]["points"] = max(states_number_period)
            options_spec["interpolation"]["flag"] = True

    def test_2(self):
        """ This test compares the results from a solution using the interpolation code for the
        special case where the number of interpolation points is exactly the number of states in
        the final period. In this case the interpolation code is run and then all predicted
        values replaced with their actual values.
        """
        # Set initial constraints
        # Set initial constraints
        constr = {"interpolation": {"flag": False}}

        params_spec, options_spec = generate_random_model(point_constr=constr,
                                                          deterministic=True)
        baseline = None

        # Solve with and without interpolation code
        for _ in range(2):


            # Process and solve
            respy_obj = RespyCls(params_spec, options_spec)
            respy_obj = simulate_observed(respy_obj)

            # Extract class attributes
            states_number_period, periods_emax = dist_class_attributes(
                respy_obj, "states_number_period", "periods_emax"
            )

            # Store and check results
            if baseline is None:
                baseline = periods_emax
            else:
                np.testing.assert_array_almost_equal(baseline, periods_emax)

            # Updates for second iteration
            options_spec["interpolation"]["points"] = max(states_number_period)
            options_spec["interpolation"]["flag"] = True
