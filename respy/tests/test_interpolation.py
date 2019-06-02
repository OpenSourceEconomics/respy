"""Test the interpolation routine."""
import numpy as np

from respy.simulate import simulate
from respy.tests.random_model import generate_random_model


def test_equality_of_full_and_interpolated_solution():
    """Test the equality of the full and interpolated solution.

    If an interpolation is requested but the number of states used for the interpolation
    is equal to the total number of states in the period, the simulated expected maximum
    utility should be equal to the full solution.

    """
    # Get results from full solution.
    constr = {"interpolation_points": -1}
    params, options = generate_random_model(point_constr=constr)

    state_space, _ = simulate(params, options)
    emaxs_full = state_space.emaxs

    # Get results from interpolated solution.
    options["interpolation_points"] = max(state_space.states_per_period)

    state_space, _ = simulate(params, options)
    emaxs_interpolated = state_space.emaxs

    np.testing.assert_array_almost_equal(emaxs_full, emaxs_interpolated)
