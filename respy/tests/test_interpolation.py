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
    constr = {"interpolation": {"flag": False}}
    params_spec, options_spec = generate_random_model(point_constr=constr)

    state_space, _ = simulate(params_spec, options_spec)
    emaxs_full = state_space.emaxs

    # Get results from interpolated solution.
    options_spec["interpolation"]["points"] = max(state_space.states_per_period)
    options_spec["interpolation"]["flag"] = True

    state_space, _ = simulate(params_spec, options_spec)
    emaxs_interpolated = state_space.emaxs

    np.testing.assert_array_almost_equal(emaxs_full, emaxs_interpolated)
