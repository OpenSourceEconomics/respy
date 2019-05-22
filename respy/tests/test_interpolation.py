"""Test the interpolation routine."""
import numpy as np

from respy.pre_processing.model_processing import process_model_spec
from respy.python.interface import minimal_simulation_interface
from respy.tests.codes.random_model import generate_random_model


def test_equality_of_full_and_interpolated_solution():
    """Test the equality of the full and interpolated solution.

    If an interpolation is requested but the number of states used for the interpolation
    is equal to the total number of states in the period, the simulated expected maximum
    utility should be equal to the full solution.

    """
    # Get results from full solution.
    constr = {"interpolation": {"flag": False}}
    params_spec, options_spec = generate_random_model(point_constr=constr)
    attr = process_model_spec(params_spec, options_spec)

    state_space, _ = minimal_simulation_interface(attr)
    emaxs_full = state_space.emaxs

    # Get results from interpolated solution.
    options_spec["interpolation"]["points"] = max(state_space.states_per_period)
    options_spec["interpolation"]["flag"] = True
    attr = process_model_spec(params_spec, options_spec)

    state_space, _ = minimal_simulation_interface(attr)
    emaxs_interpolated = state_space.emaxs

    np.testing.assert_array_almost_equal(emaxs_full, emaxs_interpolated)
