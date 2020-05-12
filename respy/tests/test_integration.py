import pytest

from respy.likelihood import get_crit_func
from respy.tests.random_model import add_noise_to_params
from respy.tests.random_model import generate_random_model
from respy.tests.random_model import simulate_truncated_data


@pytest.mark.xfail
@pytest.mark.end_to_end
def test_simulation_and_estimation_with_different_models():
    """Test the evaluation of the criterion function not at the true parameters."""
    # Simulate a dataset
    params, options = generate_random_model()
    df = simulate_truncated_data(params, options)

    # Evaluate at different points, ensuring that the simulated dataset still fits.
    crit_func = get_crit_func(params, options, df)
    params_ = add_noise_to_params(params, options)

    params.equals(params_)

    crit_func(params)


@pytest.mark.xfail
@pytest.mark.end_to_end
def test_invariant_results_for_two_estimations():
    params, options = generate_random_model()
    df = simulate_truncated_data(params, options)

    crit_func = get_crit_func(params, options, df)

    # First estimation.
    crit_val = crit_func(params)

    # Second estimation.
    crit_val_ = crit_func(params)

    assert crit_val == crit_val_
