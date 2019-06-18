import numpy as np

from respy.likelihood import get_crit_func
from respy.tests.random_model import generate_random_model
from respy.tests.random_model import simulate_truncated_data


def test_simulation_and_estimation_with_different_models():
    """Test the evaluation of the criterion function not at the true parameters."""
    # Set constraints.
    num_agents = np.random.randint(5, 100)
    constr = {
        "simulation_agents": num_agents,
        "n_periods": np.random.randint(1, 4),
        "choices": {
            "edu": {"start": [7], "max": 15, "share": [1.0], "has_experience": True}
        },
    }

    # Simulate a dataset
    params, options = generate_random_model(point_constr=constr)
    df = simulate_truncated_data(params, options)

    # Evaluate at different points, ensuring that the simulated dataset still fits.
    params, options = generate_random_model(point_constr=constr)
    crit_func = get_crit_func(params, options, df)

    crit_func(params)


def test_invariant_results_for_two_estimations():
    num_agents = np.random.randint(5, 100)
    constr = {"simulation_agents": num_agents, "n_periods": np.random.randint(1, 4)}

    # Simulate a dataset.
    params, options = generate_random_model(point_constr=constr)
    df = simulate_truncated_data(params, options)

    crit_func = get_crit_func(params, options, df)

    # First estimation.
    crit_val = crit_func(params)

    # Second estimation.
    crit_val_ = crit_func(params)

    assert crit_val == crit_val_
