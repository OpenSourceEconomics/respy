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
        "num_periods": np.random.randint(1, 4),
        "sectors": {
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
    constr = {"simulation_agents": num_agents, "num_periods": np.random.randint(1, 4)}

    # Simulate a dataset.
    params, options = generate_random_model(point_constr=constr)
    df = simulate_truncated_data(params, options)

    crit_func = get_crit_func(params, options, df)

    # First estimation.
    crit_val = crit_func(params)

    # Second estimation.
    crit_val_ = crit_func(params)

    assert crit_val == crit_val_


def test_invariance_to_initial_conditions():
    """Test invariance to initial conditions.

    We ensure that the number of initial conditions does not matter for the evaluation
    of the criterion function if a weight of one is put on the first group.

    """
    num_agents = np.random.randint(5, 100)
    constr = {
        "simulation_agents": num_agents,
        "num_periods": np.random.randint(1, 4),
        "sectors": {
            "edu": {"max": np.random.randint(15, 25, size=1)[0], "has_experience": True}
        },
        "interpolation_points": -1,
    }

    params, options = generate_random_model(point_constr=constr)

    df = simulate_truncated_data(params, options)

    edu_start_base = np.random.randint(1, 5, size=1)[0]

    # We need to ensure that the initial lagged activity always has the same
    # distribution.
    edu_lagged_base = np.random.uniform(size=5)

    likelihoods = []

    for num_edu_start in [1, 2, 3, 4]:

        # We always need to ensure that a weight of one is on the first level of
        # initial schooling.
        options["sectors"]["edu"]["share"] = [1.0] + [0.0] * (num_edu_start - 1)
        options["sectors"]["edu"]["lagged"] = edu_lagged_base[:num_edu_start]

        # We need to make sure that the baseline level of initial schooling is
        # always included. At the same time we cannot have any duplicates.
        edu_start = np.random.choice(range(1, 10), size=num_edu_start, replace=False)
        if edu_start_base in edu_start:
            edu_start = edu_start[edu_start != edu_start_base]
            edu_start = np.append(edu_start_base, edu_start)
        else:
            edu_start[0] = edu_start_base

        options["sectors"]["edu"]["start"] = edu_start

        df = simulate_truncated_data(params, options)

        crit_func = get_crit_func(params, options, df)

        likelihood = crit_func(params)

        likelihoods.append(likelihood)

    assert np.equal.reduce(likelihoods)
