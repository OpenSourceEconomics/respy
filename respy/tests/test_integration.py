import numpy as np

from respy.likelihood import get_crit_func
from respy.tests.random_model import generate_random_model
from respy.tests.random_model import simulate_truncated_data


def test_simulation_and_estimation_with_different_models():
    """Test the evaluation of the criterion function not at the true parameters."""
    # Set constraints.
    num_agents = np.random.randint(5, 100)
    constr = {
        "simulation": {"agents": num_agents},
        "num_periods": np.random.randint(1, 4),
        "edu_spec": {"start": [7], "max": 15, "share": [1.0]},
        "estimation": {"agents": num_agents},
    }

    # Simulate a dataset
    params_spec, options_spec = generate_random_model(point_constr=constr)
    df = simulate_truncated_data(params_spec, options_spec)

    # Evaluate at different points, ensuring that the simulated dataset still fits.
    params_spec, options_spec = generate_random_model(point_constr=constr)
    crit_func = get_crit_func(params_spec, options_spec, df)

    crit_func(params_spec)


def test_invariant_results_for_two_estimations():
    num_agents = np.random.randint(5, 100)
    constr = {
        "simulation": {"agents": num_agents},
        "num_periods": np.random.randint(1, 4),
        "estimation": {"agents": num_agents},
    }

    # Simulate a dataset.
    params_spec, options_spec = generate_random_model(point_constr=constr)
    df = simulate_truncated_data(params_spec, options_spec)

    crit_func = get_crit_func(params_spec, options_spec, df)

    # First estimation.
    crit_val = crit_func(params_spec)

    # Second estimation.
    crit_val_ = crit_func(params_spec)

    assert crit_val == crit_val_


def test_invariance_to_initial_conditions():
    """Test invariance to initial conditions.

    We ensure that the number of initial conditions does not matter for the evaluation
    of the criterion function if a weight of one is put on the first group.

    """
    num_agents = np.random.randint(5, 100)
    constr = {
        "simulation": {"agents": num_agents},
        "num_periods": np.random.randint(1, 4),
        "edu_spec": {"max": np.random.randint(15, 25, size=1).tolist()[0]},
        "estimation": {"agents": num_agents},
        "interpolation": {"flag": False},
    }

    params_spec, options_spec = generate_random_model(point_constr=constr)
    df = simulate_truncated_data(params_spec, options_spec)

    edu_start_base = np.random.randint(1, 5, size=1).tolist()[0]

    # We need to ensure that the initial lagged activity always has the same
    # distribution.
    edu_lagged_base = np.random.uniform(size=5).tolist()

    likelihoods = []

    for num_edu_start in [1, 2, 3, 4]:

        # We always need to ensure that a weight of one is on the first level of
        # initial schooling.
        options_spec["edu_spec"]["share"] = [1.0] + [0.0] * (num_edu_start - 1)
        options_spec["edu_spec"]["lagged"] = edu_lagged_base[:num_edu_start]

        # We need to make sure that the baseline level of initial schooling is
        # always included. At the same time we cannot have any duplicates.
        edu_start = np.random.choice(
            range(1, 10), size=num_edu_start, replace=False
        ).tolist()
        if edu_start_base in edu_start:
            edu_start.remove(edu_start_base)
            edu_start.insert(0, edu_start_base)
        else:
            edu_start[0] = edu_start_base

        options_spec["edu_spec"]["start"] = edu_start

        df = simulate_truncated_data(params_spec, options_spec)

        crit_func = get_crit_func(params_spec, options_spec, df)

        likelihood = crit_func(params_spec)

        likelihoods.append(likelihood)

    assert np.equal.reduce(likelihoods)
