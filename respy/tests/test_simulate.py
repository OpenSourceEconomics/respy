"""Test the simulation routine."""
import numpy as np
import pandas as pd
import pytest

import respy as rp
from respy.likelihood import get_crit_func
from respy.tests.random_model import generate_random_model


@pytest.mark.parametrize("seed", range(20))
def test_equality_for_myopic_agents_and_tiny_delta(seed):
    """Test equality of simulated data and likelihood with myopia and tiny delta."""
    np.random.seed(seed)

    # Get simulated data and likelihood for myopic model.
    params, options = generate_random_model(myopic=True)

    simulate = rp.get_simulate_func(params, options)
    df = simulate(params)

    crit_func = get_crit_func(params, options, df)
    likelihood = crit_func(params)

    # Get simulated data and likelihood for model with tiny delta.
    params.loc["delta", "value"] = 1e-12

    df_ = simulate(params)

    crit_func_ = rp.get_crit_func(params, options, df_)
    likelihood_ = crit_func_(params)

    pd.testing.assert_frame_equal(df, df_)
    np.testing.assert_almost_equal(likelihood, likelihood_, decimal=12)
