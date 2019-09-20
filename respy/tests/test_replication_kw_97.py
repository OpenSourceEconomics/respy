import numpy as np
import pytest

import respy as rp


@pytest.mark.wip
@pytest.mark.parametrize(
    "model, type_proportions",
    [
        (
            "kw_97_basic",
            ([0.1751, 0.2396, 0.5015, 0.0838], [0.0386, 0.4409, 0.4876, 0.0329]),
        ),
        (
            "kw_97_extended",
            ([0.0491, 0.1987, 0.4066, 0.3456], [0.2343, 0.2335, 0.3734, 0.1588]),
        ),
    ],
)
def test_type_proportions(model, type_proportions):
    nine_years_or_less = type_proportions[0]
    ten_years_or_more = type_proportions[1]

    params, options = rp.get_example_model(model, with_data=False)

    options["n_periods"] = 1
    options["simulated_agents"] = 10_000

    simulate = rp.get_simulate_func(params, options)

    df = simulate(params)

    np.testing.assert_allclose(
        df.loc[df.Experience_School.le(9), "Type"]
        .value_counts(normalize=True, sort=False)
        .sort_index(),
        nine_years_or_less,
        atol=0.05,
    )

    np.testing.assert_allclose(
        df.loc[df.Experience_School.ge(10), "Type"]
        .value_counts(normalize=True, sort=False)
        .sort_index(),
        ten_years_or_more,
        atol=0.05,
    )
