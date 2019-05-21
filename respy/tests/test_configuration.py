"""Test the configuration."""
import os

import pytest


@pytest.mark.parametrize("env_value, choices", [(True, [1, 2, 3, 4]), (False, [2])])
def test_keane_wolpin_squared_experiences(env_value, choices):
    """Test the environment variable ``RESPY_KW_SQUARED_EXPERIENCES``.

    We run the the first example, kw_data_one, with the environment variable set to
    ``True`` and to ``False``. If the variable evaluates to ``True``, all choices are
    observed in the simulated data. Otherwise, individuals will only choose occupation
    B.

    """
    os.environ["RESPY_KW_SQUARED_EXPERIENCES"] = f"{env_value}"

    # Late import ensures that the environment variable is read.
    import respy as rp

    options_spec, params_spec = rp.get_example_model("kw_data_one")

    model = rp.RespyCls(params_spec, options_spec)

    model, df = model.simulate()

    assert df.Choice.isin(choices).all()
